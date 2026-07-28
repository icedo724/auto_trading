"""파라미터 그리드 탐색.

핵심 규칙 — **모든 후보는 완전히 동일한 조건에서 경쟁한다.**

  · 같은 종목 유니버스, 같은 거래일 달력
  · 같은 평가 시작일 (모든 후보의 지표 워밍업 중 가장 긴 값으로 통일)
  · 같은 거래비용/체결 규칙(BacktestConfig)

워밍업을 통일하지 않으면 "MA5 전략은 5일째부터, MA200 전략은 200일째부터" 평가되어
서로 다른 시장 구간을 비교하게 된다. 이 모듈은 그 함정을 구조적으로 차단한다.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .engine import Backtester
from .metrics import compute_portfolio_metrics
from .strategy import Strategy, build_candidates, create_strategy

# --------------------------------------------------------------------------------
# 목적함수
# --------------------------------------------------------------------------------
Objective = Callable[[dict[str, float]], float]

#: 최소거래수 필터에 걸린 후보의 점수 (항상 순위 최하위로 밀린다)
FILTERED_SCORE = float("-inf")


def _finite(x: float) -> float:
    return float(x) if np.isfinite(x) else -1e9


def obj_sharpe(m: dict[str, float]) -> float:
    return _finite(m.get("sharpe", 0.0))


def obj_calmar(m: dict[str, float]) -> float:
    return _finite(m.get("calmar", 0.0))


def obj_cagr(m: dict[str, float]) -> float:
    return _finite(m.get("cagr", 0.0))


def obj_sortino(m: dict[str, float]) -> float:
    return _finite(m.get("sortino", 0.0))


def obj_robust(m: dict[str, float]) -> float:
    """과최적화 저항형 복합 점수 (기본값).

    Sharpe 를 뼈대로 하되,
      · 표본이 적은(거래 수가 적은) 후보에 신뢰도 할인
      · MDD 가 깊을수록 감점
      · 회전율이 과도하면 감점 (비용/체결 리스크 민감도)
    """
    sharpe = _finite(m.get("sharpe", 0.0))
    n = m.get("n_trades", 0.0)
    mdd = abs(m.get("max_drawdown", 0.0))
    turnover = m.get("turnover", 0.0)

    confidence = math.sqrt(min(n, 30.0) / 30.0)  # 30거래 이상이면 온전한 신뢰
    dd_penalty = 1.0 / (1.0 + 3.0 * max(mdd - 0.20, 0.0))  # MDD 20% 초과분에 감점
    to_penalty = 1.0 / (1.0 + 0.02 * max(turnover - 12.0, 0.0))  # 월 1회전 초과분에 감점
    return sharpe * confidence * dd_penalty * to_penalty


OBJECTIVES: dict[str, Objective] = {
    "robust": obj_robust,
    "sharpe": obj_sharpe,
    "sortino": obj_sortino,
    "calmar": obj_calmar,
    "cagr": obj_cagr,
}


def get_objective(name: str) -> Objective:
    if name not in OBJECTIVES:
        raise ValueError(f"알 수 없는 목적함수: {name!r} (가능: {sorted(OBJECTIVES)})")
    return OBJECTIVES[name]


# --------------------------------------------------------------------------------
# 결과 컨테이너
# --------------------------------------------------------------------------------
@dataclass
class EvalResult:
    """후보 1개의 평가 결과."""

    strategy: str
    params: dict[str, Any]
    score: float
    metrics: dict[str, float]
    per_symbol: dict[str, dict[str, float]] = field(default_factory=dict)
    equity: pd.Series | None = None
    #: 최소거래수 미달로 순위에서 배제되었는지 여부
    filtered: bool = False

    @property
    def label(self) -> str:
        body = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.strategy}({body})"

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"strategy": self.strategy, "label": self.label, "score": self.score}
        row.update({f"param_{k}": v for k, v in sorted(self.params.items())})
        row.update(self.metrics)
        return row


@dataclass
class OptimizationReport:
    """그리드 탐색 전체 결과."""

    results: list[EvalResult]
    objective: str
    trade_start: pd.Timestamp
    trade_end: pd.Timestamp
    symbols: list[str]
    config: BacktestConfig
    elapsed_sec: float = 0.0

    @property
    def best(self) -> EvalResult:
        """최고 점수 후보. 필터를 통과한 후보가 하나도 없으면 예외."""
        for r in self.results:
            if not r.filtered:
                return r
        raise ValueError(
            "최소거래수 필터를 통과한 후보가 없습니다. min_trades 를 낮추거나 "
            "평가 기간을 늘리세요."
        )

    def leaderboard(self, top: int | None = None) -> pd.DataFrame:
        rows = [r.to_row() for r in self.results[: top or len(self.results)]]
        df = pd.DataFrame(rows)
        return df.reset_index(drop=True)

    def best_per_strategy(self) -> list[EvalResult]:
        seen: set[str] = set()
        out: list[EvalResult] = []
        for r in self.results:  # 이미 점수 내림차순
            if r.strategy not in seen:
                seen.add(r.strategy)
                out.append(r)
        return out


# --------------------------------------------------------------------------------
# 후보 생성
# --------------------------------------------------------------------------------
def build_all_candidates(
    strategies: Sequence[str] | None = None,
    grids: dict[str, dict[str, list[Any]]] | None = None,
) -> list[Strategy]:
    """전략 이름 목록(+선택적 커스텀 그리드)에서 후보 전략 인스턴스를 만든다."""
    from .strategy import available_strategies

    names = list(strategies) if strategies else available_strategies()
    grids = grids or {}
    out: list[Strategy] = []
    for name in names:
        out.extend(build_candidates(name, grids.get(name)))
    if not out:
        raise ValueError("생성된 후보가 없습니다. 전략/그리드 설정을 확인하세요.")
    return out


def common_trade_start(
    calendar: pd.DatetimeIndex, candidates: Iterable[Strategy], *, extra_bars: int = 0
) -> pd.Timestamp:
    """모든 후보가 지표를 확보한 이후의 **공통 평가 시작일**."""
    warmup = max([c.warmup for c in candidates] + [0]) + extra_bars
    if warmup >= len(calendar):
        raise ValueError(
            f"워밍업({warmup} 봉)이 데이터 길이({len(calendar)} 봉)보다 깁니다. "
            "기간을 늘리거나 장기 파라미터를 제외하세요."
        )
    return calendar[warmup]


# --------------------------------------------------------------------------------
# 평가
# --------------------------------------------------------------------------------
def evaluate_candidate(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    config: BacktestConfig,
    trade_start: pd.Timestamp,
    objective: Objective,
    *,
    store_equity: bool = False,
    min_trades: int = 0,
) -> EvalResult:
    """후보 1개를 유니버스 전체에 적용하고 동일비중 포트폴리오로 평가."""
    engine = Backtester(config)
    results = {}
    for sym, df in data.items():
        sig = strategy.generate_signals(df)
        results[sym] = engine.run(df, sig, symbol=sym, trade_start=trade_start)

    equity, metrics = compute_portfolio_metrics(results, config)
    score = objective(metrics)

    # 벤치마크(예: buy_and_hold)는 거래가 적은 것이 정상이므로 필터에서 제외한다.
    filtered = not strategy.is_benchmark and metrics.get("n_trades", 0.0) < min_trades
    if filtered:
        score = FILTERED_SCORE  # 표본 부족 후보는 순위에서 배제

    return EvalResult(
        strategy=strategy.name,
        params=dict(strategy.params),
        score=score,
        metrics=metrics,
        per_symbol={s: r.metrics for s, r in results.items()},
        equity=equity if store_equity else None,
        filtered=filtered,
    )


# --- 멀티프로세싱 워커 -------------------------------------------------------------
_WORKER: dict[str, Any] = {}


def _init_worker(data, config_dict, trade_start, objective_name, min_trades) -> None:
    _WORKER["data"] = data
    _WORKER["config"] = BacktestConfig.from_dict(config_dict)
    _WORKER["trade_start"] = trade_start
    _WORKER["objective"] = get_objective(objective_name)
    _WORKER["min_trades"] = min_trades


def _eval_worker(spec: tuple[str, dict[str, Any]]) -> EvalResult:
    name, params = spec
    return evaluate_candidate(
        create_strategy(name, params),
        _WORKER["data"],
        _WORKER["config"],
        _WORKER["trade_start"],
        _WORKER["objective"],
        store_equity=False,
        min_trades=_WORKER["min_trades"],
    )


# --------------------------------------------------------------------------------
# 그리드 탐색
# --------------------------------------------------------------------------------
def grid_search(
    data: dict[str, pd.DataFrame],
    candidates: Sequence[Strategy],
    config: BacktestConfig,
    *,
    objective: str = "robust",
    trade_start: pd.Timestamp | None = None,
    min_trades: int = 5,
    workers: int | None = None,
    store_equity_top: int = 10,
    progress: bool = True,
) -> OptimizationReport:
    """모든 후보를 동일 조건에서 평가하고 점수순으로 정렬해 반환한다."""
    if not data:
        raise ValueError("데이터가 비어 있습니다.")
    config.validate()

    calendar = next(iter(data.values())).index
    for sym, df in data.items():
        if not df.index.equals(calendar):
            raise ValueError(
                f"{sym}: 거래일 달력이 다릅니다. data.load_universe(align=True)를 사용하세요."
            )

    if trade_start is None:
        trade_start = common_trade_start(calendar, candidates)
    obj_fn = get_objective(objective)

    t0 = time.perf_counter()
    specs = [(c.name, dict(c.params)) for c in candidates]
    n = len(specs)
    if workers is None:
        workers = max(1, min(os.cpu_count() or 1, 8))

    results: list[EvalResult] = []
    if workers > 1 and n > 4:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(data, config.to_dict(), trade_start, objective, min_trades),
        ) as pool:
            for i, res in enumerate(pool.map(_eval_worker, specs, chunksize=8), 1):
                results.append(res)
                if progress and (i % 25 == 0 or i == n):
                    _print_progress(i, n, t0)
    else:
        for i, c in enumerate(candidates, 1):
            results.append(
                evaluate_candidate(
                    c, data, config, trade_start, obj_fn, min_trades=min_trades
                )
            )
            if progress and (i % 25 == 0 or i == n):
                _print_progress(i, n, t0)

    results.sort(key=lambda r: r.score, reverse=True)

    # 상위 후보만 자산곡선을 다시 계산해 보관 (메모리 절약, 결과는 결정적이므로 동일)
    for r in results[:store_equity_top]:
        full = evaluate_candidate(
            create_strategy(r.strategy, r.params),
            data, config, trade_start, obj_fn, store_equity=True, min_trades=min_trades,
        )
        r.equity = full.equity

    return OptimizationReport(
        results=results,
        objective=objective,
        trade_start=pd.Timestamp(trade_start),
        trade_end=pd.Timestamp(calendar[-1]),
        symbols=sorted(data),
        config=config,
        elapsed_sec=time.perf_counter() - t0,
    )


def _print_progress(i: int, n: int, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    rate = i / elapsed if elapsed > 0 else 0.0
    eta = (n - i) / rate if rate > 0 else 0.0
    print(
        f"\r  평가 {i}/{n} ({100 * i / n:5.1f}%)  {rate:5.1f} 조합/초  ETA {eta:5.0f}s",
        end="",
        flush=True,
    )
    if i == n:
        print()


# --------------------------------------------------------------------------------
# 파라미터 민감도 (과최적화 진단)
# --------------------------------------------------------------------------------
def sensitivity(
    report: OptimizationReport, strategy: str, param: str
) -> pd.DataFrame:
    """특정 파라미터 값별 점수 분포.

    최적값 하나만 뾰족하게 튀어나오면 과최적화 신호,
    이웃 값들도 고르게 좋으면 견고한 영역(plateau)일 가능성이 높다.
    """
    rows = [
        {"value": r.params[param], "score": r.score, **r.metrics}
        for r in report.results
        if r.strategy == strategy and param in r.params and not r.filtered
    ]
    if not rows:
        raise ValueError(f"{strategy}.{param} 에 해당하는 결과가 없습니다.")
    df = pd.DataFrame(rows)
    return (
        df.groupby("value")
        .agg(
            n=("score", "size"),
            score_mean=("score", "mean"),
            score_max=("score", "max"),
            score_std=("score", "std"),
            cagr_mean=("cagr", "mean"),
            mdd_mean=("max_drawdown", "mean"),
        )
        .reset_index()
        .sort_values("score_mean", ascending=False)
    )
