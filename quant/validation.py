"""과최적화 검증 도구.

그리드 탐색의 1위는 "그 구간에서 가장 운이 좋았던 조합"일 수 있다.
여기서는 두 가지 방법으로 실전 생존 가능성을 점검한다.

1. **In-Sample / Out-of-Sample 분할** — 앞 구간에서 고르고 뒤 구간에서 검증
2. **Walk-Forward 분석** — 학습창을 굴리며 매번 다시 고르고, 이어붙인 OOS 곡선을 평가
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .metrics import compute_metrics, equity_from_returns
from .optimizer import (
    EvalResult,
    common_trade_start,
    evaluate_candidate,
    get_objective,
    grid_search,
)
from .strategy import Strategy, create_strategy


@dataclass
class HoldoutResult:
    """IS/OOS 검증 결과."""

    candidate: EvalResult  # IS 에서 선택된 후보 (IS 지표 포함)
    is_metrics: dict[str, float]
    oos_metrics: dict[str, float]
    split_date: pd.Timestamp
    oos_equity: pd.Series | None = None

    @property
    def degradation(self) -> dict[str, float]:
        """IS 대비 OOS 성과 감소폭. 값이 작을수록 견고하다."""
        out = {}
        for k in ("cagr", "sharpe", "calmar", "max_drawdown"):
            i, o = self.is_metrics.get(k, 0.0), self.oos_metrics.get(k, 0.0)
            out[k] = o - i
        return out

    @property
    def survived(self) -> bool:
        """OOS 에서도 실질적으로 작동했는가에 대한 간단한 판정."""
        return (
            self.oos_metrics.get("sharpe", 0.0) > 0.3
            and self.oos_metrics.get("cagr", 0.0) > 0.0
            and self.oos_metrics.get("n_trades", 0.0) >= 3
        )


def holdout_validate(
    data: dict[str, pd.DataFrame],
    candidates: Sequence[Strategy],
    config: BacktestConfig,
    split_date: str | pd.Timestamp,
    *,
    objective: str = "robust",
    top_n: int = 5,
    min_trades: int = 5,
    workers: int | None = None,
    progress: bool = False,
) -> list[HoldoutResult]:
    """학습 구간에서 상위 N개를 고른 뒤 검증 구간 성과를 측정한다."""
    ts = pd.Timestamp(split_date)
    calendar = next(iter(data.values())).index
    if not (calendar[0] < ts < calendar[-1]):
        raise ValueError(f"split_date({ts.date()})가 데이터 구간 밖입니다.")

    train = {s: df[df.index < ts] for s, df in data.items()}
    is_report = grid_search(
        train, candidates, config,
        objective=objective, min_trades=min_trades, workers=workers,
        store_equity_top=0, progress=progress,
    )

    obj_fn = get_objective(objective)
    out: list[HoldoutResult] = []
    for cand in is_report.results[:top_n]:
        if cand.filtered:
            continue
        strat = create_strategy(cand.strategy, cand.params)
        # 검증 구간: 전체 데이터를 주되 split_date 부터만 매매 (워밍업은 과거로 확보)
        oos = evaluate_candidate(
            strat, data, config, ts, obj_fn, store_equity=True, min_trades=0
        )
        out.append(
            HoldoutResult(
                candidate=cand,
                is_metrics=cand.metrics,
                oos_metrics=oos.metrics,
                split_date=ts,
                oos_equity=oos.equity,
            )
        )
    return out


@dataclass
class WalkForwardResult:
    """워크포워드 분석 결과."""

    windows: list[dict[str, Any]] = field(default_factory=list)
    oos_equity: pd.Series | None = None
    oos_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def efficiency(self) -> float:
        """워크포워드 효율 = OOS CAGR / IS CAGR 평균.

        1.0 근처면 견고, 0.5 미만이면 과최적화 의심, 음수면 폐기 대상.
        """
        is_cagr = np.mean([w["is_cagr"] for w in self.windows]) if self.windows else 0.0
        oos_cagr = self.oos_metrics.get("cagr", 0.0)
        if is_cagr <= 0:
            return 0.0
        return float(oos_cagr / is_cagr)

    def windows_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.windows)


def walk_forward(
    data: dict[str, pd.DataFrame],
    candidates: Sequence[Strategy],
    config: BacktestConfig,
    *,
    train_days: int = 504,  # 약 2년
    test_days: int = 126,  # 약 6개월
    objective: str = "robust",
    min_trades: int = 5,
    workers: int | None = None,
    progress: bool = True,
    selector: "Callable[[Any], Strategy] | None" = None,
) -> WalkForwardResult:
    """학습창을 굴리며 매 구간 전략을 재선택하고 OOS 수익률을 이어붙인다.

    ``selector`` 를 주면 "무엇을 고를지"를 바꿀 수 있다. 기본값은 점수 1등(argmax)
    이지만, 잡음 섞인 추정치의 최댓값은 편향된 추정량이므로 상위 K개 앙상블 같은
    대안이 OOS 에서 더 나을 수 있다. ``compare_selectors`` 로 직접 비교하라.
    """
    calendar = next(iter(data.values())).index
    obj_fn = get_objective(objective)
    warmup_start = common_trade_start(calendar, candidates)
    warmup_bars = int(calendar.searchsorted(warmup_start))

    result = WalkForwardResult()
    oos_chunks: list[pd.Series] = []

    start = warmup_bars
    window_no = 0
    while start + train_days + test_days <= len(calendar):
        train_end = start + train_days
        test_end = min(train_end + test_days, len(calendar))
        train_slice = {s: df.iloc[:train_end] for s, df in data.items()}
        test_start_date = calendar[train_end]
        test_end_date = calendar[test_end - 1]
        window_no += 1

        if progress:
            print(
                f"  [WF {window_no}] 학습 ~{calendar[train_end - 1].date()} / "
                f"검증 {test_start_date.date()}~{test_end_date.date()}"
            )

        is_report = grid_search(
            train_slice, candidates, config,
            objective=objective, trade_start=calendar[start],
            min_trades=min_trades, workers=workers, store_equity_top=0, progress=False,
        )
        try:
            best = is_report.best
        except ValueError:  # 이 학습창에서는 통과 후보 없음 -> 건너뛴다
            start += test_days
            continue

        strat = selector(is_report) if selector is not None else create_strategy(
            best.strategy, best.params
        )
        test_slice = {s: df.iloc[:test_end] for s, df in data.items()}
        oos = evaluate_candidate(
            strat, test_slice, config, test_start_date, obj_fn,
            store_equity=True, min_trades=0,
        )
        assert oos.equity is not None
        chunk = oos.equity.pct_change().fillna(0.0)
        oos_chunks.append(chunk)

        result.windows.append(
            {
                "window": window_no,
                "train_end": calendar[train_end - 1].strftime("%Y-%m-%d"),
                "test_start": test_start_date.strftime("%Y-%m-%d"),
                "test_end": test_end_date.strftime("%Y-%m-%d"),
                "selected": (
                    strat.describe() if selector is not None else best.label
                ),
                "is_score": round(best.score, 4),
                "is_cagr": round(best.metrics.get("cagr", 0.0), 4),
                "is_sharpe": round(best.metrics.get("sharpe", 0.0), 3),
                "oos_cagr": round(oos.metrics.get("cagr", 0.0), 4),
                "oos_sharpe": round(oos.metrics.get("sharpe", 0.0), 3),
                "oos_mdd": round(oos.metrics.get("max_drawdown", 0.0), 4),
            }
        )
        start += test_days

    if not oos_chunks:
        raise ValueError(
            "워크포워드 구간을 만들 수 없습니다. 기간을 늘리거나 "
            "train_days/test_days 를 줄이세요."
        )

    stitched = pd.concat(oos_chunks).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    equity = equity_from_returns(stitched, config.initial_cash)

    from .engine import BacktestResult

    agg = BacktestResult(
        symbol="WALK_FORWARD",
        equity=equity,
        returns=stitched,
        position=pd.Series(0.0, index=stitched.index),
        trades=[],
    )
    metrics = compute_metrics(agg, config)
    # 거래 단위 통계는 스티칭 과정에서 의미가 없으므로 제거
    for k in ("win_rate", "profit_factor", "avg_win", "avg_loss", "payoff_ratio",
              "expectancy", "n_trades", "avg_holding_days", "exposure", "turnover"):
        metrics.pop(k, None)

    result.oos_equity = equity
    result.oos_metrics = metrics
    return result


# --------------------------------------------------------------------------------
# 선택 방식 비교 — "1등을 고르는 것"이 최선인가
# --------------------------------------------------------------------------------
def compare_selectors(
    data: dict[str, pd.DataFrame],
    candidates: Sequence[Strategy],
    config: BacktestConfig,
    selectors: dict[str, "Callable[[Any], Strategy]"],
    *,
    train_days: int = 504,
    test_days: int = 126,
    objective: str = "robust",
    min_trades: int = 5,
    workers: int | None = None,
    progress: bool = True,
) -> dict[str, WalkForwardResult]:
    """여러 선택 방식을 **같은 워크포워드 창**에서 비교한다.

    창 분할·후보군·비용이 전부 동일하므로 차이는 오직 '무엇을 고르는가'에서 온다.
    """
    out: dict[str, WalkForwardResult] = {}
    for name, sel in selectors.items():
        if progress:
            print(f"\n[선택방식] {name}")
        out[name] = walk_forward(
            data, candidates, config,
            train_days=train_days, test_days=test_days, objective=objective,
            min_trades=min_trades, workers=workers, progress=False, selector=sel,
        )
    return out
