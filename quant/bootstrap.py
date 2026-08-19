"""블록 부트스트랩 — "이 성과가 얼마나 불확실한가"를 수치로 답한다.

백테스트는 **점추정치 하나**를 준다. Sharpe 0.70 이라고. 그런데 그게
0.70 ± 0.10 인지 0.70 ± 0.60 인지에 따라 결론이 완전히 달라진다.

부트스트랩은 관측된 수익률 시계열을 재표집해 **분포**를 만든다.
다만 금융 수익률은 독립이 아니다(변동성 군집 = 자기상관). 그래서 한 점씩
뽑는 단순 부트스트랩은 불확실성을 **과소평가**한다.

여기서는 **정상 부트스트랩**(Politis & Romano, 1994)을 쓴다. 기하분포 길이의
블록을 이어붙여 국소 상관구조를 보존하면서 재표집한다.

용도 두 가지:
  1. 지표의 신뢰구간 — "Sharpe 0.70 [0.12, 1.31]" 처럼
  2. 두 전략의 차이 검정 — "A 가 B 보다 나을 확률 78%"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

MetricFn = Callable[[np.ndarray], float]


# --------------------------------------------------------------------------------
# 기본 지표 (numpy 배열용 — 부트스트랩 루프에서 빠르게 돌아야 한다)
# --------------------------------------------------------------------------------
def sharpe_of(returns: np.ndarray, trading_days: int = 252) -> float:
    if returns.size < 2:
        return 0.0
    sd = returns.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(returns.mean() / sd * math.sqrt(trading_days))


def cagr_of(returns: np.ndarray, trading_days: int = 252) -> float:
    if returns.size < 2:
        return 0.0
    growth = float(np.prod(1.0 + returns))
    if growth <= 0:
        return -1.0
    years = returns.size / trading_days
    return float(growth ** (1.0 / years) - 1.0)


def max_drawdown_of(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    eq = np.cumprod(1.0 + returns)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


# --------------------------------------------------------------------------------
# 정상 부트스트랩
# --------------------------------------------------------------------------------
def optimal_block_size(returns: np.ndarray) -> float:
    """자기상관 정도에 맞춘 평균 블록 길이의 간단한 추정.

    1차 자기상관 rho 로부터 n^(1/3) 규칙을 조정한다. 정밀한 최적값은 아니지만,
    "상관이 강하면 블록을 길게"라는 방향은 맞다.
    """
    n = returns.size
    if n < 10:
        return 1.0
    r = returns - returns.mean()
    denom = float((r * r).sum())
    rho = float((r[:-1] * r[1:]).sum() / denom) if denom > 0 else 0.0
    rho = float(np.clip(abs(rho), 0.0, 0.95))
    base = n ** (1.0 / 3.0)
    return float(np.clip(base * (1.0 + 2.0 * rho), 1.0, max(n / 4.0, 1.0)))


def stationary_bootstrap(
    returns: np.ndarray, n_boot: int, block_size: float, rng: np.random.Generator
) -> np.ndarray:
    """재표집 표본 행렬 (n_boot, n) 을 만든다.

    각 위치에서 확률 1/block_size 로 새 블록을 시작하고(무작위 위치로 점프),
    아니면 직전 인덱스의 다음 값을 이어 쓴다. 인덱스는 순환한다.
    """
    n = returns.size
    p = 1.0 / max(block_size, 1.0)

    starts = rng.integers(0, n, size=(n_boot, n))
    jump = rng.random((n_boot, n)) < p
    jump[:, 0] = True  # 첫 칸은 항상 새 블록

    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(jump[:, t], starts[:, t], cont)
    return returns[idx]


@dataclass
class BootstrapCI:
    """부트스트랩 신뢰구간."""

    point: float  # 관측값
    lower: float
    upper: float
    mean: float  # 부트스트랩 평균
    std: float
    level: float
    n_boot: int
    block_size: float
    prob_positive: float  # 부트스트랩 표본 중 0보다 큰 비율

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0

    def format(self, name: str = "지표", pct: bool = False) -> str:
        f = (lambda v: f"{v:>8.2%}") if pct else (lambda v: f"{v:>8.3f}")
        mark = "유의" if self.excludes_zero else "0 포함"
        return (
            f"  {name:<12}{f(self.point)}   {int(self.level * 100)}% CI "
            f"[{f(self.lower).strip()}, {f(self.upper).strip()}]   "
            f"P(>0)={self.prob_positive:>5.1%}   {mark}"
        )


def bootstrap_ci(
    returns: pd.Series | np.ndarray,
    metric: MetricFn,
    *,
    n_boot: int = 2000,
    level: float = 0.90,
    block_size: float | None = None,
    seed: int = 0,
) -> BootstrapCI:
    """지표의 부트스트랩 신뢰구간."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if r.size < 10:
        raise ValueError(f"표본이 너무 짧습니다 ({r.size}).")

    bs = block_size if block_size is not None else optimal_block_size(r)
    rng = np.random.default_rng(seed)
    samples = stationary_bootstrap(r, n_boot, bs, rng)
    vals = np.array([metric(row) for row in samples], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("부트스트랩 표본에서 유효한 지표를 얻지 못했습니다.")

    alpha = (1.0 - level) / 2.0
    return BootstrapCI(
        point=float(metric(r)),
        lower=float(np.quantile(vals, alpha)),
        upper=float(np.quantile(vals, 1.0 - alpha)),
        mean=float(vals.mean()),
        std=float(vals.std(ddof=1)),
        level=level,
        n_boot=int(vals.size),
        block_size=float(bs),
        prob_positive=float((vals > 0).mean()),
    )


@dataclass
class BootstrapComparison:
    """두 전략의 차이에 대한 부트스트랩 검정."""

    name_a: str
    name_b: str
    metric_a: float
    metric_b: float
    diff: float
    lower: float
    upper: float
    prob_a_better: float
    level: float
    n_boot: int

    @property
    def significant(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0

    def format(self) -> str:
        verdict = (
            "차이가 유의하다" if self.significant
            else "차이를 구분할 수 없다 (표본 부족 또는 실제로 비슷함)"
        )
        return "\n".join([
            f"  {self.name_a:<28}{self.metric_a:>8.3f}",
            f"  {self.name_b:<28}{self.metric_b:>8.3f}",
            f"  {'차이 (A-B)':<28}{self.diff:>8.3f}   "
            f"{int(self.level * 100)}% CI [{self.lower:.3f}, {self.upper:.3f}]",
            f"  {'A 가 더 나을 확률':<28}{self.prob_a_better:>8.1%}",
            f"\n  판정: {verdict}",
        ])


def compare_strategies(
    returns_a: pd.Series,
    returns_b: pd.Series,
    metric: MetricFn,
    *,
    name_a: str = "A",
    name_b: str = "B",
    n_boot: int = 2000,
    level: float = 0.90,
    seed: int = 0,
) -> BootstrapComparison:
    """A 가 B 보다 나은지 **쌍대(paired) 부트스트랩**으로 검정.

    같은 날짜를 함께 재표집한다. 두 전략이 같은 시장을 겪었으므로 시장 변동이
    상쇄되어 검정력이 훨씬 높아진다 — 이것이 각각 따로 CI 를 구해 겹치는지
    보는 것보다 옳은 방법이다.
    """
    frame = pd.DataFrame({"a": pd.Series(returns_a), "b": pd.Series(returns_b)}).dropna()
    if len(frame) < 10:
        raise ValueError(f"공통 표본이 너무 짧습니다 ({len(frame)}).")

    a = frame["a"].to_numpy(dtype=float)
    b = frame["b"].to_numpy(dtype=float)
    n = a.size
    bs = optimal_block_size(a - b)
    rng = np.random.default_rng(seed)

    # 인덱스를 한 번만 만들어 두 계열에 동일하게 적용 (쌍대 구조 보존)
    p = 1.0 / max(bs, 1.0)
    starts = rng.integers(0, n, size=(n_boot, n))
    jump = rng.random((n_boot, n)) < p
    jump[:, 0] = True
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)

    diffs = np.array(
        [metric(a[row]) - metric(b[row]) for row in idx], dtype=float
    )
    diffs = diffs[np.isfinite(diffs)]
    alpha = (1.0 - level) / 2.0

    return BootstrapComparison(
        name_a=name_a,
        name_b=name_b,
        metric_a=float(metric(a)),
        metric_b=float(metric(b)),
        diff=float(metric(a) - metric(b)),
        lower=float(np.quantile(diffs, alpha)),
        upper=float(np.quantile(diffs, 1.0 - alpha)),
        prob_a_better=float((diffs > 0).mean()),
        level=level,
        n_boot=int(diffs.size),
    )
