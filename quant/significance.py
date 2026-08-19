"""통계적 유의성 — "이 성과가 운이 아니라고 말할 수 있는가".

그리드 탐색은 수백 개 조합을 시험하고 그중 1등을 고른다. 이때 1등의 Sharpe 는
**반드시** 부풀려져 있다. 동전 439개를 던져 앞면이 가장 많이 나온 동전을 고르면
그 동전이 대단해 보이는 것과 같다.

기관 리서치는 이를 **다중검정 보정**으로 처리한다. 여기서는 두 가지를 구현한다.

- **PSR** (Probabilistic Sharpe Ratio): 관측된 Sharpe 가 기준치보다 진짜로
  높을 확률. 표본 길이·왜도·첨도를 반영한다.
- **DSR** (Deflated Sharpe Ratio): 기준치를 "N번 시도했을 때 우연히 기대되는
  최대 Sharpe"로 잡은 PSR. 즉 **시도 횟수를 벌점으로 매긴 유의성**.

참고: Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014).
scipy 의존을 피하기 위해 정규분포 함수는 직접 구현한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------------
# 정규분포 (scipy 없이)
# --------------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    """표준정규 누적분포."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float, *, tol: float = 1e-12, max_iter: int = 200) -> float:
    """표준정규 분위수 (누적분포의 역함수). 이분법으로 푼다."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p 는 (0, 1) 범위여야 합니다: {p}")
    lo, hi = -40.0, 40.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------------
# 지표
# --------------------------------------------------------------------------------
def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    *,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR — 참 Sharpe 가 ``benchmark`` 를 넘을 확률.

    관측 Sharpe 가 같아도 **표본이 짧으면** 확률이 낮아지고,
    **왼쪽 꼬리가 두꺼우면**(음의 왜도·높은 첨도) 역시 낮아진다.
    Sharpe 를 그냥 비교하면 안 되는 이유다.

    sharpe / benchmark 는 **같은 주기**여야 한다(둘 다 연율 또는 둘 다 일간).
    """
    if n_obs < 2:
        return 0.0
    # Sharpe 추정량의 표준오차 (비정규성 보정 포함)
    var = 1.0 - skew * sharpe + 0.25 * (kurtosis - 1.0) * sharpe**2
    if var <= 0:
        return 0.0
    z = (sharpe - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(var)
    return norm_cdf(z)


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """참 Sharpe 가 전부 0일 때, N번 시도하면 우연히 기대되는 **최대** Sharpe.

    이것이 DSR 의 기준선이다. 시도를 많이 할수록 높아지므로,
    "많이 돌려서 찾은 1등"은 그만큼 높은 문턱을 넘어야 한다.
    """
    if n_trials < 2 or sharpe_std <= 0:
        return 0.0
    g = EULER_MASCHERONI
    a = norm_ppf(1.0 - 1.0 / n_trials)
    b = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sharpe_std * ((1.0 - g) * a + g * b)


@dataclass
class SignificanceResult:
    """유의성 판정 결과."""

    sharpe: float
    n_obs: int
    n_trials: int
    skew: float
    kurtosis: float
    sharpe_std: float
    threshold: float  # 우연히 기대되는 최대 Sharpe (DSR 기준선)
    psr: float  # 0 대비 유의확률
    dsr: float  # 다중검정 보정 후 유의확률

    @property
    def verdict(self) -> str:
        if self.dsr >= 0.95:
            return "유의 — 우연으로 보기 어렵다"
        if self.dsr >= 0.80:
            return "약한 근거 — 표본을 더 쌓을 것"
        return "유의하지 않음 — 탐색 과정의 산물일 가능성이 높다"

    def format(self) -> str:
        return "\n".join([
            "통계적 유의성 (다중검정 보정)",
            "=" * 64,
            f"  관측 Sharpe        {self.sharpe:>10.3f}",
            f"  표본 길이          {self.n_obs:>10,} 봉",
            f"  시험한 조합 수     {self.n_trials:>10,} 개",
            f"  수익률 왜도/첨도   {self.skew:>10.2f} / {self.kurtosis:.2f}",
            "",
            f"  우연 기대 최대 SR  {self.threshold:>10.3f}   ← {self.n_trials:,}번 시도하면"
            " 이 정도는 그냥 나온다",
            f"  PSR (기준 0)       {self.psr:>10.1%}",
            f"  DSR (보정 후)      {self.dsr:>10.1%}",
            "",
            f"  판정: {self.verdict}",
        ])


def deflated_sharpe_ratio(
    returns: pd.Series,
    sharpe: float,
    n_trials: int,
    sharpe_std: float,
    *,
    trading_days: int = 252,
) -> SignificanceResult:
    """DSR 계산.

    returns  : 최적 후보의 일별 수익률 (왜도·첨도 산출용)
    sharpe   : 그 후보의 **연율** Sharpe
    n_trials : 시험한 조합 수
    sharpe_std : 조합들의 Sharpe 표준편차 (시도들의 산포)
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 3:
        return SignificanceResult(sharpe, n, n_trials, 0.0, 3.0, sharpe_std, 0.0, 0.0, 0.0)

    skew = float(r.skew())
    kurt = float(r.kurtosis() + 3.0)  # pandas 는 초과첨도를 준다
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt):
        kurt = 3.0

    # PSR 공식은 관측 주기 기준이므로 연율 Sharpe 를 일간으로 되돌린다
    scale = math.sqrt(trading_days)
    sr_daily = sharpe / scale
    std_daily = sharpe_std / scale

    threshold_daily = expected_max_sharpe(n_trials, std_daily)
    psr = probabilistic_sharpe_ratio(sr_daily, n, benchmark=0.0, skew=skew, kurtosis=kurt)
    dsr = probabilistic_sharpe_ratio(
        sr_daily, n, benchmark=threshold_daily, skew=skew, kurtosis=kurt
    )

    return SignificanceResult(
        sharpe=sharpe,
        n_obs=n,
        n_trials=n_trials,
        skew=skew,
        kurtosis=kurt,
        sharpe_std=sharpe_std,
        threshold=threshold_daily * scale,  # 표시용으로 다시 연율화
        psr=psr,
        dsr=dsr,
    )


def assess_report(report, *, trading_days: int | None = None) -> SignificanceResult:
    """OptimizationReport 에서 바로 유의성 판정.

    시험한 조합 수와 Sharpe 산포를 리포트에서 그대로 가져오므로,
    "몇 개를 돌려서 1등을 골랐는가"가 자동으로 반영된다.
    """
    passed = [r for r in report.results if not r.filtered]
    if not passed:
        raise ValueError("필터를 통과한 후보가 없습니다.")

    best = passed[0]
    if best.equity is None:
        raise ValueError(
            "최적 후보의 자산곡선이 없습니다. grid_search(store_equity_top>=1) 로 실행하세요."
        )

    sharpes = np.array([r.metrics.get("sharpe", 0.0) for r in passed], dtype=float)
    sharpes = sharpes[np.isfinite(sharpes)]
    sharpe_std = float(sharpes.std(ddof=1)) if len(sharpes) > 1 else 0.0

    td = trading_days if trading_days is not None else report.config.trading_days
    return deflated_sharpe_ratio(
        best.equity.pct_change(),
        best.metrics.get("sharpe", 0.0),
        n_trials=len(passed),
        sharpe_std=sharpe_std,
        trading_days=td,
    )
