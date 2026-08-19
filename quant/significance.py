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


# --------------------------------------------------------------------------------
# PBO — 과최적화 확률 (Combinatorially Symmetric Cross-Validation)
# --------------------------------------------------------------------------------
@dataclass
class PBOResult:
    """백테스트 과최적화 확률."""

    pbo: float  # 0~1. IS 1등이 OOS 중앙값 아래로 떨어질 확률
    n_candidates: int
    n_splits: int
    n_combinations: int
    median_oos_rank: float  # IS 1등의 OOS 상대순위 중앙값 (1.0 = 최고)
    is_oos_slope: float  # IS 성과 -> OOS 성과 회귀 기울기

    @property
    def verdict(self) -> str:
        if self.pbo <= 0.20:
            return "견고 — IS 1등이 OOS 에서도 대체로 상위를 지킨다"
        if self.pbo <= 0.50:
            return "주의 — IS 1등이 절반 가까이 OOS 하위로 떨어진다"
        return "과최적화 — IS 1등이 OOS 에서 동전던지기보다 못하다"

    def format(self) -> str:
        return "\n".join([
            "과최적화 확률 (PBO · 조합적 교차검증)",
            "=" * 64,
            f"  후보 수              {self.n_candidates:>10,}",
            f"  분할 블록 / 조합 수  {self.n_splits:>10,} / {self.n_combinations:,}",
            "",
            f"  IS 1등의 OOS 상대순위 (중앙값)  {self.median_oos_rank:>8.2f}"
            "   (1.0=최고, 0.5=중간)",
            f"  IS→OOS 성과 회귀 기울기         {self.is_oos_slope:>8.3f}"
            "   (음수면 IS 좋을수록 OOS 나쁨)",
            "",
            f"  **PBO = {self.pbo:.1%}**",
            "",
            f"  판정: {self.verdict}",
        ])


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    *,
    n_splits: int = 10,
    max_combinations: int = 1000,
    seed: int = 0,
) -> PBOResult:
    """CSCV 로 PBO 를 계산한다.

    returns_matrix : (시점 × 후보) 일별 수익률 행렬

    아이디어: 기간을 S개 블록으로 쪼개고, 절반을 학습(IS)·절반을 검증(OOS)으로
    **가능한 모든 방식으로** 나눈다. 매번 IS 1등을 고른 뒤 그것이 OOS 에서
    몇 등인지 본다. IS 1등이 OOS 에서 자꾸 중간 이하로 떨어지면 = 과최적화.

    워크포워드가 "시간 순서"만 보는 것과 달리, CSCV 는 **가능한 모든 분할**을
    보므로 특정 구간의 운에 덜 좌우된다. 둘은 상호보완적이다.

    참고: Bailey, Borwein, López de Prado & Zhu (2015).
    """
    from itertools import combinations

    m = returns_matrix.dropna(axis=1, how="all").dropna()
    n_obs, n_cand = m.shape
    if n_cand < 2:
        raise ValueError("후보가 2개 이상이어야 합니다.")
    if n_splits % 2 != 0:
        raise ValueError("n_splits 는 짝수여야 합니다.")
    if n_obs < n_splits * 10:
        raise ValueError(f"표본({n_obs})이 분할 수({n_splits})에 비해 너무 짧습니다.")

    values = m.to_numpy(dtype=float)
    blocks = np.array_split(np.arange(n_obs), n_splits)

    def sharpe_cols(rows: np.ndarray) -> np.ndarray:
        sub = values[rows]
        sd = sub.std(axis=0, ddof=1)
        mu = sub.mean(axis=0)
        out = np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)
        return np.nan_to_num(out)

    half = n_splits // 2
    all_combos = list(combinations(range(n_splits), half))
    rng = np.random.default_rng(seed)
    if len(all_combos) > max_combinations:
        picked = rng.choice(len(all_combos), size=max_combinations, replace=False)
        all_combos = [all_combos[i] for i in picked]

    logits: list[float] = []
    ranks: list[float] = []
    is_perf: list[float] = []
    oos_perf: list[float] = []

    for combo in all_combos:
        train_idx = np.concatenate([blocks[i] for i in combo])
        test_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])

        is_sr = sharpe_cols(train_idx)
        oos_sr = sharpe_cols(test_idx)

        best = int(np.argmax(is_sr))
        # OOS 에서의 상대순위 (1.0 = 최고)
        rank = float((oos_sr <= oos_sr[best]).sum()) / (n_cand + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)

        ranks.append(rank)
        logits.append(math.log(rank / (1.0 - rank)))
        is_perf.append(float(is_sr[best]))
        oos_perf.append(float(oos_sr[best]))

    logit_arr = np.array(logits)
    pbo = float((logit_arr <= 0).mean())

    # IS 성과가 OOS 성과를 얼마나 설명하는가 (음수면 역효과)
    x, y = np.array(is_perf), np.array(oos_perf)
    slope = 0.0
    if x.size > 2 and x.std() > 0:
        slope = float(np.polyfit(x, y, 1)[0])

    return PBOResult(
        pbo=pbo,
        n_candidates=n_cand,
        n_splits=n_splits,
        n_combinations=len(all_combos),
        median_oos_rank=float(np.median(ranks)),
        is_oos_slope=slope,
    )


def collect_returns_matrix(
    data: dict,
    candidates,
    config,
    trade_start=None,
    *,
    max_candidates: int = 300,
) -> pd.DataFrame:
    """후보별 포트폴리오 일별 수익률 행렬 (PBO 입력용).

    후보가 많으면 메모리·시간이 커지므로 ``max_candidates`` 개까지만 균등 표집한다.
    """
    from .optimizer import common_trade_start, evaluate_candidate, obj_sharpe

    cands = list(candidates)
    if len(cands) > max_candidates:
        step = len(cands) / max_candidates
        cands = [cands[int(i * step)] for i in range(max_candidates)]

    if trade_start is None:
        trade_start = common_trade_start(next(iter(data.values())).index, cands)

    cols: dict[str, pd.Series] = {}
    for c in cands:
        res = evaluate_candidate(
            c, data, config, trade_start, obj_sharpe, store_equity=True, min_trades=0
        )
        if res.equity is not None:
            cols[c.describe()] = res.equity.pct_change()
    if not cols:
        raise ValueError("수익률을 수집한 후보가 없습니다.")
    return pd.DataFrame(cols)
