"""통계적 개선 도구 테스트 — 앙상블 · 변동성타겟 · 부트스트랩 · PBO."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.bootstrap import (
    BootstrapCI,
    bootstrap_ci,
    cagr_of,
    compare_strategies,
    max_drawdown_of,
    optimal_block_size,
    sharpe_of,
    stationary_bootstrap,
)
from quant.data import SyntheticSource
from quant.ensemble import EnsembleStrategy, VolatilityScaled, build_ensemble
from quant.significance import probability_of_backtest_overfitting as pbo
from quant.strategy import create_strategy


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return SyntheticSource().get("STAT", "2018-01-01", "2024-12-31")


# ------------------------------------------------------------------- 앙상블
def test_ensemble_averages_member_signals(sample):
    a = create_strategy("sma_cross", {"fast": 10, "slow": 60})
    b = create_strategy("momentum", {"lookback": 120})
    ens = EnsembleStrategy([a, b])

    expected = (a.generate_signals(sample) + b.generate_signals(sample)) / 2
    pd.testing.assert_series_equal(ens.generate_signals(sample), expected, check_names=False)


def test_ensemble_respects_weights(sample):
    a = create_strategy("sma_cross", {"fast": 10, "slow": 60})
    b = create_strategy("momentum", {"lookback": 120})
    ens = EnsembleStrategy([a, b], weights=[3, 1])

    assert ens.weights == [0.75, 0.25]
    expected = 0.75 * a.generate_signals(sample) + 0.25 * b.generate_signals(sample)
    pd.testing.assert_series_equal(ens.generate_signals(sample), expected, check_names=False)


def test_ensemble_warmup_is_the_longest_member(sample):
    a = create_strategy("sma_cross", {"fast": 5, "slow": 40})
    b = create_strategy("momentum", {"lookback": 200, "ma_filter": 200})
    assert EnsembleStrategy([a, b]).warmup == max(a.warmup, b.warmup)


def test_ensemble_reduces_signal_variance():
    """독립적인 잡음 신호들을 평균하면 분산이 줄어야 한다 — 앙상블의 존재 이유."""
    rng = np.random.default_rng(0)
    n, k = 2000, 10
    signals = [pd.Series(rng.random(n)) for _ in range(k)]
    mean_individual_var = float(np.mean([s.var() for s in signals]))
    ensemble_var = float(pd.concat(signals, axis=1).mean(axis=1).var())
    assert ensemble_var < mean_individual_var / 2      # 대략 1/k 로 줄어든다


def test_ensemble_rejects_bad_input():
    a = create_strategy("sma_cross")
    for bad in (
        lambda: EnsembleStrategy([]),
        lambda: EnsembleStrategy([a], weights=[1, 2]),
        lambda: EnsembleStrategy([a], weights=[-1]),
        lambda: EnsembleStrategy([a], weights=[0]),
    ):
        with pytest.raises(ValueError):
            bad()


# --------------------------------------------------------------- 변동성 타겟
def test_volatility_scaling_is_inverse_to_volatility():
    """변동성이 높은 구간에서 비중이 줄어야 한다."""
    n = 600
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(1)
    # 앞 절반은 저변동, 뒤 절반은 고변동
    vol = np.concatenate([np.full(n // 2, 0.005), np.full(n - n // 2, 0.03)])
    px = 100 * np.exp(np.cumsum(rng.normal(0, 1, n) * vol))
    df = pd.DataFrame({"open": px, "high": px, "low": px, "close": px,
                       "volume": np.full(n, 1e3)}, index=idx)

    base = create_strategy("buy_and_hold")
    scaled = VolatilityScaled(base, target_vol=0.15, window=20).generate_signals(df)

    calm = scaled.iloc[100:280].mean()
    wild = scaled.iloc[380:560].mean()
    assert wild < calm, (calm, wild)


def test_volatility_scaling_respects_max_leverage(sample):
    v = VolatilityScaled(create_strategy("buy_and_hold"), target_vol=5.0, max_leverage=1.0)
    assert v.generate_signals(sample).max() <= 1.0 + 1e-9


def test_volatility_scaling_rejects_bad_params():
    b = create_strategy("buy_and_hold")
    for kw in ({"target_vol": 0}, {"window": 2}, {"max_leverage": 0}):
        with pytest.raises(ValueError):
            VolatilityScaled(b, **kw)


def test_build_ensemble_diversifies_across_strategies():
    """같은 전략의 이웃 파라미터만 모으면 평균 효과가 없다."""
    from quant.config import BacktestConfig, CostModel
    from quant.data import load_universe
    from quant.optimizer import build_all_candidates, grid_search

    data = load_universe(SyntheticSource(), ["A", "B"], "2019-01-01", "2024-12-31")
    cands = build_all_candidates(["sma_cross", "momentum", "donchian"])
    rep = grid_search(data, cands, BacktestConfig(cost=CostModel.zero()),
                      min_trades=0, progress=False, workers=1, store_equity_top=0)

    ens = build_ensemble(rep, top_k=3, diversify=True)
    names = {m.name for m in ens.members}
    assert len(names) == 3, names                 # 전략 종류가 서로 다르다

    flat = build_ensemble(rep, top_k=3, diversify=False)
    assert len(flat.members) == 3

    for w in ("equal", "score", "rank"):
        assert sum(build_ensemble(rep, top_k=5, weighting=w).weights) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="weighting"):
        build_ensemble(rep, weighting="nope")


# ---------------------------------------------------------------- 부트스트랩
def test_block_size_grows_with_autocorrelation():
    rng = np.random.default_rng(0)
    iid = rng.normal(0, 0.01, 1500)
    ar = pd.Series(iid).ewm(alpha=0.2).mean().to_numpy()
    assert optimal_block_size(ar) > optimal_block_size(iid)


def test_stationary_bootstrap_shape_and_resampling():
    rng = np.random.default_rng(0)
    r = np.arange(100, dtype=float)
    samples = stationary_bootstrap(r, n_boot=50, block_size=5, rng=rng)
    assert samples.shape == (50, 100)
    assert set(np.unique(samples)) <= set(r)          # 원본 값만 사용
    assert not np.array_equal(samples[0], samples[1])  # 표본마다 다르다


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.01, 1500))
    ci = bootstrap_ci(r, sharpe_of, n_boot=800)
    assert ci.lower < ci.point < ci.upper
    assert 0 <= ci.prob_positive <= 1
    assert isinstance(ci, BootstrapCI) and "CI" in ci.format("Sharpe")


def test_bootstrap_ci_narrows_with_more_data():
    """표본이 길수록 불확실성이 줄어야 한다."""
    rng = np.random.default_rng(0)
    short = pd.Series(rng.normal(0.0008, 0.01, 250))
    long_ = pd.Series(rng.normal(0.0008, 0.01, 4000))
    w_short = bootstrap_ci(short, sharpe_of, n_boot=600).upper - bootstrap_ci(
        short, sharpe_of, n_boot=600).lower
    w_long = bootstrap_ci(long_, sharpe_of, n_boot=600).upper - bootstrap_ci(
        long_, sharpe_of, n_boot=600).lower
    assert w_long < w_short


def test_bootstrap_ci_rejects_short_series():
    with pytest.raises(ValueError, match="너무 짧"):
        bootstrap_ci(pd.Series([0.01] * 5), sharpe_of)


def test_paired_comparison_detects_real_difference():
    """A 가 명확히 나으면 높은 확률로 잡아내야 한다."""
    rng = np.random.default_rng(0)
    n = 2000
    common = rng.normal(0, 0.01, n)          # 같은 시장 변동
    a = pd.Series(common + 0.0012)           # A 만 꾸준한 초과수익
    b = pd.Series(common)
    cmp = compare_strategies(a, b, sharpe_of, n_boot=800)
    assert cmp.prob_a_better > 0.95
    assert cmp.diff > 0 and cmp.significant


def test_paired_comparison_finds_no_difference_when_identical():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 1500))
    cmp = compare_strategies(r, r.copy(), sharpe_of, n_boot=500)
    assert cmp.diff == pytest.approx(0.0)
    assert not cmp.significant


def test_metric_helpers():
    r = np.array([0.1, -0.05, 0.02])
    assert cagr_of(r) != 0.0
    assert max_drawdown_of(np.array([0.1, -0.5, 0.1])) < -0.4
    assert sharpe_of(np.array([0.01])) == 0.0        # 표본 부족
    assert max_drawdown_of(np.array([0.01])) == 0.0


# ----------------------------------------------------------------------- PBO
def test_pbo_high_for_pure_noise_low_for_real_edge():
    """실력 차이가 없으면 PBO 가 높고, 진짜 차이가 있으면 낮아야 한다."""
    rng = np.random.default_rng(0)
    T, N = 1500, 60
    noise = pd.DataFrame(rng.normal(0, 0.01, (T, N)))
    edge = pd.DataFrame(rng.normal(0, 0.01, (T, N)) + np.linspace(0, 0.001, N))

    p_noise = pbo(noise, n_splits=10)
    p_edge = pbo(edge, n_splits=10)

    assert p_noise.pbo > p_edge.pbo
    assert p_noise.pbo > 0.30                     # 잡음이면 상당히 높다
    assert p_edge.median_oos_rank > p_noise.median_oos_rank
    assert "PBO" in p_noise.format()


def test_pbo_verdict_bands():
    from quant.significance import PBOResult

    for value, expect in [(0.05, "견고"), (0.35, "주의"), (0.70, "과최적화")]:
        r = PBOResult(pbo=value, n_candidates=10, n_splits=10,
                      n_combinations=100, median_oos_rank=0.5, is_oos_slope=0.0)
        assert expect in r.verdict


def test_pbo_input_validation():
    rng = np.random.default_rng(0)
    m = pd.DataFrame(rng.normal(0, 0.01, (500, 10)))
    with pytest.raises(ValueError, match="짝수"):
        pbo(m, n_splits=7)
    with pytest.raises(ValueError, match="2개 이상"):
        pbo(m.iloc[:, :1], n_splits=10)
    with pytest.raises(ValueError, match="너무 짧"):
        pbo(m.iloc[:50], n_splits=10)
