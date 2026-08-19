"""다중검정 보정 테스트 — "수백 개 돌려 1등을 골랐다"를 벌점화하는 로직."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant.significance import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    norm_cdf,
    norm_ppf,
    probabilistic_sharpe_ratio,
)


# ------------------------------------------------------------------ 정규분포
def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-4)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-4)
    assert norm_cdf(1.0) == pytest.approx(0.8413, abs=1e-4)


def test_norm_ppf_inverts_cdf():
    for p in (0.01, 0.25, 0.5, 0.75, 0.975, 0.999):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-8)
    assert norm_ppf(0.975) == pytest.approx(1.95996, abs=1e-4)
    for bad in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            norm_ppf(bad)


# ------------------------------------------------------------------------ PSR
def test_psr_is_half_at_benchmark():
    """관측 Sharpe 가 기준치와 같으면 '더 높을 확률'은 50%."""
    assert probabilistic_sharpe_ratio(0.1, 500, benchmark=0.1) == pytest.approx(0.5)


def test_psr_rises_with_sample_length():
    """같은 Sharpe 라도 표본이 길수록 확신이 커진다."""
    short = probabilistic_sharpe_ratio(0.05, 50)
    long_ = probabilistic_sharpe_ratio(0.05, 2000)
    assert 0.5 < short < long_ < 1.0


def test_psr_penalizes_negative_skew_and_fat_tails():
    """왼쪽 꼬리가 두꺼우면 같은 Sharpe 도 덜 믿을 만하다."""
    base = probabilistic_sharpe_ratio(0.08, 1000, skew=0.0, kurtosis=3.0)
    skewed = probabilistic_sharpe_ratio(0.08, 1000, skew=-1.5, kurtosis=3.0)
    fat = probabilistic_sharpe_ratio(0.08, 1000, skew=0.0, kurtosis=12.0)
    assert skewed < base
    assert fat < base


def test_psr_degenerate_inputs():
    assert probabilistic_sharpe_ratio(1.0, 1) == 0.0
    assert probabilistic_sharpe_ratio(1.0, 0) == 0.0


# --------------------------------------------------- 시도 횟수에 따른 기준선
def test_expected_max_sharpe_grows_with_trials():
    """더 많이 시도할수록 '우연히 나오는 최고 성적'이 높아진다."""
    vals = [expected_max_sharpe(n, 0.5) for n in (2, 10, 100, 1000, 10000)]
    assert all(a < b for a, b in zip(vals, vals[1:]))
    assert expected_max_sharpe(1, 0.5) == 0.0        # 한 번만 시도하면 벌점 없음
    assert expected_max_sharpe(100, 0.0) == 0.0      # 산포가 없으면 벌점 없음


def test_expected_max_sharpe_scales_with_dispersion():
    assert expected_max_sharpe(100, 1.0) == pytest.approx(
        2 * expected_max_sharpe(100, 0.5)
    )


# ------------------------------------------------------------------------ DSR
def _returns(mean: float, vol: float, n: int = 1000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(mean, vol, n), index=idx)


def test_dsr_is_below_psr_when_many_trials():
    """시도 횟수를 반영하면 유의확률이 반드시 내려간다."""
    r = _returns(0.0006, 0.01)
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252))
    res = deflated_sharpe_ratio(r, sharpe, n_trials=500, sharpe_std=0.5)
    assert res.dsr < res.psr
    assert res.threshold > 0


def test_dsr_falls_as_trials_increase():
    r = _returns(0.0006, 0.01)
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252))
    dsrs = [
        deflated_sharpe_ratio(r, sharpe, n_trials=n, sharpe_std=0.5).dsr
        for n in (2, 50, 500, 5000)
    ]
    assert all(a >= b for a, b in zip(dsrs, dsrs[1:])), dsrs


def test_single_trial_leaves_psr_unchanged():
    """한 번만 시험했으면 보정할 것이 없다 -> DSR == PSR."""
    r = _returns(0.0006, 0.01)
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252))
    res = deflated_sharpe_ratio(r, sharpe, n_trials=1, sharpe_std=0.5)
    assert res.threshold == 0.0
    assert res.dsr == pytest.approx(res.psr)


def test_verdict_thresholds():
    r = _returns(0.0006, 0.01)
    res = deflated_sharpe_ratio(r, 1.0, n_trials=10, sharpe_std=0.1)
    for dsr, expect in [(0.99, "유의"), (0.85, "약한 근거"), (0.30, "유의하지 않음")]:
        res.dsr = dsr
        assert expect in res.verdict


def test_dsr_handles_short_series():
    res = deflated_sharpe_ratio(pd.Series([0.01, -0.01]), 1.0, 100, 0.5)
    assert res.dsr == 0.0 and res.psr == 0.0


# --------------------------------------------------------- 리포트 연동
def test_assess_report_uses_trial_count(tmp_path):
    """탐색한 조합 수가 그대로 벌점에 반영되는지."""
    from quant.config import BacktestConfig, CostModel
    from quant.data import SyntheticSource, load_universe
    from quant.optimizer import grid_search
    from quant.significance import assess_report
    from quant.strategy import build_candidates

    data = load_universe(SyntheticSource(), ["A", "B"], "2019-01-01", "2024-12-31")
    c = BacktestConfig(initial_cash=1_000_000, cost=CostModel.kr_stock())

    narrow = build_candidates("sma_cross", {"fast": [10], "slow": [60]})
    wide = build_candidates("sma_cross", {"fast": [5, 10, 20, 30], "slow": [40, 60, 90, 120]})

    kw = dict(min_trades=0, progress=False, workers=1, store_equity_top=1)
    a = assess_report(grid_search(data, narrow, c, **kw))
    b = assess_report(grid_search(data, wide, c, **kw))

    assert a.n_trials == 1 and b.n_trials == len(wide)
    assert b.threshold >= a.threshold      # 많이 시험할수록 문턱이 높다


def test_assess_report_requires_equity():
    from quant.optimizer import EvalResult, OptimizationReport
    from quant.config import BacktestConfig
    from quant.significance import assess_report

    r = EvalResult(strategy="x", params={}, score=1.0, metrics={"sharpe": 1.0})
    rep = OptimizationReport(
        results=[r], objective="sharpe",
        trade_start=pd.Timestamp("2020-01-01"), trade_end=pd.Timestamp("2021-01-01"),
        symbols=["A"], config=BacktestConfig(),
    )
    with pytest.raises(ValueError, match="자산곡선"):
        assess_report(rep)
