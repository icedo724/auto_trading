"""최적화 · 검증 · 데이터 계층 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.config import BacktestConfig, CostModel
from quant.data import SyntheticSource, align_to_calendar, common_calendar, load_universe
from quant.data.base import DataError, normalize_ohlcv
from quant.metrics import cagr, max_drawdown, sharpe_ratio
from quant.optimizer import (
    build_all_candidates,
    common_trade_start,
    grid_search,
    obj_robust,
    sensitivity,
)
from quant.strategy import build_candidates, create_strategy
from quant.validation import holdout_validate, walk_forward


@pytest.fixture(scope="module")
def universe() -> dict[str, pd.DataFrame]:
    return load_universe(
        SyntheticSource(), ["A", "B", "C"], "2015-01-01", "2024-12-31", min_bars=100
    )


@pytest.fixture(scope="module")
def config() -> BacktestConfig:
    return BacktestConfig(initial_cash=1_000_000, cost=CostModel.kr_stock())


# --------------------------------------------------------------------- 데이터
def test_universe_shares_one_calendar(universe):
    cals = [df.index for df in universe.values()]
    for c in cals[1:]:
        assert c.equals(cals[0])


def test_normalize_handles_korean_columns():
    idx = pd.date_range("2020-01-01", periods=3)
    raw = pd.DataFrame(
        {"시가": [1, 2, 3], "고가": [2, 3, 4], "저가": [0.5, 1, 2],
         "종가": [1.5, 2.5, 3.5], "거래량": [10, 20, 30]},
        index=idx,
    )
    out = normalize_ohlcv(raw, symbol="005930")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.name == "date"


def test_normalize_rejects_missing_columns():
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2))
    with pytest.raises(DataError, match="필수 컬럼 누락"):
        normalize_ohlcv(df, symbol="X")


def test_normalize_drops_nonpositive_prices():
    idx = pd.date_range("2020-01-01", periods=3)
    raw = pd.DataFrame(
        {"open": [1, 0, 3], "high": [1, 0, 3], "low": [1, 0, 3],
         "close": [1, 0, 3], "volume": [1, 1, 1]},
        index=idx,
    )
    assert len(normalize_ohlcv(raw, symbol="X")) == 2


def test_align_keeps_pre_listing_nan():
    a = SyntheticSource().get("A", "2020-01-01", "2021-01-01")
    b = SyntheticSource().get("B", "2020-06-01", "2021-01-01")
    aligned = align_to_calendar({"A": a, "B": b}, common_calendar({"A": a, "B": b}))
    assert aligned["B"]["close"].iloc[0] != aligned["B"]["close"].iloc[0]  # NaN
    assert aligned["B"]["close"].iloc[-1] > 0


def test_synthetic_source_is_deterministic():
    a = SyntheticSource().get("SEED-TEST", "2020-01-01", "2021-01-01")
    b = SyntheticSource().get("SEED-TEST", "2020-01-01", "2021-01-01")
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_ohlc_consistency():
    df = SyntheticSource().get("OHLC", "2020-01-01", "2022-01-01")
    assert (df["high"] >= df[["open", "close", "low"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close", "high"]].min(axis=1) + 1e-9).all()


# --------------------------------------------------------------------- 지표
def test_metric_math():
    eq = pd.Series([100.0, 110.0, 90.0, 120.0])
    assert max_drawdown(eq) == pytest.approx(90 / 110 - 1)
    flat = pd.Series(np.full(252, 100.0))
    assert cagr(flat) == pytest.approx(0.0)
    assert sharpe_ratio(pd.Series(np.zeros(100))) == 0.0


# --------------------------------------------------------------------- 최적화
def test_common_trade_start_uses_longest_warmup(universe):
    cands = [create_strategy("sma_cross", {"fast": 5, "slow": 40}),
             create_strategy("momentum", {"lookback": 200, "ma_filter": 200})]
    cal = next(iter(universe.values())).index
    start = common_trade_start(cal, cands)
    assert start == cal[max(c.warmup for c in cands)]


def test_grid_search_uses_identical_window(universe, config):
    cands = build_candidates("sma_cross", {"fast": [5, 20], "slow": [60, 120]})
    report = grid_search(
        universe, cands, config, workers=1, min_trades=0, progress=False,
        store_equity_top=len(cands),
    )
    assert len(report.results) == len(cands)
    curves = [r.equity for r in report.results if r.equity is not None]
    assert len(curves) == len(cands)
    for eq in curves[1:]:
        assert eq.index.equals(curves[0].index)  # 동일 시점 평가
        assert eq.iloc[0] == pytest.approx(curves[0].iloc[0])  # 동일 초기자본


def test_grid_search_ranks_descending(universe, config):
    cands = build_candidates("sma_cross", {"fast": [5, 10, 20], "slow": [40, 60, 120]})
    report = grid_search(universe, cands, config, workers=1, min_trades=0, progress=False)
    scores = [r.score for r in report.results]
    assert scores == sorted(scores, reverse=True)
    assert report.best.score == max(scores)


def test_grid_search_rejects_misaligned_calendars(config):
    a = SyntheticSource().get("A", "2020-01-01", "2022-01-01")
    b = SyntheticSource().get("B", "2020-06-01", "2022-01-01")
    with pytest.raises(ValueError, match="거래일 달력"):
        grid_search({"A": a, "B": b}, [create_strategy("buy_and_hold")], config,
                    workers=1, progress=False)


def test_min_trades_filter_demotes_inactive(universe, config):
    cands = build_candidates("sma_cross", {"fast": [5], "slow": [40]})
    report = grid_search(
        universe, cands, config, workers=1, min_trades=10**6, progress=False,
        store_equity_top=0,
    )
    assert all(r.filtered for r in report.results)
    with pytest.raises(ValueError, match="최소거래수"):
        _ = report.best


def test_benchmark_exempt_from_min_trades(universe, config):
    """buy_and_hold 는 거래가 적은 것이 정상 -> 필터에서 제외되어야 한다."""
    cands = [create_strategy("buy_and_hold")]
    report = grid_search(
        universe, cands, config, workers=1, min_trades=10**6, progress=False,
        store_equity_top=0,
    )
    assert not report.results[0].filtered
    assert np.isfinite(report.best.score)


def test_filtered_candidates_rank_last(universe, config):
    """필터된 후보는 통과 후보보다 항상 아래에 온다."""
    cands = build_candidates("sma_cross", {"fast": [5, 20], "slow": [40, 120]})
    report = grid_search(
        universe, cands, config, workers=1, min_trades=60, progress=False,
        store_equity_top=0,
    )
    flags = [r.filtered for r in report.results]
    assert flags == sorted(flags)  # False 들이 앞, True 들이 뒤


def test_objective_penalizes_small_samples():
    many = {"sharpe": 1.0, "n_trades": 100.0, "max_drawdown": -0.1, "turnover": 5.0}
    few = {"sharpe": 1.0, "n_trades": 3.0, "max_drawdown": -0.1, "turnover": 5.0}
    deep_dd = {"sharpe": 1.0, "n_trades": 100.0, "max_drawdown": -0.6, "turnover": 5.0}
    assert obj_robust(many) > obj_robust(few)
    assert obj_robust(many) > obj_robust(deep_dd)


def test_multi_strategy_candidates_are_unique():
    cands = build_all_candidates(["sma_cross", "rsi_reversion"])
    sigs = {c.signature() for c in cands}
    assert len(sigs) == len(cands)
    assert {c.name for c in cands} == {"sma_cross", "rsi_reversion"}


def test_sensitivity_table(universe, config):
    cands = build_candidates("sma_cross", {"fast": [5, 10, 20], "slow": [60, 120]})
    report = grid_search(universe, cands, config, workers=1, min_trades=0,
                         progress=False, store_equity_top=0)
    table = sensitivity(report, "sma_cross", "fast")
    assert set(table["value"]) == {5, 10, 20}
    assert "score_mean" in table.columns


def test_parallel_matches_serial(universe, config):
    cands = build_candidates("sma_cross", {"fast": [5, 10], "slow": [60, 120]})
    kw = dict(min_trades=0, progress=False, store_equity_top=0)
    serial = grid_search(universe, cands, config, workers=1, **kw)
    parallel = grid_search(universe, cands, config, workers=2, **kw)
    a = {r.label: round(r.score, 9) for r in serial.results}
    b = {r.label: round(r.score, 9) for r in parallel.results}
    assert a == b


# --------------------------------------------------------------------- 검증
def test_holdout_split_is_out_of_sample(universe, config):
    cands = build_candidates("sma_cross", {"fast": [5, 20], "slow": [60, 120]})
    results = holdout_validate(
        universe, cands, config, "2021-01-01", top_n=2, min_trades=0, workers=1
    )
    assert results
    for h in results:
        assert h.oos_equity is not None
        assert h.oos_equity.index[0] >= pd.Timestamp("2021-01-01")
        assert set(h.degradation) >= {"cagr", "sharpe"}


def test_walk_forward_produces_oos_curve(universe, config):
    cands = build_candidates("sma_cross", {"fast": [5, 20], "slow": [60, 120]})
    wf = walk_forward(
        universe, cands, config, train_days=400, test_days=200,
        min_trades=0, workers=1, progress=False,
    )
    assert len(wf.windows) >= 2
    assert wf.oos_equity is not None
    assert wf.oos_equity.index.is_monotonic_increasing
    assert not wf.oos_equity.index.duplicated().any()
    assert isinstance(wf.efficiency, float)
    frame = wf.windows_frame()
    assert set(frame.columns) >= {"test_start", "test_end", "selected", "oos_cagr"}
    # 각 검증창은 학습 종료 이후여야 한다
    assert (frame["test_start"] > frame["train_end"]).all()
