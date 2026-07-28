"""전략 · 지표 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import indicators as ind
from quant.data import SyntheticSource
from quant.strategy import (
    available_strategies,
    build_candidates,
    create_strategy,
    expand_grid,
    get_strategy_class,
)


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return SyntheticSource().get("TEST-A", "2018-01-01", "2024-12-31")


# --------------------------------------------------------------------- 지표
def test_sma_matches_manual():
    s = pd.Series([1.0, 2, 3, 4, 5])
    out = ind.sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_rsi_bounds(sample):
    r = ind.rsi(sample["close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_rsi_all_up_is_100():
    s = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(s, 14).iloc[-1] == pytest.approx(100.0)


def test_atr_positive(sample):
    a = ind.atr(sample, 14).dropna()
    assert (a > 0).all()


def test_donchian_excludes_current_bar(sample):
    upper, lower = ind.donchian(sample, 20)
    # 당일 고가를 포함하면 돌파가 절대 발생할 수 없다 -> 돌파가 존재해야 정상
    assert (sample["close"] > upper).any()
    valid = upper.notna() & lower.notna()
    assert (lower[valid] <= upper[valid]).all()
    # shift(1) 로 당일 값이 제외되었는지 확인
    assert upper.iloc[:20].isna().all()


def test_bollinger_ordering(sample):
    up, mid, lo = ind.bollinger(sample["close"], 20, 2.0)
    valid = mid.notna()
    assert (up[valid] >= mid[valid]).all()
    assert (mid[valid] >= lo[valid]).all()


# --------------------------------------------------------------------- 전략
def test_all_strategies_produce_valid_signals(sample):
    for name in available_strategies():
        strat = create_strategy(name)
        sig = strat.generate_signals(sample)
        assert len(sig) == len(sample), name
        assert sig.index.equals(sample.index), name
        assert np.isfinite(sig).all(), name
        assert (sig.abs() <= 1.0 + 1e-9).all(), name


def test_every_strategy_actually_trades(sample):
    for name in available_strategies():
        sig = create_strategy(name).generate_signals(sample)
        assert (sig != 0.0).any(), f"{name}: 신호가 전혀 발생하지 않음"


def test_warmup_is_sufficient_history(sample):
    """``warmup`` 봉의 과거만 주어져도 신호가 곧 원래 경로로 수렴해야 한다.

    optimizer 는 모든 후보의 warmup 최댓값 시점부터 평가를 시작하므로,
    "그만큼의 과거만 있으면 충분한가"가 핵심 성질이다.

    완전 일치를 요구할 수는 없다.
      · EMA 계열(macd, rsi)은 무한 메모리라 초기값이 미세하게 다르다.
      · 상태 보유형(bollinger, zscore, donchian)은 진입/청산 상태가 경로 의존적이라
        잘린 시계열이 포지션 도중에 시작하면 첫 청산 신호까지 어긋난다.
    두 경우 모두 **불일치는 앞쪽 구간에 한정**되고 이후로는 영구히 일치해야 한다.
    """
    cut = len(sample) // 2
    for name in available_strategies():
        strat = create_strategy(name)
        w = strat.warmup
        if w == 0 or cut - w < 0:
            continue
        full = strat.generate_signals(sample).iloc[cut:].to_numpy()
        limited = strat.generate_signals(sample.iloc[cut - w :]).iloc[w:].to_numpy()

        mismatch = np.flatnonzero(full != limited)
        if mismatch.size == 0:
            continue
        # 마지막 불일치 이후로는 완전히 동일해야 하고, 수렴은 warmup 안에 끝나야 한다
        last = int(mismatch[-1])
        assert last <= w, f"{name}: 수렴 지연 (마지막 불일치 index {last} > warmup {w})"
        assert (full[last + 1 :] == limited[last + 1 :]).all(), f"{name}: 재발산"


def test_signals_are_causal(sample):
    """미래 데이터를 잘라내도 과거 신호는 변하지 않아야 한다 (룩어헤드 없음)."""
    cut = len(sample) // 2
    for name in available_strategies():
        strat = create_strategy(name)
        full = strat.generate_signals(sample).iloc[:cut]
        partial = strat.generate_signals(sample.iloc[:cut])
        pd.testing.assert_series_equal(full, partial, check_names=False, obj=name)


def test_buy_and_hold_is_always_long(sample):
    sig = create_strategy("buy_and_hold").generate_signals(sample)
    assert (sig == 1.0).all()


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        create_strategy("sma_cross", {"fast": 60, "slow": 20})
    with pytest.raises(ValueError):
        create_strategy("sma_cross", {"nonexistent": 1})
    with pytest.raises(ValueError):
        create_strategy("rsi_reversion", {"oversold": 60, "exit_level": 50})


def test_expand_grid_cartesian():
    combos = list(expand_grid({"a": [1, 2], "b": ["x", "y"]}))
    assert len(combos) == 4
    assert {"a": 1, "b": "x"} in combos


def test_build_candidates_filters_invalid():
    cands = build_candidates("sma_cross", {"fast": [10, 50], "slow": [20, 60]})
    # fast=50, slow=20 조합은 무효 -> 제외
    assert all(c["fast"] < c["slow"] for c in cands)
    assert len(cands) == 3


def test_default_param_spaces_are_non_empty():
    for name in available_strategies():
        cls = get_strategy_class(name)
        if name == "buy_and_hold":
            continue
        assert cls.param_space, f"{name}: param_space 가 비어 있음"
        assert build_candidates(name), f"{name}: 유효 후보 0개"


def test_stateful_strategies_hold_between_signals(sample):
    """진입 후 청산 신호 전까지 포지션이 유지되는지 (0/1 사이 진동 방지)."""
    sig = create_strategy("donchian").generate_signals(sample)
    runs = (sig != sig.shift()).cumsum()
    longest_hold = sig[sig > 0].groupby(runs[sig > 0]).size().max()
    assert longest_hold > 5
