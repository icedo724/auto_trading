"""백테스트 엔진 정확성 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.config import BacktestConfig, CostModel
from quant.engine import Backtester


def make_df(closes, opens=None, highs=None, lows=None) -> pd.DataFrame:
    n = len(closes)
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    closes = np.asarray(closes, dtype=float)
    opens = np.asarray(opens, dtype=float) if opens is not None else closes.copy()
    highs = np.asarray(highs, dtype=float) if highs is not None else np.maximum(opens, closes)
    lows = np.asarray(lows, dtype=float) if lows is not None else np.minimum(opens, closes)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": np.full(n, 1000.0)},
        index=idx,
    )


def zero_cost_config(**kw) -> BacktestConfig:
    return BacktestConfig(initial_cash=1_000_000, cost=CostModel.zero(), **kw)


def test_flat_signal_keeps_cash_flat():
    df = make_df([100, 110, 90, 130, 95])
    sig = pd.Series(0.0, index=df.index)
    res = Backtester(zero_cost_config()).run(df, sig, symbol="T")
    assert res.equity.nunique() == 1
    assert res.metrics["total_return"] == pytest.approx(0.0)
    assert res.n_trades == 0


def test_full_long_matches_buy_and_hold_without_costs():
    closes = [100, 105, 103, 110, 120]
    df = make_df(closes)
    sig = pd.Series(1.0, index=df.index)
    res = Backtester(zero_cost_config()).run(df, sig, symbol="T")

    # signal_lag=1 이므로 2번째 봉 시가(=종가 105)에 진입, 마지막 종가 120에 평가
    expected = 120 / 105 - 1
    assert res.metrics["total_return"] == pytest.approx(expected, rel=1e-9)


def test_signal_lag_prevents_lookahead():
    """마지막 봉에만 켜지는 신호는 절대 수익을 만들 수 없다."""
    df = make_df([100, 100, 100, 100, 200])
    sig = pd.Series([0, 0, 0, 0, 1.0], index=df.index)
    res = Backtester(zero_cost_config()).run(df, sig, symbol="T")
    assert res.metrics["total_return"] == pytest.approx(0.0)


def test_lag_zero_is_rejected():
    with pytest.raises(ValueError, match="signal_lag"):
        zero_cost_config(signal_lag=0).validate()


def test_costs_reduce_return():
    closes = [100] * 20
    df = make_df(closes)
    # 매수/매도를 반복 -> 비용만 계속 발생
    sig = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(20)], index=df.index)

    free = Backtester(zero_cost_config()).run(df, sig, symbol="T")
    costly = Backtester(
        BacktestConfig(initial_cash=1_000_000, cost=CostModel.kr_stock())
    ).run(df, sig, symbol="T")

    assert free.metrics["total_return"] == pytest.approx(0.0, abs=1e-12)
    assert costly.metrics["total_return"] < -0.01
    assert costly.total_cost > 0


def test_stop_loss_triggers_intrabar():
    # 3번째 봉에서 저가가 -20% 까지 밀림
    df = make_df(
        closes=[100, 100, 95, 100, 100],
        opens=[100, 100, 100, 100, 100],
        highs=[100, 100, 100, 100, 100],
        lows=[100, 100, 80, 100, 100],
    )
    sig = pd.Series(1.0, index=df.index)
    cfg = zero_cost_config(stop_loss_pct=0.10)
    res = Backtester(cfg).run(df, sig, symbol="T")

    reasons = [t.exit_reason for t in res.trades]
    assert "stop_loss" in reasons
    stop_trade = next(t for t in res.trades if t.exit_reason == "stop_loss")
    assert stop_trade.exit_price == pytest.approx(90.0, rel=1e-6)
    # 손절 후에는 신호가 0으로 리셋될 때까지 재진입 금지 -> 계속 1.0이므로 재진입 없음
    assert res.position.iloc[-1] == pytest.approx(0.0)


def test_take_profit_triggers():
    df = make_df(
        closes=[100, 100, 105, 105],
        opens=[100, 100, 100, 105],
        highs=[100, 100, 130, 105],
        lows=[100, 100, 100, 105],
    )
    sig = pd.Series(1.0, index=df.index)
    res = Backtester(zero_cost_config(take_profit_pct=0.20)).run(df, sig, symbol="T")
    assert any(t.exit_reason == "take_profit" for t in res.trades)
    assert res.metrics["total_return"] == pytest.approx(0.20, rel=1e-6)


def test_trailing_stop_locks_in_gain():
    df = make_df(
        closes=[100, 100, 150, 120],
        opens=[100, 100, 100, 150],
        highs=[100, 100, 150, 150],
        lows=[100, 100, 100, 120],
    )
    sig = pd.Series(1.0, index=df.index)
    res = Backtester(zero_cost_config(trailing_stop_pct=0.10)).run(df, sig, symbol="T")
    trailing = [t for t in res.trades if t.exit_reason == "trailing_stop"]
    assert trailing
    assert trailing[0].exit_price == pytest.approx(135.0, rel=1e-6)


def test_max_holding_days_forces_exit():
    df = make_df([100] * 10)
    sig = pd.Series(1.0, index=df.index)
    res = Backtester(zero_cost_config(max_holding_days=3)).run(df, sig, symbol="T")
    assert any(t.exit_reason == "time_exit" for t in res.trades)


def test_trade_start_aligns_evaluation_window():
    df = make_df(list(range(100, 130)))
    sig = pd.Series(1.0, index=df.index)
    start = df.index[20]
    res = Backtester(zero_cost_config()).run(df, sig, symbol="T", trade_start=start)
    assert res.equity.index[0] == start
    assert res.equity.iloc[0] == pytest.approx(1_000_000)


def test_rebalance_threshold_suppresses_micro_trades():
    df = make_df([100] * 30)
    sig = pd.Series(0.5, index=df.index)
    sig.iloc[10:] = 0.52  # 2%p 변화 -> 임계치 5% 미만이므로 거래 없음
    cfg = zero_cost_config(rebalance_threshold=0.05)
    res = Backtester(cfg).run(df, sig, symbol="T")
    # 최초 진입 1회 + 종료 정리만 존재
    assert res.n_trades <= 1


def test_short_position_profits_on_decline():
    df = make_df([100, 100, 90, 80])
    sig = pd.Series(-1.0, index=df.index)
    cfg = zero_cost_config(allow_short=True)
    res = Backtester(cfg).run(df, sig, symbol="T")
    assert res.metrics["total_return"] > 0
    assert res.position.iloc[1] < 0


def test_short_blocked_when_disallowed():
    df = make_df([100, 100, 90, 80])
    sig = pd.Series(-1.0, index=df.index)
    res = Backtester(zero_cost_config(allow_short=False)).run(df, sig, symbol="T")
    assert res.metrics["total_return"] == pytest.approx(0.0)
    assert (res.position == 0).all()


def test_integer_shares_mode():
    df = make_df([333.0] * 10)
    sig = pd.Series(1.0, index=df.index)
    cfg = BacktestConfig(
        initial_cash=100_000, cost=CostModel.zero(), allow_fractional=False
    )
    res = Backtester(cfg).run(df, sig, symbol="T")
    qty = res.trades[0].quantity if res.trades else 0
    assert qty == float(int(qty))


def test_nan_prices_are_skipped():
    df = make_df([100, 105, 110, 115, 120])
    df.loc[df.index[:2], ["open", "high", "low", "close"]] = np.nan
    sig = pd.Series(1.0, index=df.index)
    res = Backtester(zero_cost_config()).run(df, sig, symbol="T")
    assert np.isfinite(res.equity).all()
    assert res.position.iloc[0] == 0.0
