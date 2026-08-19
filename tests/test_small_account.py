"""소액 적립식 운용 관련 기능 테스트 (적립 회계 · 최소주문 · 그리드 · 코인 소스)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from quant.config import BacktestConfig, CostModel
from quant.data import UpbitSource, get_source
from quant.data.base import DataError
from quant.engine import Backtester
from quant.metrics import money_weighted_return
from quant.strategy import create_strategy

N = 504
IDX = pd.bdate_range("2024-01-01", periods=N)


def flat_df(prices) -> pd.DataFrame:
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {"open": p, "high": p, "low": p, "close": p, "volume": np.full(N, 1e3)}, index=IDX
    )


def dca_config(**kw) -> BacktestConfig:
    base = dict(
        initial_cash=100_000, cost=CostModel.zero(),
        contribution=100_000, contribution_freq="M", trading_days=252,
    )
    base.update(kw)
    return BacktestConfig(**base)


# --------------------------------------------------------------- 적립 회계
def test_deposits_do_not_create_phantom_returns():
    """가격이 전혀 안 움직이면, 아무리 입금해도 수익률은 0이어야 한다."""
    res = Backtester(dca_config()).run(flat_df([100.0] * N), pd.Series(1.0, index=IDX), symbol="T")
    m = res.metrics
    assert res.contributions is not None and (res.contributions > 0).sum() == 23
    assert m["total_return"] == pytest.approx(0.0, abs=1e-12)
    assert m["net_profit"] == pytest.approx(0.0, abs=1e-6)
    assert m["total_invested"] == pytest.approx(100_000 * 24)
    assert m["final_balance"] == pytest.approx(m["total_invested"])


def test_drawdown_not_masked_by_deposits():
    """입금으로 잔고가 회복돼도 MDD 는 전략 손실을 그대로 보여야 한다."""
    prices = np.concatenate([np.full(N // 2, 100.0), np.full(N - N // 2, 50.0)])
    res = Backtester(dca_config()).run(flat_df(prices), pd.Series(1.0, index=IDX), symbol="T")
    assert res.metrics["max_drawdown"] < -0.4          # 반토막이 드러난다
    assert res.balance.iloc[-1] > res.balance.iloc[N // 2]  # 잔고 자체는 입금으로 증가


def test_twr_ignores_deposit_timing_but_mwr_does_not():
    prices = np.linspace(100, 200, N)
    res = Backtester(dca_config()).run(flat_df(prices), pd.Series(1.0, index=IDX), symbol="T")
    m = res.metrics
    # 늦게 넣은 돈은 상승을 덜 먹으므로 원금대비 수익률 < TWR
    assert m["net_profit"] / m["total_invested"] < m["total_return"]
    assert 0 < m["mwr"] < m["total_return"]


def test_no_contribution_is_backward_compatible():
    prices = np.linspace(100, 200, N)
    df = flat_df(prices)
    cfg = BacktestConfig(initial_cash=100_000, cost=CostModel.zero())
    res = Backtester(cfg).run(df, pd.Series(1.0, index=IDX), symbol="T")
    assert res.metrics["total_return"] == pytest.approx(200 / df["open"].iloc[1] - 1)
    assert "total_invested" not in res.metrics      # 적립 지표는 나오지 않는다
    assert res.contributions.sum() == 0


def test_contribution_frequencies():
    counts = {}
    for freq, expected in [("M", 23), ("Q", 7), ("W", 100)]:
        res = Backtester(dca_config(contribution_freq=freq)).run(
            flat_df([100.0] * N), pd.Series(0.0, index=IDX), symbol="T"
        )
        counts[freq] = int((res.contributions > 0).sum())
        assert counts[freq] == expected, (freq, counts[freq])
    assert counts["W"] > counts["M"] > counts["Q"]


def test_invalid_contribution_settings_rejected():
    with pytest.raises(ValueError, match="contribution"):
        BacktestConfig(contribution=-1).validate()
    with pytest.raises(ValueError, match="contribution_freq"):
        BacktestConfig(contribution_freq="Y").validate()


# ------------------------------------------------------------ 최소 주문금액
def test_min_order_value_blocks_dust_trades():
    """거래소 최소 주문금액 미만은 주문 자체가 불가능하다."""
    df = flat_df([100.0] * N)
    sig = pd.Series([1.0 if i % 2 == 0 else 0.98 for i in range(N)], index=IDX)
    common = dict(initial_cash=100_000, cost=CostModel.zero(), rebalance_threshold=0.0)

    free = Backtester(BacktestConfig(**common)).run(df, sig, symbol="T")
    gated = Backtester(BacktestConfig(min_order_value=50_000, **common)).run(df, sig, symbol="T")
    assert gated.n_trades < free.n_trades


# ------------------------------------------------------------------ MWR
def test_mwr_matches_known_case():
    # 1년 뒤 1.1배: 100원 넣어 110원 -> IRR 10%
    assert money_weighted_return([(0.0, 100.0)], 110.0, 1.0) == pytest.approx(0.10, abs=1e-6)
    # 손실도 잡아낸다
    assert money_weighted_return([(0.0, 100.0)], 90.0, 1.0) == pytest.approx(-0.10, abs=1e-6)
    # 비정상 입력은 0
    assert money_weighted_return([], 100.0, 1.0) == 0.0


# ----------------------------------------------------------------- 그리드
def test_grid_reacts_to_deviation_not_price_level():
    """그리드는 '가격이 비싼가'가 아니라 '기준선보다 위인가'로 판단한다.

    등속 상승장에서는 종가/이평 이격도가 거의 일정하므로 비중도 일정하다.
    이것은 결함이 아니라 설계다 — 추세를 타되 잔파도만 먹는다.
    """
    g = create_strategy("grid", {"ma": 20, "band": 0.20, "levels": 4})

    steady = g.generate_signals(flat_df(np.linspace(100, 130, N))).iloc[60:]
    assert steady.nunique() == 1 and steady.iloc[0] == pytest.approx(0.5)

    # 기준선 위로 튀면 덜어내고, 아래로 빠지면 담는다
    up = np.full(N, 100.0); up[300:] = 112.0
    down = np.full(N, 100.0); down[300:] = 88.0
    assert g.generate_signals(flat_df(up)).iloc[305] == pytest.approx(0.0)
    assert g.generate_signals(flat_df(down)).iloc[305] == pytest.approx(1.0)


def test_grid_is_quantized_to_levels():
    g = create_strategy("grid", {"ma": 20, "band": 0.20, "levels": 4})
    w = g.generate_signals(flat_df(100 + 10 * np.sin(np.linspace(0, 20, N)))).dropna()
    assert set(np.round(w.unique(), 6)) <= {0.0, 0.25, 0.5, 0.75, 1.0}


def test_grid_step_pct_is_the_cost_yardstick():
    assert create_strategy("grid", {"band": 0.20, "levels": 4}).step_pct == pytest.approx(0.05)
    assert create_strategy("grid", {"band": 0.02, "levels": 10}).step_pct == pytest.approx(0.002)


def test_grid_rejects_bad_params():
    for bad in ({"band": 0.0}, {"band": 1.5}, {"levels": 0}, {"ma": 1}):
        with pytest.raises(ValueError):
            create_strategy("grid", bad)


# ------------------------------------------------------------- 비용 모델
def test_crypto_cost_models():
    up, kr = CostModel.crypto_upbit(), CostModel.kr_stock()

    def rt(c):
        s, m, t = c.slippage_bps * 1e-4, c.commission_bps * 1e-4, c.sell_tax_bps * 1e-4
        return 1 - (1 - s) * (1 - m - t) / ((1 + s) * (1 + m))

    assert rt(up) == pytest.approx(0.0020, abs=1e-5)   # 왕복 20bp
    assert rt(kr) == pytest.approx(0.0031, abs=1e-5)   # 왕복 31bp
    assert rt(up) < rt(kr)                             # 코인이 더 싸다
    assert up.sell_tax_bps == 0.0                      # 증권거래세 없음
    assert CostModel.named("crypto") == up


# ------------------------------------------------------------- 업비트 소스
UPBIT_ROWS = [
    {"market": "KRW-BTC", "candle_date_time_kst": "2024-01-03T09:00:00",
     "opening_price": 58_000_000.0, "high_price": 59_000_000.0, "low_price": 57_000_000.0,
     "trade_price": 58_500_000.0, "candle_acc_trade_volume": 1234.5},
    {"market": "KRW-BTC", "candle_date_time_kst": "2024-01-02T09:00:00",
     "opening_price": 57_000_000.0, "high_price": 58_200_000.0, "low_price": 56_500_000.0,
     "trade_price": 58_000_000.0, "candle_acc_trade_volume": 2345.6},
]


def test_upbit_parses_and_sorts_ascending():
    from quant.data.base import normalize_ohlcv

    df = normalize_ohlcv(UpbitSource.parse(UPBIT_ROWS, "KRW-BTC"), symbol="KRW-BTC")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing          # 업비트는 최신순 -> 오름차순 정렬
    assert df["close"].iloc[0] == 58_000_000.0
    assert df["close"].iloc[-1] == 58_500_000.0


def test_upbit_market_code_normalization():
    s = UpbitSource()
    assert s.market_code("btc") == "KRW-BTC"
    assert s.market_code("KRW-ETH") == "KRW-ETH"


def test_upbit_errors():
    with pytest.raises(DataError, match="비어 있습니다"):
        UpbitSource.parse([], "KRW-BTC")
    with pytest.raises(DataError, match="컬럼 누락"):
        UpbitSource.parse([{"market": "KRW-BTC"}], "KRW-BTC")


def test_upbit_paginates_until_start(monkeypatch):
    """200봉 상한이 있으므로 start 에 닿을 때까지 페이징해야 한다."""
    calls = []

    def make_batch(to_ts: pd.Timestamp, n: int):
        return [
            {"market": "KRW-BTC",
             "candle_date_time_kst": (to_ts - pd.Timedelta(days=k + 1)).strftime("%Y-%m-%dT09:00:00"),
             "opening_price": 100.0, "high_price": 101.0, "low_price": 99.0,
             "trade_price": 100.0, "candle_acc_trade_volume": 1.0}
            for k in range(n)
        ]

    def fake_get(url, params=None, timeout=None):
        calls.append(pd.Timestamp(params["to"]))
        return type("R", (), {"json": lambda s: make_batch(calls[-1], 200),
                              "raise_for_status": lambda s: None})()

    import time as _time

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=fake_get))
    monkeypatch.setattr(_time, "sleep", lambda s: None)   # 유량 제한 대기 건너뛰기
    df = UpbitSource().get("KRW-BTC", "2023-01-01", "2024-01-10")

    # 1회 200봉 상한이므로 375일 구간은 2회 이상 요청해야 채워진다
    assert len(calls) >= 2
    assert calls[-1] < calls[0]                       # 커서가 과거로 이동
    assert len(df) > 200                              # 상한을 넘겨 수집
    assert df.index.is_monotonic_increasing
    # start 를 덮은 뒤에는 더 요청하지 않는다 (무한 페이징 방지)
    assert calls[-1] >= pd.Timestamp("2023-01-01") - pd.Timedelta(days=200)


# ------------------------------------------------------------------ 소스 등록
def test_upbit_source_aliases():
    for alias in ("upbit", "coin", "crypto"):
        assert isinstance(get_source(alias, cache_dir=None), UpbitSource)
