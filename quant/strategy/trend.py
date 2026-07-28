"""추세추종 계열 전략."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import Strategy, register


@register
class BuyAndHold(Strategy):
    """벤치마크: 첫 거래일부터 100% 매수 후 보유."""

    name = "buy_and_hold"
    defaults = {}
    param_space = {}
    is_benchmark = True

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(1.0, index=df.index)
        return self._finalize(sig.where(df["close"].notna()), df.index)


@register
class SmaCross(Strategy):
    """이동평균 골든/데드 크로스.

    fast > slow 이면 롱. ``trend_filter`` 기간 장기 이평 위에 있을 때만 진입.
    ``allow_short=True`` 면 데드크로스에서 숏.
    """

    name = "sma_cross"
    defaults = {"fast": 20, "slow": 60, "trend_filter": 0, "allow_short": False}
    param_space = {
        "fast": [5, 10, 20, 30],
        "slow": [40, 60, 90, 120],
        "trend_filter": [0, 200],
        "allow_short": [False],
    }

    def validate(self) -> None:
        if self["fast"] < 2 or self["slow"] < 3:
            raise ValueError("이동평균 기간이 너무 짧습니다.")
        if self["fast"] >= self["slow"]:
            raise ValueError("fast 는 slow 보다 작아야 합니다.")

    @property
    def warmup(self) -> int:
        return max(self["slow"], self["trend_filter"]) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        fast = ind.sma(close, self["fast"])
        slow = ind.sma(close, self["slow"])

        long = fast > slow
        if self["trend_filter"]:
            long &= close > ind.sma(close, self["trend_filter"])

        sig = long.astype(float)
        if self["allow_short"]:
            short = fast < slow
            if self["trend_filter"]:
                short &= close < ind.sma(close, self["trend_filter"])
            sig = sig - short.astype(float)

        # warmup 구간(지표 NaN)은 무포지션
        sig = sig.where(fast.notna() & slow.notna())
        return self._finalize(sig, df.index)


@register
class MacdTrend(Strategy):
    """MACD 히스토그램 부호로 추세 추종."""

    name = "macd"
    defaults = {"fast": 12, "slow": 26, "signal": 9, "trend_filter": 0}
    param_space = {
        "fast": [8, 12, 16],
        "slow": [21, 26, 34],
        "signal": [5, 9, 13],
        "trend_filter": [0, 120, 200],
    }

    def validate(self) -> None:
        if self["fast"] >= self["slow"]:
            raise ValueError("fast 는 slow 보다 작아야 합니다.")
        if self["signal"] < 2:
            raise ValueError("signal 기간이 너무 짧습니다.")

    @property
    def warmup(self) -> int:
        return max(self["slow"] + self["signal"], self["trend_filter"]) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        line, sig_line, hist = ind.macd(close, self["fast"], self["slow"], self["signal"])
        long = (hist > 0) & (line > 0)
        if self["trend_filter"]:
            long &= close > ind.sma(close, self["trend_filter"])
        return self._finalize(long.astype(float).where(hist.notna()), df.index)


@register
class DonchianBreakout(Strategy):
    """터틀식 채널 돌파.

    N일 신고가 돌파 시 진입, M일 신저가 이탈 시 청산.
    ``atr_filter`` > 0 이면 변동성이 과도한 구간에서는 진입하지 않는다.
    """

    name = "donchian"
    defaults = {"entry": 20, "exit": 10, "atr_period": 14, "atr_max": 0.0}
    param_space = {
        "entry": [20, 40, 55, 80],
        "exit": [10, 20, 30],
        "atr_period": [14],
        "atr_max": [0.0, 0.05],
    }

    def validate(self) -> None:
        if self["entry"] < 5 or self["exit"] < 2:
            raise ValueError("채널 기간이 너무 짧습니다.")
        if self["exit"] > self["entry"]:
            raise ValueError("exit 채널은 entry 채널보다 길 수 없습니다.")

    @property
    def warmup(self) -> int:
        return max(self["entry"], self["exit"], self["atr_period"]) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        upper, _ = ind.donchian(df, self["entry"])
        _, lower = ind.donchian(df, self["exit"])
        close = df["close"]

        entry = close > upper
        exit_ = close < lower

        if self["atr_max"] > 0:
            atr_pct = ind.atr(df, self["atr_period"]) / close
            entry &= atr_pct < self["atr_max"]

        # 상태 유지(stateful): 진입하면 청산 신호 전까지 보유
        state = np.zeros(len(df), dtype=float)
        e = entry.to_numpy()
        x = exit_.to_numpy()
        pos = 0.0
        for i in range(len(df)):
            if pos == 0.0 and e[i]:
                pos = 1.0
            elif pos == 1.0 and x[i]:
                pos = 0.0
            state[i] = pos

        sig = pd.Series(state, index=df.index).where(upper.notna())
        return self._finalize(sig, df.index)


@register
class TimeSeriesMomentum(Strategy):
    """시계열 모멘텀: 과거 lookback 수익률이 문턱을 넘으면 롱."""

    name = "momentum"
    defaults = {"lookback": 120, "skip": 0, "threshold": 0.0, "ma_filter": 0}
    param_space = {
        "lookback": [20, 60, 120, 200],
        "skip": [0, 5, 20],
        "threshold": [0.0, 0.05],
        "ma_filter": [0, 200],
    }

    def validate(self) -> None:
        if self["lookback"] < 5:
            raise ValueError("lookback 이 너무 짧습니다.")
        if self["skip"] < 0:
            raise ValueError("skip 은 0 이상이어야 합니다.")

    @property
    def warmup(self) -> int:
        return self["lookback"] + self["skip"] + max(self["ma_filter"], 0) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        mom = ind.momentum(close, self["lookback"], self["skip"])
        long = mom > self["threshold"]
        if self["ma_filter"]:
            long &= close > ind.sma(close, self["ma_filter"])
        return self._finalize(long.astype(float).where(mom.notna()), df.index)
