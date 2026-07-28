"""평균회귀 계열 전략."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import Strategy, register


def _hold_between(entry: pd.Series, exit_: pd.Series) -> np.ndarray:
    """진입 신호에서 켜고 청산 신호에서 끄는 상태 배열."""
    e = entry.fillna(False).to_numpy()
    x = exit_.fillna(False).to_numpy()
    state = np.zeros(len(e), dtype=float)
    pos = 0.0
    for i in range(len(e)):
        if pos == 0.0 and e[i]:
            pos = 1.0
        elif pos == 1.0 and x[i]:
            pos = 0.0
        state[i] = pos
    return state


@register
class RsiReversion(Strategy):
    """RSI 과매도 매수 · 회복 시 청산.

    ``trend_filter`` 기간 이평 위에 있을 때만 진입하면 하락장 물타기를 피할 수 있다.
    """

    name = "rsi_reversion"
    defaults = {"period": 14, "oversold": 30, "exit_level": 55, "trend_filter": 0}
    param_space = {
        "period": [7, 14, 21],
        "oversold": [20, 25, 30, 35],
        "exit_level": [50, 55, 60, 70],
        "trend_filter": [0, 120, 200],
    }

    def validate(self) -> None:
        if self["period"] < 2:
            raise ValueError("RSI 기간이 너무 짧습니다.")
        if not 0 < self["oversold"] < 100:
            raise ValueError("oversold 는 (0, 100) 범위여야 합니다.")
        if self["exit_level"] <= self["oversold"]:
            raise ValueError("exit_level 은 oversold 보다 커야 합니다.")

    @property
    def warmup(self) -> int:
        return max(self["period"] * 3, self["trend_filter"]) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        r = ind.rsi(close, self["period"])

        entry = r < self["oversold"]
        if self["trend_filter"]:
            entry &= close > ind.sma(close, self["trend_filter"])
        exit_ = r > self["exit_level"]

        sig = pd.Series(_hold_between(entry, exit_), index=df.index).where(r.notna())
        return self._finalize(sig, df.index)


@register
class BollingerBands(Strategy):
    """볼린저 밴드.

    mode="reversion": 하단 이탈 매수 → 중심선 회귀 시 청산
    mode="breakout" : 상단 돌파 매수 → 중심선 이탈 시 청산
    """

    name = "bollinger"
    defaults = {"period": 20, "num_std": 2.0, "mode": "reversion", "trend_filter": 0}
    param_space = {
        "period": [10, 20, 30, 60],
        "num_std": [1.5, 2.0, 2.5],
        "mode": ["reversion", "breakout"],
        "trend_filter": [0, 200],
    }

    def validate(self) -> None:
        if self["period"] < 5:
            raise ValueError("볼린저 기간이 너무 짧습니다.")
        if self["num_std"] <= 0:
            raise ValueError("num_std 는 0보다 커야 합니다.")
        if self["mode"] not in ("reversion", "breakout"):
            raise ValueError("mode 는 reversion 또는 breakout 이어야 합니다.")

    @property
    def warmup(self) -> int:
        return max(self["period"], self["trend_filter"]) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        upper, mid, lower = ind.bollinger(close, self["period"], self["num_std"])

        if self["mode"] == "reversion":
            entry, exit_ = close < lower, close > mid
        else:
            entry, exit_ = close > upper, close < mid

        if self["trend_filter"]:
            entry &= close > ind.sma(close, self["trend_filter"])

        sig = pd.Series(_hold_between(entry, exit_), index=df.index).where(mid.notna())
        return self._finalize(sig, df.index)


@register
class ZScoreReversion(Strategy):
    """가격 z-score 평균회귀. 진입/청산 문턱을 독립적으로 탐색한다."""

    name = "zscore"
    defaults = {"window": 20, "entry_z": -2.0, "exit_z": 0.0, "trend_filter": 0}
    param_space = {
        "window": [10, 20, 40, 60],
        "entry_z": [-1.5, -2.0, -2.5],
        "exit_z": [-0.5, 0.0, 0.5],
        "trend_filter": [0, 200],
    }

    def validate(self) -> None:
        if self["window"] < 5:
            raise ValueError("window 가 너무 짧습니다.")
        if self["entry_z"] >= self["exit_z"]:
            raise ValueError("entry_z 는 exit_z 보다 작아야 합니다.")

    @property
    def warmup(self) -> int:
        return max(self["window"], self["trend_filter"]) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        z = ind.zscore(close, self["window"])

        entry = z < self["entry_z"]
        if self["trend_filter"]:
            entry &= close > ind.sma(close, self["trend_filter"])
        exit_ = z > self["exit_z"]

        sig = pd.Series(_hold_between(entry, exit_), index=df.index).where(z.notna())
        return self._finalize(sig, df.index)


@register
class VolTargetTrend(Strategy):
    """변동성 타겟팅 추세추종.

    이평 위에 있을 때만 보유하되, 비중을 ``target_vol / 실현변동성`` 으로 조절한다.
    변동성이 높은 국면에서 자동으로 노출을 줄이므로 MDD 개선 효과가 있다.
    """

    name = "vol_target"
    defaults = {"ma": 120, "vol_window": 20, "target_vol": 0.15, "max_weight": 1.0}
    param_space = {
        "ma": [60, 120, 200],
        "vol_window": [10, 20, 60],
        "target_vol": [0.10, 0.15, 0.25],
        "max_weight": [1.0],
    }

    def validate(self) -> None:
        if self["ma"] < 5 or self["vol_window"] < 5:
            raise ValueError("기간 파라미터가 너무 짧습니다.")
        if self["target_vol"] <= 0:
            raise ValueError("target_vol 은 0보다 커야 합니다.")

    @property
    def warmup(self) -> int:
        return max(self["ma"], self["vol_window"]) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        trend_ok = close > ind.sma(close, self["ma"])
        vol = ind.realized_vol(close, self["vol_window"])
        weight = (self["target_vol"] / vol.replace(0.0, np.nan)).clip(
            upper=self["max_weight"]
        )
        sig = weight.where(trend_ok, 0.0).where(vol.notna())
        # 잦은 미세 리밸런싱을 줄이기 위해 비중을 0.1 단위로 양자화
        return self._finalize((sig / 0.1).round() * 0.1, df.index)
