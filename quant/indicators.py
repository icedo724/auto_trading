"""기술적 지표 모음.

모든 함수는 **해당 시점까지의 정보만** 사용한다(룩어헤드 없음).
입력은 pandas Series/DataFrame, 출력은 동일 인덱스의 Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rolling_std(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).std(ddof=0)


def zscore(s: pd.Series, window: int) -> pd.Series:
    mu = sma(s, window)
    sd = rolling_std(s, window)
    return (s - mu) / sd.replace(0.0, np.nan)


def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI (지수평활 방식)."""
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # 하락이 전혀 없던 구간은 RSI=100
    return out.where(avg_loss.ne(0.0) | avg_gain.isna(), 100.0)


def macd(
    s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(macd line, signal line, histogram)."""
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(
    s: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(상단, 중심선, 하단)."""
    mid = sma(s, window)
    sd = rolling_std(s, window)
    return mid + num_std * sd, mid, mid - num_std * sd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def donchian(
    df: pd.DataFrame, window: int
) -> tuple[pd.Series, pd.Series]:
    """(N일 최고가, N일 최저가) — 당일 값을 제외한 과거 N일 기준."""
    upper = df["high"].rolling(window, min_periods=window).max().shift(1)
    lower = df["low"].rolling(window, min_periods=window).min().shift(1)
    return upper, lower


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """추세 강도(ADX). 값이 클수록 방향성 있는 추세."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    alpha = 1.0 / period
    atr_ = true_range(df).ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


def realized_vol(s: pd.Series, window: int, trading_days: int = 252) -> pd.Series:
    """연율화 실현변동성."""
    return s.pct_change().rolling(window, min_periods=window).std(ddof=0) * np.sqrt(trading_days)


def momentum(s: pd.Series, lookback: int, skip: int = 0) -> pd.Series:
    """lookback 기간 수익률. skip>0이면 최근 skip일을 제외(단기 반전 회피)."""
    ref = s.shift(skip)
    return ref / ref.shift(lookback) - 1.0
