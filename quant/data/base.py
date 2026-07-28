"""데이터 소스 공통 규약.

모든 소스는 아래 스키마의 일봉 DataFrame을 반환한다.

    index : pd.DatetimeIndex (tz-naive, 오름차순, 중복 없음), name="date"
    cols  : open, high, low, close, volume  (float, 수정주가 기준)
"""

from __future__ import annotations

import abc
from typing import Iterable

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    """데이터 조회/검증 실패."""


class DataSource(abc.ABC):
    """일봉 OHLCV 공급자."""

    name: str = "base"

    @abc.abstractmethod
    def _fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """소스별 원본 조회. 표준화 전 DataFrame을 반환."""

    def get(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        df = self._fetch(symbol, start, end)
        return normalize_ohlcv(df, symbol=symbol)

    def get_many(
        self, symbols: Iterable[str], start: str, end: str, *, skip_errors: bool = True
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                out[sym] = self.get(sym, start, end)
            except Exception as exc:  # noqa: BLE001 - 종목 하나 때문에 전체가 죽지 않도록
                if not skip_errors:
                    raise
                print(f"[warn] {self.name}: {sym} 조회 실패 - {exc}")
        if not out:
            raise DataError("조회에 성공한 종목이 없습니다.")
        return out


def normalize_ohlcv(df: pd.DataFrame, *, symbol: str = "") -> pd.DataFrame:
    """임의의 OHLCV 형태를 표준 스키마로 변환하고 무결성을 검증한다."""
    if df is None or len(df) == 0:
        raise DataError(f"{symbol}: 데이터가 비어 있습니다.")

    df = df.copy()

    # yfinance 멀티인덱스 컬럼(단일 티커) 평탄화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in ("adj_close", "adjclose", "수정종가"):
            key = "close"
        rename[col] = {
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
        }.get(key, key)
    df = df.rename(columns=rename)

    # 중복 컬럼(예: close 와 adj_close 가 모두 close 로 매핑) 제거 — 마지막 것 사용
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{symbol}: 필수 컬럼 누락 {missing} (보유: {list(df.columns)})")

    df = df[OHLCV_COLUMNS].astype(float)

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = pd.DatetimeIndex(idx.normalize(), name="date")

    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 가격이 0/음수/결측인 행은 거래 불가로 간주하고 제거
    prices = df[["open", "high", "low", "close"]]
    valid = prices.notna().all(axis=1) & (prices > 0).all(axis=1)
    df = df[valid]
    if df.empty:
        raise DataError(f"{symbol}: 유효한 가격 데이터가 없습니다.")

    df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)

    # high/low 정합성 보정 (일부 소스에서 어긋나는 경우가 있음)
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    return df


def slice_period(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """[start, end] 구간으로 자른다(양 끝 포함)."""
    out = df
    if start:
        out = out[out.index >= pd.Timestamp(start)]
    if end:
        out = out[out.index <= pd.Timestamp(end)]
    return out


def common_calendar(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """여러 종목이 **동일 시점**에서 비교되도록 공통 거래일 인덱스를 만든다."""
    idx: pd.DatetimeIndex | None = None
    for df in data.values():
        idx = df.index if idx is None else idx.union(df.index)
    if idx is None or len(idx) == 0:
        raise DataError("공통 거래일을 만들 데이터가 없습니다.")
    return pd.DatetimeIndex(idx).sort_values()


def align_to_calendar(
    data: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex
) -> dict[str, pd.DataFrame]:
    """공통 달력에 정렬. 상장 이전 구간은 NaN으로 남겨 거래 대상에서 제외한다."""
    out: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        aligned = df.reindex(calendar)
        # 중간 휴장/결측은 직전 가격으로 채우되(거래량 0), 상장 전 구간은 NaN 유지
        first = df.index.min()
        aligned.loc[aligned.index >= first, OHLCV_COLUMNS] = aligned.loc[
            aligned.index >= first, OHLCV_COLUMNS
        ].ffill()
        aligned.loc[aligned.index < first, OHLCV_COLUMNS] = np.nan
        out[sym] = aligned
    return out
