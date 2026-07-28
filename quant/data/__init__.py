"""데이터 계층 진입점."""

from __future__ import annotations

from pathlib import Path

from .base import (
    OHLCV_COLUMNS,
    DataError,
    DataSource,
    align_to_calendar,
    common_calendar,
    normalize_ohlcv,
    slice_period,
)
from .csv_source import CachedSource, CsvSource
from .remote import KrxSource, YahooSource
from .synthetic import SyntheticSource

__all__ = [
    "OHLCV_COLUMNS",
    "CachedSource",
    "CsvSource",
    "DataError",
    "DataSource",
    "KrxSource",
    "SyntheticSource",
    "YahooSource",
    "align_to_calendar",
    "common_calendar",
    "get_source",
    "load_universe",
    "normalize_ohlcv",
    "slice_period",
]


def get_source(
    name: str,
    *,
    cache_dir: str | Path | None = "data/cache",
    refresh: bool = False,
    csv_dir: str | Path = "data/csv",
) -> DataSource:
    """이름으로 데이터 소스를 만든다.

    name: ``synthetic`` | ``yahoo`` | ``krx`` | ``csv``
    """
    name = name.lower()
    if name == "synthetic":
        return SyntheticSource()
    if name == "csv":
        return CsvSource(csv_dir)
    if name in ("yahoo", "yf", "us"):
        base: DataSource = YahooSource()
    elif name in ("krx", "kr", "pykrx"):
        base = KrxSource()
    else:
        raise ValueError(
            f"알 수 없는 데이터 소스: {name!r} (가능: synthetic, yahoo, krx, csv)"
        )
    if cache_dir:
        return CachedSource(base, cache_dir, refresh=refresh)
    return base


def load_universe(
    source: DataSource,
    symbols: list[str],
    start: str,
    end: str,
    *,
    align: bool = True,
    min_bars: int = 60,
) -> dict[str, "object"]:
    """유니버스 전체를 받아 **동일 거래일 달력**에 정렬한다.

    모든 전략/파라미터가 같은 시점·같은 종목으로 비교되도록 하는 것이 목적.
    """
    data = source.get_many(symbols, start, end)
    data = {s: df for s, df in data.items() if len(df) >= min_bars}
    if not data:
        raise DataError(f"최소 {min_bars} 거래일을 만족하는 종목이 없습니다.")
    if align and len(data) > 1:
        data = align_to_calendar(data, common_calendar(data))
    return data
