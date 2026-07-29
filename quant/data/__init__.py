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
from .remote import FdrSource, KrxSource, NaverSource, YahooSource
from .synthetic import SyntheticSource

__all__ = [
    "OHLCV_COLUMNS",
    "CachedSource",
    "CsvSource",
    "DataError",
    "DataSource",
    "FdrSource",
    "KrxSource",
    "NaverSource",
    "SOURCE_ALIASES",
    "SyntheticSource",
    "YahooSource",
    "align_to_calendar",
    "common_calendar",
    "get_source",
    "load_universe",
    "normalize_ohlcv",
    "slice_period",
]


#: 설정 파일에 쓰는 이름 -> 소스 클래스. 모두 무료 소스다.
SOURCE_ALIASES: dict[str, str] = {
    "synthetic": "synthetic",  # 오프라인 합성 시세
    "csv": "csv",  # 로컬 CSV
    "fdr": "fdr",  # FinanceDataReader (국내 권장)
    "findatareader": "fdr",
    "naver": "naver",  # 네이버 금융 (무의존)
    "krx": "krx",  # pykrx
    "kr": "krx",
    "pykrx": "krx",
    "yahoo": "yahoo",  # yfinance (해외 권장)
    "yf": "yahoo",
    "us": "yahoo",
}


def get_source(
    name: str,
    *,
    cache_dir: str | Path | None = "data/cache",
    refresh: bool = False,
    csv_dir: str | Path = "data/csv",
) -> DataSource:
    """이름으로 데이터 소스를 만든다.

    사용 가능한 이름은 ``SOURCE_ALIASES`` 참조. 원격 소스는 자동으로 디스크
    캐시로 감싸므로 같은 구간을 다시 받지 않는다.
    """
    key = SOURCE_ALIASES.get(name.lower().strip())
    if key is None:
        raise ValueError(
            f"알 수 없는 데이터 소스: {name!r} "
            f"(가능: {', '.join(sorted(set(SOURCE_ALIASES)))})"
        )

    if key == "synthetic":
        return SyntheticSource()
    if key == "csv":
        return CsvSource(csv_dir)

    base: DataSource = {
        "fdr": FdrSource,
        "naver": NaverSource,
        "krx": KrxSource,
        "yahoo": YahooSource,
    }[key]()

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
