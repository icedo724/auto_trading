"""로컬 CSV 소스 & 디스크 캐시.

캐시 레이아웃:  <cache_dir>/<source>/<safe_symbol>.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .base import DataError, DataSource, normalize_ohlcv, slice_period

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(symbol: str) -> str:
    return _SAFE.sub("_", symbol)


class CsvSource(DataSource):
    """디렉터리 안의 ``<symbol>.csv`` 파일을 읽는다."""

    name = "csv"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _path(self, symbol: str) -> Path:
        return self.directory / f"{_safe_name(symbol)}.csv"

    def _fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        path = self._path(symbol)
        if not path.exists():
            raise DataError(f"{symbol}: CSV 없음 -> {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return slice_period(df, start, end)


class CachedSource(DataSource):
    """원본 소스를 감싸 디스크에 캐시한다.

    캐시에 요청 구간이 모두 들어있으면 네트워크를 타지 않는다.
    """

    def __init__(
        self,
        source: DataSource,
        cache_dir: str | Path = "data/cache",
        *,
        refresh: bool = False,
    ) -> None:
        self.source = source
        self.cache_dir = Path(cache_dir) / source.name
        self.refresh = refresh
        self.name = f"cached:{source.name}"

    def _path(self, symbol: str) -> Path:
        return self.cache_dir / f"{_safe_name(symbol)}.csv"

    def _read_cache(self, symbol: str) -> pd.DataFrame | None:
        path = self._path(symbol)
        if not path.exists():
            return None
        try:
            return normalize_ohlcv(
                pd.read_csv(path, index_col=0, parse_dates=True), symbol=symbol
            )
        except Exception:  # noqa: BLE001 - 손상된 캐시는 무시하고 다시 받는다
            return None

    def _fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        want_start, want_end = pd.Timestamp(start), pd.Timestamp(end)

        cached = None if self.refresh else self._read_cache(symbol)
        if cached is not None and not cached.empty:
            covered = cached.index.min() <= want_start and cached.index.max() >= (
                want_end - pd.Timedelta(days=7)  # 최근 휴장/미래 요청 허용 오차
            )
            if covered:
                return slice_period(cached, start, end)

        fresh = self.source.get(symbol, start, end)
        merged = fresh if cached is None else pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(self._path(symbol))
        return slice_period(merged, start, end)
