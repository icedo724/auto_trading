"""실시세 소스 (yfinance / pykrx).

두 패키지 모두 **선택적 의존성**이다. 설치되어 있지 않거나 네트워크가 차단된 환경에서는
import 시점이 아니라 실제 조회 시점에 친절한 에러를 낸다.
"""

from __future__ import annotations

import pandas as pd

from .base import DataError, DataSource


class YahooSource(DataSource):
    """yfinance 기반. 미국/글로벌 티커 (예: AAPL, SPY, 005930.KS)."""

    name = "yahoo"

    def __init__(self, *, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    def _fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise DataError("yfinance 미설치: pip install yfinance") from exc

        # yfinance 의 end 는 배타적이므로 하루 더한다.
        end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(
            symbol,
            start=start,
            end=end_exclusive,
            progress=False,
            auto_adjust=self.auto_adjust,
            threads=False,
        )
        if df is None or df.empty:
            raise DataError(
                f"{symbol}: yfinance 응답이 비어 있습니다. "
                "(티커 오타, 기간, 또는 네트워크/프록시 차단 확인)"
            )
        return df


class KrxSource(DataSource):
    """pykrx 기반. 한국 주식 6자리 코드 (예: 005930, 035720)."""

    name = "krx"

    def __init__(self, *, adjusted: bool = True) -> None:
        self.adjusted = adjusted

    def _fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            from pykrx import stock
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise DataError("pykrx 미설치: pip install pykrx") from exc

        code = symbol.split(".")[0].zfill(6)
        df = stock.get_market_ohlcv(
            pd.Timestamp(start).strftime("%Y%m%d"),
            pd.Timestamp(end).strftime("%Y%m%d"),
            code,
            adjusted=self.adjusted,
        )
        if df is None or df.empty:
            raise DataError(
                f"{symbol}: pykrx 응답이 비어 있습니다. "
                "(종목코드, 기간, 또는 네트워크/프록시 차단 확인)"
            )
        return df
