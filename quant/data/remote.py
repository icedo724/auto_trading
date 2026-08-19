"""실시세 소스.

**모두 무료다.** 백테스트에 필요한 것은 실시간 호가가 아니라 과거 일봉(EOD)이며,
아래 소스는 전부 무료로 일봉을 제공한다. 자세한 비교는 `docs/DATA_SOURCES.md` 참조.

| 소스 | 대상 | 인증 | 의존 패키지 |
|---|---|---|---|
| `FdrSource`   | 국내·미국·지수·환율 | 불필요 | `finance-datareader` |
| `NaverSource` | 국내 주식 | 불필요 | 없음 (requests만) |
| `KrxSource`   | 국내 주식 | 선택(KRX_ID/PW) | `pykrx` |
| `YahooSource` | 미국·글로벌 | 불필요 | `yfinance` |

모든 패키지는 **선택적 의존성**이다. 미설치/네트워크 차단 환경에서는 import 시점이
아니라 실제 조회 시점에 친절한 에러를 낸다.
"""

from __future__ import annotations

import ast
import contextlib
import io
import os

import pandas as pd

from .base import DataError, DataSource

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"


def _ymd(d: str | pd.Timestamp) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


class FdrSource(DataSource):
    """FinanceDataReader — 국내 주식의 기본 권장 소스.

    인증이 필요 없고 KRX/미국/지수/환율/암호화폐를 한 인터페이스로 다룬다.

        005930    삼성전자 (국내는 6자리 코드)
        AAPL      애플
        KS11      KOSPI 지수
        USD/KRW   환율
    """

    name = "fdr"

    def __init__(self, *, exchange: str | None = None) -> None:
        self.exchange = exchange

    def _fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import FinanceDataReader as fdr
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise DataError(
                "finance-datareader 미설치: pip install finance-datareader"
            ) from exc

        try:
            df = fdr.DataReader(symbol, start, end, exchange=self.exchange)
        except Exception as exc:  # noqa: BLE001 - 라이브러리가 다양한 예외를 던진다
            raise DataError(f"{symbol}: FinanceDataReader 조회 실패 - {exc}") from exc

        if df is None or df.empty:
            raise DataError(
                f"{symbol}: FinanceDataReader 응답이 비어 있습니다. "
                "(종목코드/기간 확인, 또는 네트워크·프록시 차단)"
            )
        return df


class NaverSource(DataSource):
    """네이버 금융 차트 API — 의존 패키지 없이 국내 일봉을 받는다.

    비공식 엔드포인트이므로 스펙이 예고 없이 바뀔 수 있다.
    안정성이 필요하면 `FdrSource` 를 우선 쓰고 이쪽은 대체 경로로 둔다.
    """

    name = "naver"
    URL = "https://api.finance.naver.com/siseJson.naver"

    def __init__(self, *, timeout: int = 15) -> None:
        self.timeout = timeout

    def _fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import requests

        code = symbol.split(".")[0].zfill(6)
        params = {
            "symbol": code,
            "requestType": "1",
            "startTime": _ymd(start),
            "endTime": _ymd(end),
            "timeframe": "day",
        }
        try:
            resp = requests.get(
                self.URL, params=params, headers={"User-Agent": _UA}, timeout=self.timeout
            )
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise DataError(f"{symbol}: 네이버 금융 요청 실패 - {exc}") from exc

        return self.parse(resp.text, symbol)

    @staticmethod
    def parse(text: str, symbol: str = "") -> pd.DataFrame:
        """네이버 응답 본문을 DataFrame 으로.

        응답은 작은따옴표를 쓰는 파이썬 리터럴 형태라 ``json.loads`` 로는 못 읽는다.

            [['날짜', '시가', '고가', '저가', '종가', '거래량', '외국인소진율'],
             ['20240102', 79600, 79800, 78200, 79600, 17142847, 54.13], ...]
        """
        body = (text or "").strip()
        if not body:
            raise DataError(f"{symbol}: 네이버 응답이 비어 있습니다.")
        try:
            rows = ast.literal_eval(body)
        except (ValueError, SyntaxError) as exc:
            raise DataError(f"{symbol}: 네이버 응답 파싱 실패 - {exc}") from exc

        if not isinstance(rows, list) or len(rows) < 2:
            raise DataError(
                f"{symbol}: 네이버에 해당 기간 데이터가 없습니다. (종목코드/기간 확인)"
            )

        header = [str(c).strip() for c in rows[0]]
        df = pd.DataFrame(rows[1:], columns=header)
        df = df.set_index(pd.to_datetime(df[header[0]].astype(str), format="%Y%m%d"))
        # 컬럼명(날짜/시가/고가/저가/종가/거래량)은 normalize_ohlcv 가 영문으로 매핑한다
        return df.drop(columns=[header[0]])


class KrxSource(DataSource):
    """pykrx — KRX 정보데이터시스템 스크래핑. 국내 6자리 코드.

    pykrx 1.2 부터 KRX 로그인을 **선택적으로** 지원한다.
    환경변수 `KRX_ID` / `KRX_PW` 가 있으면 로그인 세션을 쓰고, 없으면 익명 세션으로
    폴백한다(동작은 하지만 KRX 측 제한에 더 민감하다).
    """

    name = "krx"

    def __init__(self, *, adjusted: bool = True) -> None:
        self.adjusted = adjusted

    @staticmethod
    def has_credentials() -> bool:
        return bool(os.getenv("KRX_ID") and os.getenv("KRX_PW"))

    def _fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            # pykrx 는 import 시점에 로그인 시도 메시지를 stdout 으로 출력한다.
            # 자격증명이 없을 때의 경고는 폴백이 정상 동작하므로 조용히 삼킨다.
            with contextlib.redirect_stdout(io.StringIO()):
                from pykrx import stock
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise DataError("pykrx 미설치: pip install pykrx") from exc

        code = symbol.split(".")[0].zfill(6)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                df = stock.get_market_ohlcv(
                    _ymd(start), _ymd(end), code, adjusted=self.adjusted
                )
        except Exception as exc:  # noqa: BLE001
            raise DataError(f"{symbol}: pykrx 조회 실패 - {exc}") from exc

        if df is None or df.empty:
            hint = "" if self.has_credentials() else (
                " KRX_ID/KRX_PW 환경변수를 설정하면 안정성이 올라갑니다."
            )
            raise DataError(
                f"{symbol}: pykrx 응답이 비어 있습니다. "
                f"(종목코드/기간 확인, 또는 네트워크·프록시 차단){hint}"
            )
        return df


class YahooSource(DataSource):
    """yfinance — 미국/글로벌 티커 (예: AAPL, SPY). 국내는 `005930.KS` 형태."""

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


class UpbitSource(DataSource):
    """업비트 공개 시세 API — **인증 불필요**, 원화마켓 코인 일봉.

    소액 자동매매에 코인이 유리한 이유:
      · 소수점 매매 → 10만원으로도 여러 종목 분산 가능 (주식은 1주 단위)
      · 왕복 비용 약 20bp (국내주식 31bp), 증권거래세 없음
      · 최소 주문 5,000원

    심볼은 ``KRW-BTC`` 또는 ``BTC`` (자동으로 KRW 마켓으로 해석).
    """

    name = "upbit"
    URL = "https://api.upbit.com/v1/candles/days"
    MAX_COUNT = 200  # 1회 요청 상한

    def __init__(self, *, timeout: int = 15, quote: str = "KRW") -> None:
        self.timeout = timeout
        self.quote = quote

    def market_code(self, symbol: str) -> str:
        s = symbol.strip().upper()
        return s if "-" in s else f"{self.quote}-{s}"

    def _fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import time

        import requests

        market = self.market_code(symbol)
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

        # 업비트는 최신부터 역순으로 최대 200개씩 준다. start 에 닿을 때까지 페이징.
        rows: list[dict] = []
        cursor = end_ts + pd.Timedelta(days=1)
        for _ in range(200):  # 최대 40,000봉 (약 109년) — 무한루프 방지
            params = {
                "market": market,
                "count": self.MAX_COUNT,
                "to": cursor.strftime("%Y-%m-%d %H:%M:%S"),
            }
            try:
                resp = requests.get(self.URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                batch = resp.json()
            except Exception as exc:  # noqa: BLE001
                raise DataError(f"{symbol}: 업비트 요청 실패 - {exc}") from exc

            if not batch:
                break
            rows.extend(batch)
            oldest = pd.Timestamp(batch[-1]["candle_date_time_kst"])
            if oldest <= start_ts or len(batch) < self.MAX_COUNT:
                break
            cursor = oldest
            time.sleep(0.15)  # 공개 API 유량 제한 배려

        return self.parse(rows, symbol)

    @staticmethod
    def parse(rows: list[dict], symbol: str = "") -> pd.DataFrame:
        """업비트 캔들 JSON -> OHLCV DataFrame."""
        if not rows:
            raise DataError(
                f"{symbol}: 업비트 응답이 비어 있습니다. (마켓코드 확인: KRW-BTC 형식)"
            )
        df = pd.DataFrame(rows)
        need = {
            "candle_date_time_kst": "date",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume",
        }
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise DataError(f"{symbol}: 업비트 응답에 컬럼 누락 {missing}")

        df = df[list(need)].rename(columns=need)
        df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"])
        return df[~df.index.duplicated(keep="first")].sort_index()
