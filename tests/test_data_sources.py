"""데이터 소스 어댑터 테스트.

네트워크 없이 검증하기 위해 응답을 모킹한다.
실제 필드명·형태는 각 서비스의 실제 응답을 그대로 옮긴 것이다.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from quant.data import (
    SOURCE_ALIASES,
    CachedSource,
    FdrSource,
    KrxSource,
    NaverSource,
    SyntheticSource,
    YahooSource,
    get_source,
)
from quant.data.base import DataError
from quant.data.probe import ProbeResult, format_report, probe_source

# 네이버 금융 siseJson.naver 실제 응답 형태 (작은따옴표 파이썬 리터럴)
NAVER_BODY = """[['날짜', '시가', '고가', '저가', '종가', '거래량', '외국인소진율'],
['20240102', 79600, 79800, 78200, 79600, 17142847, 54.13],
['20240103', 78800, 79000, 76500, 77000, 21598348, 54.05],
['20240104', 76100, 77300, 76100, 76600, 15324439, 54.01]]"""


# --------------------------------------------------------------------- 네이버
def test_naver_parses_real_response_shape():
    from quant.data.base import normalize_ohlcv

    out = normalize_ohlcv(NaverSource.parse(NAVER_BODY, "005930"), symbol="005930")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 3
    assert out.index[0] == pd.Timestamp("2024-01-02")
    assert out["close"].iloc[0] == 79600.0
    assert out["volume"].iloc[1] == 21598348.0


def test_naver_end_to_end_with_mocked_http(monkeypatch):
    class FakeResp:
        text = NAVER_BODY

        def raise_for_status(self):
            return None

    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured.update(url=url, params=params)
        return FakeResp()

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=fake_get))
    df = NaverSource().get("005930", "2024-01-01", "2024-01-05")

    assert captured["params"]["symbol"] == "005930"
    assert captured["params"]["startTime"] == "20240101"
    assert captured["params"]["timeframe"] == "day"
    assert len(df) == 3
    assert df["close"].iloc[-1] == 76600.0


def test_naver_pads_short_code(monkeypatch):
    captured = {}
    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(
            get=lambda url, params=None, headers=None, timeout=None: (
                captured.update(params=params),
                type("R", (), {"text": NAVER_BODY, "raise_for_status": lambda s: None})(),
            )[1]
        ),
    )
    NaverSource().get("5930", "2024-01-01", "2024-01-05")
    assert captured["params"]["symbol"] == "005930"


def test_naver_empty_and_malformed_raise():
    with pytest.raises(DataError, match="비어 있습니다"):
        NaverSource.parse("", "005930")
    with pytest.raises(DataError, match="데이터가 없습니다"):
        NaverSource.parse("[['날짜','시가','고가','저가','종가','거래량']]", "005930")
    with pytest.raises(DataError, match="파싱 실패"):
        NaverSource.parse("<html>error</html>", "005930")


# ------------------------------------------------------------------------ FDR
def _fake_fdr(df: pd.DataFrame | None, raises: Exception | None = None):
    def DataReader(symbol, start=None, end=None, exchange=None, data_source=None):
        if raises is not None:
            raise raises
        return df

    return types.SimpleNamespace(DataReader=DataReader)


def test_fdr_normalizes_columns(monkeypatch):
    # FinanceDataReader 의 KRX 응답: Open/High/Low/Close/Volume/Change
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    raw = pd.DataFrame(
        {"Open": [79600, 78800], "High": [79800, 79000], "Low": [78200, 76500],
         "Close": [79600, 77000], "Volume": [17142847, 21598348],
         "Change": [0.0006, -0.0327]},
        index=idx,
    )
    monkeypatch.setitem(sys.modules, "FinanceDataReader", _fake_fdr(raw))
    df = FdrSource().get("005930", "2024-01-01", "2024-01-05")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]  # Change 제거
    assert df.index.name == "date"
    assert df["close"].iloc[0] == 79600.0


def test_fdr_empty_and_exception_become_dataerror(monkeypatch):
    monkeypatch.setitem(sys.modules, "FinanceDataReader", _fake_fdr(pd.DataFrame()))
    with pytest.raises(DataError, match="비어 있습니다"):
        FdrSource().get("BADCODE", "2024-01-01", "2024-01-05")

    monkeypatch.setitem(
        sys.modules, "FinanceDataReader", _fake_fdr(None, raises=RuntimeError("boom"))
    )
    with pytest.raises(DataError, match="조회 실패"):
        FdrSource().get("005930", "2024-01-01", "2024-01-05")


# ---------------------------------------------------------------------- pykrx
def test_krx_suppresses_login_noise_and_maps_korean(monkeypatch, capsys):
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    raw = pd.DataFrame(
        {"시가": [79600, 78800], "고가": [79800, 79000], "저가": [78200, 76500],
         "종가": [79600, 77000], "거래량": [17142847, 21598348],
         "등락률": [0.06, -3.27]},
        index=idx,
    )

    def get_market_ohlcv(start, end, code, adjusted=True):
        print("KRX 로그인 실패: KRX_ID 또는 KRX_PW 환경 변수가 설정되지 않았습니다.")
        return raw

    monkeypatch.setitem(
        sys.modules, "pykrx", types.SimpleNamespace(stock=None)
    )
    monkeypatch.setitem(
        sys.modules,
        "pykrx.stock",
        types.SimpleNamespace(get_market_ohlcv=get_market_ohlcv),
    )
    sys.modules["pykrx"].stock = sys.modules["pykrx.stock"]

    df = KrxSource().get("005930", "2024-01-01", "2024-01-05")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[0] == 79600.0
    # 로그인 경고가 사용자 화면으로 새어 나오지 않아야 한다
    assert "KRX 로그인 실패" not in capsys.readouterr().out


def test_krx_credential_detection(monkeypatch):
    monkeypatch.delenv("KRX_ID", raising=False)
    monkeypatch.delenv("KRX_PW", raising=False)
    assert KrxSource.has_credentials() is False
    monkeypatch.setenv("KRX_ID", "u")
    monkeypatch.setenv("KRX_PW", "p")
    assert KrxSource.has_credentials() is True


# --------------------------------------------------------------------- 레지스트리
def test_get_source_aliases():
    assert isinstance(get_source("synthetic"), SyntheticSource)
    assert isinstance(get_source("fdr", cache_dir=None), FdrSource)
    assert isinstance(get_source("naver", cache_dir=None), NaverSource)
    assert isinstance(get_source("pykrx", cache_dir=None), KrxSource)
    assert isinstance(get_source("us", cache_dir=None), YahooSource)


def test_remote_sources_are_cached_by_default():
    src = get_source("fdr")
    assert isinstance(src, CachedSource)
    assert src.name == "cached:fdr"
    # 오프라인 소스는 캐시로 감싸지 않는다
    assert not isinstance(get_source("synthetic"), CachedSource)


def test_unknown_source_lists_options():
    with pytest.raises(ValueError, match="알 수 없는 데이터 소스"):
        get_source("bloomberg")
    assert {"fdr", "naver", "krx", "yahoo", "csv", "synthetic"} <= set(SOURCE_ALIASES)


# ------------------------------------------------------------------------ 진단
def test_probe_reports_success_offline():
    r = probe_source("synthetic", "PROBE", description="합성", days=60)
    assert r.ok and r.rows > 0 and r.error == ""


def test_probe_captures_failure_without_raising(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "FinanceDataReader", _fake_fdr(None, raises=RuntimeError("차단됨"))
    )
    r = probe_source("fdr", "005930", cache_dir=None)
    assert r.ok is False
    assert "차단됨" in r.error


def test_format_report_guides_when_all_remote_fail():
    results = [
        ProbeResult("synthetic", "P", "", True, rows=20, first="2024-01-01", last="2024-02-01"),
        ProbeResult("fdr", "005930", "", False, error="네트워크 차단"),
    ]
    text = format_report(results)
    assert "사용 가능한 실시세 소스가 없습니다" in text
    assert "pip install" in text

    results[1].ok = True
    results[1].rows = 20
    ok_text = format_report(results)
    assert "source: fdr" in ok_text


# --------------------------------------------------------------------- 상대 날짜
def test_relative_dates_track_the_clock():
    """스케줄 실행 시 구간이 오늘을 따라 움직여야 한다."""
    from quant.dates import resolve_date, resolve_period

    t = pd.Timestamp("2026-07-30")
    assert resolve_date("today", today=t) == "2026-07-30"
    assert resolve_date("-3y", today=t) == "2023-07-30"
    assert resolve_date("-250d", today=t) == "2025-11-22"
    assert resolve_date("-6m", today=t) == "2026-01-30"
    assert resolve_date("2024-01-01", today=t) == "2024-01-01"
    assert resolve_date(None, today=t) == "2026-07-30"
    assert resolve_period("-3y", "today", today=t) == ("2023-07-30", "2026-07-30")

    # 하루 뒤에 실행하면 구간도 하루 이동한다 (고정 날짜는 그대로)
    t2 = t + pd.Timedelta(days=1)
    assert resolve_date("-3y", today=t2) == "2023-07-31"
    assert resolve_date("2024-01-01", today=t2) == "2024-01-01"


def test_bad_dates_rejected():
    from quant.dates import resolve_date, resolve_period

    with pytest.raises(ValueError, match="해석할 수 없습니다"):
        resolve_date("어제")
    with pytest.raises(ValueError, match="시작일"):
        resolve_period("today", "-1y", today=pd.Timestamp("2026-07-30"))
