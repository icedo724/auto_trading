"""데이터 소스 가용성 진단.

"어떤 소스가 내 환경에서 실제로 되는가"를 한 번에 확인한다.
패키지 설치 여부와 네트워크 접근을 모두 실제 조회로 검증한다.
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass

import pandas as pd

#: (소스 이름, 점검용 종목, 설명)
DEFAULT_PROBES: list[tuple[str, str, str]] = [
    ("synthetic", "PROBE", "오프라인 합성 시세 (항상 동작)"),
    ("fdr", "005930", "FinanceDataReader · 삼성전자"),
    ("naver", "005930", "네이버 금융 · 삼성전자"),
    ("krx", "005930", "pykrx · 삼성전자"),
    ("yahoo", "AAPL", "yfinance · 애플"),
]


@dataclass
class ProbeResult:
    source: str
    symbol: str
    description: str
    ok: bool
    rows: int = 0
    first: str = ""
    last: str = ""
    elapsed: float = 0.0
    error: str = ""

    @property
    def status(self) -> str:
        return "정상" if self.ok else "실패"


def probe_source(
    source_name: str,
    symbol: str,
    *,
    description: str = "",
    days: int = 30,
    cache_dir: str | None = None,
) -> ProbeResult:
    """소스 하나를 실제로 조회해 본다. 예외는 삼키고 결과에 담는다."""
    from . import get_source

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=days)
    t0 = time.perf_counter()
    try:
        # 라이브러리들이 자체 경고를 stdout/stderr 로 뿜는다. 실패 사유는 어차피
        # 예외로 잡아 결과에 담으므로, 진단 표가 지저분해지지 않게 삼킨다.
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            # 캐시를 끄고 실제 경로를 확인한다 (cache_dir=None)
            src = get_source(source_name, cache_dir=cache_dir)
            df = src.get(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        return ProbeResult(
            source=source_name,
            symbol=symbol,
            description=description,
            ok=True,
            rows=len(df),
            first=str(df.index[0].date()),
            last=str(df.index[-1].date()),
            elapsed=time.perf_counter() - t0,
        )
    except Exception as exc:  # noqa: BLE001 - 진단이 목적이므로 전부 잡는다
        msg = str(exc).replace("\n", " ")
        return ProbeResult(
            source=source_name,
            symbol=symbol,
            description=description,
            ok=False,
            elapsed=time.perf_counter() - t0,
            error=msg[:160],
        )


def probe_all(
    probes: list[tuple[str, str, str]] | None = None, *, days: int = 30
) -> list[ProbeResult]:
    return [
        probe_source(name, sym, description=desc, days=days)
        for name, sym, desc in (probes or DEFAULT_PROBES)
    ]


def format_report(results: list[ProbeResult]) -> str:
    """진단 결과를 사람이 읽는 표로."""
    lines = [
        "데이터 소스 진단",
        "=" * 78,
        f"{'소스':<12}{'상태':<7}{'봉수':>6}  {'구간':<26}{'소요':>7}",
        "-" * 78,
    ]
    for r in results:
        span = f"{r.first} ~ {r.last}" if r.ok else "-"
        lines.append(
            f"{r.source:<12}{r.status:<7}{r.rows:>6}  {span:<26}{r.elapsed:>6.1f}s"
        )
        if not r.ok:
            lines.append(f"{'':<12}└ {r.error}")

    usable = [r for r in results if r.ok and r.source != "synthetic"]
    lines += ["-" * 78]
    if usable:
        best = usable[0]
        lines += [
            f"사용 가능한 실시세 소스 {len(usable)}개: "
            f"{', '.join(r.source for r in usable)}",
            "",
            "설정 파일에 다음과 같이 지정하세요:",
            "    data:",
            f"      source: {best.source}",
        ]
    else:
        lines += [
            "사용 가능한 실시세 소스가 없습니다.",
            "",
            "확인 순서:",
            "  1) 패키지 설치     pip install finance-datareader pykrx yfinance",
            "  2) 네트워크/프록시  사내망·방화벽에서 금융 사이트가 막혔는지 확인",
            "  3) 그래도 안 되면  source: synthetic 으로 파이프라인 검증은 가능",
        ]
    return "\n".join(lines)
