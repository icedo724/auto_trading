"""설정 파일의 상대 날짜 표기 해석.

스케줄러로 매일 돌리려면 `end: "2025-12-31"` 같은 고정 날짜로는 안 된다.
시간이 지나면 신호가 과거에 멈춰버리기 때문이다.

지원 표기:
    "2024-01-01"   절대 날짜
    "today"        오늘 (= "now")
    "-3y"          3년 전   ("y" 년 / "m" 월 / "w" 주 / "d" 일)
    "-250d"        250일 전
    "+1d"          내일
"""

from __future__ import annotations

import re

import pandas as pd

_REL = re.compile(r"^\s*([+-])\s*(\d+)\s*([ymwd])\s*$", re.IGNORECASE)
_UNITS = {
    "y": lambda n: pd.DateOffset(years=n),
    "m": lambda n: pd.DateOffset(months=n),
    "w": lambda n: pd.Timedelta(weeks=n),
    "d": lambda n: pd.Timedelta(days=n),
}


def resolve_date(value: str | pd.Timestamp | None, *, today: pd.Timestamp | None = None) -> str:
    """설정값을 ``YYYY-MM-DD`` 문자열로 정규화한다.

    ``today`` 를 주입할 수 있어 테스트가 시계에 의존하지 않는다.
    """
    base = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()

    if value is None:
        return base.strftime("%Y-%m-%d")

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    text = str(value).strip()
    if text.lower() in ("today", "now", "auto"):
        return base.strftime("%Y-%m-%d")

    m = _REL.match(text)
    if m:
        sign, num, unit = m.group(1), int(m.group(2)), m.group(3).lower()
        offset = _UNITS[unit](num)
        ts = base - offset if sign == "-" else base + offset
        return pd.Timestamp(ts).strftime("%Y-%m-%d")

    try:
        return pd.Timestamp(text).strftime("%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"날짜를 해석할 수 없습니다: {value!r}. "
            "'YYYY-MM-DD', 'today', '-3y', '-250d' 형식을 지원합니다."
        ) from exc


def resolve_period(
    start: str | None, end: str | None, *, today: pd.Timestamp | None = None
) -> tuple[str, str]:
    """(start, end) 쌍을 정규화하고 순서를 검증한다."""
    s = resolve_date(start, today=today)
    e = resolve_date(end, today=today)
    if pd.Timestamp(s) > pd.Timestamp(e):
        raise ValueError(f"시작일({s})이 종료일({e})보다 뒤입니다.")
    return s, e
