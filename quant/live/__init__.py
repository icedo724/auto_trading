"""페이퍼 트레이딩 (실데이터 · 가상 자금).

실제 주문은 어디에서도 내지 않는다. 브로커 API 키조차 필요 없다.
"""

from __future__ import annotations

from .journal import Journal
from .daily import build_daily_report, health_checks
from .portfolio import Decision, Fill, PaperPortfolio
from .report import (
    backtest_reference,
    equity_curve,
    format_comparison,
    format_status,
    live_metrics,
    live_period,
    save_live_report,
)
from .runner import CycleResult, PaperTrader

__all__ = [
    "CycleResult",
    "Decision",
    "Fill",
    "Journal",
    "PaperPortfolio",
    "PaperTrader",
    "backtest_reference",
    "build_daily_report",
    "health_checks",
    "equity_curve",
    "format_comparison",
    "format_status",
    "live_metrics",
    "live_period",
    "save_live_report",
]
