"""페이퍼 트레이딩 (실데이터 · 가상 자금).

실제 주문은 어디에서도 내지 않는다. 브로커 API 키조차 필요 없다.
"""

from __future__ import annotations

from .journal import Journal
from .portfolio import Fill, PaperPortfolio
from .report import (
    backtest_reference,
    equity_curve,
    format_comparison,
    format_status,
    live_metrics,
    save_live_report,
)
from .runner import CycleResult, PaperTrader

__all__ = [
    "CycleResult",
    "Fill",
    "Journal",
    "PaperPortfolio",
    "PaperTrader",
    "backtest_reference",
    "equity_curve",
    "format_comparison",
    "format_status",
    "live_metrics",
    "save_live_report",
]
