"""quant — 자동 주식매매 알고리즘 백테스트 · 파라미터 최적화 프레임워크."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import BacktestConfig, CostModel
from .engine import Backtester, BacktestResult, Trade, run_backtest
from .metrics import compute_metrics, compute_portfolio_metrics
from .optimizer import (
    EvalResult,
    OptimizationReport,
    build_all_candidates,
    common_trade_start,
    grid_search,
    sensitivity,
)
from .strategy import Strategy, available_strategies, build_candidates, create_strategy
from .validation import holdout_validate, walk_forward

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
    "CostModel",
    "EvalResult",
    "OptimizationReport",
    "Strategy",
    "Trade",
    "available_strategies",
    "build_all_candidates",
    "build_candidates",
    "common_trade_start",
    "compute_metrics",
    "compute_portfolio_metrics",
    "create_strategy",
    "grid_search",
    "holdout_validate",
    "run_backtest",
    "sensitivity",
    "walk_forward",
]
