"""성과 지표 계산."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from .config import BacktestConfig

if TYPE_CHECKING:  # pragma: no cover
    from .engine import BacktestResult, Trade

#: 리포트에 표시되는 지표 순서
METRIC_ORDER = [
    "total_return", "cagr", "ann_volatility", "sharpe", "sortino", "max_drawdown",
    "calmar", "ulcer_index", "var_95", "win_rate", "profit_factor", "avg_win",
    "avg_loss", "payoff_ratio", "expectancy", "n_trades", "avg_holding_days",
    "exposure", "turnover", "cost_drag", "best_day", "worst_day",
    # 적립식 전용 (contribution > 0 일 때만 채워진다)
    "total_invested", "final_balance", "net_profit", "mwr",
]

#: 값이 클수록 좋은 지표
HIGHER_IS_BETTER = {
    "total_return", "cagr", "sharpe", "sortino", "calmar", "win_rate",
    "profit_factor", "payoff_ratio", "expectancy", "avg_win", "best_day",
}


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak.replace(0.0, np.nan) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    dd = drawdown_series(equity)
    return float(dd.min()) if len(dd) else 0.0


def ulcer_index(equity: pd.Series) -> float:
    """드로다운의 제곱평균제곱근. 하락의 깊이와 지속기간을 함께 반영한다."""
    dd = drawdown_series(equity).fillna(0.0)
    return float(np.sqrt((dd**2).mean())) if len(dd) else 0.0


def cagr(equity: pd.Series, trading_days: int = 252) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / trading_days
    if years <= 0:
        return 0.0
    growth = equity.iloc[-1] / equity.iloc[0]
    if growth <= 0:
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / trading_days
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(trading_days))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / trading_days
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    dd = np.sqrt((downside**2).mean())
    if not np.isfinite(dd) or dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(trading_days))


def trade_stats(trades: "Iterable[Trade]") -> dict[str, float]:
    rets = np.array([t.return_pct for t in trades], dtype=float)
    pnls = np.array([t.pnl for t in trades], dtype=float)
    holds = np.array([t.holding_days for t in trades], dtype=float)
    if rets.size == 0:
        return {
            "n_trades": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, "payoff_ratio": 0.0, "expectancy": 0.0,
            "avg_holding_days": 0.0,
        }
    wins, losses = rets[rets > 0], rets[rets <= 0]
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = -pnls[pnls <= 0].sum()
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    win_rate = float(wins.size / rets.size)
    return {
        "n_trades": float(rets.size),
        "win_rate": win_rate,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        ),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0.0,
        "expectancy": float(win_rate * avg_win + (1 - win_rate) * avg_loss),
        "avg_holding_days": float(holds.mean()),
    }


def money_weighted_return(
    cashflows: list[tuple[float, float]], final_value: float, years: float
) -> float:
    """자금가중수익률(IRR, 연율). 적립식에서 "내 돈이 실제로 몇 % 벌었나"를 답한다.

    cashflows: [(경과연수, 투입금액), ...]  final_value: 최종 평가액
    TWR 과 달리 **입금 시점**의 영향을 받는다. 늦게 넣은 돈은 덜 반영된다.
    의존성 없이 이분법으로 푼다.
    """
    if not cashflows or years <= 0 or final_value <= 0:
        return 0.0

    def npv(rate: float) -> float:
        base = 1.0 + rate
        if base <= 1e-9:
            return float("inf")
        out = -sum(amt / base**t for t, amt in cashflows)
        return out + final_value / base**years

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-9:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float((lo + hi) / 2)


def contribution_metrics(
    result: "BacktestResult", config: BacktestConfig
) -> dict[str, float]:
    """적립식 지표. 적립이 없으면 빈 dict."""
    if result.contributions is None or result.balance is None:
        return {}
    deposited = float(result.contributions.sum())
    if deposited <= 0:
        return {}

    invested = config.initial_cash + deposited
    final = float(result.balance.iloc[-1])
    td = config.trading_days
    n = len(result.balance)

    idx = result.balance.index
    flows = [(0.0, config.initial_cash)]
    nz = result.contributions[result.contributions > 0]
    for date, amt in nz.items():
        flows.append((float(idx.searchsorted(date)) / td, float(amt)))

    return {
        "total_invested": invested,
        "final_balance": final,
        "net_profit": final - invested,
        "mwr": money_weighted_return(flows, final, n / td),
    }


def compute_metrics(result: "BacktestResult", config: BacktestConfig) -> dict[str, float]:
    """백테스트 결과 -> 지표 딕셔너리."""
    eq, rets, pos = result.equity, result.returns, result.position
    td, rf = config.trading_days, config.risk_free_rate

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else 0.0
    mdd = max_drawdown(eq)
    ann_ret = cagr(eq, td)
    vol = float(rets.std(ddof=1) * np.sqrt(td)) if len(rets) > 1 else 0.0

    # 회전율: 일평균 비중 변화 * 연환산 (편도 기준)
    turnover = float(pos.diff().abs().mean() * td) if len(pos) > 1 else 0.0

    metrics = {
        "total_return": total_return,
        "cagr": ann_ret,
        "ann_volatility": vol,
        "sharpe": sharpe_ratio(rets, rf, td),
        "sortino": sortino_ratio(rets, rf, td),
        "max_drawdown": mdd,
        "calmar": float(ann_ret / abs(mdd)) if mdd < 0 else 0.0,
        "ulcer_index": ulcer_index(eq),
        "var_95": float(np.percentile(rets, 5)) if len(rets) > 20 else 0.0,
        "exposure": float(pos.abs().clip(upper=1.0).mean()) if len(pos) else 0.0,
        "turnover": turnover,
        "cost_drag": float(result.total_cost / config.initial_cash),
        "best_day": float(rets.max()) if len(rets) else 0.0,
        "worst_day": float(rets.min()) if len(rets) else 0.0,
    }
    metrics.update(trade_stats(result.trades))
    metrics.update(contribution_metrics(result, config))
    return {k: metrics[k] for k in METRIC_ORDER if k in metrics}


def portfolio_returns(results: "dict[str, BacktestResult]") -> pd.Series:
    """종목별 결과를 동일비중(일별 리밸런싱) 포트폴리오 수익률로 합성.

    모든 종목이 같은 달력에 정렬되어 있다는 전제(load_universe 가 보장).
    """
    if not results:
        raise ValueError("합성할 결과가 없습니다.")
    frame = pd.DataFrame({s: r.returns for s, r in results.items()}).sort_index()
    return frame.mean(axis=1, skipna=True).fillna(0.0)


def equity_from_returns(returns: pd.Series, initial: float) -> pd.Series:
    eq = initial * (1.0 + returns).cumprod()
    eq.name = "equity"
    return eq


def compute_portfolio_metrics(
    results: "dict[str, BacktestResult]", config: BacktestConfig
) -> tuple[pd.Series, dict[str, float]]:
    """(포트폴리오 자산곡선, 지표) 반환."""
    from .engine import BacktestResult  # 지연 import

    rets = portfolio_returns(results)
    equity = equity_from_returns(rets, config.initial_cash)

    positions = pd.DataFrame({s: r.position for s, r in results.items()}).mean(axis=1)
    trades = [t for r in results.values() for t in r.trades]
    total_cost = sum(r.total_cost for r in results.values()) / max(len(results), 1)

    agg = BacktestResult(
        symbol="PORTFOLIO",
        equity=equity,
        returns=rets,
        position=positions.fillna(0.0),
        trades=trades,
        total_cost=total_cost,
    )
    return equity, compute_metrics(agg, config)
