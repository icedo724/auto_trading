"""페이퍼 트레이딩 성과 리포트.

3개월을 돌린 뒤 답해야 할 질문은 하나다:
**"실제로 돌려본 성과가 백테스트가 약속한 것과 비슷한가?"**

저널의 cycle 이벤트로 자산곡선을 복원해 지표를 계산하고, 같은 기간의
백테스트 결과와 나란히 놓는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..metrics import max_drawdown, money_weighted_return
from .journal import Journal
from .portfolio import PaperPortfolio


def equity_curve(journal: Journal) -> pd.DataFrame:
    """저널에서 자산곡선 복원. (index=시각, columns=[equity, invested, cash])"""
    rows = [r for r in journal.read() if r.get("event") == "cycle"]
    if not rows:
        return pd.DataFrame(columns=["equity", "invested", "cash"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="mixed")
    df = df.sort_values("ts")

    # 시장 봉 날짜를 우선 쓴다. 벽시계로 묶으면 같은 날 여러 번 돌린 사이클이
    # 하나로 합쳐져 곡선이 뭉개진다(특히 과거 데이터로 리허설할 때).
    if "bar" in df.columns and df["bar"].astype(str).str.len().gt(0).any():
        key = df["bar"].replace("", pd.NA).ffill()
    else:
        key = df["ts"].dt.date.astype(str)

    out = df[["equity", "invested", "cash"]].astype(float)
    out.index = pd.Index(key, name="date")
    # 같은 봉에 여러 사이클이 돌 수 있으므로 마지막 값만 남긴다
    return out.groupby(level=0).last().sort_index()


def live_period(journal: Journal) -> tuple[str, str] | None:
    """라이브가 실제로 커버한 **시장 날짜** 구간.

    포트폴리오의 created_at 은 벽시계 시각이라 시장 달력과 무관하다.
    그것을 백테스트 대조군의 시작일로 쓰면 데이터 범위를 벗어나 거래가
    하나도 안 잡히고, 결과적으로 "백테스트 0%" 라는 가짜 기준선이 나온다.
    """
    curve = equity_curve(journal)
    if len(curve) < 2:
        return None
    return str(curve.index[0]), str(curve.index[-1])


def live_metrics(
    journal: Journal, portfolio: PaperPortfolio, config: BacktestConfig
) -> dict[str, float]:
    """실제로 돌린 결과의 지표. 입금은 수익에서 제외한다(TWR)."""
    curve = equity_curve(journal)
    if len(curve) < 2:
        return {}

    eq, inv = curve["equity"], curve["invested"]
    deposits = inv.diff().fillna(0.0)  # 입금은 수익이 아니다
    prev = eq.shift(1)
    denom = (prev + deposits).replace(0.0, np.nan)
    returns = (eq / denom - 1.0).fillna(0.0)
    nav = config.initial_cash * (1.0 + returns).cumprod()

    days = len(curve)
    td = config.trading_days
    final, invested = float(eq.iloc[-1]), float(inv.iloc[-1])

    flows = [(0.0, config.initial_cash)]
    for i, (_, amt) in enumerate(deposits[deposits > 0].items()):
        flows.append((float(i + 1) / td, float(amt)))

    growth = float(nav.iloc[-1] / nav.iloc[0])
    years = days / td
    return {
        "days": float(days),
        "total_return": growth - 1.0,
        "cagr": float(growth ** (1.0 / years) - 1.0) if years > 0 and growth > 0 else 0.0,
        "ann_volatility": float(returns.std(ddof=1) * np.sqrt(td)) if days > 2 else 0.0,
        "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(td))
        if days > 2 and returns.std(ddof=1) > 0 else 0.0,
        "max_drawdown": max_drawdown(nav),
        "total_invested": invested,
        "final_balance": final,
        "net_profit": final - invested,
        "mwr": money_weighted_return(flows, final, years),
        "n_fills": float(len(portfolio.fills)),
        "total_fees": portfolio.total_fees,
        "fee_drag": portfolio.total_fees / invested if invested > 0 else 0.0,
        "missed_bars": float(sum(int(r.get("count", 0)) for r in journal.read()
                                 if r.get("event") == "missed_bars")),
    }


def backtest_reference(
    experiment: dict[str, Any], strategy, config: BacktestConfig, start: str, end: str
) -> dict[str, float]:
    """같은 기간·같은 전략을 백테스트로 돌린 기준선."""
    from ..data import get_source, load_universe
    from ..engine import run_backtest
    from ..metrics import compute_portfolio_metrics

    d = experiment["data"]
    source = get_source(
        d.get("source", "synthetic"),
        cache_dir=d.get("cache_dir", "data/cache"),
        csv_dir=d.get("csv_dir", "data/csv"),
    )
    # 지표 워밍업을 위해 과거를 넉넉히 받되 매매는 start 부터
    hist_start = (pd.Timestamp(start) - pd.Timedelta(days=max(strategy.warmup * 3, 400))).strftime("%Y-%m-%d")
    data = load_universe(source, list(d.get("symbols", [])), hist_start, end, min_bars=10)
    results = {
        s: run_backtest(df, strategy, config, symbol=s, trade_start=start)
        for s, df in data.items()
    }
    _, m = compute_portfolio_metrics(results, config)
    return m


def format_status(portfolio: PaperPortfolio, prices: dict[str, float]) -> str:
    """현재 계좌 상태를 사람이 읽는 형태로."""
    eq = portfolio.equity(prices)
    invested = portfolio.total_invested
    pnl = eq - invested
    lines = [
        "페이퍼 계좌 현황",
        "=" * 64,
        f"  시작        {portfolio.created_at[:19]}",
        f"  갱신        {portfolio.updated_at[:19]}",
        f"  투입원금    {invested:>16,.0f}원  (초기 {portfolio.initial_cash:,.0f} + 적립 {portfolio.total_deposited:,.0f})",
        f"  평가금액    {eq:>16,.0f}원",
        f"  손익        {pnl:>+16,.0f}원  ({pnl / invested:+.2%})" if invested > 0 else "",
        f"  현금        {portfolio.cash:>16,.0f}원",
        f"  누적수수료  {portfolio.total_fees:>16,.0f}원",
        f"  체결 건수   {len(portfolio.fills):>16,}건",
        "",
        "  보유 종목",
        "  " + "-" * 62,
    ]
    holdings = {s: q for s, q in portfolio.positions.items() if q}
    if not holdings:
        lines.append("    (없음 — 전액 현금)")
    else:
        lines.append(f"    {'종목':<12}{'수량':>16}{'평가금액':>14}{'비중':>8}")
        w = portfolio.weights(prices)
        for sym, qty in sorted(holdings.items()):
            px = prices.get(sym, 0.0)
            lines.append(f"    {sym:<12}{qty:>16.8f}{qty * px:>14,.0f}{w.get(sym, 0):>8.1%}")
    return "\n".join(x for x in lines if x != "")


def format_comparison(live: dict[str, float], bt: dict[str, float]) -> str:
    """라이브 vs 백테스트 비교표 — 이 프로젝트의 최종 판정."""
    if not live:
        return "아직 비교할 데이터가 부족합니다 (최소 2 사이클 필요)."

    keys = [
        ("total_return", "총수익률", True),
        ("cagr", "CAGR", True),
        ("ann_volatility", "연변동성", True),
        ("sharpe", "Sharpe", False),
        ("max_drawdown", "MDD", True),
    ]
    lines = [
        "라이브 vs 백테스트",
        "=" * 64,
        f"  {'지표':<12}{'라이브':>14}{'백테스트':>14}{'차이':>14}",
        "  " + "-" * 54,
    ]
    for key, label, pct in keys:
        lv, bv = live.get(key), bt.get(key)
        if lv is None or bv is None:
            continue
        fmt = (lambda v: f"{v:.2%}") if pct else (lambda v: f"{v:.3f}")
        lines.append(f"  {label:<12}{fmt(lv):>14}{fmt(bv):>14}{fmt(lv - bv):>14}")

    lines += ["", f"  운영일수 {live.get('days', 0):.0f}일 · 체결 {live.get('n_fills', 0):.0f}건 · "
              f"수수료 부담 {live.get('fee_drag', 0):.2%}"]

    days = live.get("days", 0)
    missed = live.get("missed_bars", 0)
    unreliable = False
    if missed:
        ratio = missed / max(days + missed, 1)
        lines.append(
            f"  놓친 봉 {missed:.0f}개 (전체의 {ratio:.0%}) — 서버가 꺼져 있던 구간"
        )
        if ratio > 0.10:
            unreliable = True

    # 판정은 자료가 믿을 만할 때만 낸다. 가동률이 낮으면 비교 자체가 무의미하다.
    if unreliable:
        lines += [
            "",
            "  ⚠ 놓친 봉이 10%를 넘어 비교를 낼 수 없다.",
            "    백테스트는 그 구간에도 매매한 것으로 계산하므로 같은 실험이 아니다.",
            "    가동률부터 고치고 다시 시작할 것 (docs/PAPER_TRADING.md '가동률').",
        ]
    elif days < 30:
        lines.append("\n  ※ 30일 미만은 표본이 너무 적다. 판단하지 말 것.")
    else:
        gap = live.get("cagr", 0) - bt.get("cagr", 0)
        if gap < -0.10:
            lines.append("\n  ⚠ 라이브가 백테스트보다 크게 부진하다. 원인 규명 전에는 실전 금지.")
            lines.append("    흔한 원인: 슬리피지 과소평가, 데이터 지연, 체결 시점 불일치")
        elif abs(gap) <= 0.10:
            lines.append("\n  ✓ 라이브와 백테스트가 비슷하다. 백테스트 가정이 현실적이었다는 뜻.")
    return "\n".join(lines)


def save_live_report(
    outdir: str | Path, journal: Journal, portfolio: PaperPortfolio, config: BacktestConfig
) -> dict[str, Path]:
    """자산곡선 CSV + 체결내역 CSV 저장."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    curve = equity_curve(journal)
    if not curve.empty:
        p = out / "live_equity.csv"
        curve.to_csv(p, encoding="utf-8-sig")
        written["equity"] = p

    if portfolio.fills:
        p = out / "live_fills.csv"
        pd.DataFrame([f.to_dict() for f in portfolio.fills]).to_csv(
            p, index=False, encoding="utf-8-sig"
        )
        written["fills"] = p
    return written
