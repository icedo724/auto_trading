"""결과 리포트 (콘솔 / 마크다운 / CSV / 차트)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import drawdown_series

_PCT = {
    "total_return", "cagr", "ann_volatility", "max_drawdown", "win_rate", "avg_win",
    "avg_loss", "expectancy", "exposure", "cost_drag", "best_day", "worst_day",
    "var_95", "ulcer_index",
}
_LABELS = {
    "total_return": "총수익률", "cagr": "CAGR", "ann_volatility": "연변동성",
    "sharpe": "Sharpe", "sortino": "Sortino", "max_drawdown": "MDD",
    "calmar": "Calmar", "ulcer_index": "Ulcer", "var_95": "VaR(95%)",
    "win_rate": "승률", "profit_factor": "손익비(PF)", "avg_win": "평균수익",
    "avg_loss": "평균손실", "payoff_ratio": "Payoff", "expectancy": "기대값",
    "n_trades": "거래수", "avg_holding_days": "평균보유일", "exposure": "노출도",
    "turnover": "연회전율", "cost_drag": "비용부담", "best_day": "최고일",
    "worst_day": "최악일",
}


def fmt_metric(key: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-" if value is None else ("∞" if value > 0 else "-∞")
    if not isinstance(value, (int, float, np.floating, np.integer)):
        return str(value)
    if key in _PCT:
        return f"{value * 100:.2f}%"
    if key in ("n_trades",):
        return f"{int(value)}"
    return f"{value:.3f}"


def label_of(key: str) -> str:
    return _LABELS.get(key, key)


def metrics_table(metrics: dict[str, float], title: str = "") -> str:
    lines = [f"{title}"] if title else []
    for k, v in metrics.items():
        lines.append(f"  {label_of(k):<12} {fmt_metric(k, v):>12}")
    return "\n".join(lines)


def sparkline(series: pd.Series, width: int = 60) -> str:
    """콘솔용 자산곡선 스파크라인."""
    blocks = "▁▂▃▄▅▆▇█"
    s = series.dropna()
    if len(s) < 2:
        return ""
    idx = np.linspace(0, len(s) - 1, min(width, len(s))).astype(int)
    v = s.iloc[idx].to_numpy(dtype=float)
    lo, hi = v.min(), v.max()
    if hi <= lo:
        return blocks[0] * len(v)
    scaled = ((v - lo) / (hi - lo) * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[i] for i in scaled)


# --------------------------------------------------------------------------------
# 마크다운
# --------------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df.empty:
        return "_(결과 없음)_"
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float) and np.isfinite(v):
                cells.append(floatfmt.format(v))
            elif isinstance(v, float):
                cells.append("∞" if v > 0 else "-∞")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def optimization_markdown(
    report: "Any",
    *,
    top: int = 20,
    sensitivity_params: Sequence[tuple[str, str]] = (),
) -> str:
    """OptimizationReport -> 마크다운 문서."""
    cfg = report.config
    board = report.leaderboard(top)

    keep = [
        c for c in ["strategy", "label", "score", "cagr", "sharpe", "sortino",
                    "max_drawdown", "calmar", "win_rate", "profit_factor",
                    "n_trades", "turnover"] if c in board.columns
    ]
    view = board[keep].copy()

    lines = [
        "# 파라미터 최적화 리포트",
        "",
        "## 실험 조건 (모든 후보 공통)",
        "",
        f"- 평가 구간: **{report.trade_start.date()} ~ {report.trade_end.date()}**",
        f"- 종목 유니버스 ({len(report.symbols)}): {', '.join(report.symbols)}",
        f"- 목적함수: `{report.objective}`",
        f"- 평가 후보 수: **{len(report.results):,}개**",
        f"- 체결: {cfg.execution} (신호 지연 {cfg.signal_lag}일)",
        f"- 비용: 수수료 {cfg.cost.commission_bps}bp/편도, "
        f"거래세 {cfg.cost.sell_tax_bps}bp(매도), 슬리피지 {cfg.cost.slippage_bps}bp",
        f"- 초기자본: {cfg.initial_cash:,.0f}",
        f"- 소요 시간: {report.elapsed_sec:.1f}초",
        "",
        "> 모든 후보는 위 조건과 **동일한 시작일**(가장 긴 지표 워밍업 기준)에서 매매를 시작한다.",
        "",
        f"## 리더보드 (상위 {len(view)})",
        "",
        _md_table(view),
        "",
        "## 전략별 최고 성과",
        "",
    ]

    per_strategy = pd.DataFrame(
        [
            {
                "strategy": r.strategy,
                "best_params": r.label.split("(", 1)[1].rstrip(")") or "-",
                "score": r.score,
                "cagr": r.metrics.get("cagr", 0.0),
                "sharpe": r.metrics.get("sharpe", 0.0),
                "max_drawdown": r.metrics.get("max_drawdown", 0.0),
                "n_trades": r.metrics.get("n_trades", 0.0),
            }
            for r in report.best_per_strategy()
        ]
    )
    lines += [_md_table(per_strategy), ""]

    best = report.best
    lines += [
        "## 최종 선택 알고리즘",
        "",
        f"```\n{best.label}\n```",
        "",
        _md_table(
            pd.DataFrame([{label_of(k): fmt_metric(k, v) for k, v in best.metrics.items()}])
        ),
        "",
    ]

    if best.per_symbol:
        lines += ["### 종목별 성과 (최종 선택)", ""]
        per_sym = pd.DataFrame(best.per_symbol).T
        cols = [c for c in ["cagr", "sharpe", "max_drawdown", "win_rate", "n_trades"]
                if c in per_sym.columns]
        per_sym = per_sym[cols].reset_index().rename(columns={"index": "symbol"})
        lines += [_md_table(per_sym), ""]

    if sensitivity_params:
        from .optimizer import sensitivity

        lines += [
            "## 파라미터 민감도",
            "",
            "> 특정 값에서만 점수가 튀면 과최적화 신호. "
            "이웃 값들도 고르게 좋아야 실전에서 재현된다.",
            "",
        ]
        for strat_name, param in sensitivity_params:
            try:
                sens = sensitivity(report, strat_name, param)
            except ValueError:
                continue
            lines += [f"### `{strat_name}` · `{param}`", "", _md_table(sens), ""]

    return "\n".join(lines)


def holdout_markdown(results: Sequence["Any"]) -> str:
    """HoldoutResult 목록 -> 마크다운."""
    if not results:
        return "## IS/OOS 검증\n\n_(결과 없음)_\n"
    rows = []
    for h in results:
        rows.append(
            {
                "candidate": h.candidate.label,
                "IS_cagr": h.is_metrics.get("cagr", 0.0),
                "IS_sharpe": h.is_metrics.get("sharpe", 0.0),
                "OOS_cagr": h.oos_metrics.get("cagr", 0.0),
                "OOS_sharpe": h.oos_metrics.get("sharpe", 0.0),
                "OOS_mdd": h.oos_metrics.get("max_drawdown", 0.0),
                "OOS_trades": h.oos_metrics.get("n_trades", 0.0),
                "생존": "O" if h.survived else "X",
            }
        )
    split = results[0].split_date.date()
    return "\n".join(
        [
            "## IS / OOS 검증",
            "",
            f"- 분할 기준일: **{split}** (이전=학습, 이후=검증)",
            "- `생존` = OOS Sharpe > 0.3 이고 CAGR > 0 이며 거래 3건 이상",
            "",
            _md_table(pd.DataFrame(rows)),
            "",
        ]
    )


def walkforward_markdown(wf: "Any") -> str:
    """WalkForwardResult -> 마크다운."""
    lines = [
        "## 워크포워드 분석",
        "",
        "> 학습창을 굴리며 매 구간 파라미터를 **재선택**하고, 그 다음 구간의 실적만 이어붙였다.",
        "> 실제 운용에 가장 가까운 추정치다.",
        "",
        _md_table(wf.windows_frame()),
        "",
        "### 통합 OOS 성과",
        "",
        _md_table(
            pd.DataFrame([{label_of(k): fmt_metric(k, v) for k, v in wf.oos_metrics.items()}])
        ),
        "",
        f"- **워크포워드 효율(WFE): {wf.efficiency:.2f}**  "
        "(1.0 근처=견고, 0.5 미만=과최적화 의심)",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------
# 저장
# --------------------------------------------------------------------------------
def save_outputs(
    outdir: str | Path,
    *,
    markdown: str | None = None,
    leaderboard: pd.DataFrame | None = None,
    best_params: dict[str, Any] | None = None,
    equity_curves: dict[str, pd.Series] | None = None,
    trades: pd.DataFrame | None = None,
    name: str = "optimization",
) -> dict[str, Path]:
    """리포트 산출물을 디렉터리에 저장하고 경로를 반환한다."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if markdown is not None:
        p = out / f"{name}.md"
        p.write_text(markdown, encoding="utf-8")
        written["markdown"] = p
    if leaderboard is not None:
        p = out / f"{name}_leaderboard.csv"
        leaderboard.to_csv(p, index=False, encoding="utf-8-sig")
        written["leaderboard"] = p
    if best_params is not None:
        p = out / f"{name}_best.json"
        p.write_text(
            json.dumps(best_params, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        written["best_params"] = p
    if equity_curves:
        p = out / f"{name}_equity.csv"
        pd.DataFrame(equity_curves).to_csv(p, encoding="utf-8-sig")
        written["equity"] = p
        chart = plot_equity(equity_curves, out / f"{name}_equity.png")
        if chart:
            written["chart"] = chart
    if trades is not None and not trades.empty:
        p = out / f"{name}_trades.csv"
        trades.to_csv(p, index=False, encoding="utf-8-sig")
        written["trades"] = p

    return written


def plot_equity(curves: dict[str, pd.Series], path: str | Path) -> Path | None:
    """자산곡선 + 드로다운 차트. matplotlib 이 없으면 건너뛴다."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for label, eq in curves.items():
        eq = eq.dropna()
        if eq.empty:
            continue
        ax1.plot(eq.index, eq / eq.iloc[0], label=label, linewidth=1.3)
        ax2.fill_between(eq.index, drawdown_series(eq) * 100, 0, alpha=0.25)

    ax1.set_title("Equity Curve (normalized)")
    ax1.set_ylabel("Growth (x)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_title("Drawdown (%)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()

    path = Path(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
