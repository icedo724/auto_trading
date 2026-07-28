#!/usr/bin/env python3
"""전체 파이프라인 데모 (네트워크 불필요).

    python scripts/run_demo.py

1. 합성 유니버스 5종목 수집
2. 전 전략 × 전 파라미터를 **동일 시점**에서 그리드 탐색
3. 1위 파라미터의 민감도 분석
4. IS/OOS + 워크포워드로 과최적화 검증
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.config import BacktestConfig, CostModel
from quant.data import SyntheticSource, load_universe
from quant.optimizer import build_all_candidates, common_trade_start, grid_search, sensitivity
from quant.report import metrics_table, save_outputs, sparkline
from quant.validation import holdout_validate, walk_forward

SYMBOLS = ["DEMO-A", "DEMO-B", "DEMO-C", "DEMO-D", "DEMO-E"]
START, END = "2016-01-01", "2025-12-31"
OUTDIR = "reports/demo_script"


def main() -> int:
    print("=" * 72)
    print("1) 데이터 수집")
    print("=" * 72)
    data = load_universe(SyntheticSource(), SYMBOLS, START, END)
    calendar = next(iter(data.values())).index
    print(f"   {len(data)}종목 · {len(calendar)}거래일 · "
          f"{calendar[0].date()} ~ {calendar[-1].date()}")

    config = BacktestConfig(
        initial_cash=10_000_000,
        cost=CostModel.kr_stock(),
        execution="next_open",
        signal_lag=1,
        stop_loss_pct=0.10,
    )

    print("\n" + "=" * 72)
    print("2) 그리드 탐색 — 모든 후보가 동일 시점에서 경쟁")
    print("=" * 72)
    candidates = build_all_candidates()
    trade_start = common_trade_start(calendar, candidates)
    print(f"   후보 {len(candidates)}개")
    print(f"   공통 평가 구간 {trade_start.date()} ~ {calendar[-1].date()} "
          f"(최장 워밍업 {max(c.warmup for c in candidates)}봉 기준)\n")

    report = grid_search(data, candidates, config, objective="robust", min_trades=10)

    board = report.leaderboard(10)
    cols = ["label", "score", "cagr", "sharpe", "max_drawdown", "n_trades"]
    print("\n── 상위 10 ──")
    print(board[cols].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    best = report.best
    print(f"\n★ 최적: {best.label}")
    print(metrics_table(best.metrics))
    if best.equity is not None:
        print(f"\n   {sparkline(best.equity)}")

    print("\n" + "=" * 72)
    print(f"3) 파라미터 민감도 — {best.strategy}")
    print("=" * 72)
    for param in sorted(best.params):
        try:
            table = sensitivity(report, best.strategy, param)
        except ValueError:
            continue
        if len(table) < 2:
            continue
        print(f"\n── {param} ──")
        print(table.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\n" + "=" * 72)
    print("4) 과최적화 검증")
    print("=" * 72)
    split = calendar[int(len(calendar) * 0.7)]
    print(f"\n[IS/OOS] 분할 {split.date()}")
    for h in holdout_validate(data, candidates, config, split, top_n=3, min_trades=10):
        print(f"  [{'O' if h.survived else 'X'}] {h.candidate.label}")
        print(f"      IS  CAGR {h.is_metrics['cagr']:7.2%}  Sharpe {h.is_metrics['sharpe']:5.2f}")
        print(f"      OOS CAGR {h.oos_metrics['cagr']:7.2%}  Sharpe {h.oos_metrics['sharpe']:5.2f}")

    print("\n[워크포워드]")
    wf = walk_forward(data, candidates, config, train_days=504, test_days=126, min_trades=10)
    print("\n" + wf.windows_frame().to_string(index=False))
    print(f"\n  통합 OOS CAGR {wf.oos_metrics['cagr']:7.2%}  "
          f"Sharpe {wf.oos_metrics['sharpe']:5.2f}  "
          f"MDD {wf.oos_metrics['max_drawdown']:7.2%}")
    print(f"  워크포워드 효율(WFE): {wf.efficiency:.2f}")

    verdict = (
        "견고 — 실전 후보로 검토 가능" if wf.efficiency >= 0.5
        else "과최적화 의심 — 그리드 1위는 표본 잡음일 가능성이 높다"
    )
    print(f"  판정: {verdict}")

    curves = {r.label: r.equity for r in report.results[:3] if r.equity is not None}
    if wf.oos_equity is not None:
        curves["Walk-Forward OOS"] = wf.oos_equity
    written = save_outputs(
        OUTDIR,
        leaderboard=report.leaderboard(),
        best_params={"strategy": best.strategy, "params": best.params,
                     "metrics": best.metrics, "walk_forward_efficiency": wf.efficiency},
        equity_curves=curves,
        name="demo",
    )
    print("\n저장:", ", ".join(str(p) for p in written.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
