#!/usr/bin/env python3
"""손실 한도의 대가 — 바닥을 만들면 무엇을 잃는가.

    python scripts/risk_limit_experiment.py

드로다운 한도는 **공짜가 아니다.** 회복했을 손실을 확정 손실로 바꾼다.
얼마나 잃는지 알고 쓰라는 것이 이 실험의 목적이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.config import BacktestConfig, CostModel, RiskLimits
from quant.data import SyntheticSource, load_universe
from quant.engine import run_backtest
from quant.metrics import compute_portfolio_metrics
from quant.optimizer import common_trade_start
from quant.strategy import create_strategy

START, END = "2016-01-01", "2025-12-31"


def main() -> int:
    data = load_universe(
        SyntheticSource(annual_vol=0.45), [f"SYM-{c}" for c in "ABCDEF"], START, END
    )
    cal = next(iter(data.values())).index
    strat = create_strategy("sma_cross", {"fast": 20, "slow": 60})
    start = common_trade_start(cal, [strat])

    def run(risk: RiskLimits) -> dict:
        cfg = BacktestConfig(
            initial_cash=10_000_000, cost=CostModel.kr_stock(), risk=risk
        )
        res = {s: run_backtest(df, strat, cfg, symbol=s, trade_start=start)
               for s, df in data.items()}
        _, m = compute_portfolio_metrics(res, cfg)
        m["halts"] = sum(len(r.halt_events) for r in res.values())
        return m

    print("=" * 76)
    print(f"손실 한도별 비교 · {strat.describe()}")
    print(f"{start.date()} ~ {cal[-1].date()} · 6종목 동일비중")
    print("=" * 76)

    print(f"\n{'설정':<28}{'CAGR':>9}{'MDD':>9}{'Sharpe':>8}{'발동':>7}")
    print("-" * 76)

    base = run(RiskLimits())
    print(f"{'한도 없음':<28}{base['cagr']:>8.2%}{base['max_drawdown']:>9.2%}"
          f"{base['sharpe']:>8.2f}{'-':>7}")

    rows = []
    for dd in (0.30, 0.20, 0.15, 0.10):
        for action, label in (("halt", "영구정지"), ("cooldown", "60일 후 재개")):
            r = RiskLimits(max_drawdown=dd, action=action, cooldown_days=60)
            m = run(r)
            rows.append((dd, action, m))
            print(f"{f'고점대비 -{dd:.0%} / {label}':<28}{m['cagr']:>8.2%}"
                  f"{m['max_drawdown']:>9.2%}{m['sharpe']:>8.2f}{m['halts']:>7.0f}")

    print()
    print("── 원금 대비 한도 (진짜 '바닥') ──")
    for ml in (0.30, 0.20, 0.15):
        m = run(RiskLimits(max_loss=ml))
        print(f"{f'원금대비 -{ml:.0%} / 영구정지':<28}{m['cagr']:>8.2%}"
              f"{m['max_drawdown']:>9.2%}{m['sharpe']:>8.2f}{m['halts']:>7.0f}")

    print("\n" + "=" * 76)
    print("해석")
    print("=" * 76)

    halts = [m for dd, a, m in rows if a == "halt"]
    cools = [m for dd, a, m in rows if a == "cooldown"]
    print("  · 한도 없음 대비 CAGR 변화")
    print(f"      영구정지    평균 {sum(m['cagr'] for m in halts)/len(halts) - base['cagr']:+.2%}p")
    print(f"      쿨다운      평균 {sum(m['cagr'] for m in cools)/len(cools) - base['cagr']:+.2%}p")
    print("  · MDD 는 설정한 한도 근처로 눌린다 — 그것이 이 장치의 목적이다.")
    print("  · **영구정지는 회복 구간을 통째로 놓친다.** 한 번 걸리면 끝이므로")
    print("    한도를 타이트하게 잡을수록 손해가 커진다.")
    print("  · 쿨다운은 절충안이지만 하락장에서 반복 발동해 조금씩 깎인다.")
    print()
    print("  ⚠️ **쿨다운 모드의 max_drawdown 은 누적 손실을 보장하지 않는다.**")
    print("     재개할 때 고점을 현재로 리셋하기 때문이다(안 그러면 즉시 재발동).")
    print("     즉 '회차별 한도'이지 '누적 바닥'이 아니다.")
    print("     실제로 -20% 한도인데 최종 MDD 가 그보다 깊게 나오는 경우가 있다.")
    print()
    print("  → **진짜 바닥은 max_loss(원금 대비)다.** 고점 리셋의 영향을 받지 않는다.")
    print("     둘을 함께 쓰는 것이 안전하다:")
    print("       max_loss      = 절대 넘지 않을 선 (영구정지)")
    print("       max_drawdown  = 그 전에 한 박자 쉬는 장치 (쿨다운)")
    print()
    print("  권장: 한도는 **백테스트 MDD 보다 넉넉하게** 잡을 것.")
    print(f"        이 전략의 백테스트 MDD 는 {base['max_drawdown']:.1%} 이므로,")
    print("        한도를 그보다 타이트하게 잡으면 정상 변동에도 발동한다.")
    print("        한도는 '전략이 예상대로 작동하는 범위'를 벗어났을 때만 걸려야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
