#!/usr/bin/env python3
"""선택 방식 실험 — "수백 개 중 1등 고르기"가 정말 최선인가.

    python scripts/selection_experiment.py

같은 워크포워드 창 · 같은 후보군 · 같은 비용에서 **무엇을 고를지만** 바꿔 비교한다.
차이가 나온다면 그것은 전적으로 선택 방식 때문이다.

배경: 관측 성과 = 실력 + 잡음. 최댓값을 고르면 '실력 큰 것'이 아니라
'실력 + 운이 큰 것'을 고르게 되고, 운은 다음 기간에 재현되지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.bootstrap import compare_strategies, sharpe_of
from quant.config import BacktestConfig, CostModel
from quant.data import SyntheticSource, load_universe
from quant.ensemble import VolatilityScaled, build_ensemble
from quant.optimizer import build_all_candidates
from quant.strategy import create_strategy
from quant.validation import compare_selectors

START, END = "2016-01-01", "2025-12-31"


def main() -> int:
    print("데이터 준비...")
    data = load_universe(
        SyntheticSource(annual_vol=0.35),
        [f"SYM-{c}" for c in "ABCDEF"], START, END,
    )
    cal = next(iter(data.values())).index
    config = BacktestConfig(
        initial_cash=10_000_000, cost=CostModel.kr_stock(), trading_days=252
    )

    # 탐색 공간을 적당히 줄여 실험 시간을 확보 (선택 방식 비교가 목적)
    grids = {
        "sma_cross": {"fast": [5, 10, 20, 30], "slow": [40, 60, 90, 120]},
        "momentum": {"lookback": [20, 60, 120], "skip": [0, 20], "ma_filter": [0, 200]},
        "donchian": {"entry": [20, 40, 55], "exit": [10, 20], "atr_period": [14],
                     "atr_max": [0.0]},
        "zscore": {"window": [10, 20, 40], "entry_z": [-1.5, -2.0],
                   "exit_z": [0.0], "trend_filter": [0, 200]},
        "bollinger": {"period": [20, 60], "num_std": [2.0], "mode": ["reversion", "breakout"],
                      "trend_filter": [0]},
    }
    candidates = build_all_candidates(list(grids), grids)
    print(f"후보 {len(candidates)}개 · 종목 {len(data)} · {cal[0].date()}~{cal[-1].date()}\n")

    selectors = {
        "1등만 (argmax)": lambda rep: create_strategy(rep.best.strategy, rep.best.params),
        "상위 5개 평균": lambda rep: build_ensemble(rep, top_k=5),
        "상위 15개 평균": lambda rep: build_ensemble(rep, top_k=15),
        "1등 + 변동성타겟": lambda rep: VolatilityScaled(
            create_strategy(rep.best.strategy, rep.best.params), target_vol=0.15
        ),
        "상위15 평균 + 변동성타겟": lambda rep: VolatilityScaled(
            build_ensemble(rep, top_k=15), target_vol=0.15
        ),
    }

    print("=" * 78)
    print("워크포워드 비교 (학습 2년 / 검증 6개월, 창을 굴리며 매번 재선택)")
    print("=" * 78)
    results = compare_selectors(
        data, candidates, config, selectors,
        train_days=504, test_days=126, min_trades=10,
    )

    print(f"\n{'선택 방식':<26}{'OOS CAGR':>10}{'Sharpe':>9}{'MDD':>9}{'WFE':>8}")
    print("-" * 78)
    rows = []
    for name, wf in results.items():
        m = wf.oos_metrics
        rows.append((name, wf))
        print(f"{name:<26}{m['cagr']:>9.2%}{m['sharpe']:>9.2f}"
              f"{m['max_drawdown']:>9.2%}{wf.efficiency:>8.2f}")

    # 1등 방식과 나머지를 쌍대 부트스트랩으로 비교
    base_name, base_wf = rows[0]
    base_r = base_wf.oos_equity.pct_change().dropna()

    print("\n" + "=" * 78)
    print(f"'{base_name}' 대비 통계적 비교 (쌍대 블록 부트스트랩, Sharpe 기준)")
    print("=" * 78)
    for name, wf in rows[1:]:
        r = wf.oos_equity.pct_change().dropna()
        try:
            cmp = compare_strategies(
                r, base_r, sharpe_of, name_a=name, name_b=base_name, n_boot=1000
            )
        except ValueError as exc:
            print(f"\n{name}: 비교 불가 ({exc})")
            continue
        flag = "◎" if cmp.significant else " "
        print(f"\n{flag} {name}")
        print(f"    Sharpe {cmp.metric_a:.3f} vs {cmp.metric_b:.3f}"
              f"  (차이 {cmp.diff:+.3f}, 90% CI [{cmp.lower:+.3f}, {cmp.upper:+.3f}])")
        print(f"    더 나을 확률 {cmp.prob_a_better:.1%}"
              f"{'   ← 유의' if cmp.significant else '   (유의하지 않음)'}")

    print("\n" + "=" * 78)
    print("해석")
    print("=" * 78)
    print("  · 합성 데이터이므로 절대 수익률에는 의미가 없다.")
    print("  · 의미 있는 것은 '선택 방식 간의 상대 차이'와 그 통계적 유의성이다.")
    print("  · 앙상블은 알파를 만들지 않는다. 선택 편향을 줄여 재현성을 높인다.")
    print("  · 변동성 타겟팅은 변동성의 자기상관(예측 가능한 유일한 것)을 이용한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
