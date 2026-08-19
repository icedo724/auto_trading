#!/usr/bin/env python3
"""월 N만원 적립 소액 자동매매 — 비용 현실과 거래 빈도의 대가.

    python scripts/small_account_sim.py

"자주 많이 거래해서 조금씩 번다"가 소액에서 성립하는지 숫자로 확인한다.
합성 시세를 쓰므로 네트워크가 없어도 돌아가고, 언제 돌려도 같은 결과가 나온다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.config import BacktestConfig, CostModel
from quant.data import SyntheticSource, load_universe
from quant.engine import run_backtest
from quant.metrics import compute_portfolio_metrics
from quant.optimizer import common_trade_start
from quant.strategy import create_strategy

MONTHLY = 100_000
START, END = "2019-01-01", "2025-12-31"


def round_trip(c: CostModel) -> float:
    s, comm, tax = c.slippage_bps * 1e-4, c.commission_bps * 1e-4, c.sell_tax_bps * 1e-4
    return 1 - (1 - s) * (1 - comm - tax) / ((1 + s) * (1 + comm))


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    section("1) 왕복 1회 거래비용 — 무엇을 거래하든 매번 떼이는 금액")
    models = [("국내주식", CostModel.kr_stock()), ("미국주식", CostModel.us_stock()),
              ("코인(업비트)", CostModel.crypto_upbit()), ("코인(바이낸스)", CostModel.crypto_binance())]
    print(f"  {'대상':<16}{'왕복비용':>10}{'10만원 거래 시':>16}")
    for name, c in models:
        rt = round_trip(c)
        print(f"  {name:<16}{rt * 100:>9.3f}%{rt * 100_000:>14,.0f}원")

    section("2) 거래 빈도별 '본전에 필요한 연 총수익률'")
    print("  거래를 많이 한다고 돈을 버는 게 아니다. 비용은 횟수에 비례해 확정적으로 나간다.")
    print(f"\n  {'빈도':<12}{'연 거래수':>9}{'국내주식':>11}{'코인':>11}")
    kr, up = round_trip(CostModel.kr_stock()), round_trip(CostModel.crypto_upbit())
    for label, n in [("월 1회", 12), ("주 1회", 52), ("주 2회", 104),
                     ("일 1회", 250), ("일 5회", 1250)]:
        print(f"  {label:<12}{n:>9}{n * kr * 100:>10.1f}%{n * up * 100:>10.1f}%")

    section("3) 월 10만원 적립 · 7년 — 전략별 실측")
    # 코인 수준의 변동성(연 70%)을 가진 합성 시세 5종목
    src = SyntheticSource(annual_vol=0.70, base_price=50_000)
    data = load_universe(src, ["COIN-A", "COIN-B", "COIN-C", "COIN-D", "COIN-E"], START, END)
    cal = next(iter(data.values())).index

    cands = [
        ("적립식 매수 (거래 안 함)", create_strategy("buy_and_hold")),
        ("그리드 4분할 (band 20%)", create_strategy("grid", {"ma": 20, "band": 0.20, "levels": 4})),
        ("그리드 8분할 (band 10%)", create_strategy("grid", {"ma": 20, "band": 0.10, "levels": 8})),
        ("추세추종 (sma 20/60)", create_strategy("sma_cross", {"fast": 20, "slow": 60})),
        ("고빈도 회귀 (zscore 5일)", create_strategy("zscore", {"window": 5, "entry_z": -1.5, "exit_z": -0.5})),
    ]
    start = common_trade_start(cal, [s for _, s in cands])
    years = len(cal[cal >= start]) / 252

    def run(strat, cost):
        cfg = BacktestConfig(
            initial_cash=MONTHLY, contribution=MONTHLY, contribution_freq="M",
            min_order_value=5_000, cost=cost, allow_fractional=True,
            rebalance_threshold=0.10, trading_days=252,
        )
        res = {s: run_backtest(df, strat, cfg, symbol=s, trade_start=start)
               for s, df in data.items()}
        _, m = compute_portfolio_metrics(res, cfg)
        # 포트폴리오 잔고 = 종목별 잔고 합계 / 종목수 (동일비중 분할 투자)
        bal = sum(r.balance.iloc[-1] for r in res.values()) / len(res)
        inv = MONTHLY + sum(r.contributions.sum() for r in res.values()) / len(res)
        return m, bal, inv

    print(f"  기간 {start.date()} ~ {cal[-1].date()} ({years:.1f}년) · 매월 {MONTHLY:,}원 적립\n")
    print(f"  {'전략':<24}{'연거래':>7}{'투입원금':>12}{'최종잔고':>12}{'손익':>12}{'MDD':>9}")
    print("  " + "-" * 74)
    rows = []
    for name, s in cands:
        m, bal, inv = run(s, CostModel.crypto_upbit())
        rows.append((name, m, bal, inv))
        print(f"  {name:<24}{m['n_trades'] / years:>7.0f}{inv:>12,.0f}{bal:>12,.0f}"
              f"{bal - inv:>+12,.0f}{m['max_drawdown']:>9.1%}")

    section("4) 거래 빈도의 대가 — 같은 그리드 전략, 분할 단계만 다르게")
    print("  분할을 잘게 쪼갤수록 거래는 늘지만, 한 칸이 비용보다 작아지면 역효과.\n")
    print(f"  {'설정':<26}{'한 칸':>7}{'연거래':>7}{'비용0 수익':>11}{'비용후 수익':>12}{'차이':>9}")
    print("  " + "-" * 74)
    for band, levels in [(0.30, 2), (0.20, 4), (0.10, 8), (0.05, 10), (0.02, 10)]:
        s = create_strategy("grid", {"ma": 20, "band": band, "levels": levels})
        g, _, _ = run(s, CostModel.zero())
        c, _, _ = run(s, CostModel.crypto_upbit())
        print(f"  {'band %.0f%% / %d분할' % (band * 100, levels):<26}{s.step_pct:>6.1%}"
              f"{c['n_trades'] / years:>7.0f}{g['cagr']:>11.1%}{c['cagr']:>12.1%}"
              f"{c['cagr'] - g['cagr']:>+9.1%}")

    section("5) 결론")
    best = max(rows, key=lambda r: r[2] - r[3])
    print(f"  · 이 시뮬레이션의 최고 성적: {best[0]}  (손익 {best[2] - best[3]:+,.0f}원)")
    print("  · 거래 빈도는 수익원이 아니라 비용 증폭기다. 엣지가 음수면 많이 거래할수록 빨리 잃는다.")
    print("  · 합성 데이터 결과이므로 절대 수익률에는 의미가 없다.")
    print("    의미 있는 것은 '비용 0 대비 비용 후'의 낙폭 — 이건 실데이터에서도 그대로 발생한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
