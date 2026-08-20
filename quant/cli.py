"""명령줄 인터페이스.

    python -m quant list                     # 등록된 전략/파라미터 공간 보기
    python -m quant fetch   -c configs/experiment_kr.yaml
    python -m quant backtest -c ... -s sma_cross -p fast=20 -p slow=60
    python -m quant optimize -c ...          # 그리드 탐색 (동일 시점 비교)
    python -m quant validate -c ...          # IS/OOS + 워크포워드 검증
    python -m quant signal   -c ...          # 최적 파라미터로 오늘의 매매 신호
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import BacktestConfig, CostModel, RiskLimits
from .data import get_source, load_universe
from .dates import resolve_period
from .engine import run_backtest
from .metrics import compute_portfolio_metrics
from .optimizer import (
    OBJECTIVES,
    build_all_candidates,
    common_trade_start,
    grid_search,
    sensitivity,
)
from .report import (
    holdout_markdown,
    metrics_table,
    optimization_markdown,
    save_outputs,
    sparkline,
    walkforward_markdown,
)
from .strategy import available_strategies, create_strategy, get_strategy_class
from .validation import holdout_validate, walk_forward

DEFAULT_CONFIG = Path("configs/experiment_kr.yaml")


# --------------------------------------------------------------------------------
# 설정 로딩
# --------------------------------------------------------------------------------
def load_experiment(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"설정 파일이 없습니다: {p}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg.setdefault("data", {})
    cfg.setdefault("backtest", {})
    cfg.setdefault("optimize", {})
    return cfg


def build_backtest_config(exp: dict[str, Any]) -> BacktestConfig:
    bt = dict(exp.get("backtest", {}))
    risk = bt.pop("risk", None)
    if isinstance(risk, dict):
        bt["risk"] = RiskLimits(**risk)
    cost = bt.pop("cost", "kr")
    if isinstance(cost, str):
        bt["cost"] = CostModel.named(cost)
    elif isinstance(cost, dict):
        bt["cost"] = CostModel(**cost)
    return BacktestConfig(**bt)


def load_data(exp: dict[str, Any], *, refresh: bool = False) -> dict[str, pd.DataFrame]:
    d = exp["data"]
    source = get_source(
        d.get("source", "synthetic"),
        cache_dir=d.get("cache_dir", "data/cache"),
        refresh=refresh,
        csv_dir=d.get("csv_dir", "data/csv"),
    )
    symbols = list(d.get("symbols") or [])
    if not symbols:
        raise SystemExit("설정의 data.symbols 가 비어 있습니다.")

    # 'today' / '-3y' 같은 상대 표기를 실제 날짜로. 스케줄 실행 시 구간이 따라 움직인다.
    start, end = resolve_period(d.get("start"), d.get("end"))
    print(
        f"[데이터] source={source.name} 종목={len(symbols)}개 기간={start}~{end}"
    )
    data = load_universe(
        source, symbols, start, end, min_bars=int(d.get("min_bars", 60))
    )
    cal = next(iter(data.values())).index
    print(f"[데이터] 거래일 {len(cal)}봉 · {cal[0].date()} ~ {cal[-1].date()}")
    return data


def build_candidates_from(exp: dict[str, Any]):
    o = exp.get("optimize", {})
    return build_all_candidates(o.get("strategies"), o.get("grids"))


def _parse_params(pairs: list[str] | None, strategy: str) -> dict[str, Any]:
    """``-p fast=20 -p mode=reversion`` 를 타입에 맞춰 파싱."""
    if not pairs:
        return {}
    defaults = get_strategy_class(strategy).defaults
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"파라미터 형식 오류: {pair!r} (key=value)")
        k, v = pair.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in defaults:
            raise SystemExit(f"{strategy}: 알 수 없는 파라미터 {k!r}")
        ref = defaults[k]
        try:
            if isinstance(ref, bool):
                out[k] = v.lower() in ("1", "true", "yes", "y", "t")
            elif isinstance(ref, int):
                out[k] = int(v)
            elif isinstance(ref, float):
                out[k] = float(v)
            else:
                out[k] = v
        except ValueError as exc:
            raise SystemExit(f"{k}={v!r} 를 {type(ref).__name__} 로 변환할 수 없습니다.") from exc
    return out


# --------------------------------------------------------------------------------
# 서브커맨드
# --------------------------------------------------------------------------------
def cmd_list(args: argparse.Namespace) -> int:
    print("등록된 전략\n" + "=" * 60)
    for name in available_strategies():
        cls = get_strategy_class(name)
        n_combos = 1
        for v in cls.param_space.values():
            n_combos *= max(len(v), 1)
        print(f"\n● {name}   (그리드 조합 {n_combos}개)")
        doc = (cls.__doc__ or "").strip().splitlines()
        if doc:
            print(f"  {doc[0]}")
        for k, default in sorted(cls.defaults.items()):
            space = cls.param_space.get(k, [])
            print(f"    - {k:<14} 기본={default!r:<12} 후보={space}")
    print(f"\n목적함수: {', '.join(sorted(OBJECTIVES))}")
    return 0


def cmd_check_data(args: argparse.Namespace) -> int:
    """어떤 데이터 소스가 이 환경에서 실제로 동작하는지 진단한다."""
    from .data.probe import DEFAULT_PROBES, format_report, probe_all

    probes = DEFAULT_PROBES
    if args.source:
        probes = [p for p in DEFAULT_PROBES if p[0] in args.source]
        if not probes:
            probes = [(s, args.symbol or "005930", "사용자 지정") for s in args.source]
    elif args.symbol:
        probes = [(n, args.symbol, d) for n, _, d in DEFAULT_PROBES if n != "synthetic"]

    print("소스별로 최근 30일 일봉을 실제로 받아봅니다. 잠시 걸릴 수 있습니다.\n")
    results = probe_all(probes, days=args.days)
    print(format_report(results))
    return 0 if any(r.ok and r.source != "synthetic" for r in results) else 1


def cmd_fetch(args: argparse.Namespace) -> int:
    exp = load_experiment(args.config)
    data = load_data(exp, refresh=args.refresh)
    print("\n종목별 수집 결과")
    for sym, df in data.items():
        valid = df["close"].notna().sum()
        print(f"  {sym:<12} {valid:>5}봉  {df.index[0].date()} ~ {df.index[-1].date()}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    exp = load_experiment(args.config)
    config = build_backtest_config(exp)
    data = load_data(exp)

    params = _parse_params(args.param, args.strategy)
    strat = create_strategy(args.strategy, params)
    calendar = next(iter(data.values())).index
    trade_start = common_trade_start(calendar, [strat])

    print(f"\n[백테스트] {strat.describe()}")
    print(f"[백테스트] 매매 시작 {trade_start.date()} (워밍업 {strat.warmup}봉)\n")

    results = {
        sym: run_backtest(df, strat, config, symbol=sym, trade_start=trade_start)
        for sym, df in data.items()
    }
    equity, metrics = compute_portfolio_metrics(results, config)

    print(metrics_table(metrics, "── 포트폴리오 성과 (동일비중) ──"))
    print(f"\n  {sparkline(equity)}")
    print(f"  {equity.index[0].date()} → {equity.index[-1].date()}   "
          f"{config.initial_cash:,.0f} → {equity.iloc[-1]:,.0f}\n")

    print("── 종목별 ──")
    per_sym = pd.DataFrame({s: r.metrics for s, r in results.items()}).T
    cols = [c for c in ["cagr", "sharpe", "max_drawdown", "win_rate", "n_trades"]
            if c in per_sym.columns]
    print(per_sym[cols].to_string(float_format=lambda v: f"{v:8.3f}"))

    if args.out:
        trades = pd.concat(
            [r.trades_frame() for r in results.values()], ignore_index=True
        )
        written = save_outputs(
            args.out,
            leaderboard=per_sym.reset_index().rename(columns={"index": "symbol"}),
            best_params={"strategy": strat.name, "params": strat.params,
                         "metrics": metrics},
            equity_curves={strat.describe(): equity},
            trades=trades,
            name=f"backtest_{strat.name}",
        )
        print("\n저장:", ", ".join(str(p) for p in written.values()))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    exp = load_experiment(args.config)
    config = build_backtest_config(exp)
    data = load_data(exp)
    candidates = build_candidates_from(exp)

    o = exp.get("optimize", {})
    objective = args.objective or o.get("objective", "robust")
    calendar = next(iter(data.values())).index
    trade_start = common_trade_start(calendar, candidates)

    print(f"\n[최적화] 후보 {len(candidates):,}개 · 목적함수={objective}")
    print(f"[최적화] 공통 평가 구간 {trade_start.date()} ~ {calendar[-1].date()} "
          f"(모든 후보 동일)\n")

    report = grid_search(
        data, candidates, config,
        objective=objective,
        trade_start=trade_start,
        min_trades=int(o.get("min_trades", 5)),
        workers=args.workers,
        store_equity_top=int(o.get("store_equity_top", 8)),
    )

    top = args.top
    print(f"\n── 상위 {top}개 ──")
    board = report.leaderboard(top)
    show = [c for c in ["label", "score", "cagr", "sharpe", "max_drawdown",
                        "n_trades"] if c in board.columns]
    print(board[show].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    best = report.best
    print(f"\n★ 최적 알고리즘: {best.label}")
    print(metrics_table(best.metrics))
    if best.equity is not None:
        print(f"\n  {sparkline(best.equity)}")

    # 다중검정 보정 — 수백 개를 돌려 1등을 골랐다는 사실 자체가 성과를 부풀린다
    try:
        from .significance import assess_report

        sig = assess_report(report)
        print("\n" + sig.format())
    except ValueError as exc:
        print(f"\n[유의성 판정 생략] {exc}")

    outdir = args.out or o.get("output_dir", "reports")
    sens_params = [
        (best.strategy, p)
        for p in sorted(get_strategy_class(best.strategy).param_space)
    ]
    md = optimization_markdown(report, top=top, sensitivity_params=sens_params)
    curves = {r.label: r.equity for r in report.results[:5] if r.equity is not None}
    written = save_outputs(
        outdir,
        markdown=md,
        leaderboard=report.leaderboard(),
        best_params={
            "strategy": best.strategy,
            "params": best.params,
            "score": best.score,
            "objective": objective,
            "metrics": best.metrics,
            "evaluation": {
                "start": str(report.trade_start.date()),
                "end": str(report.trade_end.date()),
                "symbols": report.symbols,
            },
            "backtest_config": config.to_dict(),
        },
        equity_curves=curves,
        name="optimization",
    )
    print("\n저장:")
    for k, p in written.items():
        print(f"  {k:<12} {p}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    exp = load_experiment(args.config)
    config = build_backtest_config(exp)
    data = load_data(exp)
    candidates = build_candidates_from(exp)

    o = exp.get("optimize", {})
    v = exp.get("validate", {})
    objective = args.objective or o.get("objective", "robust")
    calendar = next(iter(data.values())).index

    split = args.split or v.get("split_date")
    if not split:
        split = calendar[int(len(calendar) * float(v.get("split_ratio", 0.7)))]
    split = pd.Timestamp(split)

    print(f"\n[검증] IS/OOS 분할 기준일 {split.date()}")
    holdout = holdout_validate(
        data, candidates, config, split,
        objective=objective, top_n=int(v.get("top_n", 5)),
        min_trades=int(o.get("min_trades", 5)), workers=args.workers,
    )
    for h in holdout:
        mark = "O" if h.survived else "X"
        print(
            f"  [{mark}] {h.candidate.label}\n"
            f"      IS  CAGR {h.is_metrics.get('cagr', 0):7.2%}  "
            f"Sharpe {h.is_metrics.get('sharpe', 0):5.2f}\n"
            f"      OOS CAGR {h.oos_metrics.get('cagr', 0):7.2%}  "
            f"Sharpe {h.oos_metrics.get('sharpe', 0):5.2f}  "
            f"MDD {h.oos_metrics.get('max_drawdown', 0):7.2%}"
        )

    md_parts = [holdout_markdown(holdout)]
    curves = {f"OOS · {h.candidate.label}": h.oos_equity
              for h in holdout[:5] if h.oos_equity is not None}

    if not args.no_walkforward:
        print("\n[검증] 워크포워드 분석")
        wf = walk_forward(
            data, candidates, config,
            train_days=int(v.get("train_days", 504)),
            test_days=int(v.get("test_days", 126)),
            objective=objective,
            min_trades=int(o.get("min_trades", 5)),
            workers=args.workers,
        )
        print("\n" + wf.windows_frame().to_string(index=False))
        print("\n── 통합 OOS 성과 ──")
        print(metrics_table(wf.oos_metrics))
        print(f"\n  워크포워드 효율(WFE): {wf.efficiency:.2f}")
        md_parts.append(walkforward_markdown(wf))
        if wf.oos_equity is not None:
            curves["Walk-Forward OOS"] = wf.oos_equity

    outdir = args.out or o.get("output_dir", "reports")
    written = save_outputs(
        outdir,
        markdown="# 검증 리포트\n\n" + "\n".join(md_parts),
        equity_curves=curves,
        name="validation",
    )
    print("\n저장:", ", ".join(str(p) for p in written.values()))
    return 0


def cmd_signal(args: argparse.Namespace) -> int:
    """최적(또는 지정) 파라미터로 **가장 최근 봉의 매매 신호**를 출력한다."""
    exp = load_experiment(args.config)
    data = load_data(exp, refresh=args.refresh)

    if args.best_file:
        payload = json.loads(Path(args.best_file).read_text(encoding="utf-8"))
        strat = create_strategy(payload["strategy"], payload.get("params", {}))
    else:
        if not args.strategy:
            raise SystemExit("--strategy 또는 --best-file 중 하나가 필요합니다.")
        strat = create_strategy(args.strategy, _parse_params(args.param, args.strategy))

    print(f"\n[신호] {strat.describe()}")
    rows = []
    for sym, df in data.items():
        sig = strat.generate_signals(df).dropna()
        if sig.empty:
            continue
        today, prev = sig.iloc[-1], (sig.iloc[-2] if len(sig) > 1 else 0.0)
        action = "HOLD"
        if today > prev + 1e-9:
            action = "BUY" if prev <= 0 else "ADD"
        elif today < prev - 1e-9:
            action = "SELL" if today <= 0 else "REDUCE"
        elif today == 0:
            action = "FLAT"
        rows.append(
            {
                "symbol": sym,
                "date": sig.index[-1].strftime("%Y-%m-%d"),
                "close": round(float(df["close"].iloc[-1]), 2),
                "target_weight": round(float(today), 3),
                "prev_weight": round(float(prev), 3),
                "action": action,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["action", "symbol"])
    print("\n" + frame.to_string(index=False))
    print(
        "\n※ 이 신호는 마지막 봉 종가 기준이며, 백테스트 규칙상 "
        f"'{'다음 거래일 시가' if exp['backtest'].get('execution', 'next_open') == 'next_open' else '다음 거래일 종가'}'에 체결하는 것을 가정한다."
    )
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        p = Path(args.out) / "signals.csv"
        frame.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"저장: {p}")
    return 0


def _paper_setup(args: argparse.Namespace):
    """설정 · 전략 · 트레이더를 구성한다."""
    from .live import PaperTrader

    exp = load_experiment(args.config)
    config = build_backtest_config(exp)

    if args.best_file:
        payload = json.loads(Path(args.best_file).read_text(encoding="utf-8"))
        strat = create_strategy(payload["strategy"], payload.get("params", {}))
    elif args.strategy:
        strat = create_strategy(args.strategy, _parse_params(args.param, args.strategy))
    else:
        raise SystemExit("--strategy 또는 --best-file 중 하나가 필요합니다.")

    trader = PaperTrader(
        exp, strat, config,
        state_dir=args.state_dir,
        name=args.name,
        report_dir=args.out or "reports/live",
        auto_report=not getattr(args, "no_report", False),
    )
    return exp, config, strat, trader


def cmd_paper(args: argparse.Namespace) -> int:
    """실데이터 · 가상자금 페이퍼 트레이딩. 실제 주문은 절대 내지 않는다."""
    exp, config, strat, trader = _paper_setup(args)

    print(f"[페이퍼] {strat.describe()}")
    print(f"[페이퍼] 상태 {trader.state_path} · 저널 {trader.journal.path}")
    print(f"[페이퍼] 초기자본 {config.initial_cash:,.0f}원"
          + (f" · 적립 {config.contribution:,.0f}원/{config.contribution_freq}"
             if config.contribution > 0 else "")
          + "  ※ 가상 자금 — 실제 주문 없음\n")

    if args.loop:
        return trader.run_forever(
            interval_sec=args.interval, stop_file=args.stop_file, max_cycles=args.max_cycles
        )

    res = trader.run_once(force=args.force)
    if res.errors:
        for e in res.errors:
            print(f"  [오류] {e}")
    print(f"  처리 {len(res.processed)}종목 · 건너뜀 {len(res.skipped)} · 체결 {len(res.fills)}건")
    for f in res.fills:
        print(f"    {f['side']:<4} {f['symbol']:<10} {f['quantity']:.8f} @ {f['price']:,.2f}"
              f"  수수료 {f['fee']:,.0f}원")
    if res.deposited:
        print(f"  적립 입금 {res.deposited:,.0f}원")
    print(f"  평가금액 {res.equity:,.0f}원")
    if res.report_path:
        print(f"  리포트 {res.report_path}")
    return 1 if res.errors and not res.processed else 0


def cmd_paper_status(args: argparse.Namespace) -> int:
    from .live import format_status

    exp, config, strat, trader = _paper_setup(args)
    data = load_data(exp)
    prices = {s: float(df["close"].iloc[-1]) for s, df in data.items()}
    print()
    print(format_status(trader.portfolio, prices))
    return 0


def cmd_paper_report(args: argparse.Namespace) -> int:
    from .live import (
        backtest_reference, format_comparison, format_status,
        live_metrics, live_period, save_live_report,
    )

    exp, config, strat, trader = _paper_setup(args)
    data = load_data(exp)
    prices = {s: float(df["close"].iloc[-1]) for s, df in data.items()}

    print()
    print(format_status(trader.portfolio, prices))

    live = live_metrics(trader.journal, trader.portfolio, config)
    if not live:
        print("\n아직 사이클이 부족해 성과를 낼 수 없습니다.")
        return 0

    # 벽시계 시각이 아니라 라이브가 실제로 커버한 시장 날짜를 써야 한다
    period = live_period(trader.journal)
    try:
        if period is None:
            raise ValueError("시장 날짜 구간을 알 수 없습니다 (사이클 부족)")
        bt = backtest_reference(exp, strat, config, *period)
    except Exception as exc:  # noqa: BLE001 - 비교는 부가 기능이므로 실패해도 진행
        print(f"\n[경고] 백테스트 기준선 계산 실패: {exc}")
        bt = {}

    print()
    print(format_comparison(live, bt))

    outdir = args.out or exp.get("optimize", {}).get("output_dir", "reports") + "/live"
    written = save_live_report(outdir, trader.journal, trader.portfolio, config)
    if written:
        print("\n저장:", ", ".join(str(v) for v in written.values()))
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    exp = load_experiment(args.config)
    config = build_backtest_config(exp)
    data = load_data(exp)
    candidates = [c for c in build_candidates_from(exp) if c.name == args.strategy]
    if not candidates:
        raise SystemExit(f"{args.strategy} 후보가 설정의 optimize.strategies 에 없습니다.")

    report = grid_search(
        data, candidates, config,
        objective=args.objective or exp["optimize"].get("objective", "robust"),
        workers=args.workers, store_equity_top=0,
    )
    for param in sorted(get_strategy_class(args.strategy).param_space):
        try:
            table = sensitivity(report, args.strategy, param)
        except ValueError:
            continue
        print(f"\n── {args.strategy}.{param} ──")
        print(table.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    return 0


# --------------------------------------------------------------------------------
# 파서
# --------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quant", description="자동매매 알고리즘 백테스트 · 파라미터 최적화 프레임워크"
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help="실험 설정 YAML")
        sp.add_argument("--out", default=None, help="산출물 디렉터리")
        sp.add_argument("--workers", type=int, default=None, help="병렬 프로세스 수")

    sp = sub.add_parser("list", help="등록된 전략과 파라미터 공간 보기")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("check-data", help="어떤 데이터 소스가 동작하는지 진단")
    sp.add_argument("--source", action="append", help="점검할 소스 (반복 가능)")
    sp.add_argument("--symbol", default=None, help="점검용 종목코드")
    sp.add_argument("--days", type=int, default=30, help="조회할 최근 일수")
    sp.set_defaults(func=cmd_check_data)

    sp = sub.add_parser("fetch", help="시세 수집 및 캐시")
    add_common(sp)
    sp.add_argument("--refresh", action="store_true", help="캐시 무시하고 다시 받기")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("backtest", help="단일 전략/파라미터 백테스트")
    add_common(sp)
    sp.add_argument("-s", "--strategy", required=True)
    sp.add_argument("-p", "--param", action="append", help="key=value (반복 가능)")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("optimize", help="파라미터 그리드 탐색 (동일 시점 비교)")
    add_common(sp)
    sp.add_argument("--objective", choices=sorted(OBJECTIVES), default=None)
    sp.add_argument("--top", type=int, default=20, help="콘솔/리포트에 표시할 상위 개수")
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("validate", help="IS/OOS + 워크포워드 검증")
    add_common(sp)
    sp.add_argument("--objective", choices=sorted(OBJECTIVES), default=None)
    sp.add_argument("--split", default=None, help="IS/OOS 분할 기준일 (YYYY-MM-DD)")
    sp.add_argument("--no-walkforward", action="store_true")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("sensitivity", help="파라미터 민감도 분석")
    add_common(sp)
    sp.add_argument("-s", "--strategy", required=True)
    sp.add_argument("--objective", choices=sorted(OBJECTIVES), default=None)
    sp.set_defaults(func=cmd_sensitivity)

    def add_paper_args(sp: argparse.ArgumentParser) -> None:
        add_common(sp)
        sp.add_argument("-s", "--strategy", default=None)
        sp.add_argument("-p", "--param", action="append")
        sp.add_argument("--best-file", default=None,
                        help="optimize 가 저장한 optimization_best.json")
        sp.add_argument("--state-dir", default="state", help="상태·저널 저장 위치")
        sp.add_argument("--name", default="paper", help="여러 실험을 병행할 때의 구분 이름")

    sp = sub.add_parser("paper", help="실데이터·가상자금 페이퍼 트레이딩 (실제 주문 없음)")
    add_paper_args(sp)
    sp.add_argument("--loop", action="store_true", help="주기 실행 (서버 상주)")
    sp.add_argument("--interval", type=int, default=3600, help="--loop 시 실행 간격(초)")
    sp.add_argument("--max-cycles", type=int, default=0, help="0이면 무한")
    sp.add_argument("--stop-file", default="STOP", help="이 파일이 생기면 안전 종료")
    sp.add_argument("--force", action="store_true",
                    help="이미 처리한 봉도 다시 처리 (디버깅용)")
    sp.add_argument("--no-report", action="store_true",
                    help="일일 리포트 자동 생성 끄기")
    sp.set_defaults(func=cmd_paper)

    sp = sub.add_parser("paper-status", help="페이퍼 계좌 현황")
    add_paper_args(sp)
    sp.set_defaults(func=cmd_paper_status)

    sp = sub.add_parser("paper-report", help="페이퍼 성과 + 백테스트 비교")
    add_paper_args(sp)
    sp.set_defaults(func=cmd_paper_report)

    sp = sub.add_parser("signal", help="최신 봉 기준 매매 신호 산출")
    add_common(sp)
    sp.add_argument("-s", "--strategy", default=None)
    sp.add_argument("-p", "--param", action="append")
    sp.add_argument("--best-file", default=None,
                    help="optimize 가 저장한 optimization_best.json 경로")
    sp.add_argument("--refresh", action="store_true")
    sp.set_defaults(func=cmd_signal)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n중단됨.", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as exc:
        print(f"\n오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
