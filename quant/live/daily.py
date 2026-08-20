"""일일 리포트 생성 — **읽고 판단하기 위한** 문서.

설계 원칙: 사람이 매일 열어보고 "계속할까 / 멈출까 / 고칠까"를 결정할 수 있어야 한다.
그래서 숫자 나열이 아니라 **판정과 근거**를 먼저 놓는다.

  1. 신호등     지금 상태가 정상인가 (규칙 기반 자동 판정)
  2. 오늘 요약   무슨 일이 있었나
  3. 판단 근거   종목별로 왜 그렇게 했는가 — **거래 안 한 이유 포함**
  4. 성과       불확실성(신뢰구간)까지
  5. 대조군     백테스트 대비 · 벤치마크 대비
  6. 기록       보유·체결 이력

마크다운이라 GitHub 에서 바로 읽힌다(Actions 로 돌리면 커밋되어 폰에서도 보인다).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from functools import partial

from ..bootstrap import bootstrap_ci, sharpe_of
from .report import equity_curve, live_metrics, live_period

SPARK = "▁▂▃▄▅▆▇█"


def _spark(series: pd.Series, width: int = 50) -> str:
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return ""
    idx = np.linspace(0, len(s) - 1, min(width, len(s))).astype(int)
    v = s.iloc[idx].to_numpy(dtype=float)
    lo, hi = v.min(), v.max()
    if hi <= lo:
        return SPARK[0] * len(v)
    return "".join(SPARK[i] for i in ((v - lo) / (hi - lo) * 7).round().astype(int))


def _table(rows: list[dict[str, Any]], headers: list[str] | None = None) -> str:
    if not rows:
        return "_(없음)_"
    cols = headers or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------------
# 신호등 — 자동 판정
# --------------------------------------------------------------------------------
def health_checks(
    trader, live: dict[str, float], bt: dict[str, float]
) -> list[tuple[str, str, str]]:
    """(등급, 항목, 설명) 목록. 등급은 OK / 주의 / 경고."""
    checks: list[tuple[str, str, str]] = []
    days = live.get("days", 0)
    pf = trader.portfolio
    risk = trader.config.risk

    # 0) 손실 한도 — 발동했으면 다른 무엇보다 먼저 봐야 한다
    if risk.enabled:
        if pf.halted:
            checks.append(("경고", "손실 한도", "**발동 — 영구 정지 중.** 사람이 판단해야 재개"))
        elif pf.halt_until:
            checks.append(("경고", "손실 한도", f"발동 — {pf.halt_until} 까지 정지"))
        else:
            dd = pf.nav / pf.peak_nav - 1.0 if pf.peak_nav > 0 else 0.0
            room = risk.max_drawdown + dd if risk.max_drawdown else None
            note = f"현재 고점대비 {dd:.1%}"
            if room is not None:
                note += f" · 한도까지 {room:.1%} 남음"
            grade = "주의" if (room is not None and room < 0.05) else "OK"
            checks.append((grade, "손실 한도", note))
    else:
        checks.append(("주의", "손실 한도", "**설정 없음** — 계좌 전체 손실 바닥이 없다"))

    # 1) 가동률
    missed = live.get("missed_bars", 0)
    ratio = missed / max(days + missed, 1)
    if missed == 0:
        checks.append(("OK", "가동률", "놓친 봉 없음"))
    elif ratio <= 0.10:
        checks.append(("주의", "가동률", f"놓친 봉 {missed:.0f}개 ({ratio:.0%})"))
    else:
        checks.append(
            ("경고", "가동률", f"놓친 봉 {missed:.0f}개 ({ratio:.0%}) — 비교 신뢰 불가")
        )

    # 2) 표본 충분성
    if days < 30:
        checks.append(("주의", "표본", f"{days:.0f}일 — 30일 미만은 판단 불가"))
    elif days < 60:
        checks.append(("주의", "표본", f"{days:.0f}일 — 아직 짧다"))
    else:
        checks.append(("OK", "표본", f"{days:.0f}일"))

    # 3) 백테스트 대비 괴리
    if bt and days >= 30:
        gap = live.get("cagr", 0.0) - bt.get("cagr", 0.0)
        if gap < -0.10:
            checks.append(
                ("경고", "백테스트 괴리", f"라이브가 {abs(gap):.1%}p 부진 — 원인 규명 필요")
            )
        elif gap < -0.05:
            checks.append(("주의", "백테스트 괴리", f"라이브가 {abs(gap):.1%}p 부진"))
        elif gap > 0.10:
            # 좋은 것도 의심 대상이다. 3개월은 짧고, 설정 차이나 버그일 수 있다
            checks.append(
                ("주의", "백테스트 괴리", f"라이브가 {gap:.1%}p 우수 — 운이거나 설정 차이일 수 있음")
            )
        else:
            checks.append(("OK", "백테스트 괴리", f"{gap:+.1%}p"))

    # 4) MDD 가 백테스트 가정을 넘었는가 — 가장 중요한 위험 신호
    if bt:
        live_mdd, bt_mdd = live.get("max_drawdown", 0.0), bt.get("max_drawdown", 0.0)
        if bt_mdd < 0 and live_mdd < bt_mdd:
            checks.append(
                ("경고", "MDD", f"{live_mdd:.1%} — 백테스트 {bt_mdd:.1%} 를 초과")
            )
        else:
            checks.append(("OK", "MDD", f"{live_mdd:.1%} (백테스트 {bt_mdd:.1%})"))

    # 5) 비용 부담
    drag = live.get("fee_drag", 0.0)
    if drag > 0.02:
        checks.append(("경고", "수수료", f"원금의 {drag:.2%} — 거래가 과하다"))
    elif drag > 0.01:
        checks.append(("주의", "수수료", f"원금의 {drag:.2%}"))
    else:
        checks.append(("OK", "수수료", f"원금의 {drag:.2%}"))

    # 6) 임계치가 배분 단위보다 큰가 — 전략의 일부가 구조적으로 죽는 설정
    n_sym = len(trader.exp.get("data", {}).get("symbols", []) or [])
    if n_sym:
        alloc = 1.0 / n_sym
        thr = trader.config.rebalance_threshold
        if thr >= alloc:
            checks.append((
                "경고", "임계치 설정",
                f"리밸런싱 임계치 {thr:.0%} ≥ 종목당 배분 {alloc:.0%}"
                " — 신호가 최대여도 거래되지 않는다",
            ))
        elif thr > alloc / 2:
            checks.append((
                "주의", "임계치 설정",
                f"임계치 {thr:.0%} 가 종목당 배분 {alloc:.0%} 의 절반을 넘는다"
                " — 약한 신호는 영원히 체결되지 않는다",
            ))
        else:
            checks.append(("OK", "임계치 설정", f"임계치 {thr:.0%} < 배분 {alloc:.0%}"))

    # 7) 장기 무거래 — 전략이 죽었거나 설정이 잘못됐을 수 있다
    fills = trader.portfolio.fills
    if days >= 30 and not fills:
        checks.append(("경고", "거래", "30일 넘게 체결이 하나도 없다 — 설정 확인"))
    elif fills:
        last = pd.Timestamp(fills[-1].timestamp)
        idle = (pd.Timestamp.now(tz="UTC") - last).days
        if idle > 45:
            checks.append(("주의", "거래", f"{idle}일째 무거래"))
        else:
            checks.append(("OK", "거래", f"총 {len(fills)}건 · 최근 {idle}일 전"))

    return checks


def overall_verdict(checks: list[tuple[str, str, str]]) -> str:
    grades = [g for g, _, _ in checks]
    if "경고" in grades:
        return "🔴 **확인 필요** — 아래 경고 항목을 먼저 볼 것"
    if grades.count("주의") >= 2:
        return "🟡 **주의** — 아직 판단하기 이르거나 살펴볼 항목이 있음"
    return "🟢 **정상** — 계획대로 진행 중"


# --------------------------------------------------------------------------------
# 리포트 본문
# --------------------------------------------------------------------------------
def build_daily_report(trader, data, prices: dict[str, float], cycle=None) -> str:
    """트레이더 상태로 마크다운 일일 리포트를 만든다."""
    from .report import backtest_reference, format_status

    pf = trader.portfolio
    cfg = trader.config
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")

    live = live_metrics(trader.journal, pf, cfg)
    curve = equity_curve(trader.journal)

    bt: dict[str, float] = {}
    period = live_period(trader.journal)
    if live and period:
        try:
            bt = backtest_reference(trader.exp, trader.strategy, cfg, *period)
        except Exception:  # noqa: BLE001 - 대조군 실패해도 리포트는 나와야 한다
            bt = {}

    equity = pf.equity(prices)
    invested = pf.total_invested
    pnl = equity - invested

    L: list[str] = [
        f"# 페이퍼 트레이딩 리포트 · {trader.name}",
        "",
        f"_{now} 생성 · 가상 자금 · 실제 주문 없음_",
        "",
    ]

    # 정지 상태면 최상단에 크게 알린다
    if pf.halted or pf.halt_until:
        last = pf.halt_events[-1] if pf.halt_events else {}
        L += [
            "> 🛑 **매매 정지 중**",
            ">",
            f"> 사유: `{last.get('reason', '?')}` "
            f"{last.get('value', 0):.1%} ({last.get('date', '?')})",
            ">",
            "> " + (
                "**영구 정지.** 설정을 재검토하고 직접 재개해야 한다."
                if pf.halted else f"{pf.halt_until} 까지 자동 정지."
            ),
            "",
        ]

    # ---------------------------------------------------------------- 1. 신호등
    if live:
        checks = health_checks(trader, live, bt)
        L += ["## 판정", "", overall_verdict(checks), ""]
        icon = {"OK": "🟢", "주의": "🟡", "경고": "🔴"}
        L += [_table(
            [{"": icon[g], "항목": k, "상태": v} for g, k, v in checks],
            ["", "항목", "상태"],
        ), ""]

    # ------------------------------------------------------------- 2. 오늘 요약
    L += ["## 오늘", ""]
    if cycle is not None:
        traded = [d for d in cycle.decisions if d.traded]
        L += [
            f"- 처리 {len(cycle.processed)}종목 · 건너뜀 {len(cycle.skipped)}"
            f" · **체결 {len(traded)}건**",
        ]
        if cycle.deposited:
            L.append(f"- 적립 입금 **{cycle.deposited:,.0f}원**")
        if cycle.missed_bars:
            L.append(f"- ⚠️ 놓친 봉 {cycle.missed_bars}개 (서버 정지 구간)")
        for e in cycle.errors:
            L.append(f"- 🔴 오류: {e}")
    L += [
        f"- 평가금액 **{equity:,.0f}원** · 투입원금 {invested:,.0f}원"
        f" · 손익 **{pnl:+,.0f}원 ({pnl / invested:+.2%})**" if invested > 0 else "",
        "",
    ]

    # ---------------------------------------------------- 3. 판단 근거 (핵심)
    if cycle is not None and cycle.decisions:
        L += [
            "## 오늘의 판단 — 종목별",
            "",
            "> 거래하지 **않은** 이유까지 남긴다. \"왜 아무것도 안 했지?\"에 답하기 위함.",
            "",
        ]
        rows = []
        for d in sorted(cycle.decisions, key=lambda x: (not x.traded, x.symbol)):
            rows.append({
                "종목": d.symbol,
                "가격": f"{d.price:,.2f}",
                "신호": f"{d.signal:.2f}",
                "목표비중": f"{d.target_weight:.1%}",
                "현재비중": f"{d.current_weight:.1%}",
                "행동": ("**" + d.action + "**") if d.traded else d.action,
                "사유": d.detail,
            })
        L += [_table(rows), ""]

    # ------------------------------------------------------------------ 4. 성과
    if live:
        L += ["## 성과", ""]
        rows = [
            {"지표": "운영일수", "값": f"{live['days']:.0f}일"},
            {"지표": "총수익률 (TWR)", "값": f"{live['total_return']:+.2%}"},
            {"지표": "연환산 (CAGR)", "값": f"{live['cagr']:+.2%}"},
            {"지표": "자금가중 (MWR)", "값": f"{live['mwr']:+.2%}"},
            {"지표": "Sharpe", "값": f"{live['sharpe']:.2f}"},
            {"지표": "MDD", "값": f"{live['max_drawdown']:.2%}"},
            {"지표": "연변동성", "값": f"{live['ann_volatility']:.2%}"},
            {"지표": "체결 건수", "값": f"{live['n_fills']:.0f}건"},
            {"지표": "누적 수수료", "값": f"{live['total_fees']:,.0f}원 ({live['fee_drag']:.2%})"},
        ]
        L += [_table(rows, ["지표", "값"]), ""]

        # 불확실성 — 점추정치만 보면 안 된다
        if len(curve) >= 30:
            try:
                rets = curve["equity"].pct_change().dropna()
                # 성과표와 같은 연율화 기준을 써야 두 숫자가 일치한다
                ci = bootstrap_ci(
                    rets, partial(sharpe_of, trading_days=cfg.trading_days), n_boot=500
                )
                verdict = (
                    "0을 제외 — 통계적으로 유의"
                    if ci.excludes_zero
                    else "**0을 포함 — 아직 운과 구분할 수 없다**"
                )
                L += [
                    "### 불확실성 (블록 부트스트랩)",
                    "",
                    f"- Sharpe **{ci.point:.2f}**, 90% 신뢰구간 "
                    f"**[{ci.lower:.2f}, {ci.upper:.2f}]**",
                    f"- 참 Sharpe > 0 일 확률 **{ci.prob_positive:.1%}** — {verdict}",
                    "",
                ]
            except ValueError:
                pass

        if len(curve) >= 2:
            L += [
                "### 자산곡선",
                "",
                "```",
                _spark(curve["equity"]),
                f"{curve.index[0]}  →  {curve.index[-1]}",
                f"{curve['equity'].iloc[0]:,.0f}원  →  {curve['equity'].iloc[-1]:,.0f}원",
                "```",
                "",
            ]

    # --------------------------------------------------------------- 5. 대조군
    if live and bt:
        L += [
            "## 백테스트 대비",
            "",
            f"> 같은 기간({period[0]} ~ {period[1]})·같은 전략을 백테스트로 돌린 값과 비교.",
            "",
        ]
        rows = []
        for key, label, pct in [
            ("total_return", "총수익률", True), ("cagr", "CAGR", True),
            ("sharpe", "Sharpe", False), ("max_drawdown", "MDD", True),
            ("ann_volatility", "연변동성", True),
        ]:
            lv, bv = live.get(key), bt.get(key)
            if lv is None or bv is None:
                continue
            f = (lambda v: f"{v:+.2%}") if pct else (lambda v: f"{v:.2f}")
            rows.append({"지표": label, "라이브": f(lv), "백테스트": f(bv), "차이": f(lv - bv)})
        L += [_table(rows, ["지표", "라이브", "백테스트", "차이"]), ""]

    # ------------------------------------------------------- 5.5 손실 한도
    if cfg.risk.enabled:
        dd = pf.nav / pf.peak_nav - 1.0 if pf.peak_nav > 0 else 0.0
        L += [
            "## 손실 한도 (서킷브레이커)",
            "",
            _table([
                {"항목": "설정", "값": cfg.risk.describe()},
                {"항목": "현재 고점 대비", "값": f"{dd:.2%}"},
                {"항목": "상태", "값": (
                    "🛑 영구 정지" if pf.halted else
                    f"🛑 {pf.halt_until} 까지 정지" if pf.halt_until else "🟢 정상 가동"
                )},
                {"항목": "누적 발동", "값": f"{len(pf.halt_events)}회"},
            ], ["항목", "값"]),
            "",
        ]
        if pf.halt_events:
            L += ["### 발동 이력", "", _table([
                {"일자": e["date"], "사유": e["reason"],
                 "값": f"{e['value']:.2%}", "평가금액": f"{e['equity']:,.0f}",
                 "조치": e["action"]}
                for e in pf.halt_events[-10:]
            ]), ""]

    # ------------------------------------------------------------ 6. 보유·체결
    L += ["## 보유 현황", "", "```", format_status(pf, prices), "```", ""]

    if pf.fills:
        L += ["## 최근 체결 (최대 15건)", ""]
        rows = []
        for f in pf.fills[-15:][::-1]:
            rows.append({
                "일시": f.timestamp[:16].replace("T", " "),
                "종목": f.symbol,
                "구분": f.side,
                "수량": f"{f.quantity:.8f}".rstrip("0").rstrip("."),
                "체결가": f"{f.price:,.2f}",
                "금액": f"{f.notional:,.0f}",
                "수수료": f"{f.fee:,.0f}",
            })
        L += [_table(rows), ""]

    # -------------------------------------------------------------- 7. 실험 조건
    L += [
        "## 실험 조건",
        "",
        _table([
            {"항목": "전략", "값": f"`{trader.strategy.describe()}`"},
            {"항목": "종목", "값": ", ".join(sorted(prices))},
            {"항목": "초기자본", "값": f"{cfg.initial_cash:,.0f}원"},
            {"항목": "적립", "값": (f"{cfg.contribution:,.0f}원 / {cfg.contribution_freq}"
                                  if cfg.contribution > 0 else "없음")},
            {"항목": "비용", "값": f"수수료 {cfg.cost.commission_bps}bp · "
                                f"거래세 {cfg.cost.sell_tax_bps}bp · "
                                f"슬리피지 {cfg.cost.slippage_bps}bp"},
            {"항목": "리밸런싱 임계치", "값": f"{cfg.rebalance_threshold:.0%}"},
            {"항목": "손실 한도", "값": cfg.risk.describe()},
            {"항목": "최소 주문", "값": f"{cfg.min_order_value:,.0f}원"},
            {"항목": "시작일", "값": pf.created_at[:10]},
        ], ["항목", "값"]),
        "",
        "---",
        "",
        "_이 리포트는 매 사이클 자동 생성된다. "
        "판단 기준은 [`docs/PAPER_TRADING.md`](../../docs/PAPER_TRADING.md) 참조._",
    ]

    return "\n".join(x for x in L if x is not None)
