"""일일 리포트 테스트 — 판단에 필요한 정보가 실제로 담기는지."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant.config import BacktestConfig, CostModel
from quant.data import SyntheticSource
from quant.live import Journal, PaperPortfolio, PaperTrader, health_checks, live_period
from quant.strategy import create_strategy


def cfg(**kw) -> BacktestConfig:
    base = dict(initial_cash=400_000, cost=CostModel.crypto_upbit(),
                min_order_value=5_000, rebalance_threshold=0.08, trading_days=365)
    base.update(kw)
    return BacktestConfig(**base)


@pytest.fixture
def env(tmp_path):
    """CSV 소스로 오프라인 라이브 환경을 만든다."""
    csv = tmp_path / "csv"
    csv.mkdir()
    syms = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
    for s in syms:
        SyntheticSource(annual_vol=0.6).get(s, "2024-01-01", "2025-06-30").to_csv(csv / f"{s}.csv")
    exp = {"data": {"source": "csv", "csv_dir": str(csv), "symbols": syms,
                    "start": "2024-01-01", "end": "2025-06-30", "cache_dir": None},
           "backtest": {}, "optimize": {}}
    return exp, tmp_path


def make_trader(env, **kw) -> PaperTrader:
    exp, tmp = env
    return PaperTrader(
        exp, create_strategy("sma_cross", {"fast": 10, "slow": 40}), cfg(),
        state_dir=tmp / "state", name="t", report_dir=tmp / "reports", **kw
    )


# ------------------------------------------------------ 거래하지 않은 이유 기록
def test_decisions_record_why_no_trade(env):
    t = make_trader(env)
    res = t.run_once()

    assert res.decisions, "판단이 기록되지 않았다"
    assert len(res.decisions) == 3
    for d in res.decisions:
        assert d.action in ("BUY", "SELL", "HOLD")
        assert d.reason and d.detail          # 사유가 사람이 읽는 문구로 나온다
        assert d.price > 0

    # 저널에도 체결 여부와 무관하게 남는다
    events = [e for e in t.journal.read() if e["event"] == "decision"]
    assert len(events) == 3
    assert all("reason" in e and "signal" in e for e in events)


def test_hold_reason_is_specific(env):
    """'그냥 안 샀다'가 아니라 어떤 규칙에 걸렸는지 알 수 있어야 한다."""
    exp, tmp = env
    t = PaperTrader(exp, create_strategy("buy_and_hold"),
                    cfg(min_order_value=10_000_000),      # 최소주문을 크게
                    state_dir=tmp / "state", name="t", report_dir=tmp / "reports")
    res = t.run_once()
    assert {d.reason for d in res.decisions} == {"below_min_order"}
    assert all("최소 주문금액" in d.detail for d in res.decisions)


# --------------------------------------------------------------- 리포트 생성
def test_report_is_written_automatically(env):
    t = make_trader(env)
    res = t.run_once()

    assert res.report_path is not None and res.report_path.exists()
    assert res.report_path.name == "t_latest.md"

    archive = list((t.report_dir / "daily").glob("t_*.md"))
    assert len(archive) == 1                          # 날짜별 보관본도 남는다
    assert archive[0].read_text(encoding="utf-8") == res.report_path.read_text(encoding="utf-8")


def test_report_contains_judgement_sections(env):
    t = make_trader(env)
    text = t.run_once().report_path.read_text(encoding="utf-8")

    for section in ["# 페이퍼 트레이딩 리포트", "## 오늘", "## 오늘의 판단 — 종목별",
                    "## 보유 현황", "## 실험 조건"]:
        assert section in text, section
    assert "가상 자금 · 실제 주문 없음" in text
    for sym in ("KRW-BTC", "KRW-ETH", "KRW-XRP"):
        assert sym in text


def test_report_can_be_disabled(env):
    t = make_trader(env, auto_report=False)
    res = t.run_once()
    assert res.report_path is None
    assert not (t.report_dir / "t_latest.md").exists()


def test_report_failure_does_not_break_trading(env, monkeypatch):
    """리포트가 깨져도 매매·상태저장은 계속돼야 한다."""
    import quant.live.daily as daily

    t = make_trader(env)
    monkeypatch.setattr(daily, "build_daily_report",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = t.run_once()
    assert res.report_path is None
    assert res.processed                                  # 매매는 정상 진행
    assert t.state_path.exists()
    assert any(e["event"] == "report_error" for e in t.journal.read())


# --------------------------------------------------------------- 시장 날짜
def test_live_period_uses_market_dates_not_wall_clock(tmp_path):
    """벽시계 시각을 백테스트 시작일로 쓰면 데이터 범위를 벗어나 기준선이 0이 된다."""
    j = Journal(tmp_path / "j.jsonl")
    for d in ("2026-01-05", "2026-01-06", "2026-01-07"):
        j.write("cycle", bar=d, equity=100.0, invested=100.0, cash=0.0)

    period = live_period(j)
    assert period == ("2026-01-05", "2026-01-07")
    assert not period[0].startswith(str(pd.Timestamp.now().year) + "-" + "99")


def test_live_period_none_when_too_few_cycles(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    j.write("cycle", bar="2026-01-05", equity=100.0, invested=100.0, cash=0.0)
    assert live_period(j) is None


# --------------------------------------------------------------- 신호등 판정
def _trader_stub(tmp_path, fills=()):
    class Stub:
        def __init__(self):
            self.portfolio = PaperPortfolio(cash=0.0, initial_cash=100_000)
            self.portfolio.fills = list(fills)
    return Stub()


def test_health_flags_poor_uptime(tmp_path):
    t = _trader_stub(tmp_path)
    checks = health_checks(t, {"days": 100, "missed_bars": 40, "fee_drag": 0.001}, {})
    grade = dict((k, (g, v)) for g, k, v in checks)
    assert grade["가동률"][0] == "경고"


def test_health_flags_mdd_exceeding_backtest(tmp_path):
    t = _trader_stub(tmp_path)
    live = {"days": 100, "missed_bars": 0, "fee_drag": 0.001, "cagr": 0.1,
            "max_drawdown": -0.35}
    bt = {"cagr": 0.12, "max_drawdown": -0.20}
    grade = dict((k, (g, v)) for g, k, v in health_checks(t, live, bt))
    assert grade["MDD"][0] == "경고"
    assert "초과" in grade["MDD"][1]


def test_health_flags_underperformance_and_overperformance(tmp_path):
    t = _trader_stub(tmp_path)
    base = {"days": 100, "missed_bars": 0, "fee_drag": 0.001, "max_drawdown": -0.1}
    bt = {"cagr": 0.20, "max_drawdown": -0.20}

    under = dict((k, g) for g, k, _ in health_checks(t, {**base, "cagr": 0.02}, bt))
    assert under["백테스트 괴리"] == "경고"

    over = dict((k, (g, v)) for g, k, v in health_checks(t, {**base, "cagr": 0.45}, bt))
    assert over["백테스트 괴리"][0] == "주의"      # 너무 좋은 것도 의심 대상
    assert "우수" in over["백테스트 괴리"][1]


def test_health_flags_excessive_fees_and_no_trades(tmp_path):
    t = _trader_stub(tmp_path)
    checks = health_checks(t, {"days": 60, "missed_bars": 0, "fee_drag": 0.05}, {})
    grade = dict((k, g) for g, k, _ in checks)
    assert grade["수수료"] == "경고"
    assert grade["거래"] == "경고"          # 60일간 체결 0건


def test_overall_verdict_escalates(tmp_path):
    from quant.live.daily import overall_verdict

    assert "정상" in overall_verdict([("OK", "a", ""), ("OK", "b", "")])
    assert "주의" in overall_verdict([("주의", "a", ""), ("주의", "b", "")])
    assert "확인 필요" in overall_verdict([("OK", "a", ""), ("경고", "b", "")])
