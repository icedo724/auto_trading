"""페이퍼 트레이딩 테스트 — 3개월 무인 운용에서 실제로 터질 것들을 검증한다."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant.config import BacktestConfig, CostModel
from quant.live import Journal, PaperPortfolio, PaperTrader, equity_curve, live_metrics
from quant.strategy import create_strategy

PRICES = {"A": 100.0, "B": 200.0}


def cfg(**kw) -> BacktestConfig:
    base = dict(initial_cash=100_000, cost=CostModel.crypto_upbit(),
                allow_fractional=True, rebalance_threshold=0.10, trading_days=365)
    base.update(kw)
    return BacktestConfig(**base)


def new_portfolio(c: BacktestConfig) -> PaperPortfolio:
    return PaperPortfolio.create(c, "2026-01-01T00:00:00+00:00")


# ------------------------------------------------------------------- 포트폴리오
def test_execute_applies_slippage_and_fee():
    c = cfg()
    p = new_portfolio(c)
    f = p.execute("A", 1.0, 100.0, c, "t", PRICES)
    assert f is not None and f.side == "BUY"
    assert f.price == pytest.approx(100.0 * 1.0005)      # 슬리피지 5bp
    assert f.fee == pytest.approx(f.quantity * f.price * 0.0005)  # 수수료 5bp
    assert p.total_fees == pytest.approx(f.fee)


def test_execute_respects_rebalance_threshold():
    c = cfg(rebalance_threshold=0.20)
    p = new_portfolio(c)
    p.execute("A", 0.50, 100.0, c, "t", PRICES)
    n = len(p.fills)
    assert p.execute("A", 0.55, 100.0, c, "t", PRICES) is None   # 5%p 변화 -> 생략
    assert len(p.fills) == n


def test_execute_respects_min_order_value():
    c = cfg(min_order_value=50_000, rebalance_threshold=0.0)
    p = new_portfolio(c)
    assert p.execute("A", 0.01, 100.0, c, "t", PRICES) is None   # 1,000원 주문 -> 거부
    assert p.execute("A", 1.0, 100.0, c, "t", PRICES) is not None


def test_cash_never_goes_negative():
    """가상 계좌라도 없는 돈을 쓰면 안 된다."""
    c = cfg(max_weight=1.0)
    p = new_portfolio(c)
    for sym in ("A", "B"):
        p.execute(sym, 1.0, PRICES[sym], c, "t", PRICES)
    assert p.cash >= -1e-6


def test_full_exit_ignores_threshold():
    c = cfg(rebalance_threshold=0.50)
    p = new_portfolio(c)
    p.execute("A", 1.0, 100.0, c, "t", PRICES)
    f = p.execute("A", 0.0, 100.0, c, "t", PRICES)
    assert f is not None and f.side == "SELL"
    assert p.positions["A"] == pytest.approx(0.0)


def test_state_round_trip(tmp_path):
    c = cfg()
    p = new_portfolio(c)
    p.execute("A", 0.5, 100.0, c, "t", PRICES)
    p.last_bar["A"] = "2026-01-01"
    p.deposit(100_000, "t")
    path = p.save(tmp_path / "s.json")

    q = PaperPortfolio.load(path)
    assert q.cash == pytest.approx(p.cash)
    assert q.positions == p.positions
    assert q.last_bar == {"A": "2026-01-01"}
    assert q.total_deposited == 100_000
    assert len(q.fills) == 1 and q.fills[0].symbol == "A"


def test_save_is_atomic(tmp_path):
    """저장 중 죽어도 기존 상태가 깨지지 않아야 한다 (임시파일 + rename)."""
    c = cfg()
    path = tmp_path / "s.json"
    new_portfolio(c).save(path)
    before = path.read_text(encoding="utf-8")

    p = PaperPortfolio.load(path)
    p.cash = float("inf")                       # JSON 직렬화 불가로 실패 유도
    p.fills.append(object())                    # type: ignore[arg-type]
    with pytest.raises(Exception):
        p.save(path)
    assert path.read_text(encoding="utf-8") == before      # 원본 보존
    assert not list(tmp_path.glob("*.tmp"))                # 임시파일 정리


def test_load_rejects_newer_version(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"version": 999, "cash": 1.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="버전"):
        PaperPortfolio.load(path)


# ----------------------------------------------------------------------- 저널
def test_journal_appends_and_survives_truncation(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    j.write("cycle", equity=100.0)
    j.write("fill", symbol="A")
    with j.path.open("a", encoding="utf-8") as fh:
        fh.write('{"event": "cut')                 # 크래시로 잘린 줄
    events = list(j.read())
    assert [e["event"] for e in events] == ["cycle", "fill"]


# ---------------------------------------------------------------------- 러너
@pytest.fixture
def live_env(tmp_path):
    """CSV 소스로 오프라인 라이브 환경을 만든다."""
    from quant.data import SyntheticSource

    csv = tmp_path / "csv"
    csv.mkdir()
    syms = ["A", "B", "C"]
    for s in syms:
        SyntheticSource().get(s, "2024-01-01", "2025-06-30").to_csv(csv / f"{s}.csv")
    exp = {"data": {"source": "csv", "csv_dir": str(csv), "symbols": syms,
                    "start": "2024-01-01", "end": "2025-06-30", "cache_dir": None},
           "backtest": {}, "optimize": {}}
    return exp, tmp_path


def test_runner_is_idempotent(live_env):
    exp, tmp = live_env
    c = cfg()
    strat = create_strategy("buy_and_hold")
    t = PaperTrader(exp, strat, c, state_dir=tmp / "state", name="t")

    first = t.run_once()
    assert first.processed and first.fills

    again = t.run_once()
    assert again.fills == []                 # 같은 봉 -> 아무것도 하지 않는다
    assert len(again.skipped) == 3

    forced = t.run_once(force=True)
    assert forced.processed                  # --force 는 다시 처리


def test_runner_survives_restart(live_env):
    exp, tmp = live_env
    c = cfg()
    strat = create_strategy("buy_and_hold")
    t1 = PaperTrader(exp, strat, c, state_dir=tmp / "state", name="t")
    t1.run_once()
    equity_before = t1.portfolio.equity({s: 1.0 for s in "ABC"})

    t2 = PaperTrader(exp, strat, c, state_dir=tmp / "state", name="t")   # 재시작
    assert t2.portfolio.equity({s: 1.0 for s in "ABC"}) == pytest.approx(equity_before)
    assert t2.run_once().fills == []          # 이미 처리한 봉을 다시 사지 않는다


def test_capital_is_split_across_symbols(live_env):
    """백테스트는 동일비중 합성이다. 한 종목이 자본을 독식하면 안 된다."""
    exp, tmp = live_env
    c = cfg()
    t = PaperTrader(exp, create_strategy("buy_and_hold"), c,
                    state_dir=tmp / "state", name="t")
    t.run_once()
    held = {s: q for s, q in t.portfolio.positions.items() if q}
    assert len(held) == 3, f"분산 실패: {held}"

    prices = {f["symbol"]: f["price"] for f in [x.to_dict() for x in t.portfolio.fills]}
    w = t.portfolio.weights(prices)
    for sym, weight in w.items():
        assert 0.2 < weight < 0.45, f"{sym} 비중 이상: {weight}"


def test_contribution_happens_once_per_period(live_env):
    exp, tmp = live_env
    c = cfg(contribution=50_000, contribution_freq="M")
    t = PaperTrader(exp, create_strategy("buy_and_hold"), c,
                    state_dir=tmp / "state", name="t")

    assert t.run_once().deposited == 0.0          # 최초는 initial_cash 로 갈음
    period = t.portfolio.last_contribution
    assert period

    t.portfolio.last_contribution = "1999-01"     # 새 달이 온 것처럼
    assert t.run_once(force=True).deposited == 50_000
    assert t.run_once(force=True).deposited == 0.0   # 같은 달 두 번 입금 금지


def test_data_failure_does_not_crash(live_env):
    exp, tmp = live_env
    exp = {**exp, "data": {**exp["data"], "symbols": ["NOPE"]}}
    t = PaperTrader(exp, create_strategy("buy_and_hold"), cfg(),
                    state_dir=tmp / "state", name="t")
    res = t.run_once()
    assert res.errors and res.fills == []          # 예외 대신 결과로 보고


def test_no_real_orders_are_possible():
    """페이퍼 모듈은 주문 API 를 아예 갖고 있지 않다."""
    import quant.live.portfolio as pf
    import quant.live.runner as rn

    src = Path(pf.__file__).read_text() + Path(rn.__file__).read_text()
    for banned in ("order-cash", "requests.post", "api_key", "APP_SECRET"):
        assert banned not in src, f"주문 관련 코드 발견: {banned}"


# -------------------------------------------------------------------- 성과
def test_equity_curve_and_metrics_exclude_deposits(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    # 잔고가 입금으로만 늘어난 경우: 수익률은 0이어야 한다
    for i, (eq, inv) in enumerate([(100_000, 100_000), (200_000, 200_000),
                                   (300_000, 300_000)]):
        j.write("cycle", bar=f"2026-01-0{i + 1}", equity=eq, invested=inv, cash=0.0)

    curve = equity_curve(j)
    assert len(curve) == 3 and list(curve.index) == ["2026-01-01", "2026-01-02", "2026-01-03"]

    p = PaperPortfolio(cash=0.0, initial_cash=100_000, total_deposited=200_000)
    m = live_metrics(j, p, cfg())
    assert m["total_return"] == pytest.approx(0.0, abs=1e-9)
    assert m["net_profit"] == pytest.approx(0.0)
    assert m["days"] == 3


def test_equity_curve_dedupes_same_bar(tmp_path):
    """한 봉에 여러 사이클이 돌아도 곡선은 봉당 한 점이다."""
    j = Journal(tmp_path / "j.jsonl")
    for eq in (100.0, 110.0, 120.0):
        j.write("cycle", bar="2026-01-01", equity=eq, invested=100.0, cash=0.0)
    curve = equity_curve(j)
    assert len(curve) == 1 and curve["equity"].iloc[0] == 120.0   # 마지막 값
