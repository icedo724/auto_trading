"""계좌 전체 손실 한도(서킷브레이커) 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.config import BacktestConfig, CostModel, RiskLimits
from quant.engine import Backtester

N = 400
IDX = pd.bdate_range("2024-01-01", periods=N)


def frame(prices) -> pd.DataFrame:
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {"open": p, "high": p, "low": p, "close": p, "volume": np.full(N, 1e3)}, index=IDX
    )


def run(prices, risk: RiskLimits, **kw):
    cfg = BacktestConfig(initial_cash=1_000_000, cost=CostModel.zero(), risk=risk, **kw)
    return Backtester(cfg).run(frame(prices), pd.Series(1.0, index=IDX), symbol="T")


DECLINE_THEN_RECOVER = np.concatenate([np.linspace(100, 60, 200), np.linspace(60, 120, 200)])


# ----------------------------------------------------------------- 설정 검증
def test_limits_disabled_by_default():
    assert not RiskLimits().enabled
    assert RiskLimits().describe() == "없음"
    assert RiskLimits(max_drawdown=0.2).enabled


def test_invalid_limits_rejected():
    for bad in ({"max_drawdown": 1.5}, {"max_loss": -0.1}, {"daily_loss": 1.0},
                {"action": "nope"}, {"cooldown_days": -1}):
        with pytest.raises(ValueError):
            RiskLimits(**bad).validate()


def test_config_round_trip_preserves_risk():
    r = RiskLimits(max_drawdown=0.2, max_loss=0.15, action="cooldown", cooldown_days=30)
    c = BacktestConfig(risk=r)
    assert BacktestConfig.from_dict(c.to_dict()).risk == r


# --------------------------------------------------------------- 발동 동작
def test_no_limit_means_no_halt():
    res = run(DECLINE_THEN_RECOVER, RiskLimits())
    assert res.halt_events == []
    assert res.metrics["max_drawdown"] < -0.35        # 그대로 다 맞는다


def test_drawdown_limit_caps_loss_but_forfeits_recovery():
    """한도는 손실을 막지만 **회복도 포기시킨다** — 공짜가 아니다."""
    free = run(DECLINE_THEN_RECOVER, RiskLimits())
    capped = run(DECLINE_THEN_RECOVER, RiskLimits(max_drawdown=0.20))

    assert len(capped.halt_events) == 1
    assert capped.halt_events[0]["reason"] == "max_drawdown"
    assert capped.metrics["max_drawdown"] > -0.25      # 손실은 한도 근처에서 멈췄다
    assert capped.metrics["total_return"] < free.metrics["total_return"]  # 회복은 놓쳤다


def test_halt_is_permanent():
    res = run(DECLINE_THEN_RECOVER, RiskLimits(max_drawdown=0.20, action="halt"))
    assert len(res.halt_events) == 1                  # 한 번 걸리면 끝
    assert res.position.iloc[-1] == pytest.approx(0.0)  # 끝까지 현금


def test_cooldown_resumes_trading():
    res = run(DECLINE_THEN_RECOVER,
              RiskLimits(max_drawdown=0.20, action="cooldown", cooldown_days=20))
    assert len(res.halt_events) >= 1
    assert res.position.iloc[-1] > 0                  # 재개해서 회복 구간에 올라탔다
    assert res.metrics["total_return"] > run(
        DECLINE_THEN_RECOVER, RiskLimits(max_drawdown=0.20, action="halt")
    ).metrics["total_return"]


def test_max_loss_is_the_real_floor():
    """원금 대비 한도는 고점 리셋의 영향을 받지 않는 절대 바닥이다."""
    res = run(np.linspace(100, 40, N), RiskLimits(max_loss=0.20))
    assert res.halt_events and res.halt_events[0]["reason"] == "max_loss"
    # 원금 대비 손실이 한도 근처에서 멈춘다 (다음 봉 시가 체결이라 약간의 초과는 정상)
    assert res.metrics["total_return"] > -0.30


def test_daily_loss_limit_catches_crash():
    px = np.full(N, 100.0)
    px[200:] = 70.0                                    # 하루 만에 -30%
    res = run(px, RiskLimits(daily_loss=0.15))
    assert res.halt_events and res.halt_events[0]["reason"] == "daily_loss"


def test_halt_event_records_context():
    res = run(DECLINE_THEN_RECOVER, RiskLimits(max_drawdown=0.20))
    e = res.halt_events[0]
    assert set(e) == {"date", "reason", "value", "equity", "action"}
    assert e["value"] <= -0.20 and e["equity"] > 0
    pd.Timestamp(e["date"])                            # 파싱 가능한 날짜


def test_n_halts_metric_only_when_enabled():
    assert "n_halts" not in run(DECLINE_THEN_RECOVER, RiskLimits()).metrics
    assert run(DECLINE_THEN_RECOVER, RiskLimits(max_drawdown=0.20)).metrics["n_halts"] == 1


def test_deposits_do_not_mask_drawdown_for_limit():
    """적립 입금으로 잔고가 유지돼도 한도는 발동해야 한다 (TWR 기준)."""
    res = run(np.linspace(100, 55, N), RiskLimits(max_drawdown=0.20),
              contribution=200_000, contribution_freq="M")
    assert res.halt_events, "입금이 드로다운을 가려 한도가 발동하지 않았다"


# ------------------------------------------------------- 페이퍼 트레이딩 연동
def test_paper_portfolio_halts_and_persists(tmp_path):
    from quant.live import PaperPortfolio

    cfg = BacktestConfig(initial_cash=100_000, cost=CostModel.zero(),
                         risk=RiskLimits(max_drawdown=0.20))
    p = PaperPortfolio.create(cfg, "2026-01-01T00:00:00+00:00")

    assert p.check_risk_limits(cfg, 100_000, 0.0, "2026-01-01") is None
    assert p.is_trading_halted("2026-01-02") == (False, "")

    event = p.check_risk_limits(cfg, 75_000, 0.0, "2026-01-02")   # -25%
    assert event and event["reason"] == "max_drawdown"
    halted, why = p.is_trading_halted("2026-01-03")
    assert halted and "영구" in why

    q = PaperPortfolio.load(p.save(tmp_path / "s.json"))          # 재시작해도 유지
    assert q.halted and len(q.halt_events) == 1


def test_paper_cooldown_expires(tmp_path):
    from quant.live import PaperPortfolio

    cfg = BacktestConfig(initial_cash=100_000, cost=CostModel.zero(),
                         risk=RiskLimits(max_drawdown=0.20, action="cooldown",
                                         cooldown_days=10))
    p = PaperPortfolio.create(cfg, "2026-01-01T00:00:00+00:00")
    p.check_risk_limits(cfg, 100_000, 0.0, "2026-01-01")
    p.check_risk_limits(cfg, 75_000, 0.0, "2026-01-02")

    assert p.is_trading_halted("2026-01-05")[0] is True
    assert p.is_trading_halted("2026-01-20")[0] is False     # 쿨다운 만료
    assert p.peak_nav == pytest.approx(p.nav)                # 고점 리셋
