"""백테스트 실행 환경 설정 (거래비용 · 체결 규칙 · 리스크 관리)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BPS = 1e-4

ExecutionMode = Literal["next_open", "next_close"]


@dataclass(frozen=True)
class CostModel:
    """편도(one-way) 거래비용 모델. 단위는 bps(0.01%)."""

    commission_bps: float = 1.5  # 매수/매도 각각 부과되는 위탁수수료
    sell_tax_bps: float = 0.0  # 매도 시에만 부과되는 증권거래세
    slippage_bps: float = 5.0  # 체결 가격 불리하게 밀리는 정도

    @classmethod
    def kr_stock(cls) -> "CostModel":
        """국내 주식 기본값: 수수료 0.015%, 거래세 0.18%(매도), 슬리피지 0.05%."""
        return cls(commission_bps=1.5, sell_tax_bps=18.0, slippage_bps=5.0)

    @classmethod
    def us_stock(cls) -> "CostModel":
        """미국 주식 기본값: 수수료 0.01%, 거래세 없음, 슬리피지 0.03%."""
        return cls(commission_bps=1.0, sell_tax_bps=0.0, slippage_bps=3.0)

    @classmethod
    def crypto_upbit(cls) -> "CostModel":
        """업비트 원화마켓: 수수료 0.05%/편도, 거래세 없음, 슬리피지 0.05%.

        왕복 약 20bp 로 국내주식(31bp)의 2/3 수준. 소수점 매매가 되므로
        소액 계좌에서 분산이 가능한 것이 더 큰 장점이다.
        수수료율은 거래소·이벤트에 따라 바뀌므로 실제 값을 확인할 것.
        """
        return cls(commission_bps=5.0, sell_tax_bps=0.0, slippage_bps=5.0)

    @classmethod
    def crypto_binance(cls) -> "CostModel":
        """바이낸스 현물: 수수료 0.1%/편도(BNB 할인 미적용), 슬리피지 0.03%."""
        return cls(commission_bps=10.0, sell_tax_bps=0.0, slippage_bps=3.0)

    @classmethod
    def zero(cls) -> "CostModel":
        """비용 0 (전략 로직 검증용)."""
        return cls(commission_bps=0.0, sell_tax_bps=0.0, slippage_bps=0.0)

    @classmethod
    def named(cls, name: str) -> "CostModel":
        table = {
            "kr": cls.kr_stock,
            "us": cls.us_stock,
            "zero": cls.zero,
            "upbit": cls.crypto_upbit,
            "crypto": cls.crypto_upbit,
            "binance": cls.crypto_binance,
        }
        if name not in table:
            raise ValueError(f"알 수 없는 비용 모델: {name!r} (가능: {sorted(table)})")
        return table[name]()

    def buy_cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) * BPS

    def sell_cost_rate(self) -> float:
        return (self.commission_bps + self.sell_tax_bps + self.slippage_bps) * BPS


@dataclass(frozen=True)
class RiskLimits:
    """계좌 전체 손실 한도 — **매몰되지 않기 위한 바닥**.

    종목별 손절(``stop_loss_pct``)만으로는 계좌를 지키지 못한다.
    10종목이 각각 9%씩 빠지면(손절선 10% 미달) 계좌는 -9%인데 아무 손절도
    발동하지 않는다. 반복 손절로 조금씩 녹는 경우도 마찬가지다.
    그래서 **포트폴리오 단위 서킷브레이커**가 따로 필요하다.

    한도에 닿으면 전량 청산하고 매매를 멈춘다.

    ⚠️ 공짜가 아니다. 드로다운 한도는 **회복했을 손실을 확정 손실로 바꾼다.**
       추세추종 전략에 하드 스톱을 걸면 대개 총수익이 줄어든다.
       이것은 수익을 늘리는 장치가 아니라 **최악을 제한하는 장치**다.
       그 대가를 알고 쓰라 (`scripts/risk_limit_experiment.py` 로 직접 확인).
    """

    #: 고점 대비 손실 한도. 0.20 = 고점에서 -20% 되면 정지. 0이면 미사용.
    #: 적립 입금이 하락을 가리지 않도록 **시간가중(TWR) 기준**으로 잰다.
    max_drawdown: float = 0.0
    #: 투입원금 대비 누적손실 한도. 0.15 = 원금의 -15% 되면 정지
    max_loss: float = 0.0
    #: 하루 손실 한도. 0.10 = 하루 -10% 되면 정지 (플래시 크래시 대응)
    daily_loss: float = 0.0

    #: halt = 영구 정지(사람이 판단할 때까지) / cooldown = 일정 기간 쉬었다 재개
    action: str = "halt"
    cooldown_days: int = 20

    @property
    def enabled(self) -> bool:
        return self.max_drawdown > 0 or self.max_loss > 0 or self.daily_loss > 0

    def validate(self) -> None:
        for name in ("max_drawdown", "max_loss", "daily_loss"):
            v = getattr(self, name)
            if not 0 <= v < 1:
                raise ValueError(f"{name} 는 [0, 1) 범위여야 합니다: {v}")
        if self.action not in ("halt", "cooldown"):
            raise ValueError("action 은 halt 또는 cooldown 이어야 합니다.")
        if self.cooldown_days < 0:
            raise ValueError("cooldown_days 는 0 이상이어야 합니다.")

    def describe(self) -> str:
        if not self.enabled:
            return "없음"
        parts = []
        if self.max_drawdown:
            parts.append(f"고점대비 -{self.max_drawdown:.0%}")
        if self.max_loss:
            parts.append(f"원금대비 -{self.max_loss:.0%}")
        if self.daily_loss:
            parts.append(f"일일 -{self.daily_loss:.0%}")
        tail = "영구정지" if self.action == "halt" else f"{self.cooldown_days}일 정지 후 재개"
        return " / ".join(parts) + f" → {tail}"


@dataclass(frozen=True)
class BacktestConfig:
    """백테스트 엔진 설정.

    모든 전략/파라미터 조합은 **동일한 BacktestConfig + 동일한 기간**으로 평가되어야
    비교가 공정하다. optimizer 는 이 규칙을 강제한다.
    """

    initial_cash: float = 10_000_000.0
    cost: CostModel = field(default_factory=CostModel.kr_stock)

    # 체결 규칙: t일 종가까지의 정보로 만든 신호를 t+1일 시가(또는 종가)에 체결
    execution: ExecutionMode = "next_open"
    signal_lag: int = 1  # 룩어헤드 방지용 시그널 지연(일). 1 미만이면 미래정보 사용.

    # 포지션
    allow_short: bool = False
    max_weight: float = 1.0  # 종목당 최대 절대 비중 (1.0 = 100%)
    rebalance_threshold: float = 0.05  # 목표비중 변화가 이보다 작으면 거래 생략
    allow_fractional: bool = True  # False면 정수 주 단위로만 매매

    # 리스크 관리 (0 또는 None이면 미사용)
    stop_loss_pct: float = 0.0  # 진입가 대비 손절 (예: 0.08 = -8%)
    take_profit_pct: float = 0.0  # 진입가 대비 익절
    trailing_stop_pct: float = 0.0  # 고점 대비 추적 손절
    max_holding_days: int = 0  # 최대 보유 영업일

    #: 계좌 전체 손실 한도 (서킷브레이커)
    risk: RiskLimits = field(default_factory=RiskLimits)

    # 적립식 (월 N만원씩 입금하는 운용)
    contribution: float = 0.0  # 1회 입금액 (0이면 적립 없음)
    contribution_freq: str = "M"  # M=월 / W=주 / Q=분기
    min_order_value: float = 0.0  # 최소 주문금액 (업비트 5000원 등). 미만이면 주문 생략

    # 성과 지표
    trading_days: int = 252
    risk_free_rate: float = 0.0  # 연율 (예: 0.03 = 3%)

    def validate(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash 는 0보다 커야 합니다.")
        if self.signal_lag < 1:
            raise ValueError(
                "signal_lag 는 1 이상이어야 합니다. 0이면 당일 종가 정보로 "
                "당일 체결하는 룩어헤드 바이어스가 발생합니다."
            )
        if self.max_weight <= 0:
            raise ValueError("max_weight 는 0보다 커야 합니다.")
        if not 0 <= self.rebalance_threshold < 1:
            raise ValueError("rebalance_threshold 는 [0, 1) 범위여야 합니다.")
        if self.contribution < 0:
            raise ValueError("contribution 은 0 이상이어야 합니다.")
        if self.contribution_freq not in ("M", "W", "Q"):
            raise ValueError("contribution_freq 는 M / W / Q 중 하나여야 합니다.")
        if self.min_order_value < 0:
            raise ValueError("min_order_value 는 0 이상이어야 합니다.")
        if self.trading_days <= 0:
            raise ValueError("trading_days 는 0보다 커야 합니다.")
        self.risk.validate()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost"] = asdict(self.cost)
        d["risk"] = asdict(self.risk)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BacktestConfig":
        d = dict(d)
        risk = d.pop("risk", None)
        risk_model = RiskLimits(**risk) if isinstance(risk, dict) else RiskLimits()
        cost = d.pop("cost", None)
        if isinstance(cost, str):
            cost_model = CostModel.named(cost)
        elif isinstance(cost, dict):
            cost_model = CostModel(**cost)
        else:
            cost_model = CostModel.kr_stock()
        known = {f for f in cls.__dataclass_fields__ if f not in ("cost", "risk")}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"알 수 없는 백테스트 설정 항목: {sorted(unknown)}")
        return cls(cost=cost_model, risk=risk_model, **d)
