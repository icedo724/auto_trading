"""가상 포트폴리오 — 페이퍼 트레이딩용 계좌.

백테스트 엔진과 **똑같은 체결·비용 규칙**을 쓴다(docs/ALGORITHM.md §3.2~3.4).
그래야 3개월 뒤 "실전 성과 vs 백테스트 예상"을 비교하는 것이 의미를 갖는다.

상태는 JSON 으로 디스크에 저장되며, 서버 재시작·크래시 후에도 이어진다.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import BPS, BacktestConfig

STATE_VERSION = 1


@dataclass
class Fill:
    """가상 체결 1건."""

    timestamp: str
    symbol: str
    side: str  # BUY / SELL
    quantity: float
    price: float  # 슬리피지 반영 체결가
    fee: float
    notional: float  # quantity * price
    reason: str = "signal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperPortfolio:
    """가상 계좌. 현금 + 종목별 수량."""

    cash: float
    positions: dict[str, float] = field(default_factory=dict)  # symbol -> quantity
    avg_price: dict[str, float] = field(default_factory=dict)  # 비용 포함 평균 단가
    total_deposited: float = 0.0  # 적립 입금 누적 (초기자본 제외)
    initial_cash: float = 0.0
    total_fees: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    #: 종목별 마지막으로 처리한 봉 날짜 — 중복 실행 방지의 핵심
    last_bar: dict[str, str] = field(default_factory=dict)
    #: 마지막 적립 입금 기간 ("2026-01"). 같은 달에 두 번 넣지 않기 위함
    last_contribution: str = ""
    created_at: str = ""
    updated_at: str = ""

    # ------------------------------------------------------------------ 평가
    def position_value(self, prices: dict[str, float]) -> float:
        return sum(
            qty * prices[sym]
            for sym, qty in self.positions.items()
            if qty and sym in prices
        )

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.position_value(prices)

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {s: 0.0 for s in self.positions}
        return {
            sym: (qty * prices[sym] / eq) if sym in prices else 0.0
            for sym, qty in self.positions.items()
        }

    @property
    def total_invested(self) -> float:
        return self.initial_cash + self.total_deposited

    def deposit(self, amount: float, timestamp: str) -> None:
        if amount <= 0:
            return
        self.cash += amount
        self.total_deposited += amount
        self.updated_at = timestamp

    # ------------------------------------------------------------------ 체결
    def execute(
        self,
        symbol: str,
        target_weight: float,
        price: float,
        config: BacktestConfig,
        timestamp: str,
        prices: dict[str, float],
        *,
        reason: str = "signal",
    ) -> Fill | None:
        """목표 비중에 맞춰 가상 체결. 실제로 거래가 없으면 None.

        백테스트 엔진의 리밸런싱 규칙을 그대로 따른다:
        임계치 미만이면 생략, 최소주문금액 미만이면 생략, 슬리피지·수수료 반영.
        """
        if price <= 0:
            return None

        comm = config.cost.commission_bps * BPS
        tax = config.cost.sell_tax_bps * BPS
        slip = config.cost.slippage_bps * BPS

        equity = self.equity(prices)
        if equity <= 0:
            return None

        held = self.positions.get(symbol, 0.0)
        current_w = held * price / equity

        need_flat = target_weight == 0.0 and held != 0.0
        if not need_flat and abs(target_weight - current_w) < config.rebalance_threshold:
            return None

        target_qty = target_weight * equity / price
        if not config.allow_fractional:
            target_qty = float(int(target_qty))
        delta = target_qty - held
        if delta == 0.0:
            return None

        notional = abs(delta) * price
        if config.min_order_value > 0 and notional < config.min_order_value:
            return None

        buying = delta > 0
        fill_price = price * (1.0 + slip) if buying else price * (1.0 - slip)
        qty = abs(delta)
        fee = qty * fill_price * (comm + (0.0 if buying else tax))

        cost = delta * fill_price + fee
        if buying and cost > self.cash:
            # 현금 부족: 살 수 있는 만큼만 (가상이라도 마이너스 통장은 없다)
            affordable = self.cash / (fill_price * (1.0 + comm))
            if config.min_order_value > 0 and affordable * fill_price < config.min_order_value:
                return None
            if not config.allow_fractional:
                affordable = float(int(affordable))
            if affordable <= 0:
                return None
            qty, delta = affordable, affordable
            fee = qty * fill_price * comm
            cost = delta * fill_price + fee

        self.cash -= cost
        self.total_fees += fee
        new_qty = held + delta

        if buying:
            eff = fill_price * (1.0 + comm)
            prev = self.avg_price.get(symbol, 0.0)
            self.avg_price[symbol] = (
                eff if held <= 0 else (prev * held + eff * delta) / new_qty
            )
        if abs(new_qty) < 1e-12:
            new_qty = 0.0
            self.avg_price.pop(symbol, None)
        self.positions[symbol] = new_qty

        fill = Fill(
            timestamp=timestamp,
            symbol=symbol,
            side="BUY" if buying else "SELL",
            quantity=qty,
            price=fill_price,
            fee=fee,
            notional=qty * fill_price,
            reason=reason,
        )
        self.fills.append(fill)
        self.updated_at = timestamp
        return fill

    # ------------------------------------------------------------------ 저장
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fills"] = [f.to_dict() for f in self.fills]
        d["version"] = STATE_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaperPortfolio":
        d = dict(d)
        version = d.pop("version", 0)
        if version > STATE_VERSION:
            raise ValueError(
                f"상태 파일 버전({version})이 이 코드({STATE_VERSION})보다 새것입니다."
            )
        fills = [Fill(**f) for f in d.pop("fills", [])]
        known = set(cls.__dataclass_fields__) - {"fills"}
        return cls(fills=fills, **{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> Path:
        """원자적 저장 — 쓰다가 죽어도 기존 상태가 깨지지 않는다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PaperPortfolio | None":
        path = Path(path)
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def create(cls, config: BacktestConfig, timestamp: str) -> "PaperPortfolio":
        return cls(
            cash=config.initial_cash,
            initial_cash=config.initial_cash,
            created_at=timestamp,
            updated_at=timestamp,
        )
