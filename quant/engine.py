"""백테스트 엔진.

설계 원칙
---------
1. **룩어헤드 금지** — 전략이 t일 종가까지의 정보로 만든 신호는 ``signal_lag`` 만큼
   지연되어 t+1일 시가(기본값)에 체결된다.
2. **동일 시점 비교** — ``trade_start`` 를 지정하면 그 시점부터만 매매한다.
   지표 워밍업 길이가 서로 다른 파라미터들도 완전히 같은 구간에서 경쟁하게 된다.
3. **현실적 비용** — 슬리피지(체결가 악화) + 수수료(양방향) + 거래세(매도)를 모두 반영.
4. **장중 리스크 관리** — 손절/익절/추적손절은 당일 고가·저가로 체결 여부를 판정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import BPS, BacktestConfig

ExitReason = Literal["signal", "stop_loss", "take_profit", "trailing_stop", "time_exit", "eod"]


@dataclass
class Trade:
    """청산이 발생한 라운드트립(부분 청산 포함) 1건."""

    symbol: str
    direction: int  # +1 롱, -1 숏
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float  # 비용 포함 평균 진입 체결가
    exit_price: float  # 비용 포함 청산 체결가
    quantity: float
    pnl: float  # 비용 차감 후 손익 (통화 단위)
    return_pct: float  # 비용 차감 후 수익률
    holding_days: int
    exit_reason: ExitReason

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["entry_date"] = self.entry_date.strftime("%Y-%m-%d")
        d["exit_date"] = self.exit_date.strftime("%Y-%m-%d")
        return d


@dataclass
class BacktestResult:
    """단일 종목 백테스트 결과."""

    symbol: str
    equity: pd.Series  # 일별 평가자산 (trade_start 이후)
    returns: pd.Series  # 일별 수익률
    position: pd.Series  # 일별 보유 비중
    trades: list[Trade] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0  # 누적 거래비용 (통화 단위)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "symbol", "direction", "entry_date", "exit_date", "entry_price",
                    "exit_price", "quantity", "pnl", "return_pct", "holding_days",
                    "exit_reason",
                ]
            )
        return pd.DataFrame([t.to_dict() for t in self.trades])


class Backtester:
    """일봉 단일 종목 백테스터."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.config.validate()

    # ------------------------------------------------------------------ public
    def run(
        self,
        df: pd.DataFrame,
        signal: pd.Series,
        *,
        symbol: str = "",
        trade_start: pd.Timestamp | str | None = None,
    ) -> BacktestResult:
        from .metrics import compute_metrics  # 순환 import 방지

        cfg = self.config
        if len(df) < 2:
            raise ValueError(f"{symbol}: 봉 개수가 부족합니다 ({len(df)}).")

        signal = signal.reindex(df.index)
        target = signal.shift(cfg.signal_lag).fillna(0.0).astype(float)
        target = target.clip(-cfg.max_weight, cfg.max_weight)
        if not cfg.allow_short:
            target = target.clip(lower=0.0)

        start_idx = self._resolve_start(df.index, trade_start)

        sim = self._simulate(df, target.to_numpy(), start_idx, symbol)

        equity = sim["equity"].iloc[start_idx:]
        if equity.empty:
            raise ValueError(f"{symbol}: 평가 구간이 비어 있습니다.")
        # 평가 구간 시작점을 초기자본으로 정규화 (파라미터 간 공정 비교)
        equity = equity / equity.iloc[0] * cfg.initial_cash
        returns = equity.pct_change().fillna(0.0)

        trades = [t for t in sim["trades"] if t.exit_date >= equity.index[0]]
        result = BacktestResult(
            symbol=symbol,
            equity=equity,
            returns=returns,
            position=sim["position"].iloc[start_idx:],
            trades=trades,
            total_cost=sim["total_cost"],
        )
        result.metrics = compute_metrics(result, cfg)
        return result

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _resolve_start(index: pd.DatetimeIndex, trade_start: Any) -> int:
        if trade_start is None:
            return 0
        if isinstance(trade_start, (int, np.integer)):
            return int(np.clip(trade_start, 0, len(index) - 1))
        ts = pd.Timestamp(trade_start)
        pos = int(index.searchsorted(ts, side="left"))
        return int(np.clip(pos, 0, len(index) - 1))

    def _simulate(
        self, df: pd.DataFrame, target: np.ndarray, start_idx: int, symbol: str
    ) -> dict[str, Any]:
        cfg = self.config
        comm = cfg.cost.commission_bps * BPS
        tax = cfg.cost.sell_tax_bps * BPS
        slip = cfg.cost.slippage_bps * BPS

        op = df["open"].to_numpy(dtype=float)
        hi = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        cl = df["close"].to_numpy(dtype=float)
        dates = df.index
        n = len(df)

        use_open = cfg.execution == "next_open"

        cash = float(cfg.initial_cash)
        shares = 0.0
        entry_price = 0.0  # 비용 포함 평균 진입가
        entry_idx = -1
        peak = 0.0  # 진입 후 최고가(롱) / 최저가(숏)
        blocked = False  # 강제청산 후 재진입 차단 (신호가 0으로 돌아올 때까지)

        equity_arr = np.empty(n)
        pos_arr = np.zeros(n)
        trades: list[Trade] = []
        total_cost = 0.0
        last_equity = cash

        def fill_price(raw: float, buying: bool) -> float:
            return raw * (1.0 + slip) if buying else raw * (1.0 - slip)

        def trade_fee(qty_abs: float, price: float, selling: bool) -> float:
            return qty_abs * price * (comm + (tax if selling else 0.0))

        for i in range(n):
            price_close = cl[i]
            if not np.isfinite(price_close) or price_close <= 0:
                # 상장 전/데이터 결측 구간: 매매하지 않고 자산만 유지
                equity_arr[i] = last_equity
                pos_arr[i] = 0.0
                continue

            raw_exec = op[i] if use_open else price_close
            if not np.isfinite(raw_exec) or raw_exec <= 0:
                raw_exec = price_close

            # ---------------- 1) 목표 비중 확정 ----------------
            desired = float(target[i]) if i >= start_idx else 0.0
            if blocked:
                if desired == 0.0:
                    blocked = False
                else:
                    desired = 0.0

            # ---------------- 2) 리밸런싱 체결 ----------------
            equity_at_exec = cash + shares * raw_exec
            if equity_at_exec <= 0:  # 파산
                equity_arr[i:] = 0.0
                pos_arr[i:] = 0.0
                break

            current_w = shares * raw_exec / equity_at_exec
            need_flat = desired == 0.0 and shares != 0.0
            if need_flat or abs(desired - current_w) >= cfg.rebalance_threshold:
                tgt_shares = desired * equity_at_exec / raw_exec
                if not cfg.allow_fractional:
                    tgt_shares = float(np.trunc(tgt_shares))
                delta = tgt_shares - shares
                if delta != 0.0:
                    buying = delta > 0
                    fp = fill_price(raw_exec, buying)
                    qty = abs(delta)
                    fee = trade_fee(qty, fp, selling=not buying)
                    total_cost += fee + qty * abs(fp - raw_exec)

                    closing = min(qty, abs(shares)) if np.sign(delta) != np.sign(shares) and shares != 0 else 0.0
                    if closing > 0:
                        trades.append(
                            self._make_trade(
                                symbol, shares, entry_price, entry_idx, closing,
                                fp, comm, tax, i, dates, "signal",
                            )
                        )
                    if shares == 0 or np.sign(delta) == np.sign(shares):
                        # 신규/추가 진입 -> 비용 포함 평균단가 갱신
                        eff = fp * (1.0 + comm) if buying else fp * (1.0 - comm - tax)
                        new_shares = shares + delta
                        if shares == 0:
                            entry_price, entry_idx, peak = eff, i, raw_exec
                        else:
                            entry_price = (entry_price * shares + eff * delta) / new_shares

                    cash -= delta * fp + fee
                    shares = tgt_shares
                    if shares == 0.0:
                        entry_price, entry_idx, peak = 0.0, -1, 0.0

            # ---------------- 3) 장중 리스크 관리 ----------------
            if shares != 0.0:
                direction = 1 if shares > 0 else -1
                peak = max(peak, hi[i]) if direction > 0 else min(peak or lo[i], lo[i])

                stop_px, reason = self._stop_level(direction, entry_price, peak)
                hit_px: float | None = None
                hit_reason: ExitReason | None = None

                if stop_px is not None:
                    if (direction > 0 and lo[i] <= stop_px) or (
                        direction < 0 and hi[i] >= stop_px
                    ):
                        hit_px, hit_reason = float(np.clip(stop_px, lo[i], hi[i])), reason

                if hit_px is None and cfg.take_profit_pct > 0:
                    tp = entry_price * (
                        1 + cfg.take_profit_pct * (1 if direction > 0 else -1)
                    )
                    if (direction > 0 and hi[i] >= tp) or (direction < 0 and lo[i] <= tp):
                        hit_px, hit_reason = float(np.clip(tp, lo[i], hi[i])), "take_profit"

                if (
                    hit_px is None
                    and cfg.max_holding_days > 0
                    and entry_idx >= 0
                    and (i - entry_idx) >= cfg.max_holding_days
                ):
                    hit_px, hit_reason = price_close, "time_exit"

                if hit_px is not None and hit_reason is not None:
                    selling = shares > 0
                    fp = fill_price(hit_px, buying=not selling)
                    qty = abs(shares)
                    fee = trade_fee(qty, fp, selling=selling)
                    total_cost += fee + qty * abs(fp - hit_px)
                    trades.append(
                        self._make_trade(
                            symbol, shares, entry_price, entry_idx, qty,
                            fp, comm, tax, i, dates, hit_reason,
                        )
                    )
                    cash += shares * fp - fee
                    shares, entry_price, entry_idx, peak = 0.0, 0.0, -1, 0.0
                    blocked = True

            # ---------------- 4) 종가 평가 ----------------
            last_equity = cash + shares * price_close
            equity_arr[i] = last_equity
            pos_arr[i] = 0.0 if last_equity <= 0 else shares * price_close / last_equity

        # 미청산 포지션은 마지막 종가로 정리해 거래 통계에 포함
        if shares != 0.0:
            i = n - 1
            selling = shares > 0
            fp = fill_price(cl[i], buying=not selling)
            trades.append(
                self._make_trade(
                    symbol, shares, entry_price, entry_idx, abs(shares),
                    fp, comm, tax, i, dates, "eod",
                )
            )

        return {
            "equity": pd.Series(equity_arr, index=dates, name="equity"),
            "position": pd.Series(pos_arr, index=dates, name="position"),
            "trades": trades,
            "total_cost": total_cost,
        }

    def _stop_level(
        self, direction: int, entry_price: float, peak: float
    ) -> tuple[float | None, ExitReason]:
        """손절/추적손절 중 더 보수적인(먼저 걸리는) 가격을 반환."""
        cfg = self.config
        levels: list[tuple[float, ExitReason]] = []
        if cfg.stop_loss_pct > 0:
            levels.append(
                (entry_price * (1 - cfg.stop_loss_pct * direction), "stop_loss")
            )
        if cfg.trailing_stop_pct > 0 and peak > 0:
            levels.append(
                (peak * (1 - cfg.trailing_stop_pct * direction), "trailing_stop")
            )
        if not levels:
            return None, "signal"
        # 롱은 높은 손절가가 먼저 걸리고, 숏은 낮은 손절가가 먼저 걸린다
        return max(levels) if direction > 0 else min(levels)

    @staticmethod
    def _make_trade(
        symbol: str,
        shares: float,
        entry_price: float,
        entry_idx: int,
        qty: float,
        exit_fill: float,
        comm: float,
        tax: float,
        i: int,
        dates: pd.DatetimeIndex,
        reason: ExitReason,
    ) -> Trade:
        direction = 1 if shares > 0 else -1
        # 청산 측 비용까지 반영한 실효 청산가
        eff_exit = exit_fill * (1 - comm - tax) if direction > 0 else exit_fill * (1 + comm)
        pnl = (eff_exit - entry_price) * qty * direction
        ret = (eff_exit / entry_price - 1.0) * direction if entry_price > 0 else 0.0
        e_idx = entry_idx if entry_idx >= 0 else i
        return Trade(
            symbol=symbol,
            direction=direction,
            entry_date=dates[e_idx],
            exit_date=dates[i],
            entry_price=entry_price,
            exit_price=eff_exit,
            quantity=qty,
            pnl=pnl,
            return_pct=ret,
            holding_days=int(i - e_idx),
            exit_reason=reason,
        )


def run_backtest(
    df: pd.DataFrame,
    strategy: "Any",
    config: BacktestConfig | None = None,
    *,
    symbol: str = "",
    trade_start: pd.Timestamp | str | None = None,
) -> BacktestResult:
    """전략 객체로 바로 백테스트를 돌리는 편의 함수."""
    signal = strategy.generate_signals(df)
    return Backtester(config).run(df, signal, symbol=symbol, trade_start=trade_start)
