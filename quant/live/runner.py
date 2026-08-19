"""페이퍼 트레이딩 실행기.

실데이터 · 가상 자금으로 전략을 24시간 돌린다. **실제 주문은 절대 내지 않는다.**

한 사이클이 하는 일:
  1. 최신 시세를 받는다 (캐시 갱신)
  2. 새로 마감된 봉이 있는 종목만 골라낸다 (없으면 아무것도 하지 않음)
  3. 적립일이면 가상 계좌에 입금한다
  4. 전략 신호를 계산해 가상 체결한다
  5. 상태를 원자적으로 저장하고 저널에 남긴다

**멱등성**: 종목별로 마지막 처리 봉을 기록하므로, 같은 사이클을 여러 번 돌려도
중복 체결되지 않는다. cron 이 겹쳐 돌거나 서버가 재시작돼도 안전하다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import BacktestConfig
from ..data import DataError, get_source
from ..dates import resolve_period
from ..strategy import Strategy
from .journal import Journal
from .portfolio import PaperPortfolio


@dataclass
class CycleResult:
    """한 사이클의 결과 요약."""

    timestamp: str
    processed: list[str]
    skipped: list[str]
    fills: list[dict[str, Any]]
    deposited: float
    equity: float
    errors: list[str]
    #: 서버가 꺼져 있던 동안 지나간 봉 수 (소급 체결하지 않고 기록만 한다)
    missed_bars: int = 0

    @property
    def acted(self) -> bool:
        return bool(self.fills or self.deposited)


class PaperTrader:
    """가상 자금 · 실데이터 트레이더."""

    def __init__(
        self,
        experiment: dict[str, Any],
        strategy: Strategy,
        config: BacktestConfig,
        *,
        state_dir: str | Path = "state",
        name: str = "paper",
    ) -> None:
        self.exp = experiment
        self.strategy = strategy
        self.config = config
        self.config.validate()
        self.name = name
        self.state_path = Path(state_dir) / f"{name}.json"
        self.journal = Journal(Path(state_dir) / f"{name}.jsonl")

        self.portfolio = PaperPortfolio.load(self.state_path)
        if self.portfolio is None:
            self.portfolio = PaperPortfolio.create(config, self._now())
            self.portfolio.save(self.state_path)
            self.journal.write(
                "start",
                strategy=strategy.describe(),
                initial_cash=config.initial_cash,
                symbols=list(experiment["data"].get("symbols", [])),
                cost=config.cost.__dict__,
            )

    # ------------------------------------------------------------------ 유틸
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _load_data(self) -> dict[str, pd.DataFrame]:
        d = self.exp["data"]
        source = get_source(
            d.get("source", "synthetic"),
            cache_dir=d.get("cache_dir", "data/cache"),
            refresh=True,  # 라이브는 항상 최신을 받아야 한다
            csv_dir=d.get("csv_dir", "data/csv"),
        )
        start, end = resolve_period(d.get("start"), d.get("end"))
        out: dict[str, pd.DataFrame] = {}
        for sym in d.get("symbols", []):
            try:
                out[sym] = source.get(sym, start, end)
            except Exception as exc:  # noqa: BLE001 - 한 종목 실패로 전체가 멈추면 안 된다
                self.journal.write("data_error", symbol=sym, error=str(exc)[:300])
        if not out:
            raise DataError("모든 종목의 시세 수집에 실패했습니다.")
        return out

    def _maybe_deposit(self, now: str) -> float:
        """적립일이면 입금. 같은 기간에 두 번 넣지 않는다."""
        cfg = self.config
        if cfg.contribution <= 0:
            return 0.0
        ts = pd.Timestamp(now).tz_localize(None)  # to_period 는 tz 를 버리며 경고한다
        period = str(ts.to_period({"M": "M", "W": "W", "Q": "Q"}[cfg.contribution_freq]))
        if self.portfolio.last_contribution == period:
            return 0.0
        # 최초 실행은 initial_cash 가 이미 들어가 있으므로 기간만 기록하고 넘어간다
        first_run = not self.portfolio.last_contribution
        self.portfolio.last_contribution = period
        if first_run:
            return 0.0
        self.portfolio.deposit(cfg.contribution, now)
        self.journal.write("deposit", amount=cfg.contribution, period=period)
        return cfg.contribution

    # ------------------------------------------------------------------ 사이클
    def run_once(self, *, force: bool = False) -> CycleResult:
        now = self._now()
        processed: list[str] = []
        skipped: list[str] = []
        fills: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            data = self._load_data()
        except Exception as exc:  # noqa: BLE001
            self.journal.write("cycle_error", error=str(exc)[:300])
            return CycleResult(now, [], [], [], 0.0, 0.0, [str(exc)[:300]], 0)

        prices = {s: float(df["close"].iloc[-1]) for s, df in data.items()}

        # 새 봉이 마감된 종목만 처리 (멱등성)
        pending = {}
        missed_total = 0
        for sym, df in data.items():
            bar = str(df.index[-1].date())
            prev = self.portfolio.last_bar.get(sym)
            if not force and prev == bar:
                skipped.append(sym)
                continue

            # 서버가 꺼져 있던 동안 지나간 봉 수. 과거 가격으로 소급 체결하는 것은
            # 페이퍼 트레이딩이 아니라 백테스트이므로, 재현하지 않고 '기록'만 한다.
            # 이 값이 쌓이면 라이브 vs 백테스트 비교의 신뢰도가 떨어진다.
            missed = 0
            if prev is not None:
                pos = df.index.searchsorted(pd.Timestamp(prev), side="right")
                missed = max(int(len(df) - pos - 1), 0)
            if missed:
                missed_total += missed
                self.journal.write(
                    "missed_bars", symbol=sym, count=missed, last=prev, current=bar
                )
            pending[sym] = (df, bar)

        deposited = self._maybe_deposit(now) if pending else 0.0

        # 백테스트는 종목별 독립 계좌를 '동일비중'으로 합성한다(metrics.portfolio_returns).
        # 라이브는 계좌가 하나이므로, 종목당 배분을 1/N 로 나눠야 같은 포트폴리오가 된다.
        # 이 스케일이 없으면 먼저 처리된 종목이 자본을 전부 가져간다.
        allocation = 1.0 / max(len(data), 1)

        for sym, (df, bar) in pending.items():
            try:
                sig = self.strategy.generate_signals(df)
                if sig.empty:
                    errors.append(f"{sym}: 신호 없음")
                    continue
                # 백테스트와 동일: 마지막 '마감된' 봉의 신호를 지금(= 다음 봉 시가) 체결
                raw_target = max(0.0, min(float(sig.iloc[-1]), self.config.max_weight))
                target = raw_target * allocation

                fill = self.portfolio.execute(
                    sym, target, prices[sym], self.config, now, prices
                )
                self.portfolio.last_bar[sym] = bar
                processed.append(sym)
                if fill is not None:
                    fills.append(fill.to_dict())
                    self.journal.write(
                        "fill", bar=bar, signal=raw_target, target_weight=target,
                        **fill.to_dict(),
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{sym}: {exc}")
                self.journal.write("symbol_error", symbol=sym, error=str(exc)[:300])

        equity = self.portfolio.equity(prices)
        self.portfolio.updated_at = now
        self.portfolio.save(self.state_path)

        bars = [self.portfolio.last_bar[s] for s in data if s in self.portfolio.last_bar]
        self.journal.write(
            "cycle",
            bar=max(bars) if bars else "",
            missed=missed_total,
            processed=processed,
            skipped=len(skipped),
            n_fills=len(fills),
            deposited=deposited,
            equity=round(equity, 2),
            cash=round(self.portfolio.cash, 2),
            invested=round(self.portfolio.total_invested, 2),
            prices={k: round(v, 4) for k, v in prices.items()},
        )
        return CycleResult(
            now, processed, skipped, fills, deposited, equity, errors, missed_total
        )

    def run_forever(
        self, interval_sec: int = 3600, *, stop_file: str | Path = "STOP", max_cycles: int = 0
    ) -> int:
        """주기 실행. ``STOP`` 파일이 생기면 안전하게 멈춘다."""
        stop = Path(stop_file)
        cycles = 0
        self.journal.write("loop_start", interval_sec=interval_sec)
        try:
            while True:
                if stop.exists():
                    self.journal.write("stopped", reason="STOP 파일")
                    print(f"[{self._now()}] STOP 파일 감지 — 종료")
                    return 0
                res = self.run_once()
                mark = "*" if res.acted else "."
                print(
                    f"[{res.timestamp}] {mark} 처리 {len(res.processed)} · "
                    f"체결 {len(res.fills)} · 평가 {res.equity:,.0f}원"
                    + (f" · 놓친봉 {res.missed_bars}" if res.missed_bars else "")
                    + (f" · 오류 {len(res.errors)}" if res.errors else "")
                )
                cycles += 1
                if max_cycles and cycles >= max_cycles:
                    return 0
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            self.journal.write("stopped", reason="KeyboardInterrupt")
            print("\n중단됨.")
            return 130
