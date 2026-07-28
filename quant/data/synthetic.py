"""오프라인/재현 가능한 합성 시세 생성기.

외부 네트워크 없이 전체 파이프라인(백테스트·최적화·검증·리포트)을 돌려보기 위한 소스.
종목 코드로부터 시드를 결정하므로 **언제 돌려도 동일한 시세**가 나온다.

특징:
  - 강세/약세/횡보 3개 국면을 마르코프 체인으로 전환 (추세·평균회귀 전략 모두 의미 있음)
  - GARCH(1,1) 유사 변동성 클러스터링
  - 시가 갭 + 장중 고저 + 거래량 (변동성과 양의 상관)
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .base import DataSource

# (연율 드리프트, 변동성 배수, 다음 국면 전이확률)
_REGIMES = {
    0: dict(name="bull", drift=0.18, vol_mult=0.85),
    1: dict(name="bear", drift=-0.16, vol_mult=1.45),
    2: dict(name="chop", drift=0.01, vol_mult=1.00),
}
_TRANSITION = np.array(
    [
        [0.985, 0.006, 0.009],  # bull ->
        [0.012, 0.972, 0.016],  # bear ->
        [0.011, 0.009, 0.980],  # chop ->
    ]
)


def _stationary_distribution(P: np.ndarray) -> np.ndarray:
    """마르코프 전이행렬의 정상분포 (좌고유벡터, 고유값 1)."""
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return v / v.sum()


_STATIONARY = _stationary_distribution(_TRANSITION)
_VOL_MULT_RMS = float(
    np.sqrt(sum(_STATIONARY[r] * _REGIMES[r]["vol_mult"] ** 2 for r in _REGIMES))
)
#: 팻테일 충격(1% 확률로 3배 추가 정규충격)의 RMS -> Var[z] = 1 + 0.01 * 3^2
_JUMP_PROB, _JUMP_SCALE = 0.01, 3.0
_SHOCK_RMS = float(np.sqrt(1.0 + _JUMP_PROB * _JUMP_SCALE**2))


def _seed_for(symbol: str) -> int:
    return int(hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:8], 16)


class SyntheticSource(DataSource):
    """합성 시세 소스."""

    name = "synthetic"

    def __init__(
        self,
        *,
        base_price: float = 50_000.0,
        annual_vol: float = 0.28,
        trading_days: int = 252,
        seed_offset: int = 0,
    ) -> None:
        self.base_price = base_price
        self.annual_vol = annual_vol
        self.trading_days = trading_days
        self.seed_offset = seed_offset

    def _fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.bdate_range(start=start, end=end)
        n = len(dates)
        if n < 2:
            raise ValueError(f"기간이 너무 짧습니다: {start} ~ {end}")

        rng = np.random.default_rng(_seed_for(symbol) + self.seed_offset)

        dt = 1.0 / self.trading_days
        # 국면별 변동성 배수 때문에 실현변동성이 목표치보다 커지므로,
        # 정상상태 국면분포 기준 RMS 배수로 나누어 보정한다.
        base_sigma = self.annual_vol * np.sqrt(dt) / (_VOL_MULT_RMS * _SHOCK_RMS)

        # --- 국면 경로 ---
        regime = np.empty(n, dtype=int)
        regime[0] = rng.choice(3, p=[0.45, 0.2, 0.35])
        u = rng.random(n)
        for t in range(1, n):
            regime[t] = np.searchsorted(_TRANSITION[regime[t - 1]].cumsum(), u[t])

        # --- 변동성 클러스터링: AR(1) 로그변동성 (확률변동성 모형) ---
        # GARCH 의 제곱수익률 피드백과 달리 발산하지 않으면서 군집성은 그대로 재현한다.
        phi, eta = 0.97, 0.12
        stat_var = eta**2 / (1.0 - phi**2)  # 정상상태 분산
        # sigma 는 로그정규이므로 E[sigma^2] = exp(2*mu + 2*stat_var).
        # 실현 *분산* 을 목표치에 맞추려면 mu = log(base) - stat_var 이어야 한다.
        log_mean = np.log(base_sigma) - stat_var

        log_sigma = np.empty(n)
        log_sigma[0] = log_mean
        vol_noise = rng.standard_normal(n) * eta
        for t in range(1, n):
            log_sigma[t] = log_mean + phi * (log_sigma[t - 1] - log_mean) + vol_noise[t]

        vol_mult = np.array([_REGIMES[r]["vol_mult"] for r in regime])
        daily_sigma = np.exp(log_sigma) * vol_mult

        # 팻테일: 가끔 큰 충격 (±5σ 로 제한)
        shocks = np.clip(
            rng.standard_normal(n)
            + (rng.random(n) < _JUMP_PROB) * rng.standard_normal(n) * _JUMP_SCALE,
            -5.0,
            5.0,
        )
        drift = np.array([_REGIMES[r]["drift"] for r in regime]) * dt
        # 일간 로그수익률을 ±40% 로 제한 (가격지수 발산 방지)
        log_ret = np.clip(drift - 0.5 * daily_sigma**2 + daily_sigma * shocks, -0.4, 0.4)

        close = self.base_price * np.exp(np.cumsum(log_ret))

        # --- 시가 갭 & 장중 고저 ---
        prev_close = np.concatenate([[self.base_price], close[:-1]])
        gap = rng.standard_normal(n) * daily_sigma * 0.35
        open_ = prev_close * np.exp(gap)

        span = np.abs(rng.standard_normal(n)) * daily_sigma * 0.9
        hi_base = np.maximum(open_, close)
        lo_base = np.minimum(open_, close)
        high = hi_base * (1.0 + span * rng.uniform(0.3, 1.0, n))
        low = lo_base * (1.0 - span * rng.uniform(0.3, 1.0, n))

        volume = np.exp(
            rng.normal(np.log(300_000), 0.45, n) + np.minimum(4.0 * np.abs(log_ret), 3.0)
        ).round()

        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=pd.DatetimeIndex(dates, name="date"),
        )
