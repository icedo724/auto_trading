"""전략 결합 — 잡음이 섞인 추정치의 '최댓값'을 고르지 않는 방법.

## 왜 필요한가

그리드 탐색은 439개 조합의 성과를 추정하고 **최댓값**을 고른다. 그런데
추정치에 잡음이 있으면 최댓값은 **체계적으로 위쪽으로 편향된** 추정량이다.
관측 성과 = 실력 + 잡음 이므로, 최댓값을 고르는 것은 "실력이 큰 것"이 아니라
"실력 + 운이 큰 것"을 고르는 일이고, 운은 다음 기간에 재현되지 않는다.

통계학의 표준 해법은 두 가지다.

1. **평균(앙상블)** — 여러 후보의 신호를 평균하면 잡음이 상쇄된다.
   서로 독립인 K개를 평균하면 잡음 분산이 1/K 로 줄지만 실력은 유지된다.
2. **축소(shrinkage)** — 개별 추정치를 전체 평균 쪽으로 당긴다.
   (여기서는 상위 K개 평균이 그 역할을 겸한다)

## 무엇이 개선되나

앙상블은 **알파를 만들지 않는다.** 대신 **선택 편향을 줄여** 실전 성과가
백테스트에 가까워지게 만든다. 즉 기대수익이 아니라 **재현성**이 개선된다.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import indicators as ind
from .strategy.base import Strategy


class EnsembleStrategy(Strategy):
    """여러 전략의 목표비중을 가중평균한다.

    레지스트리에 등록하지 않는다 — 그리드 탐색의 **결과물**로 만들어지는 것이지
    탐색 대상이 아니기 때문이다.
    """

    name = "ensemble"

    def __init__(
        self, members: Sequence[Strategy], weights: Sequence[float] | None = None
    ) -> None:
        if not members:
            raise ValueError("앙상블 멤버가 비어 있습니다.")

        w = np.ones(len(members)) if weights is None else np.asarray(weights, dtype=float)
        if len(w) != len(members):
            raise ValueError("weights 길이가 members 와 다릅니다.")
        if np.any(w < 0):
            raise ValueError("가중치는 음수일 수 없습니다.")
        total = w.sum()
        if total <= 0:
            raise ValueError("가중치 합이 0입니다.")

        self.members = list(members)
        self.weights = (w / total).tolist()
        # Strategy 의 파라미터 검증 체계를 우회한다(멤버가 곧 파라미터)
        self.params: dict[str, Any] = {
            "n_members": len(self.members),
            "members": [m.describe() for m in self.members],
        }

    @property
    def warmup(self) -> int:
        return max(m.warmup for m in self.members)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        acc = None
        for strat, w in zip(self.members, self.weights):
            s = strat.generate_signals(df) * w
            acc = s if acc is None else acc + s
        return self._finalize(acc, df.index)

    def describe(self) -> str:
        return f"ensemble({len(self.members)}개: " + " + ".join(
            f"{w:.0%}×{m.name}" for m, w in zip(self.members, self.weights)
        ) + ")"

    __repr__ = describe

    def signature(self) -> tuple:
        return ("ensemble", tuple(m.signature() for m in self.members))


class VolatilityScaled(Strategy):
    """어떤 전략이든 감싸서 **변동성 타겟팅**을 씌운다.

    비중을 ``target_vol / 실현변동성`` 으로 조절한다.

    이것이 통계적으로 정당한 이유: **변동성은 자기상관이 매우 높다**(오늘 변동성이
    크면 내일도 크다). 수익률의 방향은 거의 예측 불가능하지만 변동성의 크기는
    상당히 예측 가능하다. 예측 가능한 것만 이용하는 것이므로 과최적화 위험이 낮고,
    실증적으로 가장 재현성 높은 개선 수단이다.

    효과는 기대수익 증가가 아니라 **변동성 안정화 → Sharpe 개선 · MDD 축소**다.
    """

    name = "vol_scaled"

    def __init__(
        self,
        base: Strategy,
        *,
        target_vol: float = 0.15,
        window: int = 20,
        max_leverage: float = 1.0,
        trading_days: int = 252,
        quantize: float = 0.1,
    ) -> None:
        if target_vol <= 0:
            raise ValueError("target_vol 은 0보다 커야 합니다.")
        if window < 5:
            raise ValueError("window 가 너무 짧습니다.")
        if max_leverage <= 0:
            raise ValueError("max_leverage 는 0보다 커야 합니다.")

        self.base = base
        self.target_vol = target_vol
        self.window = window
        self.max_leverage = max_leverage
        self.trading_days = trading_days
        self.quantize = quantize
        self.params = {
            "base": base.describe(),
            "target_vol": target_vol,
            "window": window,
            "max_leverage": max_leverage,
        }

    @property
    def warmup(self) -> int:
        return max(self.base.warmup, self.window) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raw = self.base.generate_signals(df)
        vol = ind.realized_vol(df["close"], self.window, self.trading_days)
        scale = (self.target_vol / vol.replace(0.0, np.nan)).clip(upper=self.max_leverage)
        out = raw * scale
        if self.quantize > 0:
            # 미세 비중 변동으로 회전율이 폭증하는 것을 막는다
            out = (out / self.quantize).round() * self.quantize
        return self._finalize(out.where(vol.notna()), df.index)

    def describe(self) -> str:
        return f"vol_scaled({self.base.describe()}, target={self.target_vol:.0%})"

    __repr__ = describe

    def signature(self) -> tuple:
        return ("vol_scaled", self.base.signature(), self.target_vol, self.window)


# --------------------------------------------------------------------------------
# 리포트에서 앙상블 만들기
# --------------------------------------------------------------------------------
def build_ensemble(
    report,
    *,
    top_k: int = 10,
    weighting: str = "equal",
    diversify: bool = True,
) -> EnsembleStrategy:
    """그리드 탐색 결과 상위 K개로 앙상블을 구성한다.

    weighting:
        equal  — 동일가중 (가장 보수적, 기본값)
        score  — 점수 비례 (양수 점수만)
        rank   — 순위 역수 비례

    diversify=True 면 **전략 종류별로 하나씩** 먼저 채운다. 같은 전략의 이웃
    파라미터만 K개 모으면 서로 거의 같은 신호라 평균 효과가 없기 때문이다.
    """
    from .strategy import create_strategy

    passed = [r for r in report.results if not r.filtered]
    if not passed:
        raise ValueError("필터를 통과한 후보가 없습니다.")

    if diversify:
        picked, seen = [], set()
        for r in passed:  # 점수 내림차순
            if r.strategy not in seen:
                seen.add(r.strategy)
                picked.append(r)
            if len(picked) >= top_k:
                break
        for r in passed:  # 남는 자리는 점수순으로
            if len(picked) >= top_k:
                break
            if r not in picked:
                picked.append(r)
    else:
        picked = passed[:top_k]

    members = [create_strategy(r.strategy, r.params) for r in picked]

    if weighting == "equal":
        weights = None
    elif weighting == "score":
        raw = np.array([max(r.score, 0.0) for r in picked])
        weights = raw if raw.sum() > 0 else None
    elif weighting == "rank":
        weights = 1.0 / np.arange(1, len(picked) + 1)
    else:
        raise ValueError(f"알 수 없는 weighting: {weighting!r} (equal/score/rank)")

    return EnsembleStrategy(members, weights)
