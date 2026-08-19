"""그리드(분할 매매) 계열 — "자주 조금씩"을 비용 안에서 하는 방법.

추세추종이 "한 번 크게"라면, 그리드는 **기준선 대비 싸지면 사고 비싸지면 파는**
분할 매매다. 횡보장에서 잔파도를 반복해서 먹는 구조라 소액 적립식과 궁합이 좋다.

다만 거래 빈도가 곧 비용이므로, ``levels`` 로 분할 단계를 거칠게 잡아
**밴드 한 칸이 왕복 비용보다 충분히 크도록** 설계해야 한다.
한 칸이 비용보다 작으면 거래할수록 손해다.
"""

from __future__ import annotations

import pandas as pd

from .. import indicators as ind
from .base import Strategy, register


@register
class GridTrading(Strategy):
    """기준선(이동평균) 대비 이격도로 비중을 계단식 조절.

    기준선에서 ``band``/2 만큼 내려가면 만기 비중, ``band``/2 올라가면 0.
    그 사이를 ``levels`` 단계로 나눠 계단식으로 사고판다.

        w_t = quantize( clip( 0.5 - (P_t / MA_t - 1) / band, 0, 1 ), levels )

    한 칸의 가격 폭 = band / levels. 이 값이 왕복 비용(코인 약 0.2%)의
    몇 배는 되어야 의미가 있다. ``step_pct`` 프로퍼티로 확인할 수 있다.

    주의 — **가격 수준이 아니라 기준선 대비 이격도로 판단한다.**
    기준선이 가격을 따라가므로, 등속 상승장에서는 이격도가 일정해 비중도
    일정하게 유지된다(중립 0.5). 즉 추세를 타되 잔파도만 먹는 구조다.
    "오르면 무조건 판다"를 원한다면 그리드가 아니라 추세추종을 써야 한다.
    """

    name = "grid"
    defaults = {"ma": 20, "band": 0.20, "levels": 4, "trend_filter": 0}
    param_space = {
        "ma": [10, 20, 60],
        "band": [0.10, 0.20, 0.30],
        "levels": [2, 4, 8],
        "trend_filter": [0, 200],
    }

    def validate(self) -> None:
        if self["ma"] < 3:
            raise ValueError("기준선 기간이 너무 짧습니다.")
        if not 0 < self["band"] <= 1.0:
            raise ValueError("band 는 (0, 1] 범위여야 합니다.")
        if self["levels"] < 1:
            raise ValueError("levels 는 1 이상이어야 합니다.")

    @property
    def step_pct(self) -> float:
        """한 칸을 움직이는 데 필요한 가격 변동폭. 왕복 비용과 비교할 기준."""
        return float(self["band"]) / float(self["levels"])

    @property
    def warmup(self) -> int:
        return max(self["ma"], self["trend_filter"]) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ref = ind.sma(close, self["ma"])

        # 기준선 대비 이격도를 비중으로: 쌀수록 많이, 비쌀수록 적게
        gap = close / ref - 1.0
        raw = (0.5 - gap / self["band"]).clip(0.0, 1.0)

        # 계단식 양자화 — 이게 없으면 매일 미세 조정하느라 비용만 나간다
        step = 1.0 / self["levels"]
        w = (raw / step).round() * step

        if self["trend_filter"]:
            # 장기 추세 아래에서는 물타기를 멈춘다 (하락장 무한 매수 방지)
            w = w.where(close > ind.sma(close, self["trend_filter"]), 0.0)

        return self._finalize(w.where(ref.notna()), df.index)
