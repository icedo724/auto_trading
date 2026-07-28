"""전략 패키지.

새 전략을 추가하려면 이 패키지 안에 모듈을 만들고 ``@register`` 를 붙인 뒤
아래 import 목록에 추가하면 CLI/최적화기에서 자동으로 인식된다.
"""

from __future__ import annotations

from .base import (
    Strategy,
    available_strategies,
    build_candidates,
    create_strategy,
    expand_grid,
    get_strategy_class,
    register,
)
from .reversion import (  # noqa: F401 - 레지스트리 등록 목적
    BollingerBands,
    RsiReversion,
    VolTargetTrend,
    ZScoreReversion,
)
from .trend import (  # noqa: F401 - 레지스트리 등록 목적
    BuyAndHold,
    DonchianBreakout,
    MacdTrend,
    SmaCross,
    TimeSeriesMomentum,
)

__all__ = [
    "Strategy",
    "available_strategies",
    "build_candidates",
    "create_strategy",
    "expand_grid",
    "get_strategy_class",
    "register",
]
