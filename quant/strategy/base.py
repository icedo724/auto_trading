"""전략 베이스 클래스와 레지스트리."""

from __future__ import annotations

import abc
import itertools
from typing import Any, Iterator

import pandas as pd


class Strategy(abc.ABC):
    """일봉 전략의 공통 인터페이스.

    구현체는 ``generate_signals`` 에서 각 시점의 **목표 비중**(target weight)을 낸다.

      +1.0 = 자본 100% 롱, 0.0 = 현금, -1.0 = 100% 숏

    중요: t행의 값은 **t일 종가까지의 정보만** 사용해야 한다.
    실제 체결은 엔진이 ``signal_lag`` 만큼 미뤄서 t+1일에 수행하므로,
    이 규칙만 지키면 룩어헤드 바이어스가 발생하지 않는다.
    """

    name: str = "base"
    #: 기본 그리드 탐색 공간 {파라미터명: [후보값...]}
    param_space: dict[str, list[Any]] = {}
    #: 기본 파라미터 {파라미터명: 값}
    defaults: dict[str, Any] = {}
    #: 벤치마크 전략은 거래 수가 적어도 최소거래수 필터에서 제외된다(비교 기준이므로).
    is_benchmark: bool = False

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(
                f"{self.name}: 알 수 없는 파라미터 {sorted(unknown)} "
                f"(가능: {sorted(self.defaults)})"
            )
        self.params: dict[str, Any] = {**self.defaults, **params}
        self.validate()

    # --- 하위 클래스가 채우는 부분 -------------------------------------------------
    def validate(self) -> None:
        """파라미터 조합이 유효하지 않으면 ValueError. (예: fast >= slow)"""

    @abc.abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """목표 비중 Series (index = df.index, dtype = float)."""

    @property
    def warmup(self) -> int:
        """지표 계산에 필요한 최소 봉 수. 이 구간은 성과 평가에서 제외된다."""
        return 0

    # --- 공통 유틸 ---------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.params[key]

    def describe(self) -> str:
        body = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({body})"

    __repr__ = describe

    def signature(self) -> tuple:
        return (self.name, tuple(sorted(self.params.items())))

    @staticmethod
    def _finalize(sig: pd.Series, index: pd.Index) -> pd.Series:
        """NaN 을 0(현금)으로 채우고 float Series 로 정리."""
        return (
            pd.Series(sig, index=index, dtype="float64")
            .replace([float("inf"), float("-inf")], 0.0)
            .fillna(0.0)
        )


# --------------------------------------------------------------------------------
# 레지스트리
# --------------------------------------------------------------------------------
_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """전략 클래스를 이름으로 등록하는 데코레이터."""
    if cls.name in _REGISTRY:
        raise ValueError(f"전략 이름 중복: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy_class(name: str) -> type[Strategy]:
    if name not in _REGISTRY:
        raise ValueError(f"알 수 없는 전략: {name!r} (등록됨: {sorted(_REGISTRY)})")
    return _REGISTRY[name]


def create_strategy(name: str, params: dict[str, Any] | None = None) -> Strategy:
    return get_strategy_class(name)(**(params or {}))


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


def expand_grid(space: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """{a:[1,2], b:[3]} -> {a:1,b:3}, {a:2,b:3}."""
    if not space:
        yield {}
        return
    keys = sorted(space)
    for combo in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, combo))


def build_candidates(
    name: str, space: dict[str, list[Any]] | None = None
) -> list[Strategy]:
    """그리드를 펼쳐 유효한 파라미터 조합만 전략 인스턴스로 만든다.

    ``validate()`` 에서 걸러진 조합(예: fast >= slow)은 조용히 제외된다.
    """
    cls = get_strategy_class(name)
    grid = space if space is not None else cls.param_space
    out: list[Strategy] = []
    seen: set[tuple] = set()
    for params in expand_grid(grid):
        try:
            strat = cls(**params)
        except ValueError:
            continue
        sig = strat.signature()
        if sig in seen:
            continue
        seen.add(sig)
        out.append(strat)
    return out
