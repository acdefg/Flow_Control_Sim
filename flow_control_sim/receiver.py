"""Receiver service-rate models."""

from __future__ import annotations

import random
from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple

from .config import ServiceSegment, SimulationConfig


class ReceiverProfile(Protocol):
    def rate(self, cycle: int) -> float:
        ...

    @property
    def change_cycles(self) -> Tuple[int, ...]:
        ...


@dataclass(frozen=True)
class ConstantReceiver:
    service_rate: float

    def __post_init__(self) -> None:
        if self.service_rate < 0:
            raise ValueError("service_rate must be >= 0")

    def rate(self, cycle: int) -> float:
        return self.service_rate

    @property
    def change_cycles(self) -> Tuple[int, ...]:
        return (0,)


class StepReceiver:
    """Piecewise-constant receiver profile."""

    def __init__(self, segments: Sequence[ServiceSegment]) -> None:
        if not segments:
            raise ValueError("segments must not be empty")
        self._segments = tuple(sorted(segments, key=lambda item: item.start_cycle))
        if self._segments[0].start_cycle != 0:
            raise ValueError("segments must start at cycle 0")
        self._starts = tuple(segment.start_cycle for segment in self._segments)
        self._rates = tuple(segment.rate for segment in self._segments)

    def rate(self, cycle: int) -> float:
        if cycle < 0:
            raise ValueError("cycle must be >= 0")
        idx = bisect_right(self._starts, cycle) - 1
        return self._rates[idx]

    @property
    def change_cycles(self) -> Tuple[int, ...]:
        return self._starts

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "StepReceiver":
        return cls(config.service_profile)


class RandomReceiver:
    """Receiver with uniformly jittered service rate."""

    def __init__(self, mean_rate: float, jitter: float, seed: int = 1, min_rate: float = 0.0) -> None:
        if mean_rate < 0:
            raise ValueError("mean_rate must be >= 0")
        if jitter < 0:
            raise ValueError("jitter must be >= 0")
        if min_rate < 0:
            raise ValueError("min_rate must be >= 0")
        self.mean_rate = mean_rate
        self.jitter = jitter
        self.min_rate = min_rate
        self._rng = random.Random(seed)

    def rate(self, cycle: int) -> float:
        del cycle
        value = self.mean_rate + self._rng.uniform(-self.jitter, self.jitter)
        return max(self.min_rate, value)

    @property
    def change_cycles(self) -> Tuple[int, ...]:
        return (0,)
