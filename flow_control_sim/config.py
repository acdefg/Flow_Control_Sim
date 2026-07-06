"""Configuration objects for the flow-control simulator."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class DelayConfig:
    """Pipeline delay configuration in cycles."""

    forward_sn: int = 80
    forward_nr: int = 20
    return_rn: int = 20
    return_ns: int = 80
    throttle: int = 44

    def __post_init__(self) -> None:
        for name in ("forward_sn", "forward_nr", "return_rn", "return_ns", "throttle"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} delay must be >= 0")

    @property
    def request_path(self) -> int:
        return self.forward_sn + self.forward_nr

    @property
    def response_path(self) -> int:
        return self.return_rn + self.return_ns

    @property
    def round_trip(self) -> int:
        return self.request_path + self.response_path


@dataclass(frozen=True, order=True)
class ServiceSegment:
    """Receiver service rate starting at a cycle."""

    start_cycle: int
    rate: float

    def __post_init__(self) -> None:
        if self.start_cycle < 0:
            raise ValueError("service segment start_cycle must be >= 0")
        if self.rate < 0:
            raise ValueError("service segment rate must be >= 0")


def default_service_profile() -> Tuple[ServiceSegment, ...]:
    return (
        ServiceSegment(0, 3.0),
        ServiceSegment(1000, 1.0),
        ServiceSegment(2000, 3.0),
    )


@dataclass(frozen=True)
class TrafficConfig:
    """Discrete request traffic model.

    Units:
    - Rates are bytes/cycle.
    - Outstanding, queues, and capacities are request counts.
    """

    request_bytes: int = 32
    sender_burst_reqs: int = 16
    sender_check_interval: int = 44
    sender_burst_period: int = 176
    sender_mode: str = "fluid"
    noc_queue_capacity_reqs: int = 128
    sender_noc_inflight_capacity_reqs: int = 256
    noc_issue_reqs_per_cycle: int = 1
    noc_queue_delay: int = 770
    noc_queue_poll_period: int = 0

    def __post_init__(self) -> None:
        for name in (
            "request_bytes",
            "sender_burst_reqs",
            "sender_check_interval",
            "sender_burst_period",
            "noc_queue_capacity_reqs",
            "sender_noc_inflight_capacity_reqs",
            "noc_issue_reqs_per_cycle",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.noc_queue_delay < 0:
            raise ValueError("noc_queue_delay must be >= 0")
        if self.noc_queue_poll_period < 0:
            raise ValueError("noc_queue_poll_period must be >= 0")
        if self.sender_burst_period < self.sender_burst_reqs:
            raise ValueError("sender_burst_period must be >= sender_burst_reqs")
        if self.sender_mode not in {"fluid", "burst"}:
            raise ValueError("sender_mode must be either 'fluid' or 'burst'")

    @property
    def sender_average_rate(self) -> float:
        return self.sender_burst_reqs * self.request_bytes / self.sender_burst_period


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level experiment configuration."""

    total_cycles: int = 3000
    max_send_rate: float = 3.0
    sender_threshold: float = 128.0
    noc_threshold: float = 128.0
    queue_capacity: Optional[float] = None
    delays: DelayConfig = field(default_factory=DelayConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    service_profile: Tuple[ServiceSegment, ...] = field(default_factory=default_service_profile)

    settling_window: int = 100
    settling_tail_window: int = 200
    settling_rel_tolerance: float = 0.05
    settling_abs_tolerance: float = 1.0
    oscillation_window: int = 500

    def __post_init__(self) -> None:
        if self.total_cycles <= 0:
            raise ValueError("total_cycles must be > 0")
        if self.max_send_rate < 0:
            raise ValueError("max_send_rate must be >= 0")
        if self.sender_threshold <= 0:
            raise ValueError("sender_threshold must be > 0")
        if self.noc_threshold <= 0:
            raise ValueError("noc_threshold must be > 0")
        if self.queue_capacity is not None and self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be > 0 when set")
        if self.settling_window <= 0:
            raise ValueError("settling_window must be > 0")
        if self.settling_tail_window <= 0:
            raise ValueError("settling_tail_window must be > 0")
        if self.oscillation_window <= 0:
            raise ValueError("oscillation_window must be > 0")

        normalized = self._normalize_profile(self.service_profile)
        object.__setattr__(self, "service_profile", normalized)

    @staticmethod
    def _normalize_profile(profile: Iterable[ServiceSegment]) -> Tuple[ServiceSegment, ...]:
        segments = tuple(sorted(profile, key=lambda item: item.start_cycle))
        if not segments:
            raise ValueError("service_profile must contain at least one segment")
        if segments[0].start_cycle != 0:
            raise ValueError("service_profile must start at cycle 0")
        for prev, curr in zip(segments, segments[1:]):
            if curr.start_cycle == prev.start_cycle:
                raise ValueError(f"duplicate service segment at cycle {curr.start_cycle}")
        return segments

    def with_threshold(self, threshold: float, controller_kind: str) -> "SimulationConfig":
        normalized = controller_kind.lower().strip()
        if normalized in {"sender", "sender_outstanding", "controller1"}:
            return replace(self, sender_threshold=threshold)
        if normalized in {"noc", "noc_watermark", "controller2", "smith", "smith_pi", "pi", "controller3"}:
            return replace(self, noc_threshold=threshold)
        raise ValueError(f"threshold sweep is not defined for controller kind {controller_kind!r}")

    def with_throttle_delay(self, delay: int) -> "SimulationConfig":
        return replace(self, delays=replace(self.delays, throttle=delay))

    def with_service_step(
        self,
        low_rate: float,
        drop_cycle: int = 1000,
        recover_cycle: int = 2000,
        high_rate: Optional[float] = None,
    ) -> "SimulationConfig":
        if high_rate is None:
            high_rate = self.max_send_rate
        return replace(
            self,
            service_profile=(
                ServiceSegment(0, high_rate),
                ServiceSegment(drop_cycle, low_rate),
                ServiceSegment(recover_cycle, high_rate),
            ),
        )
