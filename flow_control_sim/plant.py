"""Plant model: fluid sender bandwidth, NOC queue, delay pipelines, and receiver polling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from .config import DelayConfig, TrafficConfig
from .controller import ControlDecision

EPSILON = 1e-9


class DelayFifo:
    """Fixed-cycle delay line with zero initial fill."""

    def __init__(self, delay: int, initial: float = 0.0) -> None:
        if delay < 0:
            raise ValueError("delay must be >= 0")
        self.delay = delay
        self._fifo: Deque[float] = deque([initial] * delay)

    def push(self, value: float) -> float:
        if self.delay == 0:
            return value
        self._fifo.append(value)
        return self._fifo.popleft()

    def reset(self, initial: float = 0.0) -> None:
        self._fifo = deque([initial] * self.delay)

    @property
    def occupancy(self) -> float:
        return sum(self._fifo)


class PollingDelayQueue:
    """Variable NOC queue delay caused by a round-robin polling slot.

    An accepted request waits until the next poll slot. With period N, the
    residence delay ranges from 0 to N-1 cycles depending on arrival phase.
    """

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be > 0")
        self.period = period
        self._buckets = [0.0] * period

    def push(self, value: float, cycle: int) -> float:
        if cycle < 0:
            raise ValueError("cycle must be >= 0")
        idx = cycle % self.period
        released = self._buckets[idx]
        self._buckets[idx] = 0.0

        wait = (-cycle) % self.period
        if wait == 0:
            released += value
        else:
            self._buckets[(cycle + wait) % self.period] += value
        return released

    def reset(self) -> None:
        self._buckets = [0.0] * self.period

    @property
    def occupancy(self) -> float:
        return sum(self._buckets)


@dataclass(frozen=True)
class PlantState:
    cycle: int
    sender_outstanding: float
    noc_outstanding: float
    receiver_queue: float
    noc_queue: float
    sender_noc_inflight: float


@dataclass(frozen=True)
class CycleRecord:
    cycle: int
    send_rate: float
    sent: float
    burst_period: float
    commanded_bandwidth: float
    service_rate: float
    serviced: float
    sender_outstanding: float
    noc_outstanding: float
    receiver_queue: float
    arrival_noc: float
    arrival_receiver: float
    response_noc: float
    response_sender: float
    raw_throttle: bool
    delayed_throttle: bool
    queue_over_capacity: bool
    sent_reqs: float = 0.0
    sent_bytes: float = 0.0
    serviced_reqs: float = 0.0
    serviced_bytes: float = 0.0
    forward_sn_occupancy: float = 0.0
    pending_noc_arrivals: float = 0.0
    noc_queue: float = 0.0
    forward_nr_occupancy: float = 0.0
    return_rn_occupancy: float = 0.0
    return_ns_occupancy: float = 0.0
    sender_noc_inflight: float = 0.0
    sender_noc_os_gap: float = 0.0
    noc_queue_over_capacity: bool = False
    sender_inflight_over_capacity: bool = False
    sender_backpressure: bool = False
    sender_check: bool = False
    burst_active: bool = False


class FlowControlPlant:
    """Cycle-accurate plant update using request-count state variables.

    Sender traffic can be modeled either as a fluid equivalent bandwidth or as
    discrete bursts. In burst mode, the sender emits one request per cycle for
    ``sender_burst_reqs`` consecutive cycles, then waits until the next
    ``burst_period`` slot.
    """

    def __init__(
        self,
        delays: DelayConfig,
        max_send_rate: float,
        traffic: Optional[TrafficConfig] = None,
        queue_capacity: Optional[float] = None,
    ) -> None:
        self.delays = delays
        self.max_send_rate = max_send_rate
        self.traffic = traffic or TrafficConfig()
        self.queue_capacity = queue_capacity

        self.forward_sn = DelayFifo(delays.forward_sn)
        self.forward_nr = DelayFifo(delays.forward_nr)
        self.return_rn = DelayFifo(delays.return_rn)
        self.return_ns = DelayFifo(delays.return_ns)
        self.noc_queue_delay = DelayFifo(self.traffic.noc_queue_delay)
        self.noc_poll_queue: Optional[PollingDelayQueue] = (
            PollingDelayQueue(self.traffic.noc_queue_poll_period)
            if self.traffic.noc_queue_poll_period > 0
            else None
        )

        self.sender_outstanding = 0.0
        self.noc_outstanding = 0.0
        self.receiver_queue = 0.0
        self.noc_queue = 0.0
        self.sender_noc_inflight = 0.0
        self._pending_noc_arrivals = 0.0
        self._receiver_credit_bytes = 0.0
        self._burst_remaining = 0
        self._next_burst_cycle = 0
        self.cycle = 0

    def reset(self) -> None:
        self.forward_sn.reset()
        self.forward_nr.reset()
        self.return_rn.reset()
        self.return_ns.reset()
        self.noc_queue_delay.reset()
        if self.noc_poll_queue is not None:
            self.noc_poll_queue.reset()
        self.sender_outstanding = 0.0
        self.noc_outstanding = 0.0
        self.receiver_queue = 0.0
        self.noc_queue = 0.0
        self.sender_noc_inflight = 0.0
        self._pending_noc_arrivals = 0.0
        self._receiver_credit_bytes = 0.0
        self._burst_remaining = 0
        self._next_burst_cycle = 0
        self.cycle = 0

    def snapshot(self) -> PlantState:
        return PlantState(
            cycle=self.cycle,
            sender_outstanding=self.sender_outstanding,
            noc_outstanding=self.noc_outstanding,
            receiver_queue=self.receiver_queue,
            noc_queue=self.noc_queue,
            sender_noc_inflight=self.sender_noc_inflight,
        )

    def step(self, decision: ControlDecision, service_rate: float) -> CycleRecord:
        """Execute one cycle in the fixed plant order."""

        if service_rate < 0:
            raise ValueError("service_rate must be >= 0")

        sender_check = self.cycle % self.traffic.sender_check_interval == 0
        sender_backpressure = self.sender_noc_inflight >= (
            self.traffic.sender_noc_inflight_capacity_reqs - EPSILON
        )
        controller_allows = decision.send_rate > EPSILON
        burst_period = self._burst_period_from_decision(decision)
        equivalent_bandwidth = self.traffic.sender_burst_reqs * self.traffic.request_bytes / burst_period
        commanded_bandwidth = min(
            self.max_send_rate,
            decision.send_rate,
            equivalent_bandwidth,
        ) if controller_allows else 0.0

        sender_capacity_reqs = max(
            0.0,
            self.traffic.sender_noc_inflight_capacity_reqs - self.sender_noc_inflight,
        )
        sent_reqs = 0.0
        if controller_allows and not sender_backpressure:
            max_allowed_reqs = sender_capacity_reqs
            if decision.sender_outstanding_limit is not None:
                sender_os_credit = max(0.0, decision.sender_outstanding_limit - self.sender_outstanding)
                max_allowed_reqs = min(max_allowed_reqs, sender_os_credit)
            desired_reqs = self._desired_sender_reqs(commanded_bandwidth, burst_period, max_allowed_reqs)
            sent_reqs = min(desired_reqs, max_allowed_reqs)

        sent_bytes = sent_reqs * self.traffic.request_bytes
        self.sender_outstanding += sent_reqs
        self.sender_noc_inflight += sent_reqs

        # Sender -> Forward SN -> NOC.
        arrival_noc = self.forward_sn.push(sent_reqs)
        self._pending_noc_arrivals += arrival_noc

        # NOC accepts as many pending upstream arrivals as it has queue space.
        # Requests are counted in NOC OS as soon as they enter the NOC queue,
        # not only after they leave the queue toward the receiver.
        self.noc_queue = self._noc_queue_occupancy()
        noc_space = max(0.0, self.traffic.noc_queue_capacity_reqs - self.noc_queue)
        accepted_noc = min(
            self._pending_noc_arrivals,
            noc_space,
            float(self.traffic.noc_issue_reqs_per_cycle),
        )
        self._pending_noc_arrivals = _clean_non_negative(self._pending_noc_arrivals - accepted_noc)
        self.sender_noc_inflight = _clean_non_negative(self.sender_noc_inflight - accepted_noc)
        self.noc_outstanding += accepted_noc

        # NOC queue residence delay -> Forward NR. Fixed delay and polling
        # delay are modeled in series, so a 64-cycle FIFO plus a 64-cycle poll
        # slot produces a 64..127-cycle residence time.
        noc_issued = self._push_noc_queue(accepted_noc)
        self.noc_queue = self._noc_queue_occupancy()
        arrival_receiver = self.forward_nr.push(noc_issued)

        # Receiver queue and polling service. Service rate is bytes/cycle; each
        # completed poll consumes one fixed-size request.
        self.receiver_queue += arrival_receiver
        if self.receiver_queue > EPSILON:
            self._receiver_credit_bytes += service_rate
        else:
            self._receiver_credit_bytes = 0.0
        serviced_reqs = 0.0
        if self.receiver_queue >= 1.0 - EPSILON and self._receiver_credit_bytes >= self.traffic.request_bytes - EPSILON:
            serviced_reqs = 1.0
            self.receiver_queue = _clean_non_negative(self.receiver_queue - serviced_reqs)
            self._receiver_credit_bytes = _clean_non_negative(
                self._receiver_credit_bytes - self.traffic.request_bytes
            )
            if self.receiver_queue <= EPSILON:
                self._receiver_credit_bytes = 0.0

        serviced_bytes = serviced_reqs * self.traffic.request_bytes

        # Receiver -> Return RN -> NOC.
        response_noc = self.return_rn.push(serviced_reqs)

        # NOC response side.
        self.noc_outstanding = _clean_non_negative(self.noc_outstanding - response_noc)
        response_sender = self.return_ns.push(response_noc)

        # Return NS -> Sender.
        self.sender_outstanding = _clean_non_negative(self.sender_outstanding - response_sender)

        receiver_over_capacity = (
            self.queue_capacity is not None and self.receiver_queue > self.queue_capacity + EPSILON
        )
        noc_over_capacity = self.noc_queue > self.traffic.noc_queue_capacity_reqs + EPSILON
        inflight_over_capacity = (
            self.sender_noc_inflight > self.traffic.sender_noc_inflight_capacity_reqs + EPSILON
        )
        sender_noc_os_gap = _clean_non_negative(self.sender_outstanding - self.noc_outstanding)

        record = CycleRecord(
            cycle=self.cycle,
            send_rate=sent_bytes,
            sent=sent_bytes,
            burst_period=burst_period,
            commanded_bandwidth=commanded_bandwidth,
            service_rate=service_rate,
            serviced=serviced_bytes,
            sender_outstanding=self.sender_outstanding,
            noc_outstanding=self.noc_outstanding,
            receiver_queue=self.receiver_queue,
            arrival_noc=accepted_noc,
            arrival_receiver=arrival_receiver,
            response_noc=response_noc,
            response_sender=response_sender,
            raw_throttle=decision.raw_throttle,
            delayed_throttle=decision.delayed_throttle,
            queue_over_capacity=receiver_over_capacity,
            sent_reqs=sent_reqs,
            sent_bytes=sent_bytes,
            serviced_reqs=serviced_reqs,
            serviced_bytes=serviced_bytes,
            forward_sn_occupancy=self.forward_sn.occupancy,
            pending_noc_arrivals=self._pending_noc_arrivals,
            noc_queue=self.noc_queue,
            forward_nr_occupancy=self.forward_nr.occupancy,
            return_rn_occupancy=self.return_rn.occupancy,
            return_ns_occupancy=self.return_ns.occupancy,
            sender_noc_inflight=self.sender_noc_inflight,
            sender_noc_os_gap=sender_noc_os_gap,
            noc_queue_over_capacity=noc_over_capacity,
            sender_inflight_over_capacity=inflight_over_capacity,
            sender_backpressure=sender_backpressure,
            sender_check=sender_check,
            burst_active=sent_reqs > EPSILON,
        )
        self.cycle += 1
        return record

    def _burst_period_from_decision(self, decision: ControlDecision) -> float:
        period = (
            float(self.traffic.sender_burst_period)
            if decision.burst_period is None
            else float(decision.burst_period)
        )
        if period <= 0:
            raise ValueError("burst_period must be > 0")
        if period < self.traffic.sender_burst_reqs:
            raise ValueError("burst_period must be >= sender_burst_reqs")
        return period

    def _desired_sender_reqs(
        self,
        commanded_bandwidth: float,
        burst_period: float,
        max_allowed_reqs: float,
    ) -> float:
        if self.traffic.sender_mode == "fluid":
            return commanded_bandwidth / self.traffic.request_bytes
        if max_allowed_reqs < 1.0 - EPSILON:
            return 0.0

        period = max(self.traffic.sender_burst_reqs, int(round(burst_period)))
        if self._burst_remaining <= 0 and self.cycle >= self._next_burst_cycle:
            self._burst_remaining = self.traffic.sender_burst_reqs
            self._next_burst_cycle = self.cycle + period
        if self._burst_remaining <= 0:
            return 0.0

        self._burst_remaining -= 1
        return 1.0

    def _push_noc_queue(self, accepted_noc: float) -> float:
        delayed = self.noc_queue_delay.push(accepted_noc)
        if self.noc_poll_queue is not None:
            return self.noc_poll_queue.push(delayed, self.cycle)
        return delayed

    def _noc_queue_occupancy(self) -> float:
        fixed_occupancy = self.noc_queue_delay.occupancy
        if self.noc_poll_queue is not None:
            return fixed_occupancy + self.noc_poll_queue.occupancy
        return fixed_occupancy


def _clean_non_negative(value: float) -> float:
    if -EPSILON < value < EPSILON:
        return 0.0
    if value < 0:
        raise RuntimeError(f"plant counter became negative: {value}")
    return value
