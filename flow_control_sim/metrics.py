"""History recording and post-simulation metrics."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pvariance
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import SimulationConfig
from .plant import CycleRecord, EPSILON


@dataclass
class SimulationHistory:
    records: List[CycleRecord] = field(default_factory=list)

    def append(self, record: CycleRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def cycles(self) -> List[int]:
        return [record.cycle for record in self.records]

    @property
    def sender_outstanding(self) -> List[float]:
        return [record.sender_outstanding for record in self.records]

    @property
    def noc_outstanding(self) -> List[float]:
        return [record.noc_outstanding for record in self.records]

    @property
    def receiver_queue(self) -> List[float]:
        return [record.receiver_queue for record in self.records]

    @property
    def send_rate(self) -> List[float]:
        return [record.send_rate for record in self.records]

    @property
    def burst_period(self) -> List[float]:
        return [record.burst_period for record in self.records]

    @property
    def commanded_bandwidth(self) -> List[float]:
        return [record.commanded_bandwidth for record in self.records]

    @property
    def service_rate(self) -> List[float]:
        return [record.service_rate for record in self.records]

    @property
    def serviced(self) -> List[float]:
        return [record.serviced for record in self.records]

    @property
    def sent(self) -> List[float]:
        return [record.sent for record in self.records]

    @property
    def sent_reqs(self) -> List[float]:
        return [record.sent_reqs for record in self.records]

    @property
    def serviced_reqs(self) -> List[float]:
        return [record.serviced_reqs for record in self.records]

    @property
    def noc_queue(self) -> List[float]:
        return [record.noc_queue for record in self.records]

    @property
    def forward_sn_occupancy(self) -> List[float]:
        return [record.forward_sn_occupancy for record in self.records]

    @property
    def pending_noc_arrivals(self) -> List[float]:
        return [record.pending_noc_arrivals for record in self.records]

    @property
    def forward_nr_occupancy(self) -> List[float]:
        return [record.forward_nr_occupancy for record in self.records]

    @property
    def return_rn_occupancy(self) -> List[float]:
        return [record.return_rn_occupancy for record in self.records]

    @property
    def return_ns_occupancy(self) -> List[float]:
        return [record.return_ns_occupancy for record in self.records]

    @property
    def sender_noc_inflight(self) -> List[float]:
        return [record.sender_noc_inflight for record in self.records]

    @property
    def sender_noc_os_gap(self) -> List[float]:
        return [record.sender_outstanding - record.noc_outstanding for record in self.records]

    @property
    def raw_throttle(self) -> List[bool]:
        return [record.raw_throttle for record in self.records]

    @property
    def delayed_throttle(self) -> List[bool]:
        return [record.delayed_throttle for record in self.records]


FIELDNAMES = [
    "cycle",
    "send_rate",
    "sent",
    "burst_period",
    "commanded_bandwidth",
    "service_rate",
    "serviced",
    "sender_outstanding",
    "noc_outstanding",
    "receiver_queue",
    "arrival_noc",
    "arrival_receiver",
    "response_noc",
    "response_sender",
    "raw_throttle",
    "delayed_throttle",
    "queue_over_capacity",
    "sent_reqs",
    "sent_bytes",
    "serviced_reqs",
    "serviced_bytes",
    "forward_sn_occupancy",
    "pending_noc_arrivals",
    "noc_queue",
    "forward_nr_occupancy",
    "return_rn_occupancy",
    "return_ns_occupancy",
    "sender_noc_inflight",
    "sender_noc_os_gap",
    "noc_queue_over_capacity",
    "sender_inflight_over_capacity",
    "sender_backpressure",
    "sender_check",
    "burst_active",
]


def summarize_history(
    history: SimulationHistory,
    config: SimulationConfig,
    controller_name: str,
    receiver_change_cycles: Sequence[int],
) -> Dict[str, Optional[float]]:
    if not history.records:
        raise ValueError("history is empty")

    total_cycles = len(history)
    total_sent = sum(history.sent)
    total_serviced = sum(history.serviced)
    total_sent_reqs = sum(history.sent_reqs)
    total_serviced_reqs = sum(history.serviced_reqs)
    max_sender = max(history.sender_outstanding)
    max_noc = max(history.noc_outstanding)
    max_queue = max(history.receiver_queue)
    max_noc_queue = max(history.noc_queue)
    max_sender_noc_inflight = max(history.sender_noc_inflight)
    max_sender_noc_os_gap = max(history.sender_noc_os_gap)
    max_burst_period = max(history.burst_period)
    positive_commanded_bandwidth = [value for value in history.commanded_bandwidth if value > EPSILON]
    tail_window = min(config.oscillation_window, total_cycles)
    tail_sender = history.sender_outstanding[-tail_window:]
    tail_noc = history.noc_outstanding[-tail_window:]
    tail_queue = history.receiver_queue[-tail_window:]

    settling_start = receiver_change_cycles[-1] if receiver_change_cycles else 0
    sender_settling = settling_time(
        cycles=history.cycles,
        values=history.sender_outstanding,
        start_cycle=settling_start,
        stable_window=config.settling_window,
        tail_window=config.settling_tail_window,
        rel_tolerance=config.settling_rel_tolerance,
        abs_tolerance=config.settling_abs_tolerance,
        threshold=config.sender_threshold,
    )
    noc_settling = settling_time(
        cycles=history.cycles,
        values=history.noc_outstanding,
        start_cycle=settling_start,
        stable_window=config.settling_window,
        tail_window=config.settling_tail_window,
        rel_tolerance=config.settling_rel_tolerance,
        abs_tolerance=config.settling_abs_tolerance,
        threshold=config.noc_threshold,
    )

    sender_overshoot = max(0.0, max_sender - config.sender_threshold)
    noc_overshoot = max(0.0, max_noc - config.noc_threshold)
    queue_over_capacity_cycles = sum(1 for record in history.records if record.queue_over_capacity)
    noc_queue_over_capacity_cycles = sum(1 for record in history.records if record.noc_queue_over_capacity)
    sender_inflight_over_capacity_cycles = sum(
        1 for record in history.records if record.sender_inflight_over_capacity
    )
    sender_backpressure_cycles = sum(1 for record in history.records if record.sender_backpressure)

    return {
        "controller": controller_name,
        "total_cycles": float(total_cycles),
        "total_sent": total_sent,
        "total_serviced": total_serviced,
        "total_sent_bytes": total_sent,
        "total_serviced_bytes": total_serviced,
        "total_sent_reqs": total_sent_reqs,
        "total_serviced_reqs": total_serviced_reqs,
        "throughput": total_serviced / total_cycles,
        "throughput_reqs_per_cycle": total_serviced_reqs / total_cycles,
        "offered_load": total_sent / total_cycles,
        "offered_reqs_per_cycle": total_sent_reqs / total_cycles,
        "mean_commanded_bandwidth": mean(history.commanded_bandwidth),
        "min_positive_commanded_bandwidth": (
            min(positive_commanded_bandwidth) if positive_commanded_bandwidth else 0.0
        ),
        "max_commanded_bandwidth": max(history.commanded_bandwidth),
        "max_burst_period": max_burst_period,
        "idle_ratio": sum(1 for value in history.sent if value <= EPSILON) / total_cycles,
        "raw_throttle_ratio": sum(1 for value in history.raw_throttle if value) / total_cycles,
        "delayed_throttle_ratio": sum(1 for value in history.delayed_throttle if value) / total_cycles,
        "max_sender_outstanding": max_sender,
        "max_noc_outstanding": max_noc,
        "max_queue": max_queue,
        "max_noc_queue": max_noc_queue,
        "max_sender_noc_inflight": max_sender_noc_inflight,
        "max_sender_noc_os_gap": max_sender_noc_os_gap,
        "mean_queue": mean(history.receiver_queue),
        "mean_noc_queue": mean(history.noc_queue),
        "mean_sender_noc_inflight": mean(history.sender_noc_inflight),
        "mean_sender_noc_os_gap": mean(history.sender_noc_os_gap),
        "sender_overshoot": sender_overshoot,
        "sender_overshoot_pct": sender_overshoot / config.sender_threshold,
        "noc_overshoot": noc_overshoot,
        "noc_overshoot_pct": noc_overshoot / config.noc_threshold,
        "queue_capacity": config.queue_capacity,
        "queue_over_capacity_cycles": float(queue_over_capacity_cycles),
        "noc_queue_capacity_reqs": float(config.traffic.noc_queue_capacity_reqs),
        "sender_noc_inflight_capacity_reqs": float(config.traffic.sender_noc_inflight_capacity_reqs),
        "noc_queue_over_capacity_cycles": float(noc_queue_over_capacity_cycles),
        "sender_inflight_over_capacity_cycles": float(sender_inflight_over_capacity_cycles),
        "sender_backpressure_cycles": float(sender_backpressure_cycles),
        "sender_backpressure_ratio": sender_backpressure_cycles / total_cycles,
        "send_rate_settle_to_receiver_time": send_rate_settle_time(
            cycles=history.cycles,
            send_bytes=history.sent,
            service_rate=history.service_rate,
            start_cycle=receiver_change_cycles[1] if len(receiver_change_cycles) > 1 else 0,
            window=config.traffic.sender_burst_period,
            hold_window=config.traffic.sender_burst_period,
        ),
        "sender_settling_time": sender_settling,
        "noc_settling_time": noc_settling,
        "tail_sender_variance": safe_pvariance(tail_sender),
        "tail_noc_variance": safe_pvariance(tail_noc),
        "tail_queue_variance": safe_pvariance(tail_queue),
        "tail_queue_delta": tail_queue[-1] - tail_queue[0] if tail_queue else 0.0,
    }


def settling_time(
    cycles: Sequence[int],
    values: Sequence[float],
    start_cycle: int,
    stable_window: int,
    tail_window: int,
    rel_tolerance: float,
    abs_tolerance: float,
    threshold: float,
) -> Optional[float]:
    if not cycles or not values or len(cycles) != len(values):
        return None
    if len(values) < stable_window:
        return None

    tail = values[-min(tail_window, len(values)) :]
    target = mean(tail)
    tolerance = max(abs_tolerance, abs(target) * rel_tolerance, threshold * rel_tolerance)

    start_idx = 0
    while start_idx < len(cycles) and cycles[start_idx] < start_cycle:
        start_idx += 1

    end_idx = len(values) - stable_window + 1
    for idx in range(start_idx, end_idx):
        window = values[idx : idx + stable_window]
        if all(abs(value - target) <= tolerance for value in window):
            return float(cycles[idx] - start_cycle)
    return None


def safe_pvariance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return pvariance(values)


def send_rate_settle_time(
    cycles: Sequence[int],
    send_bytes: Sequence[float],
    service_rate: Sequence[float],
    start_cycle: int,
    window: int,
    hold_window: int,
) -> Optional[float]:
    if not cycles or not send_bytes or not service_rate:
        return None
    avg_send = fixed_window_average(send_bytes, window)
    start_idx = 0
    while start_idx < len(cycles) and cycles[start_idx] < start_cycle:
        start_idx += 1
    end_idx = len(cycles) - hold_window + 1
    for idx in range(start_idx, max(start_idx, end_idx)):
        if idx + hold_window > len(cycles):
            break
        if all(avg_send[j] <= service_rate[j] + EPSILON for j in range(idx, idx + hold_window)):
            return float(cycles[idx] - start_cycle)
    return None


def fixed_window_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 0:
        raise ValueError("window must be > 0")
    result: List[float] = []
    running = 0.0
    fifo: List[float] = []
    for value in values:
        fifo.append(value)
        running += value
        if len(fifo) > window:
            running -= fifo.pop(0)
        result.append(running / window)
    return result


def write_history_csv(history: SimulationHistory, path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in history.records:
            writer.writerow(asdict(record))
    return output


def read_history_csv(path: Path | str) -> SimulationHistory:
    history = SimulationHistory()
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            history.append(
                CycleRecord(
                    cycle=int(row["cycle"]),
                    send_rate=float(row["send_rate"]),
                    sent=float(row["sent"]),
                    burst_period=_float_row(row, "burst_period"),
                    commanded_bandwidth=_float_row(row, "commanded_bandwidth"),
                    service_rate=float(row["service_rate"]),
                    serviced=float(row["serviced"]),
                    sender_outstanding=float(row["sender_outstanding"]),
                    noc_outstanding=float(row["noc_outstanding"]),
                    receiver_queue=float(row["receiver_queue"]),
                    arrival_noc=float(row["arrival_noc"]),
                    arrival_receiver=float(row["arrival_receiver"]),
                    response_noc=float(row["response_noc"]),
                    response_sender=float(row["response_sender"]),
                    raw_throttle=_parse_bool(row["raw_throttle"]),
                    delayed_throttle=_parse_bool(row["delayed_throttle"]),
                    queue_over_capacity=_parse_bool(row["queue_over_capacity"]),
                    sent_reqs=_float_row(row, "sent_reqs"),
                    sent_bytes=_float_row(row, "sent_bytes", fallback="sent"),
                    serviced_reqs=_float_row(row, "serviced_reqs"),
                    serviced_bytes=_float_row(row, "serviced_bytes", fallback="serviced"),
                    forward_sn_occupancy=_float_row(row, "forward_sn_occupancy"),
                    pending_noc_arrivals=_float_row(row, "pending_noc_arrivals"),
                    noc_queue=_float_row(row, "noc_queue"),
                    forward_nr_occupancy=_float_row(row, "forward_nr_occupancy"),
                    return_rn_occupancy=_float_row(row, "return_rn_occupancy"),
                    return_ns_occupancy=_float_row(row, "return_ns_occupancy"),
                    sender_noc_inflight=_float_row(row, "sender_noc_inflight"),
                    sender_noc_os_gap=_float_row(row, "sender_noc_os_gap"),
                    noc_queue_over_capacity=_bool_row(row, "noc_queue_over_capacity"),
                    sender_inflight_over_capacity=_bool_row(row, "sender_inflight_over_capacity"),
                    sender_backpressure=_bool_row(row, "sender_backpressure"),
                    sender_check=_bool_row(row, "sender_check"),
                    burst_active=_bool_row(row, "burst_active"),
                )
            )
    return history


def write_summary_json(summary: Dict[str, object], path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def write_summary_csv(summaries: Iterable[Dict[str, object]], path: Path | str) -> Path:
    rows = list(summaries)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output.open("w", encoding="utf-8") as handle:
            handle.write("")
        return output
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _float_row(row: Dict[str, str], key: str, fallback: Optional[str] = None) -> float:
    value = row.get(key)
    if value in (None, "") and fallback is not None:
        value = row.get(fallback)
    if value in (None, ""):
        return 0.0
    return float(value)


def _bool_row(row: Dict[str, str], key: str) -> bool:
    value = row.get(key)
    if value is None:
        return False
    return _parse_bool(value)
