"""Single-run simulation entry points and CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from .config import DelayConfig, ServiceSegment, SimulationConfig, TrafficConfig
from .controller import Controller, build_controller
from .metrics import (
    SimulationHistory,
    summarize_history,
    write_history_csv,
    write_summary_csv,
    write_summary_json,
)
from .plant import FlowControlPlant
from .receiver import ReceiverProfile, StepReceiver


@dataclass
class SimulationResult:
    config: SimulationConfig
    controller_name: str
    history: SimulationHistory
    summary: Dict[str, object]


def run_simulation(
    config: SimulationConfig,
    controller: Controller,
    receiver: Optional[ReceiverProfile] = None,
) -> SimulationResult:
    plant = FlowControlPlant(
        delays=config.delays,
        max_send_rate=config.max_send_rate,
        traffic=config.traffic,
        queue_capacity=config.queue_capacity,
    )
    receiver_profile = receiver or StepReceiver.from_config(config)

    plant.reset()
    controller.reset()
    history = SimulationHistory()

    for _ in range(config.total_cycles):
        state = plant.snapshot()
        decision = controller.update(state)
        service_rate = receiver_profile.rate(state.cycle)
        record = plant.step(decision, service_rate)
        history.append(record)

    summary = summarize_history(
        history=history,
        config=config,
        controller_name=controller.name,
        receiver_change_cycles=receiver_profile.change_cycles,
    )
    return SimulationResult(
        config=config,
        controller_name=controller.name,
        history=history,
        summary=summary,
    )


def run_comparison(
    config: SimulationConfig,
    controller_kinds: Iterable[str] = ("sender", "noc"),
) -> Dict[str, SimulationResult]:
    results: Dict[str, SimulationResult] = {}
    for kind in controller_kinds:
        controller = build_controller(kind, config)
        results[kind] = run_simulation(config, controller)
    return results


def save_results(results: Mapping[str, SimulationResult], output_dir: Path | str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for label, result in results.items():
        write_history_csv(result.history, output / f"{label}.csv")
        write_summary_json(result.summary, output / f"{label}_summary.json")
        summaries.append(result.summary)
    write_summary_csv(summaries, output / "comparison_summary.csv")


def build_config_from_args(args: argparse.Namespace) -> SimulationConfig:
    delays = DelayConfig(
        forward_sn=args.forward_sn_delay,
        forward_nr=args.forward_nr_delay,
        return_rn=args.return_rn_delay,
        return_ns=args.return_ns_delay,
        throttle=args.throttle_delay,
    )
    profile = (
        ServiceSegment(0, args.high_rate),
        ServiceSegment(args.drop_cycle, args.low_rate),
        ServiceSegment(args.recover_cycle, args.high_rate),
    )
    return SimulationConfig(
        total_cycles=args.cycles,
        max_send_rate=args.max_send_rate,
        sender_threshold=args.sender_threshold,
        noc_threshold=args.noc_threshold,
        queue_capacity=args.queue_capacity,
        delays=delays,
        traffic=TrafficConfig(
            request_bytes=args.request_bytes,
            sender_burst_reqs=args.sender_burst_reqs,
            sender_check_interval=args.sender_check_interval,
            sender_burst_period=args.sender_burst_period,
            sender_mode=args.sender_mode,
            noc_queue_capacity_reqs=args.noc_queue_capacity_reqs,
            sender_noc_inflight_capacity_reqs=args.sender_noc_inflight_capacity_reqs,
            noc_issue_reqs_per_cycle=args.noc_issue_reqs_per_cycle,
            noc_queue_delay=args.noc_queue_delay,
            noc_queue_poll_period=args.noc_queue_poll_period,
        ),
        service_profile=profile,
    )


def print_summary_table(results: Mapping[str, SimulationResult]) -> None:
    columns = [
        "controller",
        "max_sender_outstanding",
        "max_noc_outstanding",
        "max_queue",
        "max_noc_queue",
        "max_sender_noc_os_gap",
        "max_sender_noc_inflight",
        "sender_overshoot",
        "noc_overshoot",
        "throughput",
        "throughput_reqs_per_cycle",
        "idle_ratio",
        "sender_backpressure_ratio",
        "sender_settling_time",
        "tail_noc_variance",
    ]
    rows = [result.summary for result in results.values()]
    widths = {
        column: max(len(column), *(len(_format_value(row.get(column))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(_format_value(row.get(column)).ljust(widths[column]) for column in columns))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run flow-control simulations.")
    parser.add_argument("--controller", choices=["open", "sender", "noc", "p", "smith_pi", "all"], default="all")
    parser.add_argument("--cycles", type=int, default=3000)
    parser.add_argument("--max-send-rate", type=float, default=3.0)
    parser.add_argument("--sender-threshold", type=float, default=128.0)
    parser.add_argument("--noc-threshold", type=float, default=128.0)
    parser.add_argument("--queue-capacity", type=float, default=None)
    parser.add_argument("--high-rate", type=float, default=3.0)
    parser.add_argument("--low-rate", type=float, default=1.0)
    parser.add_argument("--drop-cycle", type=int, default=1000)
    parser.add_argument("--recover-cycle", type=int, default=2000)
    parser.add_argument("--forward-sn-delay", type=int, default=DelayConfig.forward_sn)
    parser.add_argument("--forward-nr-delay", type=int, default=DelayConfig.forward_nr)
    parser.add_argument("--return-rn-delay", type=int, default=DelayConfig.return_rn)
    parser.add_argument("--return-ns-delay", type=int, default=DelayConfig.return_ns)
    parser.add_argument("--throttle-delay", type=int, default=DelayConfig.throttle)
    parser.add_argument("--request-bytes", type=int, default=TrafficConfig.request_bytes)
    parser.add_argument("--sender-burst-reqs", type=int, default=TrafficConfig.sender_burst_reqs)
    parser.add_argument("--sender-check-interval", type=int, default=TrafficConfig.sender_check_interval)
    parser.add_argument("--sender-burst-period", type=int, default=TrafficConfig.sender_burst_period)
    parser.add_argument("--sender-mode", choices=["fluid", "burst"], default=TrafficConfig.sender_mode)
    parser.add_argument("--noc-queue-capacity-reqs", type=int, default=TrafficConfig.noc_queue_capacity_reqs)
    parser.add_argument(
        "--sender-noc-inflight-capacity-reqs",
        type=int,
        default=TrafficConfig.sender_noc_inflight_capacity_reqs,
    )
    parser.add_argument("--noc-issue-reqs-per-cycle", type=int, default=TrafficConfig.noc_issue_reqs_per_cycle)
    parser.add_argument("--noc-queue-delay", type=int, default=TrafficConfig.noc_queue_delay)
    parser.add_argument("--noc-queue-poll-period", type=int, default=TrafficConfig.noc_queue_poll_period)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = build_config_from_args(args)
    kinds = ("sender", "noc") if args.controller == "all" else (args.controller,)
    results = run_comparison(config, kinds)
    save_results(results, args.output_dir)
    print_summary_table(results)
    print(f"\nWrote CSV and JSON outputs to {args.output_dir}")
    return 0


def _format_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
