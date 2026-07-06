"""Parameter sweeps for thresholds, throttle delay, and receiver speed."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import SimulationConfig
from .controller import build_controller
from .simulate import run_simulation


def sweep_thresholds(
    config: SimulationConfig,
    thresholds: Iterable[float],
    controller_kind: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for threshold in thresholds:
        sweep_config = config.with_threshold(threshold, controller_kind)
        result = run_simulation(sweep_config, build_controller(controller_kind, sweep_config))
        rows.append({"sweep": "threshold", "threshold": threshold, **result.summary})
    return rows


def sweep_throttle_delays(
    config: SimulationConfig,
    delays: Iterable[int],
    threshold: Optional[float] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for delay in delays:
        sweep_config = config.with_throttle_delay(delay)
        if threshold is not None:
            sweep_config = replace(sweep_config, noc_threshold=threshold)
        result = run_simulation(sweep_config, build_controller("noc", sweep_config))
        rows.append({"sweep": "throttle_delay", "throttle_delay": delay, **result.summary})
    return rows


def sweep_receiver_speeds(
    config: SimulationConfig,
    low_rates: Iterable[float],
    controller_kinds: Iterable[str] = ("sender", "noc"),
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for low_rate in low_rates:
        sweep_config = config.with_service_step(low_rate=low_rate)
        for controller_kind in controller_kinds:
            result = run_simulation(sweep_config, build_controller(controller_kind, sweep_config))
            rows.append(
                {
                    "sweep": "receiver_low_rate",
                    "receiver_low_rate": low_rate,
                    "controller_kind": controller_kind,
                    **result.summary,
                }
            )
    return rows


def write_rows_csv(rows: Iterable[Dict[str, object]], path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run flow-control parameter sweeps.")
    parser.add_argument("--kind", choices=["threshold", "delay", "service"], required=True)
    parser.add_argument("--values", nargs="+", type=float, required=True)
    parser.add_argument("--controller", choices=["sender", "noc", "smith_pi"], default="noc")
    parser.add_argument("--cycles", type=int, default=3000)
    parser.add_argument("--sender-threshold", type=float, default=128.0)
    parser.add_argument("--noc-threshold", type=float, default=128.0)
    parser.add_argument("--max-send-rate", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/sweep.csv"))
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = SimulationConfig(
        total_cycles=args.cycles,
        max_send_rate=args.max_send_rate,
        sender_threshold=args.sender_threshold,
        noc_threshold=args.noc_threshold,
    )

    if args.kind == "threshold":
        rows = sweep_thresholds(config, args.values, args.controller)
    elif args.kind == "delay":
        rows = sweep_throttle_delays(config, (int(value) for value in args.values))
    else:
        rows = sweep_receiver_speeds(config, args.values)

    output = write_rows_csv(rows, args.output)
    print(f"Wrote {len(rows)} sweep rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
