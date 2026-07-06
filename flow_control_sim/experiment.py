"""YAML/JSON experiment runner with automatic strategy comparison."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import DelayConfig, ServiceSegment, SimulationConfig, TrafficConfig
from .controller import build_controller
from .metrics import write_history_csv, write_summary_json
from .simulate import SimulationResult, run_simulation


@dataclass(frozen=True)
class ControllerSpec:
    kind: str
    label: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlotConfig:
    enabled: bool = True
    mode: str = "comparison"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: Path
    simulation: SimulationConfig
    controllers: Tuple[ControllerSpec, ...] = field(default_factory=tuple)
    plots: PlotConfig = field(default_factory=PlotConfig)


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    results: Dict[str, SimulationResult]
    output_dir: Path
    long_summary_path: Path
    wide_summary_path: Path
    markdown_summary_path: Path
    plot_paths: List[Path] = field(default_factory=list)


COMPARISON_METRICS: Tuple[Tuple[str, str], ...] = (
    ("最大 Sender Outstanding", "max_sender_outstanding"),
    ("最大 NOC Outstanding", "max_noc_outstanding"),
    ("最大 Queue", "max_queue"),
    ("最大 NOC Queue", "max_noc_queue"),
    ("最大 Sender-NOC Inflight", "max_sender_noc_os_gap"),
    ("最大物理前向 Inflight", "max_sender_noc_inflight"),
    ("Sender Overshoot", "sender_overshoot"),
    ("NOC Overshoot", "noc_overshoot"),
    ("Throughput", "throughput"),
    ("Throughput Req/Cycle", "throughput_reqs_per_cycle"),
    ("平均命令带宽", "mean_commanded_bandwidth"),
    ("最小正命令带宽", "min_positive_commanded_bandwidth"),
    ("最大 Burst Period", "max_burst_period"),
    ("Sender Settling Time", "sender_settling_time"),
    ("NOC Settling Time", "noc_settling_time"),
    ("Send Rate <= Receiver Time", "send_rate_settle_to_receiver_time"),
    ("Idle Ratio", "idle_ratio"),
    ("Sender Backpressure Ratio", "sender_backpressure_ratio"),
)


def load_experiment(path: Path | str) -> ExperimentConfig:
    config_path = Path(path)
    data = _load_config_mapping(config_path)
    return experiment_from_mapping(data, base_dir=config_path.parent)


def experiment_from_mapping(data: Mapping[str, Any], base_dir: Path | str = Path(".")) -> ExperimentConfig:
    base = Path(base_dir)
    name = str(data.get("name", "flow_control_experiment"))

    outputs = _mapping(data.get("outputs", {}), "outputs")
    output_dir_value = data.get("output_dir", outputs.get("output_dir", Path("outputs") / name))
    output_dir = _resolve_path(output_dir_value, base)

    simulation = simulation_from_mapping(_mapping(data.get("simulation", {}), "simulation"))
    controllers = tuple(_parse_controllers(data.get("controllers", ("noc", "sender"))))
    if not controllers:
        raise ValueError("experiment must define at least one controller")

    plots = PlotConfig(
        enabled=bool(outputs.get("plots", True)),
        mode=str(outputs.get("plot_mode", "comparison")),
    )
    if plots.mode not in {"basic", "os", "state", "path", "rate", "control", "comparison", "all"}:
        raise ValueError(
            "outputs.plot_mode must be one of: basic, os, state, path, rate, control, comparison, all"
        )

    return ExperimentConfig(
        name=name,
        output_dir=output_dir,
        simulation=simulation,
        controllers=controllers,
        plots=plots,
    )


def simulation_from_mapping(data: Mapping[str, Any]) -> SimulationConfig:
    thresholds = _mapping(data.get("thresholds", {}), "simulation.thresholds")
    delays_data = _mapping(data.get("delays", {}), "simulation.delays")
    traffic_data = _mapping(data.get("traffic", {}), "simulation.traffic")
    receiver_data = _mapping(data.get("receiver", {}), "simulation.receiver")
    service_profile_value = data.get("service_profile", receiver_data.get("service_profile", receiver_data.get("profile")))

    delays = DelayConfig(
        forward_sn=_int_value(delays_data.get("forward_sn", DelayConfig.forward_sn), "delays.forward_sn"),
        forward_nr=_int_value(delays_data.get("forward_nr", DelayConfig.forward_nr), "delays.forward_nr"),
        return_rn=_int_value(delays_data.get("return_rn", DelayConfig.return_rn), "delays.return_rn"),
        return_ns=_int_value(delays_data.get("return_ns", DelayConfig.return_ns), "delays.return_ns"),
        throttle=_int_value(delays_data.get("throttle", DelayConfig.throttle), "delays.throttle"),
    )

    return SimulationConfig(
        total_cycles=_int_value(data.get("total_cycles", data.get("cycles", 3000)), "simulation.total_cycles"),
        max_send_rate=_float_value(data.get("max_send_rate", 3.0), "simulation.max_send_rate"),
        sender_threshold=_float_value(
            data.get("sender_threshold", thresholds.get("sender", 128.0)),
            "simulation.sender_threshold",
        ),
        noc_threshold=_float_value(
            data.get("noc_threshold", thresholds.get("noc", 128.0)),
            "simulation.noc_threshold",
        ),
        queue_capacity=_optional_float(data.get("queue_capacity"), "simulation.queue_capacity"),
        delays=delays,
        traffic=TrafficConfig(
            request_bytes=_int_value(
                traffic_data.get("request_bytes", TrafficConfig.request_bytes),
                "traffic.request_bytes",
            ),
            sender_burst_reqs=_int_value(
                traffic_data.get("sender_burst_reqs", TrafficConfig.sender_burst_reqs),
                "traffic.sender_burst_reqs",
            ),
            sender_check_interval=_int_value(
                traffic_data.get("sender_check_interval", TrafficConfig.sender_check_interval),
                "traffic.sender_check_interval",
            ),
            sender_burst_period=_int_value(
                traffic_data.get("sender_burst_period", TrafficConfig.sender_burst_period),
                "traffic.sender_burst_period",
            ),
            sender_mode=str(traffic_data.get("sender_mode", TrafficConfig.sender_mode)).strip(),
            noc_queue_capacity_reqs=_int_value(
                traffic_data.get("noc_queue_capacity_reqs", TrafficConfig.noc_queue_capacity_reqs),
                "traffic.noc_queue_capacity_reqs",
            ),
            sender_noc_inflight_capacity_reqs=_int_value(
                traffic_data.get(
                    "sender_noc_inflight_capacity_reqs",
                    TrafficConfig.sender_noc_inflight_capacity_reqs,
                ),
                "traffic.sender_noc_inflight_capacity_reqs",
            ),
            noc_issue_reqs_per_cycle=_int_value(
                traffic_data.get("noc_issue_reqs_per_cycle", TrafficConfig.noc_issue_reqs_per_cycle),
                "traffic.noc_issue_reqs_per_cycle",
            ),
            noc_queue_delay=_int_value(
                traffic_data.get("noc_queue_delay", TrafficConfig.noc_queue_delay),
                "traffic.noc_queue_delay",
            ),
            noc_queue_poll_period=_int_value(
                traffic_data.get("noc_queue_poll_period", TrafficConfig.noc_queue_poll_period),
                "traffic.noc_queue_poll_period",
            ),
        ),
        service_profile=_parse_service_profile(service_profile_value),
        settling_window=_int_value(data.get("settling_window", 100), "simulation.settling_window"),
        settling_tail_window=_int_value(data.get("settling_tail_window", 200), "simulation.settling_tail_window"),
        settling_rel_tolerance=_float_value(
            data.get("settling_rel_tolerance", 0.05),
            "simulation.settling_rel_tolerance",
        ),
        settling_abs_tolerance=_float_value(
            data.get("settling_abs_tolerance", 1.0),
            "simulation.settling_abs_tolerance",
        ),
        oscillation_window=_int_value(data.get("oscillation_window", 500), "simulation.oscillation_window"),
    )


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)

    results: Dict[str, SimulationResult] = {}
    for spec in config.controllers:
        controller = build_controller(spec.kind, config.simulation, spec.params)
        result = run_simulation(config.simulation, controller)
        result.summary["strategy"] = spec.label
        result.summary["controller_kind"] = spec.kind
        results[spec.label] = result

        slug = _slugify(spec.label)
        write_history_csv(result.history, output / f"{slug}.csv")
        write_summary_json(result.summary, output / f"{slug}_summary.json")

    _write_resolved_config(config, output / "experiment_resolved.json")
    long_path = write_long_summary_csv(results, output / "comparison_long.csv")
    wide_rows = build_wide_comparison_rows(results)
    wide_path = write_wide_comparison_csv(wide_rows, output / "comparison_wide.csv")
    md_path = write_wide_comparison_markdown(wide_rows, output / "comparison.md")

    plot_paths: List[Path] = []
    if config.plots.enabled:
        plot_paths = _write_plots(config, results, output / "plots")

    return ExperimentResult(
        config=config,
        results=results,
        output_dir=output,
        long_summary_path=long_path,
        wide_summary_path=wide_path,
        markdown_summary_path=md_path,
        plot_paths=plot_paths,
    )


def build_wide_comparison_rows(
    results: Mapping[str, SimulationResult],
    metrics: Sequence[Tuple[str, str]] = COMPARISON_METRICS,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for metric_label, summary_key in metrics:
        row = {"指标": metric_label}
        for strategy, result in results.items():
            row[strategy] = _format_metric_value(result.summary.get(summary_key))
        rows.append(row)
    return rows


def write_long_summary_csv(results: Mapping[str, SimulationResult], path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.summary for result in results.values()]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_wide_comparison_csv(rows: Sequence[Mapping[str, str]], path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["指标"]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_wide_comparison_markdown(rows: Sequence[Mapping[str, str]], path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = format_wide_comparison_markdown(rows)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
    return output


def format_wide_comparison_markdown(rows: Sequence[Mapping[str, str]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def print_wide_comparison(rows: Sequence[Mapping[str, str]]) -> None:
    print(format_wide_comparison_markdown(rows))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a YAML/JSON flow-control experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation even when YAML enables it.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_experiment(args.config)
    if args.no_plots:
        config = ExperimentConfig(
            name=config.name,
            output_dir=config.output_dir,
            simulation=config.simulation,
            controllers=config.controllers,
            plots=PlotConfig(enabled=False, mode=config.plots.mode),
        )

    result = run_experiment(config)
    rows = build_wide_comparison_rows(result.results)
    print_wide_comparison(rows)
    print(f"\nWrote experiment outputs to {result.output_dir}")
    return 0


def _load_config_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML experiment files require PyYAML. Install dependencies with: "
                "pip install -r requirements.txt"
            ) from exc
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    return _mapping(data, str(path))


def _parse_controllers(value: Any) -> List[ControllerSpec]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        raise ValueError("controllers must be a list")
    controllers: List[ControllerSpec] = []
    for idx, item in enumerate(value):
        if isinstance(item, str):
            kind = item
            label = _default_label_for_controller(kind)
            params: Dict[str, Any] = {}
        elif isinstance(item, Mapping):
            kind = str(item.get("kind", item.get("type", ""))).strip()
            if not kind:
                raise ValueError(f"controllers[{idx}] must define kind")
            label = str(item.get("label", _default_label_for_controller(kind)))
            params_data = dict(_mapping(item.get("params", {}), f"controllers[{idx}].params"))
            inline_params = {
                str(key): value
                for key, value in item.items()
                if key not in {"kind", "type", "label", "params"}
            }
            params = {**params_data, **inline_params}
        else:
            raise ValueError(f"controllers[{idx}] must be a string or mapping")
        controllers.append(ControllerSpec(kind=kind, label=label, params=params))
    return controllers


def _default_label_for_controller(kind: str) -> str:
    normalized = kind.lower().strip()
    if normalized in {"noc", "noc_watermark", "controller2"}:
        return "方案一（NOC Watermark）"
    if normalized in {"sender", "sender_outstanding", "controller1"}:
        return "方案二（Sender Outstanding）"
    if normalized in {"open", "open_loop", "none", "no_control"}:
        return "无流控"
    if normalized in {"smith", "smith_pi", "pi", "controller3"}:
        return "方案三（Smith PI）"
    return kind


def _parse_service_profile(value: Any) -> Tuple[ServiceSegment, ...]:
    if value is None:
        return (
            ServiceSegment(0, 3.0),
            ServiceSegment(1000, 1.0),
            ServiceSegment(2000, 3.0),
        )
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("receiver service_profile must be a list")
    segments: List[ServiceSegment] = []
    for idx, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"service_profile[{idx}] must be a mapping")
        start_cycle = item.get("start_cycle", item.get("cycle"))
        if start_cycle is None:
            raise ValueError(f"service_profile[{idx}] must define start_cycle or cycle")
        segments.append(
            ServiceSegment(
                start_cycle=_int_value(start_cycle, f"service_profile[{idx}].start_cycle"),
                rate=_float_value(item.get("rate"), f"service_profile[{idx}].rate"),
            )
        )
    return tuple(segments)


def _write_plots(
    config: ExperimentConfig,
    results: Mapping[str, SimulationResult],
    output_dir: Path,
) -> List[Path]:
    from .plot import (
        plot_control_response,
        plot_history,
        plot_os_rate_coupling,
        plot_os_response,
        plot_path_occupancy,
        plot_state_stack,
        plot_strategy_comparison,
    )

    paths: List[Path] = []
    if config.plots.mode in {"basic", "all"}:
        for label, result in results.items():
            paths.extend(
                plot_history(
                    result.history,
                    output_dir=output_dir,
                    prefix=_slugify(label),
                    sender_threshold=config.simulation.sender_threshold,
                    noc_threshold=config.simulation.noc_threshold,
                )
            )
    if config.plots.mode in {"state", "all"}:
        paths.append(
            plot_state_stack(
                histories=[result.history for result in results.values()],
                labels=list(results.keys()),
                output_path=output_dir / f"{_slugify(config.name)}_state_stack.png",
                sender_threshold=config.simulation.sender_threshold,
                noc_threshold=config.simulation.noc_threshold,
                sender_inflight_capacity=None,
                rate_window=config.simulation.traffic.sender_burst_period,
            )
        )
    if config.plots.mode in {"os", "all"}:
        os_results = _without_open_loop(results)
        paths.append(
            plot_os_response(
                histories=[result.history for _, result in os_results],
                labels=[label for label, _ in os_results],
                output_path=output_dir / f"{_slugify(config.name)}_os_response.png",
                sender_threshold=config.simulation.sender_threshold,
                noc_threshold=config.simulation.noc_threshold,
                rate_window=config.simulation.traffic.sender_burst_period,
            )
        )
    if config.plots.mode in {"control", "all"}:
        for label, result in results.items():
            paths.append(
                plot_control_response(
                    result.history,
                    output_path=output_dir / f"{_slugify(label)}_control_response.png",
                    title=label,
                    sender_threshold=config.simulation.sender_threshold,
                    noc_threshold=config.simulation.noc_threshold,
                )
            )
    if config.plots.mode in {"path", "all"}:
        for label, result in results.items():
            paths.append(
                plot_path_occupancy(
                    result.history,
                    output_path=output_dir / f"{_slugify(label)}_path_occupancy.png",
                    title=label,
                )
            )
    if config.plots.mode in {"rate", "all"}:
        for label, result in results.items():
            paths.append(
                plot_os_rate_coupling(
                    result.history,
                    output_path=output_dir / f"{_slugify(label)}_os_rate_coupling.png",
                    title=label,
                    sender_threshold=config.simulation.sender_threshold,
                    noc_threshold=config.simulation.noc_threshold,
                    rate_window=config.simulation.traffic.sender_burst_period,
                )
            )
    if len(results) > 1 and config.plots.mode in {"comparison", "all"}:
        paths.append(
            plot_strategy_comparison(
                histories=[result.history for result in results.values()],
                labels=list(results.keys()),
                output_path=output_dir / f"{_slugify(config.name)}_strategy_comparison.png",
                sender_threshold=config.simulation.sender_threshold,
                noc_threshold=config.simulation.noc_threshold,
                rate_window=config.simulation.traffic.sender_burst_period,
            )
        )
    return paths


def _without_open_loop(results: Mapping[str, SimulationResult]) -> List[Tuple[str, SimulationResult]]:
    filtered = [
        (label, result)
        for label, result in results.items()
        if str(result.summary.get("controller_kind", "")).lower() not in {"open", "open_loop", "none", "no_control"}
    ]
    return filtered if filtered else list(results.items())


def _write_resolved_config(config: ExperimentConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": config.name,
        "output_dir": str(config.output_dir),
        "simulation": asdict(config.simulation),
        "controllers": [asdict(item) for item in config.controllers],
        "plots": asdict(config.plots),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base_dir / path


def _int_value(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float_value(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def _optional_float(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    return _float_value(value, name)


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) < 1e-12:
            return "0"
        if 0 < abs(value) < 1:
            return f"{value:.4f}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", value).strip("_")
    return slug or "strategy"


if __name__ == "__main__":
    raise SystemExit(main())
