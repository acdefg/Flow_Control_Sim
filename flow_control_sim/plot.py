"""Plot simulation histories and strategy comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .metrics import SimulationHistory, read_history_csv

PLOT_MAX_CYCLE = 30000
LEGEND_FONT_SIZE = 8


@dataclass(frozen=True)
class PlotEvents:
    service_drop: Optional[int]
    service_recover: Optional[int]
    raw_throttle: Optional[int]
    delayed_throttle: Optional[int]
    send_drop: Optional[int]
    send_recover: Optional[int]


@dataclass(frozen=True)
class PlotMetrics:
    label: str
    max_sender: float
    max_noc: float
    max_inflight: float
    max_sender_noc_gap: float
    slowdown_throughput: Optional[float]
    recovery_throughput: Optional[float]
    slowdown_overshoot: Optional[float]
    slowdown_settling: Optional[int]
    recovery_settling: Optional[int]
    idle_ratio: float
    raw_delay: Optional[int]
    sender_delay: Optional[int]
    feedback_delay: Optional[int]


def plot_history(
    history: SimulationHistory,
    output_dir: Path | str,
    prefix: str = "sim",
    sender_threshold: Optional[float] = 128.0,
    noc_threshold: Optional[float] = 128.0,
) -> List[Path]:
    """Generate the five basic signal plots."""

    plt = _load_pyplot()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = [
        _plot_line(
            plt,
            history.cycles,
            [(history.sender_outstanding, "Sender Outstanding")],
            "Sender Outstanding",
            "Requests",
            output / f"{prefix}_sender_outstanding.png",
        ),
        _plot_line(
            plt,
            history.cycles,
            [(history.noc_outstanding, "NOC Outstanding")],
            "NOC Outstanding",
            "Requests",
            output / f"{prefix}_noc_outstanding.png",
        ),
        _plot_line(
            plt,
            history.cycles,
            [(history.receiver_queue, "Receiver Queue")],
            "Receiver Queue",
            "Requests",
            output / f"{prefix}_receiver_queue.png",
        ),
        _plot_line(
            plt,
            history.cycles,
            [
                (history.send_rate, "Send Rate"),
                (history.service_rate, "Receiver Rate"),
            ],
            "Send Rate vs Receiver Rate",
            "Byte/cycle",
            output / f"{prefix}_send_vs_receiver_rate.png",
        ),
        _plot_line(
            plt,
            history.cycles,
            [
                ([1.0 if value else 0.0 for value in history.raw_throttle], "Raw Throttle"),
                ([1.0 if value else 0.0 for value in history.delayed_throttle], "Delayed Throttle"),
            ],
            "Throttle",
            "0/1",
            output / f"{prefix}_throttle.png",
        ),
    ]
    return paths


def plot_os_inflight(
    history: SimulationHistory,
    output_path: Path | str,
    title: str,
    sender_threshold: Optional[float] = 128.0,
    noc_threshold: Optional[float] = 128.0,
    sender_inflight_capacity: Optional[float] = None,
) -> Path:
    """Plot Sender OS, NOC OS, and the Sender-NOC OS gap."""

    plt = _load_pyplot()
    events = detect_events(history)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    cycles = history.cycles

    ax.plot(cycles, history.sender_outstanding, label="Sender OS", color="#2166ac", linewidth=1.8)
    ax.plot(cycles, history.noc_outstanding, label="NOC OS", color="#b2182b", linewidth=1.8)
    ax.plot(
        cycles,
        history.sender_noc_os_gap,
        label="Sender-NOC OS gap",
        color="#404040",
        linewidth=1.7,
    )

    if sender_threshold is not None:
        ax.axhline(sender_threshold, color="#2166ac", linestyle="--", linewidth=1.0, alpha=0.6)
    if noc_threshold is not None:
        ax.axhline(noc_threshold, color="#b2182b", linestyle="--", linewidth=1.0, alpha=0.6)
    if sender_inflight_capacity is not None:
        ax.axhline(sender_inflight_capacity, color="#404040", linestyle=":", linewidth=1.0, alpha=0.55)

    _shade_degraded_region(ax, events)
    _draw_event_markers(ax, events, include_control=False)
    _annotate_peak(ax, cycles, history.sender_outstanding, "Sender OS peak", sender_threshold, "#2166ac")
    _annotate_peak(ax, cycles, history.noc_outstanding, "NOC OS peak", noc_threshold, "#b2182b")
    _annotate_peak(
        ax,
        cycles,
        history.sender_noc_os_gap,
        "OS gap peak",
        sender_inflight_capacity,
        "#404040",
    )

    ax.set_title(f"{title}: Sender OS / NOC OS / Sender-NOC OS gap")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Requests")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_state_stack(
    histories: Sequence[SimulationHistory],
    labels: Sequence[str],
    output_path: Path | str,
    sender_threshold: Optional[float] = 128.0,
    noc_threshold: Optional[float] = 128.0,
    sender_inflight_capacity: Optional[float] = None,
    rate_window: int = 176,
) -> Path:
    """Plot Sender OS, NOC OS, Sender-NOC OS gap, and averaged send rate."""

    if len(histories) != len(labels):
        raise ValueError("histories and labels must have the same length")
    if not histories:
        raise ValueError("at least one history is required")
    if rate_window <= 0:
        raise ValueError("rate_window must be > 0")

    plt = _load_pyplot()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#2166ac", "#b2182b", "#4d9221", "#7b3294", "#f4a582", "#92c5de"]
    base_events = detect_events(histories[0])

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(13, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.25, 1.25, 1.05]},
    )
    ax_sender, ax_noc, ax_inflight, ax_rate = axes

    for idx, (history, label) in enumerate(zip(histories, labels)):
        color = colors[idx % len(colors)]
        ax_sender.plot(history.cycles, history.sender_outstanding, label=label, color=color, linewidth=1.8)
        ax_noc.plot(history.cycles, history.noc_outstanding, label=label, color=color, linewidth=1.8)
        ax_inflight.plot(history.cycles, history.sender_noc_os_gap, label=label, color=color, linewidth=1.8)
        ax_rate.plot(
            history.cycles,
            block_average(history.sent, rate_window),
            label=f"{label} actual send bw/bin",
            color=color,
            linewidth=1.6,
        )

    ax_rate.step(
        histories[0].cycles,
        histories[0].service_rate,
        label="Receiver service",
        color="#404040",
        linewidth=1.5,
        linestyle="--",
        where="post",
    )

    if sender_threshold is not None:
        ax_sender.axhline(sender_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.65)
    if noc_threshold is not None:
        ax_noc.axhline(noc_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.65)
    if sender_inflight_capacity is not None:
        ax_inflight.axhline(sender_inflight_capacity, color="#666666", linestyle=":", linewidth=1.0, alpha=0.65)

    ax_sender.set_title("Sender OS / NOC OS / Sender-NOC OS gap")
    ax_sender.set_ylabel("Sender OS (req)")
    ax_noc.set_ylabel("NOC OS (req)")
    ax_inflight.set_ylabel("OS gap (req)")
    ax_rate.set_ylabel(f"Avg Rate\n({rate_window} cyc)")
    ax_rate.set_xlabel("Cycle")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(
            loc="lower right",
            ncol=2 if ax is ax_rate else 1,
            framealpha=0.82,
            fontsize=LEGEND_FONT_SIZE,
        )
        _shade_degraded_region(ax, base_events)
        _draw_event_markers(ax, base_events, include_control=False)

    _apply_cycle_limit(axes, histories[0].cycles, max_cycle=PLOT_MAX_CYCLE)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_os_response(
    histories: Sequence[SimulationHistory],
    labels: Sequence[str],
    output_path: Path | str,
    sender_threshold: Optional[float] = 64.0,
    noc_threshold: Optional[float] = 64.0,
    rate_window: int = 176,
) -> Path:
    """Plot OS and OS-gap response to receiver slowdown/recovery."""

    if len(histories) != len(labels):
        raise ValueError("histories and labels must have the same length")
    if not histories:
        raise ValueError("at least one history is required")
    if rate_window <= 0:
        raise ValueError("rate_window must be > 0")

    plt = _load_pyplot()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#8c8c8c", "#2166ac", "#b2182b", "#4d9221", "#7b3294", "#f4a582"]
    base_events = detect_events(histories[0])

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(13, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.25, 1.15, 1.0]},
    )
    ax_sender, ax_noc, ax_gap, ax_rate = axes

    for idx, (history, label) in enumerate(zip(histories, labels)):
        color = colors[idx % len(colors)]
        ax_sender.plot(history.cycles, history.sender_outstanding, label=label, color=color, linewidth=1.8)
        ax_noc.plot(history.cycles, history.noc_outstanding, label=label, color=color, linewidth=1.8)
        ax_gap.plot(history.cycles, history.sender_noc_os_gap, label=label, color=color, linewidth=1.7)
        ax_rate.plot(
            history.cycles,
            block_average(history.sent, rate_window),
            label=f"{label} actual send bw/bin",
            color=color,
            linewidth=1.5,
        )

    ax_rate.step(
        histories[0].cycles,
        histories[0].service_rate,
        label="Receiver service",
        color="#404040",
        linewidth=1.6,
        linestyle="--",
        where="post",
    )

    if sender_threshold is not None:
        ax_sender.axhline(sender_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.65)
    if noc_threshold is not None:
        ax_noc.axhline(noc_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.65)

    ax_sender.set_title("OS response to receiver slowdown/recovery")
    ax_sender.set_ylabel("Sender OS (req)")
    ax_noc.set_ylabel("NOC OS (req)")
    ax_gap.set_ylabel("OS gap (req)")
    ax_rate.set_ylabel(f"Avg Rate\n({rate_window} cyc)")
    ax_rate.set_xlabel("Cycle")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(
            loc="lower right",
            ncol=2 if ax is ax_rate else 1,
            framealpha=0.82,
            fontsize=LEGEND_FONT_SIZE,
        )
        _shade_degraded_region(ax, base_events)
        _draw_event_markers(ax, base_events, include_control=False)

    _apply_cycle_limit(axes, histories[0].cycles, max_cycle=PLOT_MAX_CYCLE)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_path_occupancy(
    history: SimulationHistory,
    output_path: Path | str,
    title: str,
) -> Path:
    """Plot where outstanding requests reside along the request/response path."""

    plt = _load_pyplot()
    events = detect_events(history)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cycles = history.cycles

    sender_to_noc = [
        forward + pending
        for forward, pending in zip(history.forward_sn_occupancy, history.pending_noc_arrivals)
    ]
    components = [
        sender_to_noc,
        history.noc_queue,
        history.forward_nr_occupancy,
        history.receiver_queue,
        history.return_rn_occupancy,
        history.return_ns_occupancy,
    ]
    labels = [
        "Sender->NOC delay",
        "NOC queue",
        "NOC->Receiver delay",
        "Receiver queue",
        "Receiver->NOC return",
        "NOC->Sender return",
    ]
    colors = ["#8dd3c7", "#bebada", "#80b1d3", "#b3de69", "#fdb462", "#fb8072"]

    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.stackplot(cycles, components, labels=labels, colors=colors, alpha=0.75)
    ax.plot(cycles, history.sender_outstanding, label="Sender OS", color="#202020", linewidth=1.7)
    ax.plot(cycles, history.noc_outstanding, label="NOC OS", color="#b2182b", linewidth=1.5, linestyle="--")

    _shade_degraded_region(ax, events)
    _draw_event_markers(ax, events, include_control=False)
    ax.set_title(f"{title}: outstanding path occupancy")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Requests")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_os_rate_coupling(
    history: SimulationHistory,
    output_path: Path | str,
    title: str,
    sender_threshold: Optional[float] = None,
    noc_threshold: Optional[float] = None,
    rate_window: int = 176,
) -> Path:
    """Plot OS waterline, throttle state, and resulting effective bandwidth."""

    if rate_window <= 0:
        raise ValueError("rate_window must be > 0")

    plt = _load_pyplot()
    events = detect_events(history)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cycles = history.cycles

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(13, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 0.75, 1.0]},
    )
    ax_os, ax_throttle, ax_rate = axes

    ax_os.plot(cycles, history.sender_outstanding, label="Sender OS", color="#2166ac", linewidth=1.8)
    ax_os.plot(cycles, history.noc_outstanding, label="NOC OS", color="#b2182b", linewidth=1.8)
    if sender_threshold is not None:
        ax_os.axhline(sender_threshold, color="#2166ac", linestyle="--", linewidth=1.0, alpha=0.65)
    if noc_threshold is not None:
        ax_os.axhline(noc_threshold, color="#b2182b", linestyle="--", linewidth=1.0, alpha=0.65)
    _annotate_peak(ax_os, cycles, history.sender_outstanding, "Sender peak", sender_threshold, "#2166ac")
    _annotate_peak(ax_os, cycles, history.noc_outstanding, "NOC peak", noc_threshold, "#b2182b")
    ax_os.set_ylabel("OS (req)")
    ax_os.set_title(f"{title}: OS-limited effective bandwidth")
    ax_os.legend(loc="upper right")

    raw = [1.0 if value else 0.0 for value in history.raw_throttle]
    delayed = [1.0 if value else 0.0 for value in history.delayed_throttle]
    ax_throttle.step(cycles, raw, label="Throttle generated", color="#b2182b", linewidth=1.4, where="post")
    ax_throttle.step(cycles, delayed, label="Sender gated", color="#ef8a62", linewidth=1.4, where="post")
    ax_throttle.set_yticks([0, 1])
    ax_throttle.set_ylim(-0.15, 1.2)
    ax_throttle.set_ylabel("Gate")
    ax_throttle.legend(loc="upper right")

    ax_rate.plot(
        cycles,
        block_average(history.commanded_bandwidth, rate_window),
        label=f"OS-gated bandwidth ({rate_window} cyc bin)",
        color="#f4a582",
        linewidth=1.8,
    )
    ax_rate.step(
        cycles,
        history.service_rate,
        label="Receiver service",
        color="#404040",
        linewidth=1.5,
        linestyle="--",
        where="post",
    )
    ax_rate.set_ylabel("B/cycle")
    ax_rate.set_xlabel("Cycle")
    ax_rate.legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        _shade_degraded_region(ax, events)
        _draw_event_markers(ax, events, include_control=False)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_control_response(
    history: SimulationHistory,
    output_path: Path | str,
    title: str,
    sender_threshold: Optional[float] = 128.0,
    noc_threshold: Optional[float] = 128.0,
    rate_window: int = 176,
) -> Path:
    """Plot the causal flow-control response for one strategy."""

    plt = _load_pyplot()
    events = detect_events(history)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.25, 1.25, 0.9]},
    )

    ax_os, ax_queue, ax_rate, ax_throttle = axes
    cycles = history.cycles

    ax_os.plot(cycles, history.sender_outstanding, label="Sender OS / in-flight", color="#2166ac", linewidth=1.8)
    ax_os.plot(cycles, history.noc_outstanding, label="NOC OS / queued in NOC", color="#b2182b", linewidth=1.8)
    ax_os.plot(
        cycles,
        history.sender_noc_os_gap,
        label="Sender-NOC OS gap",
        color="#404040",
        linewidth=1.4,
        alpha=0.8,
    )
    if sender_threshold is not None:
        ax_os.axhline(sender_threshold, color="#2166ac", linestyle="--", linewidth=1.0, alpha=0.65)
    if noc_threshold is not None:
        ax_os.axhline(noc_threshold, color="#b2182b", linestyle="--", linewidth=1.0, alpha=0.65)
    _annotate_peak(ax_os, cycles, history.sender_outstanding, "Sender peak", sender_threshold, "#2166ac")
    _annotate_peak(ax_os, cycles, history.noc_outstanding, "NOC peak", noc_threshold, "#b2182b")
    ax_os.set_ylabel("Outstanding (req)")
    ax_os.set_title(f"{title}: flow-control step response")
    ax_os.legend(loc="upper right")

    ax_queue.plot(cycles, history.receiver_queue, label="Receiver Queue", color="#4d9221", linewidth=1.8)
    ax_queue.plot(cycles, history.noc_queue, label="NOC Queue", color="#762a83", linewidth=1.5)
    _annotate_peak(ax_queue, cycles, history.receiver_queue, "Queue peak", None, "#4d9221")
    _annotate_peak(ax_queue, cycles, history.noc_queue, "NOC queue peak", None, "#762a83")
    ax_queue.set_ylabel("Queue (req)")
    ax_queue.legend(loc="upper right")

    ax_rate.step(cycles, history.service_rate, label="Receiver service rate", color="#7b3294", linewidth=1.8, where="post")
    ax_rate.plot(
        cycles,
        block_average(history.commanded_bandwidth, rate_window),
        label=f"Sender command bw/bin ({rate_window} cyc)",
        color="#f4a582",
        linewidth=1.8,
    )
    ax_rate.set_ylabel("Avg Rate (B/cycle)")
    ax_rate.legend(loc="upper right")

    raw = [1.0 if value else 0.0 for value in history.raw_throttle]
    delayed = [1.0 if value else 0.0 for value in history.delayed_throttle]
    ax_throttle.step(cycles, raw, label="Throttle generated", color="#b2182b", linewidth=1.5, where="post")
    ax_throttle.step(cycles, delayed, label="Throttle seen by sender", color="#ef8a62", linewidth=1.5, where="post")
    ax_throttle.set_yticks([0, 1])
    ax_throttle.set_ylim(-0.15, 1.25)
    ax_throttle.set_ylabel("Throttle")
    ax_throttle.set_xlabel("Cycle")
    ax_throttle.legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        _shade_degraded_region(ax, events)
        _draw_event_markers(ax, events)

    _annotate_event_text(ax_rate, events)
    _annotate_delay_brackets(ax_throttle, events)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_strategy_comparison(
    histories: Sequence[SimulationHistory],
    labels: Sequence[str],
    output_path: Path | str,
    sender_threshold: Optional[float] = 128.0,
    noc_threshold: Optional[float] = 128.0,
    rate_window: int = 176,
) -> Path:
    """Plot several flow-control strategies on shared axes with a metric table."""

    if len(histories) != len(labels):
        raise ValueError("histories and labels must have the same length")
    if not histories:
        raise ValueError("at least one history is required")

    plt = _load_pyplot()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#2166ac", "#b2182b", "#4d9221", "#7b3294", "#f4a582", "#92c5de"]
    base_events = detect_events(histories[0])

    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(5, 1, height_ratios=[1.35, 1.35, 1.35, 1.15, 1.05])
    ax_sender = fig.add_subplot(grid[0, 0])
    ax_noc = fig.add_subplot(grid[1, 0], sharex=ax_sender)
    ax_queue = fig.add_subplot(grid[2, 0], sharex=ax_sender)
    ax_rate = fig.add_subplot(grid[3, 0], sharex=ax_sender)
    ax_table = fig.add_subplot(grid[4, 0])

    for idx, (history, label) in enumerate(zip(histories, labels)):
        color = colors[idx % len(colors)]
        ax_sender.plot(history.cycles, history.sender_outstanding, label=label, color=color, linewidth=1.8)
        ax_noc.plot(history.cycles, history.noc_outstanding, label=label, color=color, linewidth=1.8)
        ax_queue.plot(history.cycles, history.receiver_queue, label=label, color=color, linewidth=1.8)
        ax_rate.plot(
            history.cycles,
            block_average(history.sent, rate_window),
            label=f"{label} actual send bw/bin",
            color=color,
            linewidth=1.4,
        )

    ax_rate.step(
        histories[0].cycles,
        histories[0].service_rate,
        label="Receiver service",
        color="#404040",
        linewidth=1.8,
        linestyle="--",
        where="post",
    )

    if sender_threshold is not None:
        ax_sender.axhline(sender_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.65)
    if noc_threshold is not None:
        ax_noc.axhline(noc_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.65)

    ax_sender.set_title("Strategy comparison under the same receiver-speed step")
    ax_sender.set_ylabel("Sender OS (req)")
    ax_noc.set_ylabel("NOC OS (req)")
    ax_queue.set_ylabel("Queue (req)")
    ax_rate.set_ylabel(f"Avg Rate\n({rate_window} cyc)")
    ax_rate.set_xlabel("Cycle")

    for ax in (ax_sender, ax_noc, ax_queue, ax_rate):
        ax.grid(True, alpha=0.25)
        ax.legend(
            loc="lower right",
            ncol=2 if ax is ax_rate else 1,
            framealpha=0.82,
            fontsize=LEGEND_FONT_SIZE,
        )
        _shade_degraded_region(ax, base_events)
        _draw_event_markers(ax, base_events, include_control=False)

    _apply_cycle_limit((ax_sender, ax_noc, ax_queue, ax_rate), histories[0].cycles, max_cycle=PLOT_MAX_CYCLE)

    metrics = [
        compute_plot_metrics(history, label, noc_threshold=noc_threshold)
        for history, label in zip(histories, labels)
    ]
    _draw_metric_table(ax_table, metrics)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def detect_events(history: SimulationHistory) -> PlotEvents:
    service_drop, service_recover = _service_step_events(history)
    raw_throttle = _first_true_cycle(history.cycles, history.raw_throttle, after=service_drop)
    delayed_throttle = _first_true_cycle(history.cycles, history.delayed_throttle, after=service_drop)
    # In burst mode the send-rate signal naturally drops to zero between bursts,
    # so a rate edge is not a reliable control-action marker. The delayed
    # throttle is the cycle where the sender actually sees the stop signal.
    send_drop = delayed_throttle
    send_recover = _first_false_cycle(history.cycles, history.delayed_throttle, after=send_drop)
    return PlotEvents(
        service_drop=service_drop,
        service_recover=service_recover,
        raw_throttle=raw_throttle,
        delayed_throttle=delayed_throttle,
        send_drop=send_drop,
        send_recover=send_recover,
    )


def compute_plot_metrics(
    history: SimulationHistory,
    label: str,
    noc_threshold: Optional[float] = None,
) -> PlotMetrics:
    events = detect_events(history)
    total_cycles = max(len(history), 1)
    idle_ratio = sum(1 for value in history.sent if value <= 1e-9) / total_cycles
    raw_delay = _delta(events.service_drop, events.raw_throttle)
    sender_delay = _delta(events.service_drop, events.send_drop)
    feedback_delay = _delta(events.raw_throttle, events.delayed_throttle)
    slowdown_throughput = _interval_average(
        cycles=history.cycles,
        values=history.serviced,
        start_cycle=events.service_drop,
        end_cycle=events.service_recover,
    )
    recovery_throughput = _interval_average(
        cycles=history.cycles,
        values=history.sent,
        start_cycle=events.service_recover,
        end_cycle=None,
    )
    if noc_threshold is None:
        slowdown_overshoot = None
        slowdown_settling = None
    else:
        slowdown_overshoot = _slowdown_overshoot_above_target(
            cycles=history.cycles,
            values=history.noc_outstanding,
            start_cycle=events.service_drop,
            end_cycle=events.service_recover,
            target=noc_threshold,
        )
        slowdown_settling = _settle_below_target_after_peak_time(
            cycles=history.cycles,
            values=history.noc_outstanding,
            start_cycle=events.service_drop,
            end_cycle=events.service_recover,
            target=noc_threshold,
        )
    recovery_settling = _settle_to_tail_time(
        cycles=history.cycles,
        values=history.noc_outstanding,
        start_cycle=events.service_recover,
        end_cycle=None,
    )
    return PlotMetrics(
        label=label,
        max_sender=max(history.sender_outstanding),
        max_noc=max(history.noc_outstanding),
        max_inflight=max(history.sender_noc_inflight),
        max_sender_noc_gap=max(history.sender_noc_os_gap),
        slowdown_throughput=slowdown_throughput,
        recovery_throughput=recovery_throughput,
        slowdown_overshoot=slowdown_overshoot,
        slowdown_settling=slowdown_settling,
        recovery_settling=recovery_settling,
        idle_ratio=idle_ratio,
        raw_delay=raw_delay,
        sender_delay=sender_delay,
        feedback_delay=feedback_delay,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot simulation CSV files.")
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--sender-threshold", type=float, default=128.0)
    parser.add_argument("--noc-threshold", type=float, default=128.0)
    parser.add_argument(
        "--mode",
        choices=["basic", "os", "state", "path", "rate", "control", "comparison", "all"],
        default="all",
        help="basic: five simple plots; os: OS-only response; state: Sender OS/NOC OS/inflight; path: per-stage occupancy; rate: OS/gate/bandwidth coupling; control: annotated response; comparison: strategy comparison.",
    )
    parser.add_argument("--sender-inflight-capacity", type=float, default=None)
    parser.add_argument("--rate-window", type=int, default=176)
    args = parser.parse_args(list(argv) if argv is not None else None)

    histories = [read_history_csv(path) for path in args.csv]
    labels = _resolve_labels(args.csv, args.labels)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    if args.mode in {"basic", "all"}:
        for path, history, label in zip(args.csv, histories, labels):
            prefix = args.prefix or path.stem
            if len(histories) > 1 and args.prefix:
                prefix = f"{args.prefix}_{label}"
            paths.extend(
                plot_history(
                    history=history,
                    output_dir=output,
                    prefix=prefix,
                    sender_threshold=args.sender_threshold,
                    noc_threshold=args.noc_threshold,
                )
            )

    if args.mode in {"state", "all"}:
        prefix = args.prefix or ("state" if len(histories) > 1 else args.csv[0].stem)
        paths.append(
            plot_state_stack(
                histories=histories,
                labels=labels,
                output_path=output / f"{prefix}_state_stack.png",
                sender_threshold=args.sender_threshold,
                noc_threshold=args.noc_threshold,
                sender_inflight_capacity=args.sender_inflight_capacity,
                rate_window=args.rate_window,
            )
        )

    if args.mode in {"os", "all"}:
        prefix = args.prefix or ("os_response" if len(histories) > 1 else args.csv[0].stem)
        paths.append(
            plot_os_response(
                histories=histories,
                labels=labels,
                output_path=output / f"{prefix}_os_response.png",
                sender_threshold=args.sender_threshold,
                noc_threshold=args.noc_threshold,
                rate_window=args.rate_window,
            )
        )

    if args.mode in {"control", "all"}:
        for path, history, label in zip(args.csv, histories, labels):
            prefix = args.prefix or path.stem
            if len(histories) > 1 and args.prefix:
                prefix = f"{args.prefix}_{label}"
            paths.append(
                plot_control_response(
                    history=history,
                    output_path=output / f"{prefix}_control_response.png",
                    title=label,
                    sender_threshold=args.sender_threshold,
                    noc_threshold=args.noc_threshold,
                )
            )

    if args.mode in {"path", "all"}:
        for path, history, label in zip(args.csv, histories, labels):
            prefix = args.prefix or path.stem
            if len(histories) > 1 and args.prefix:
                prefix = f"{args.prefix}_{label}"
            paths.append(
                plot_path_occupancy(
                    history=history,
                    output_path=output / f"{prefix}_path_occupancy.png",
                    title=label,
                )
            )

    if args.mode in {"rate", "all"}:
        for path, history, label in zip(args.csv, histories, labels):
            prefix = args.prefix or path.stem
            if len(histories) > 1 and args.prefix:
                prefix = f"{args.prefix}_{label}"
            paths.append(
                plot_os_rate_coupling(
                    history=history,
                    output_path=output / f"{prefix}_os_rate_coupling.png",
                    title=label,
                    sender_threshold=args.sender_threshold,
                    noc_threshold=args.noc_threshold,
                    rate_window=args.rate_window,
                )
            )

    if len(histories) > 1 and args.mode in {"comparison", "all"}:
        prefix = args.prefix or "comparison"
        paths.append(
            plot_strategy_comparison(
                histories=histories,
                labels=labels,
                output_path=output / f"{prefix}_strategy_comparison.png",
                sender_threshold=args.sender_threshold,
                noc_threshold=args.noc_threshold,
                rate_window=args.rate_window,
            )
        )

    for path in _dedupe_paths(paths):
        print(path)
    return 0


def _plot_line(plt, cycles, series, title: str, ylabel: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    for values, label in series:
        ax.plot(cycles, values, label=label, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Cycle")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 0:
        raise ValueError("window must be > 0")
    result: List[float] = []
    running = 0.0
    queue: List[float] = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        result.append(running / window)
    return result


def block_average(values: Sequence[float], window: int) -> List[float]:
    """Average non-overlapping windows and repeat each value across its block."""

    if window <= 0:
        raise ValueError("window must be > 0")
    result: List[float] = []
    for start in range(0, len(values), window):
        block = values[start : start + window]
        if not block:
            break
        avg = sum(block) / window
        result.extend([avg] * len(block))
    return result


def _plot_commanded_bandwidth_if_dynamic(
    ax,
    history: SimulationHistory,
    label: str,
    color: str,
    window: Optional[int] = None,
) -> None:
    if not _has_dynamic_commanded_bandwidth(history):
        return
    values = history.commanded_bandwidth
    suffix = "command bw"
    if window is not None:
        values = block_average(values, window)
        suffix = "command bw avg/bin"
    ax.plot(
        history.cycles,
        values,
        label=f"{label} {suffix}",
        color=color,
        linewidth=1.2,
        linestyle=":",
        alpha=0.9,
    )


def _has_dynamic_commanded_bandwidth(history: SimulationHistory) -> bool:
    positive = [value for value in history.commanded_bandwidth if value > 1e-9]
    if not positive:
        return False
    return max(positive) - min(positive) > 1e-6


def _annotate_peak(ax, cycles: Sequence[int], values: Sequence[float], label: str, threshold, color: str) -> None:
    if not cycles or not values:
        return
    peak = max(values)
    idx = values.index(peak)
    cycle = cycles[idx]
    if threshold is None:
        text = f"{label}: {peak:.0f}"
    else:
        overshoot = max(0.0, peak - threshold)
        text = f"{label}: {peak:.0f} (+{overshoot:.0f})"
    ax.scatter([cycle], [peak], s=28, color=color, zorder=5)
    ax.annotate(
        text,
        xy=(cycle, peak),
        xytext=(8, 8),
        textcoords="offset points",
        color=color,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.8},
    )


def _shade_degraded_region(ax, events: PlotEvents) -> None:
    if events.service_drop is None:
        return
    end = events.service_recover if events.service_recover is not None else ax.get_xlim()[1]
    ax.axvspan(events.service_drop, end, color="#fddbc7", alpha=0.25, zorder=0)


def _apply_cycle_limit(axes, cycles: Sequence[int], max_cycle: int) -> None:
    if not cycles:
        return
    left = cycles[0]
    right = min(cycles[-1], max_cycle)
    if right <= left:
        return
    for ax in axes:
        ax.set_xlim(left, right)


def _slowdown_zoom_xlim(
    events: PlotEvents,
    cycles: Sequence[int],
    rate_window: int,
) -> Optional[Tuple[int, int]]:
    if events.service_drop is None or events.service_recover is None or not cycles:
        return None
    if events.service_recover <= events.service_drop:
        return None

    slowdown_width = events.service_recover - events.service_drop
    pre_context = max(rate_window * 4, int(slowdown_width * 0.15))
    post_context = max(rate_window * 10, int(slowdown_width * 0.5))
    left = max(cycles[0], events.service_drop - pre_context)
    right = min(cycles[-1], events.service_recover + post_context)
    if right <= left:
        return None
    return left, right


def _draw_event_markers(ax, events: PlotEvents, include_control: bool = True) -> None:
    markers: List[Tuple[Optional[int], str, str]] = [
        (events.service_drop, "Receiver slows", "#7b3294"),
        (events.service_recover, "Receiver recovers", "#7b3294"),
    ]
    if include_control:
        markers.extend(
            [
                (events.raw_throttle, "Throttle generated", "#b2182b"),
                (events.delayed_throttle, "Sender sees throttle", "#ef8a62"),
                (events.send_drop, "Send rate drops", "#f4a582"),
            ]
        )
    seen = set()
    for cycle, label, color in markers:
        if cycle is None or (cycle, label) in seen:
            continue
        seen.add((cycle, label))
        ax.axvline(cycle, color=color, linestyle=":", linewidth=1.2, alpha=0.85)


def _annotate_event_text(ax, events: PlotEvents) -> None:
    y_top = ax.get_ylim()[1]
    annotations = [
        (events.service_drop, "Rx speed down", "#7b3294", 0.94),
        (events.raw_throttle, "signal asserted", "#b2182b", 0.78),
        (events.delayed_throttle, "sender throttled", "#ef8a62", 0.62),
        (events.service_recover, "Rx recovers", "#7b3294", 0.46),
    ]
    for cycle, text, color, y_frac in annotations:
        if cycle is None:
            continue
        ax.text(
            cycle,
            y_top * y_frac,
            f"{text}\n@{cycle}",
            color=color,
            fontsize=8,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.75},
        )


def _annotate_delay_brackets(ax, events: PlotEvents) -> None:
    y = 1.12
    pairs = [
        (events.service_drop, events.raw_throttle, "detection"),
        (events.raw_throttle, events.delayed_throttle, "feedback"),
        (events.service_drop, events.send_drop, "sender response"),
    ]
    for start, end, label in pairs:
        delta = _delta(start, end)
        if start is None or end is None or delta is None or delta <= 0:
            continue
        ax.annotate(
            "",
            xy=(start, y),
            xytext=(end, y),
            arrowprops={"arrowstyle": "<->", "color": "#404040", "linewidth": 1.0},
        )
        ax.text((start + end) / 2, y + 0.03, f"{label}: {delta} cycles", ha="center", fontsize=8)
        y -= 0.18


def _draw_metric_table(ax, metrics: Sequence[PlotMetrics]) -> None:
    ax.axis("off")
    columns = [
        "Strategy",
        "Max Sender OS",
        "Max NOC OS",
        "Max Inflight",
        "Max S-N Gap",
        "Throughput\nSlowdown",
        "Throughput\nRecovery",
        "Overshoot\nSlowdown",
        "Settling\nSlowdown",
        "Settling\nRecovery",
        "Idle",
        "Detect Delay",
        "Sender Delay",
        "Feedback Delay",
    ]
    rows = [
        [
            item.label,
            f"{item.max_sender:.0f}",
            f"{item.max_noc:.0f}",
            f"{item.max_inflight:.0f}",
            f"{item.max_sender_noc_gap:.0f}",
            _format_rate(item.slowdown_throughput),
            _format_rate(item.recovery_throughput),
            _format_req(item.slowdown_overshoot),
            _format_cycles(item.slowdown_settling),
            _format_cycles(item.recovery_settling),
            f"{item.idle_ratio:.1%}",
            _format_cycles(item.raw_delay),
            _format_cycles(item.sender_delay),
            _format_cycles(item.feedback_delay),
        ]
        for item in metrics
    ]
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.3)


def _service_step_events(history: SimulationHistory) -> Tuple[Optional[int], Optional[int]]:
    rates = history.service_rate
    cycles = history.cycles
    if len(rates) <= 1:
        return None, None
    drop = None
    recover = None
    for idx in range(1, len(rates)):
        if rates[idx] < rates[idx - 1] and drop is None:
            drop = cycles[idx]
        if drop is not None and rates[idx] > rates[idx - 1]:
            recover = cycles[idx]
            break
    return drop, recover


def _first_true_cycle(cycles: Sequence[int], values: Sequence[bool], after: Optional[int]) -> Optional[int]:
    for cycle, value in zip(cycles, values):
        if after is not None and cycle < after:
            continue
        if value:
            return cycle
    return None


def _first_rate_drop_cycle(cycles: Sequence[int], rates: Sequence[float], after: Optional[int]) -> Optional[int]:
    for idx in range(1, len(rates)):
        cycle = cycles[idx]
        if after is not None and cycle < after:
            continue
        if rates[idx] < rates[idx - 1] - 1e-9:
            return cycle
    return None


def _first_rate_recover_cycle(cycles: Sequence[int], rates: Sequence[float], after: Optional[int]) -> Optional[int]:
    if after is None:
        return None
    for idx in range(1, len(rates)):
        cycle = cycles[idx]
        if cycle <= after:
            continue
        if rates[idx] > rates[idx - 1] + 1e-9:
            return cycle
    return None


def _first_false_cycle(cycles: Sequence[int], values: Sequence[bool], after: Optional[int]) -> Optional[int]:
    if after is None:
        return None
    for cycle, value in zip(cycles, values):
        if cycle <= after:
            continue
        if not value:
            return cycle
    return None


def _delta(start: Optional[int], end: Optional[int]) -> Optional[int]:
    if start is None or end is None:
        return None
    return end - start


def _interval_average(
    cycles: Sequence[int],
    values: Sequence[float],
    start_cycle: Optional[int],
    end_cycle: Optional[int],
) -> Optional[float]:
    if start_cycle is None:
        return None
    total = 0.0
    count = 0
    for cycle, value in zip(cycles, values):
        if cycle < start_cycle:
            continue
        if end_cycle is not None and cycle >= end_cycle:
            continue
        total += value
        count += 1
    if count <= 0:
        return None
    return total / count


def _slowdown_overshoot_above_target(
    cycles: Sequence[int],
    values: Sequence[float],
    start_cycle: Optional[int],
    end_cycle: Optional[int],
    target: float,
) -> Optional[float]:
    if start_cycle is None or not cycles or not values:
        return None
    interval_values = [
        value
        for cycle, value in zip(cycles, values)
        if cycle >= start_cycle and (end_cycle is None or cycle < end_cycle)
    ]
    if not interval_values:
        return None
    return max(0.0, max(interval_values) - target)


def _settle_below_target_after_peak_time(
    cycles: Sequence[int],
    values: Sequence[float],
    start_cycle: Optional[int],
    end_cycle: Optional[int],
    target: float,
    tolerance_ratio: float = 0.02,
) -> Optional[int]:
    if start_cycle is None or not cycles or not values:
        return None
    samples = [
        (cycle, value)
        for cycle, value in zip(cycles, values)
        if cycle >= start_cycle and (end_cycle is None or cycle < end_cycle)
    ]
    if len(samples) < 2:
        return None
    tolerance = max(0.5, abs(target) * tolerance_ratio)
    limit = target + tolerance
    peak_idx = max(range(len(samples)), key=lambda idx: samples[idx][1])
    if samples[peak_idx][1] <= limit:
        return 0

    for cycle, value in samples[peak_idx:]:
        if value <= limit:
            return cycle - start_cycle
    return None


def _settle_to_tail_time(
    cycles: Sequence[int],
    values: Sequence[float],
    start_cycle: Optional[int],
    end_cycle: Optional[int],
    tail_window: int = 176,
    tolerance_ratio: float = 0.02,
) -> Optional[int]:
    if start_cycle is None or not cycles or not values:
        return None
    samples = [
        (cycle, value)
        for cycle, value in zip(cycles, values)
        if cycle >= start_cycle and (end_cycle is None or cycle < end_cycle)
    ]
    if len(samples) < 2:
        return None
    tail_count = max(1, min(tail_window, len(samples)))
    tail_values = [value for _, value in samples[-tail_count:]]
    target = sum(tail_values) / len(tail_values)
    tolerance = max(0.5, abs(target) * tolerance_ratio)

    for cycle, value in samples:
        if abs(value - target) <= tolerance:
            return cycle - start_cycle
    return None


def _format_cycles(value: Optional[int]) -> str:
    if value is None:
        return "-"
    return f"{value}"


def _format_rate(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3g}"


def _format_req(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2g}"


def _resolve_labels(paths: Sequence[Path], labels: Optional[Sequence[str]]) -> List[str]:
    if labels:
        if len(labels) != len(paths):
            raise ValueError("--labels length must match --csv length")
        return list(labels)
    return [path.stem for path in paths]


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    result: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install it with: pip install matplotlib") from exc
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


if __name__ == "__main__":
    raise SystemExit(main())
