"""Cycle-accurate flow-control simulator."""

from .config import DelayConfig, SimulationConfig, ServiceSegment, TrafficConfig
from .controller import (
    Controller,
    NocWatermarkController,
    OpenLoopController,
    ProportionalOutstandingController,
    SenderOutstandingController,
    build_controller,
)

__all__ = [
    "Controller",
    "DelayConfig",
    "NocWatermarkController",
    "OpenLoopController",
    "ProportionalOutstandingController",
    "SenderOutstandingController",
    "ServiceSegment",
    "TrafficConfig",
    "ControllerSpec",
    "ExperimentConfig",
    "ExperimentResult",
    "PlotConfig",
    "SimulationConfig",
    "SimulationResult",
    "build_controller",
    "experiment_from_mapping",
    "load_experiment",
    "run_comparison",
    "run_experiment",
    "run_simulation",
]


def __getattr__(name: str):
    if name in {"SimulationResult", "run_comparison", "run_simulation"}:
        from . import simulate

        return getattr(simulate, name)
    if name in {
        "ControllerSpec",
        "ExperimentConfig",
        "ExperimentResult",
        "PlotConfig",
        "experiment_from_mapping",
        "load_experiment",
        "run_experiment",
    }:
        from . import experiment

        return getattr(experiment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
