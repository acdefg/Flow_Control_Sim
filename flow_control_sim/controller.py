"""Flow-control controller implementations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Mapping, Optional, Protocol

from .config import SimulationConfig

DEFAULT_SMITH_KP = 40.0
DEFAULT_SMITH_KI = 0.0
DEFAULT_SMITH_KD = 0.0


@dataclass(frozen=True)
class ControlDecision:
    send_rate: float
    raw_throttle: bool = False
    delayed_throttle: bool = False
    burst_period: Optional[float] = None
    sender_outstanding_limit: Optional[float] = None


class StateView(Protocol):
    cycle: int
    sender_outstanding: float
    noc_outstanding: float
    receiver_queue: float


class Controller(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def update(self, state: StateView) -> ControlDecision:
        ...


class OpenLoopController:
    """Always allow sender traffic; useful as a no-flow-control baseline."""

    def __init__(self, max_send_rate: float) -> None:
        self.max_send_rate = max_send_rate
        self.name = "open_loop"

    def reset(self) -> None:
        return None

    def update(self, state: StateView) -> ControlDecision:
        del state
        return ControlDecision(send_rate=self.max_send_rate)


class SenderOutstandingController:
    """Limit sender traffic using a local outstanding-credit window."""

    def __init__(
        self,
        threshold: float,
        max_send_rate: float,
        burst_reqs: int = 1,
        check_interval: int = 1,
    ) -> None:
        if check_interval <= 0:
            raise ValueError("check_interval must be > 0")
        self.threshold = threshold
        self.max_send_rate = max_send_rate
        self.burst_reqs = burst_reqs
        self.check_interval = check_interval
        self.name = f"sender_outstanding_A{threshold:g}"

    def reset(self) -> None:
        return None

    def update(self, state: StateView) -> ControlDecision:
        throttle = state.sender_outstanding >= self.threshold
        return ControlDecision(
            send_rate=0.0 if throttle else self.max_send_rate,
            raw_throttle=throttle,
            delayed_throttle=throttle,
            sender_outstanding_limit=self.threshold,
        )


class NocWatermarkController:
    """Throttle sender using a delayed NOC outstanding watermark signal."""

    def __init__(self, threshold: float, max_send_rate: float, throttle_delay: int = 44) -> None:
        if throttle_delay < 0:
            raise ValueError("throttle_delay must be >= 0")
        self.threshold = threshold
        self.max_send_rate = max_send_rate
        self.throttle_delay = throttle_delay
        self.name = f"noc_watermark_A{threshold:g}_D{throttle_delay:g}"
        self._delay: Deque[bool] = deque([False] * throttle_delay)
        self._raw_throttle_latched = False

    def reset(self) -> None:
        self._delay = deque([False] * self.throttle_delay)
        self._raw_throttle_latched = False

    def update(self, state: StateView) -> ControlDecision:
        if state.noc_outstanding > self.threshold:
            self._raw_throttle_latched = True
        elif state.noc_outstanding < self.threshold:
            self._raw_throttle_latched = False
        raw_throttle = self._raw_throttle_latched
        if self.throttle_delay == 0:
            delayed_throttle = raw_throttle
        else:
            self._delay.append(raw_throttle)
            delayed_throttle = self._delay.popleft()
        return ControlDecision(
            send_rate=0.0 if delayed_throttle else self.max_send_rate,
            raw_throttle=raw_throttle,
            delayed_throttle=delayed_throttle,
        )


class ProportionalOutstandingController:
    """Simple P-style controller for later experiments."""

    def __init__(
        self,
        target: float,
        gain: float,
        max_send_rate: float,
        signal: str = "sender_outstanding",
    ) -> None:
        if gain < 0:
            raise ValueError("gain must be >= 0")
        if signal not in {"sender_outstanding", "noc_outstanding", "receiver_queue"}:
            raise ValueError(f"unsupported signal: {signal}")
        self.target = target
        self.gain = gain
        self.max_send_rate = max_send_rate
        self.signal = signal
        self.name = f"p_{signal}_target{target:g}_k{gain:g}"

    def reset(self) -> None:
        return None

    def update(self, state: StateView) -> ControlDecision:
        measured = getattr(state, self.signal)
        send_rate = self.max_send_rate - self.gain * (measured - self.target)
        send_rate = max(0.0, min(self.max_send_rate, send_rate))
        return ControlDecision(
            send_rate=send_rate,
            raw_throttle=send_rate <= 0.0,
            delayed_throttle=send_rate <= 0.0,
        )


class SmithPIBurstPeriodController:
    """Smith-predicted PI-style controller that changes sender burst spacing.

    The controlled signal is predicted as:

        z_hat(k) = z(k) + (z(k) - z(k - prediction_delay))

    and the commanded burst period is:

        I(k) = clamp(I(k - 1) + deadband(z_hat(k) - A))

        T(k) = base_period + kp * max(0, deadband(z_hat(k) - A))
               + ki * I(k)
               + kd * (z_hat(k) - z_hat(k - 1))

    ``T`` is clamped so the sender never exceeds the configured maximum
    16-request-per-176-cycle offered bandwidth.
    """

    SUPPORTED_SIGNALS = {
        "sender_outstanding",
        "noc_outstanding",
        "receiver_queue",
        "noc_queue",
        "sender_noc_inflight",
        "delayed_noc_outstanding",
        "sender_delayed_noc_gap",
        "sender_or_delayed_noc_error",
    }

    def __init__(
        self,
        threshold: float,
        max_send_rate: float,
        base_period: int,
        kp: float = DEFAULT_SMITH_KP,
        ki: float = DEFAULT_SMITH_KI,
        kd: float = DEFAULT_SMITH_KD,
        prediction_delay: int = 40,
        feedback_delay: int = 44,
        min_period: Optional[int] = None,
        max_period: Optional[int] = None,
        integral_limit: Optional[float] = None,
        signal: str = "sender_outstanding",
        sender_threshold: Optional[float] = None,
        noc_threshold: Optional[float] = None,
        measurement_filter_alpha: float = 1.0,
        error_deadband: float = 0.0,
        max_period_step: Optional[float] = None,
        reset_filter_on_nonpositive: bool = False,
    ) -> None:
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        if base_period <= 0:
            raise ValueError("base_period must be > 0")
        if prediction_delay < 0:
            raise ValueError("prediction_delay must be >= 0")
        if feedback_delay < 0:
            raise ValueError("feedback_delay must be >= 0")
        if ki < 0:
            raise ValueError("ki must be >= 0")
        if not 0.0 < measurement_filter_alpha <= 1.0:
            raise ValueError("measurement_filter_alpha must be in (0, 1]")
        if error_deadband < 0:
            raise ValueError("error_deadband must be >= 0")
        if integral_limit is not None and integral_limit <= 0:
            raise ValueError("integral_limit must be > 0 when set")
        if max_period_step is not None and max_period_step <= 0:
            raise ValueError("max_period_step must be > 0 when set")
        if signal not in self.SUPPORTED_SIGNALS:
            raise ValueError(f"unsupported signal: {signal}")
        self.threshold = threshold
        self.max_send_rate = max_send_rate
        self.base_period = base_period
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prediction_delay = prediction_delay
        self.feedback_delay = feedback_delay
        self.min_period = min_period if min_period is not None else base_period
        self.max_period = max_period if max_period is not None else base_period * 64
        self.signal = signal
        self.sender_threshold = sender_threshold
        self.noc_threshold = noc_threshold
        self.measurement_filter_alpha = measurement_filter_alpha
        self.error_deadband = error_deadband
        self.max_period_step = max_period_step
        self.reset_filter_on_nonpositive = reset_filter_on_nonpositive
        if self.min_period <= 0:
            raise ValueError("min_period must be > 0")
        if self.max_period < self.min_period:
            raise ValueError("max_period must be >= min_period")
        if integral_limit is None and ki > 0:
            integral_limit = (float(self.max_period) - float(self.min_period)) / ki
        self.integral_limit = integral_limit
        self.name = (
            f"smith_pi_{signal}_A{threshold:g}_P{kp:g}_I{ki:g}_D{kd:g}_pred{prediction_delay:g}"
        )
        self._history: Deque[float] = deque([0.0] * prediction_delay)
        self._noc_feedback_delay: Deque[float] = deque([0.0] * feedback_delay)
        self._prev_z_hat: Optional[float] = None
        self._filtered_measurement: Optional[float] = None
        self._prev_period: Optional[float] = None
        self._integral_error = 0.0

    def reset(self) -> None:
        self._history = deque([0.0] * self.prediction_delay)
        self._noc_feedback_delay = deque([0.0] * self.feedback_delay)
        self._prev_z_hat = None
        self._filtered_measurement = None
        self._prev_period = None
        self._integral_error = 0.0

    def update(self, state: StateView) -> ControlDecision:
        measured = self._filtered_measure(self._measure(state))
        if self.prediction_delay == 0:
            delayed = measured
        else:
            delayed = self._history.popleft()
            self._history.append(measured)

        z_hat = measured + (measured - delayed)
        delta_z = 0.0 if self._prev_z_hat is None else z_hat - self._prev_z_hat
        self._prev_z_hat = z_hat

        signed_error = self._apply_deadband(z_hat - self.threshold)
        proportional_error = max(0.0, signed_error)
        effective_delta_z = delta_z if proportional_error > 0.0 else 0.0
        self._integral_error = self._clamp_integral(self._integral_error + signed_error)

        period = (
            self.base_period
            + self.kp * proportional_error
            + self.ki * self._integral_error
            + self.kd * effective_delta_z
        )
        period = max(float(self.min_period), min(float(self.max_period), period))
        if self.max_period_step is not None and self._prev_period is not None:
            lower = self._prev_period - self.max_period_step
            upper = self._prev_period + self.max_period_step
            period = max(lower, min(upper, period))
        period = round(period)
        self._prev_period = float(period)
        active = period > self.base_period
        return ControlDecision(
            send_rate=self.max_send_rate,
            raw_throttle=active,
            delayed_throttle=active,
            burst_period=float(period),
        )

    def _measure(self, state: StateView) -> float:
        if self.signal != "sender_delayed_noc_gap":
            if self.signal == "delayed_noc_outstanding":
                return self._delayed_noc(state)
            if self.signal == "sender_or_delayed_noc_error":
                delayed_noc = self._delayed_noc(state)
                sender_limit = (
                    self.threshold if self.sender_threshold is None else self.sender_threshold
                )
                noc_limit = self.threshold if self.noc_threshold is None else self.noc_threshold
                return max(
                    float(state.sender_outstanding) - sender_limit,
                    delayed_noc - noc_limit,
                )
            return float(getattr(state, self.signal))
        delayed_noc = self._delayed_noc(state)
        return float(state.sender_outstanding) - delayed_noc

    def _delayed_noc(self, state: StateView) -> float:
        if self.feedback_delay == 0:
            return float(state.noc_outstanding)
        delayed_noc = self._noc_feedback_delay.popleft()
        self._noc_feedback_delay.append(float(state.noc_outstanding))
        return delayed_noc

    def _filtered_measure(self, measured: float) -> float:
        if self.reset_filter_on_nonpositive and measured <= 0.0:
            self._filtered_measurement = 0.0
            return 0.0
        if self.measurement_filter_alpha >= 1.0:
            return measured
        if self._filtered_measurement is None:
            self._filtered_measurement = measured
        else:
            alpha = self.measurement_filter_alpha
            self._filtered_measurement = alpha * measured + (1.0 - alpha) * self._filtered_measurement
        return self._filtered_measurement

    def _apply_deadband(self, error: float) -> float:
        if error > self.error_deadband:
            return error - self.error_deadband
        if error < -self.error_deadband:
            return error + self.error_deadband
        return 0.0

    def _clamp_integral(self, value: float) -> float:
        if self.integral_limit is None:
            return value
        return max(-self.integral_limit, min(self.integral_limit, value))


def build_controller(
    kind: str,
    config: SimulationConfig,
    params: Optional[Mapping[str, Any]] = None,
) -> Controller:
    params = params or {}
    normalized = kind.lower().strip()
    if normalized in {"open", "open_loop", "none", "no_control"}:
        return OpenLoopController(max_send_rate=config.max_send_rate)
    if normalized in {"sender", "sender_outstanding", "controller1"}:
        return SenderOutstandingController(
            threshold=config.sender_threshold,
            max_send_rate=config.max_send_rate,
            burst_reqs=config.traffic.sender_burst_reqs,
            check_interval=config.traffic.sender_check_interval,
        )
    if normalized in {"noc", "noc_watermark", "controller2"}:
        return NocWatermarkController(
            threshold=config.noc_threshold,
            max_send_rate=config.max_send_rate,
            throttle_delay=config.delays.throttle,
        )
    if normalized in {"p", "proportional"}:
        return ProportionalOutstandingController(
            target=config.sender_threshold,
            gain=0.02,
            max_send_rate=config.max_send_rate,
        )
    if normalized in {"smith", "smith_pi", "pi", "controller3"}:
        signal = str(params.get("signal", "sender_outstanding"))
        if signal == "sender_delayed_noc_gap":
            default_threshold = config.sender_threshold - config.noc_threshold
        elif signal == "sender_or_delayed_noc_error":
            default_threshold = 0.0
        else:
            default_threshold = (
                config.sender_threshold if signal == "sender_outstanding" else config.noc_threshold
            )
        kp_default = _float_param(params, "ks", DEFAULT_SMITH_KP)
        return SmithPIBurstPeriodController(
            threshold=_float_param(params, "threshold", default_threshold),
            max_send_rate=config.max_send_rate,
            base_period=_int_param(params, "base_period", config.traffic.sender_burst_period),
            kp=_float_param(params, "kp", kp_default),
            ki=_float_param(params, "ki", DEFAULT_SMITH_KI),
            kd=_float_param(params, "kd", DEFAULT_SMITH_KD),
            prediction_delay=_int_param(params, "prediction_delay", 40),
            feedback_delay=_int_param(params, "feedback_delay", config.delays.throttle),
            min_period=_optional_int_param(params, "min_period", config.traffic.sender_burst_period),
            max_period=_optional_int_param(params, "max_period", config.traffic.sender_burst_period * 64),
            integral_limit=_optional_float_param(params, "integral_limit", None),
            signal=signal,
            sender_threshold=_optional_float_param(params, "sender_threshold", config.sender_threshold),
            noc_threshold=_optional_float_param(params, "noc_threshold", config.noc_threshold),
            measurement_filter_alpha=_float_param(params, "measurement_filter_alpha", 1.0),
            error_deadband=_float_param(params, "error_deadband", 0.0),
            max_period_step=_optional_float_param(params, "max_period_step", None),
            reset_filter_on_nonpositive=_bool_param(params, "reset_filter_on_nonpositive", False),
        )
    raise ValueError(f"unknown controller kind: {kind!r}")


def _float_param(params: Mapping[str, Any], name: str, default: float) -> float:
    return float(params.get(name, default))


def _int_param(params: Mapping[str, Any], name: str, default: int) -> int:
    return int(params.get(name, default))


def _optional_int_param(params: Mapping[str, Any], name: str, default: Optional[int]) -> Optional[int]:
    value = params.get(name, default)
    if value is None:
        return None
    return int(value)


def _optional_float_param(params: Mapping[str, Any], name: str, default: Optional[float]) -> Optional[float]:
    value = params.get(name, default)
    if value is None:
        return None
    return float(value)


def _bool_param(params: Mapping[str, Any], name: str, default: bool) -> bool:
    value = params.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
