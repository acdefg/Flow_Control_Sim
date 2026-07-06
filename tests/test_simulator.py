import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from flow_control_sim.config import DelayConfig, ServiceSegment, SimulationConfig, TrafficConfig
from flow_control_sim.controller import (
    ControlDecision,
    NocWatermarkController,
    SenderOutstandingController,
    SmithPIBurstPeriodController,
    build_controller,
)
from flow_control_sim.experiment import (
    build_wide_comparison_rows,
    experiment_from_mapping,
    format_wide_comparison_markdown,
    run_experiment,
)
from flow_control_sim.plant import DelayFifo, FlowControlPlant, PollingDelayQueue
from flow_control_sim.simulate import run_comparison, run_simulation


class DelayFifoTests(unittest.TestCase):
    def test_zero_delay_is_passthrough(self):
        fifo = DelayFifo(0)
        self.assertEqual(fifo.push(5.0), 5.0)

    def test_delay_two_releases_after_two_cycles(self):
        fifo = DelayFifo(2)
        self.assertEqual(fifo.push(5.0), 0.0)
        self.assertEqual(fifo.push(6.0), 0.0)
        self.assertEqual(fifo.push(7.0), 5.0)

    def test_polling_delay_queue_releases_on_next_poll_slot(self):
        queue = PollingDelayQueue(period=4)

        self.assertEqual(queue.push(1.0, cycle=0), 1.0)
        self.assertEqual(queue.push(2.0, cycle=1), 0.0)
        self.assertEqual(queue.push(0.0, cycle=2), 0.0)
        self.assertEqual(queue.push(0.0, cycle=3), 0.0)
        self.assertEqual(queue.push(0.0, cycle=4), 2.0)


class ControllerTests(unittest.TestCase):
    class State:
        def __init__(self, sender_outstanding=0.0, noc_outstanding=0.0, receiver_queue=0.0, cycle=0):
            self.cycle = cycle
            self.sender_outstanding = sender_outstanding
            self.noc_outstanding = noc_outstanding
            self.receiver_queue = receiver_queue
            self.noc_queue = 0.0
            self.sender_noc_inflight = 0.0

    def test_sender_controller_uses_hard_outstanding_credit(self):
        controller = SenderOutstandingController(
            threshold=64,
            max_send_rate=3.0,
            burst_reqs=16,
            check_interval=44,
        )

        open_decision = controller.update(self.State(sender_outstanding=63.9, cycle=1))
        closed_decision = controller.update(self.State(sender_outstanding=64, cycle=1))

        self.assertGreater(open_decision.send_rate, 0.0)
        self.assertEqual(open_decision.sender_outstanding_limit, 64)
        self.assertEqual(closed_decision.send_rate, 0.0)
        self.assertTrue(closed_decision.raw_throttle)

    def test_noc_watermark_latches_and_delays_feedback(self):
        controller = NocWatermarkController(threshold=10, max_send_rate=3.0, throttle_delay=2)

        first = controller.update(self.State(noc_outstanding=11))
        second = controller.update(self.State(noc_outstanding=10))
        third = controller.update(self.State(noc_outstanding=10))
        fourth = controller.update(self.State(noc_outstanding=9))

        self.assertTrue(first.raw_throttle)
        self.assertFalse(first.delayed_throttle)
        self.assertTrue(second.raw_throttle)
        self.assertFalse(second.delayed_throttle)
        self.assertTrue(third.delayed_throttle)
        self.assertFalse(fourth.raw_throttle)

    def test_smith_pi_increases_burst_period_when_predicted_os_is_high(self):
        controller = SmithPIBurstPeriodController(
            threshold=10,
            max_send_rate=3.0,
            base_period=176,
            prediction_delay=2,
            signal="noc_outstanding",
        )

        baseline = controller.update(self.State(noc_outstanding=5))
        increased = controller.update(self.State(noc_outstanding=20))

        self.assertEqual(baseline.burst_period, 176.0)
        self.assertGreater(increased.burst_period, 176.0)
        self.assertTrue(increased.raw_throttle)

    def test_smith_pi_accepts_ks_alias_for_proportional_gain(self):
        config = SimulationConfig(total_cycles=1)
        controller = build_controller("smith_pi", config, {"ks": 42, "kd": 9})

        self.assertEqual(controller.kp, 42)
        self.assertEqual(controller.kd, 9)

    def test_smith_pi_can_use_delayed_noc_feedback_gap(self):
        controller = SmithPIBurstPeriodController(
            threshold=10,
            max_send_rate=3.0,
            base_period=176,
            kp=10,
            kd=0,
            prediction_delay=0,
            feedback_delay=2,
            signal="sender_delayed_noc_gap",
        )

        # First two cycles see the initially empty feedback path, so the gap is
        # sender_os - 0. The current high NOC OS is visible only after 2 cycles.
        first = controller.update(self.State(sender_outstanding=20, noc_outstanding=15))
        second = controller.update(self.State(sender_outstanding=20, noc_outstanding=15))
        third = controller.update(self.State(sender_outstanding=20, noc_outstanding=15))

        self.assertGreater(first.burst_period, 176.0)
        self.assertGreater(second.burst_period, 176.0)
        self.assertEqual(third.burst_period, 176.0)

    def test_smith_pi_combines_sender_and_delayed_noc_waterlines(self):
        controller = SmithPIBurstPeriodController(
            threshold=0,
            max_send_rate=3.0,
            base_period=176,
            kp=10,
            kd=0,
            prediction_delay=0,
            feedback_delay=2,
            signal="sender_or_delayed_noc_error",
            sender_threshold=20,
            noc_threshold=10,
        )

        first = controller.update(self.State(sender_outstanding=19, noc_outstanding=12))
        second = controller.update(self.State(sender_outstanding=19, noc_outstanding=12))
        third = controller.update(self.State(sender_outstanding=19, noc_outstanding=12))

        self.assertEqual(first.burst_period, 176.0)
        self.assertEqual(second.burst_period, 176.0)
        self.assertGreater(third.burst_period, 176.0)

    def test_smith_pi_can_filter_deadband_and_slew_limit(self):
        controller = SmithPIBurstPeriodController(
            threshold=0,
            max_send_rate=3.0,
            base_period=176,
            kp=100,
            kd=0,
            prediction_delay=0,
            signal="sender_outstanding",
            measurement_filter_alpha=0.5,
            error_deadband=1.0,
            max_period_step=10,
        )

        baseline = controller.update(self.State(sender_outstanding=0))
        first_step = controller.update(self.State(sender_outstanding=10))
        second_step = controller.update(self.State(sender_outstanding=10))

        self.assertEqual(baseline.burst_period, 176.0)
        self.assertEqual(first_step.burst_period, 186.0)
        self.assertEqual(second_step.burst_period, 196.0)

    def test_smith_pi_can_reset_filter_when_error_clears(self):
        controller = SmithPIBurstPeriodController(
            threshold=0,
            max_send_rate=3.0,
            base_period=176,
            kp=100,
            kd=0,
            prediction_delay=0,
            signal="sender_outstanding",
            measurement_filter_alpha=0.1,
            reset_filter_on_nonpositive=True,
        )

        high = controller.update(self.State(sender_outstanding=10))
        cleared = controller.update(self.State(sender_outstanding=0))

        self.assertGreater(high.burst_period, 176.0)
        self.assertEqual(cleared.burst_period, 176.0)


class PlantFluidSenderTests(unittest.TestCase):
    def test_sender_injects_equivalent_bandwidth_continuously(self):
        traffic = TrafficConfig(
            request_bytes=32,
            sender_burst_reqs=4,
            sender_check_interval=4,
            sender_burst_period=16,
            noc_queue_capacity_reqs=128,
            sender_noc_inflight_capacity_reqs=256,
        )
        plant = FlowControlPlant(
            delays=DelayConfig(forward_sn=1, forward_nr=1, return_rn=1, return_ns=1, throttle=1),
            max_send_rate=3.0,
            traffic=traffic,
        )

        records = [
            plant.step(ControlDecision(send_rate=3.0), service_rate=3.0),
            plant.step(ControlDecision(send_rate=0.0, raw_throttle=True, delayed_throttle=True), service_rate=3.0),
            plant.step(ControlDecision(send_rate=0.0, raw_throttle=True, delayed_throttle=True), service_rate=3.0),
            plant.step(ControlDecision(send_rate=0.0, raw_throttle=True, delayed_throttle=True), service_rate=3.0),
        ]

        self.assertEqual(records[0].sent_reqs, 3.0 / 32.0)
        self.assertEqual([record.sent_reqs for record in records[1:]], [0.0, 0.0, 0.0])

    def test_dynamic_period_controls_equivalent_bandwidth(self):
        traffic = TrafficConfig(
            request_bytes=32,
            sender_burst_reqs=2,
            sender_check_interval=1,
            sender_burst_period=4,
            noc_queue_capacity_reqs=128,
            sender_noc_inflight_capacity_reqs=256,
            noc_queue_delay=0,
        )
        plant = FlowControlPlant(
            delays=DelayConfig(forward_sn=1, forward_nr=1, return_rn=1, return_ns=1, throttle=1),
            max_send_rate=3.0,
            traffic=traffic,
        )

        records = [plant.step(ControlDecision(send_rate=3.0, burst_period=8), service_rate=32.0)]
        records.extend(
            plant.step(ControlDecision(send_rate=3.0, burst_period=8), service_rate=32.0)
            for _ in range(8)
        )

        self.assertEqual(records[0].burst_period, 8.0)
        self.assertEqual(records[0].commanded_bandwidth, 3.0)
        self.assertEqual(
            records[0].sender_noc_os_gap,
            records[0].sender_outstanding - records[0].noc_outstanding,
        )
        self.assertEqual(
            [record.sent_reqs for record in records],
            [3.0 / 32.0] * 9,
        )

    def test_burst_sender_emits_contiguous_requests_then_waits(self):
        traffic = TrafficConfig(
            request_bytes=32,
            sender_burst_reqs=3,
            sender_check_interval=1,
            sender_burst_period=8,
            sender_mode="burst",
            noc_queue_capacity_reqs=128,
            sender_noc_inflight_capacity_reqs=256,
            noc_queue_delay=0,
        )
        plant = FlowControlPlant(
            delays=DelayConfig(forward_sn=1, forward_nr=1, return_rn=1, return_ns=1, throttle=1),
            max_send_rate=32.0,
            traffic=traffic,
        )

        records = [plant.step(ControlDecision(send_rate=32.0), service_rate=32.0) for _ in range(10)]

        self.assertEqual([record.sent_reqs for record in records], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        self.assertEqual(records[0].commanded_bandwidth, 12.0)

    def test_sender_outstanding_limit_caps_partial_cycle_send(self):
        traffic = TrafficConfig(
            request_bytes=32,
            sender_burst_reqs=16,
            sender_check_interval=1,
            sender_burst_period=176,
            noc_queue_capacity_reqs=128,
            sender_noc_inflight_capacity_reqs=256,
            noc_queue_delay=0,
        )
        plant = FlowControlPlant(
            delays=DelayConfig(forward_sn=1, forward_nr=1, return_rn=1, return_ns=1, throttle=1),
            max_send_rate=3.0,
            traffic=traffic,
        )
        plant.sender_outstanding = 0.99

        record = plant.step(
            ControlDecision(send_rate=3.0, sender_outstanding_limit=1.0),
            service_rate=0.0,
        )

        self.assertAlmostEqual(record.sent_reqs, 0.01)
        self.assertLessEqual(record.sender_outstanding, 1.0)

    def test_noc_fixed_and_polling_queue_delays_are_composed(self):
        traffic = TrafficConfig(
            request_bytes=32,
            sender_burst_reqs=1,
            sender_check_interval=1,
            sender_burst_period=1,
            noc_queue_capacity_reqs=128,
            sender_noc_inflight_capacity_reqs=256,
            noc_queue_delay=2,
            noc_queue_poll_period=4,
        )
        plant = FlowControlPlant(
            delays=DelayConfig(forward_sn=0, forward_nr=0, return_rn=1, return_ns=1, throttle=1),
            max_send_rate=32.0,
            traffic=traffic,
        )

        records = [plant.step(ControlDecision(send_rate=32.0), service_rate=0.0)]
        records.extend(plant.step(ControlDecision(send_rate=0.0), service_rate=0.0) for _ in range(4))

        self.assertEqual([record.arrival_receiver for record in records], [0.0, 0.0, 0.0, 0.0, 1.0])


class SimulationTests(unittest.TestCase):
    def test_sender_controller_runs_expected_number_of_cycles(self):
        config = SimulationConfig(
            total_cycles=300,
            sender_threshold=32.0,
            noc_threshold=32.0,
            delays=DelayConfig(forward_sn=2, forward_nr=2, return_rn=2, return_ns=2, throttle=4),
            traffic=TrafficConfig(noc_queue_delay=0),
            service_profile=(
                ServiceSegment(0, 3.0),
                ServiceSegment(100, 1.0),
                ServiceSegment(200, 3.0),
            ),
        )
        result = run_simulation(config, build_controller("sender", config))
        self.assertEqual(len(result.history), 300)
        self.assertGreater(result.summary["throughput"], 0.0)
        self.assertLessEqual(result.summary["throughput"], config.max_send_rate)

    def test_comparison_runs_sender_and_noc(self):
        config = SimulationConfig(total_cycles=120)
        results = run_comparison(config)
        self.assertEqual(set(results), {"sender", "noc"})
        self.assertEqual(len(results["sender"].history), 120)
        self.assertEqual(len(results["noc"].history), 120)


class ExperimentTests(unittest.TestCase):
    def test_experiment_runs_comparison_and_writes_wide_table(self):
        with TemporaryDirectory() as tmp:
            data = {
                "name": "unit_experiment",
                "output_dir": "out",
                "simulation": {
                    "total_cycles": 120,
                    "max_send_rate": 3.0,
                    "thresholds": {"sender": 64, "noc": 64},
                    "delays": {
                        "forward_sn": 2,
                        "forward_nr": 2,
                        "return_rn": 2,
                        "return_ns": 2,
                        "throttle": 4,
                    },
                    "receiver": {
                        "service_profile": [
                            {"start_cycle": 0, "rate": 3.0},
                            {"start_cycle": 40, "rate": 1.0},
                            {"start_cycle": 80, "rate": 3.0},
                        ]
                    },
                },
                "controllers": [
                    {"kind": "noc", "label": "方案一（NOC Watermark）"},
                    {"kind": "sender", "label": "方案二（Sender Outstanding）"},
                    {
                        "kind": "smith_pi",
                        "label": "方案三（Smith PI）",
                        "params": {"prediction_delay": 2, "max_period": 512},
                    },
                ],
                "outputs": {"plots": False},
            }
            config = experiment_from_mapping(data, base_dir=Path(tmp))
            result = run_experiment(config)

            self.assertEqual(
                set(result.results),
                {"方案一（NOC Watermark）", "方案二（Sender Outstanding）", "方案三（Smith PI）"},
            )
            self.assertTrue(result.wide_summary_path.exists())
            self.assertTrue(result.markdown_summary_path.exists())

            rows = build_wide_comparison_rows(result.results)
            markdown = format_wide_comparison_markdown(rows)
            self.assertIn("最大 Sender Outstanding", markdown)
            self.assertIn("方案三（Smith PI）", markdown)


if __name__ == "__main__":
    unittest.main()
