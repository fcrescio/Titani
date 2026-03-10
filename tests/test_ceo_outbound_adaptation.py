import os
import unittest
from unittest.mock import patch

from titani.ceo import _next_outbound_adaptation_state, _resolve_outbound_adaptation_start
from titani.ceo_components.config import CeoConfig


class TestCeoOutboundAdaptationStart(unittest.TestCase):
    def test_uses_track_snapshot_when_no_explicit_override(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = CeoConfig()

        prebuffer_target, max_buffer_target_ms = _resolve_outbound_adaptation_start(
            startup_snapshot={"prebuffer_chunks": 4, "max_buffer_ms": 280},
            cfg=cfg,
        )

        self.assertEqual(prebuffer_target, 4)
        self.assertEqual(max_buffer_target_ms, 280)

    def test_uses_explicit_config_when_provided(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CEO_OUTBOUND_ADAPTIVE_START_PREBUFFER_CHUNKS": "6",
                "CEO_OUTBOUND_ADAPTIVE_START_MAX_BUFFER_MS": "300",
            },
            clear=True,
        ):
            cfg = CeoConfig()

        prebuffer_target, max_buffer_target_ms = _resolve_outbound_adaptation_start(
            startup_snapshot={"prebuffer_chunks": 2, "max_buffer_ms": 100},
            cfg=cfg,
        )

        self.assertEqual(prebuffer_target, 6)
        self.assertEqual(max_buffer_target_ms, 300)

    def test_clamps_values_to_runtime_bounds(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CEO_OUTBOUND_ADAPTIVE_START_PREBUFFER_CHUNKS": "999",
                "CEO_OUTBOUND_ADAPTIVE_START_MAX_BUFFER_MS": "9999",
            },
            clear=True,
        ):
            cfg = CeoConfig()

        prebuffer_target, max_buffer_target_ms = _resolve_outbound_adaptation_start(
            startup_snapshot={"prebuffer_chunks": 0, "max_buffer_ms": 0},
            cfg=cfg,
        )

        self.assertEqual(prebuffer_target, cfg.outbound_adaptation.prebuffer_max_chunks)
        self.assertEqual(max_buffer_target_ms, cfg.outbound_adaptation.buffer_max_ms)

        with patch.dict(
            os.environ,
            {
                "CEO_OUTBOUND_ADAPTIVE_START_PREBUFFER_CHUNKS": "-1",
                "CEO_OUTBOUND_ADAPTIVE_START_MAX_BUFFER_MS": "1",
            },
            clear=True,
        ):
            cfg = CeoConfig()

        prebuffer_target, max_buffer_target_ms = _resolve_outbound_adaptation_start(
            startup_snapshot={"prebuffer_chunks": 0, "max_buffer_ms": 0},
            cfg=cfg,
        )

        self.assertEqual(prebuffer_target, cfg.outbound_adaptation.prebuffer_min_chunks)
        self.assertEqual(max_buffer_target_ms, cfg.outbound_adaptation.buffer_min_ms)


class TestCeoOutboundAdaptationStateTransitions(unittest.TestCase):
    def test_network_degraded_increases_buffer_targets(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = CeoConfig()

        prebuffer, max_buffer_ms, reason = _next_outbound_adaptation_state(
            prebuffer_target=3,
            max_buffer_target_ms=200,
            jitter_s=cfg.outbound_adaptation.jitter_high_s,
            round_trip_time_s=0.0,
            packets_lost=0,
            cfg=cfg,
        )

        self.assertEqual(reason, "network-degraded")
        self.assertEqual(prebuffer, 4)
        self.assertEqual(max_buffer_ms, 221)

    def test_network_stable_reduces_buffer_targets(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = CeoConfig()

        prebuffer, max_buffer_ms, reason = _next_outbound_adaptation_state(
            prebuffer_target=4,
            max_buffer_target_ms=300,
            jitter_s=cfg.outbound_adaptation.stable_jitter_s,
            round_trip_time_s=cfg.outbound_adaptation.stable_rtt_s,
            packets_lost=0,
            cfg=cfg,
        )

        self.assertEqual(reason, "network-stable")
        self.assertEqual(prebuffer, 3)
        self.assertEqual(max_buffer_ms, 279)


if __name__ == "__main__":
    unittest.main()
