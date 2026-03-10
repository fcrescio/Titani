import os
import unittest
from unittest.mock import patch

from titani.ceo import (
    OUTBOUND_MAX_BUFFER_MAX_MS,
    OUTBOUND_MAX_BUFFER_MIN_MS,
    OUTBOUND_PREBUFFER_MAX_CHUNKS,
    OUTBOUND_PREBUFFER_MIN_CHUNKS,
    _resolve_outbound_adaptation_start,
)
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

        self.assertEqual(prebuffer_target, OUTBOUND_PREBUFFER_MAX_CHUNKS)
        self.assertEqual(max_buffer_target_ms, OUTBOUND_MAX_BUFFER_MAX_MS)

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

        self.assertEqual(prebuffer_target, OUTBOUND_PREBUFFER_MIN_CHUNKS)
        self.assertEqual(max_buffer_target_ms, OUTBOUND_MAX_BUFFER_MIN_MS)


if __name__ == "__main__":
    unittest.main()
