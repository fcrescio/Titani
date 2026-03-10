import os
import unittest
from unittest.mock import patch

from titani.ceo_components.config import CeoConfig


class TestCeoConfigNoopEnv(unittest.TestCase):
    def test_warns_when_unsupported_tts_language_env_is_set(self) -> None:
        with patch.dict(os.environ, {"CEO_TTS_LANGUAGE": "Italian"}, clear=False):
            with self.assertLogs("titani.ceo_components.config", level="WARNING") as captured:
                CeoConfig()

        self.assertTrue(
            any("CEO_TTS_LANGUAGE='Italian' ignorato" in line for line in captured.output),
            msg=f"log non trovato: {captured.output}",
        )

    def test_no_warning_when_unsupported_env_var_is_absent(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertNoLogs("titani.ceo_components.config", level="WARNING"):
                CeoConfig()


class TestCeoIngressProfiles(unittest.TestCase):
    def test_balanced_profile_is_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = CeoConfig()

        self.assertEqual(cfg.ingress.ingress_profile, "balanced")
        self.assertEqual(cfg.ingress.start_speech_chunks, 10)
        self.assertEqual(cfg.ingress.speech_majority_ratio, 0.5)
        self.assertEqual(cfg.ingress.speech_subchunk_min_count, 2)
        self.assertEqual(cfg.ingress.vad_min_rms, 0.0)

    def test_profile_overrides_vad_defaults(self) -> None:
        with patch.dict(os.environ, {"CEO_INGRESS_PROFILE": "noisy"}, clear=True):
            cfg = CeoConfig()

        self.assertEqual(cfg.ingress.start_speech_chunks, 12)
        self.assertEqual(cfg.ingress.speech_majority_ratio, 0.65)
        self.assertEqual(cfg.ingress.speech_subchunk_min_count, 3)
        self.assertEqual(cfg.ingress.vad_min_rms, 0.02)

    def test_manual_overrides_are_ignored_without_advanced_tuning(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CEO_INGRESS_PROFILE": "balanced",
                "CEO_START_SPEECH_CHUNKS": "2",
                "CEO_SPEECH_MAJORITY_RATIO": "0.1",
                "CEO_SPEECH_SUBCHUNK_MIN_COUNT": "1",
                "CEO_VAD_MIN_RMS": "0.5",
            },
            clear=True,
        ):
            with self.assertLogs("titani.ceo_components.config", level="WARNING") as captured:
                cfg = CeoConfig()

        self.assertEqual(cfg.ingress.start_speech_chunks, 10)
        self.assertEqual(cfg.ingress.speech_majority_ratio, 0.5)
        self.assertEqual(cfg.ingress.speech_subchunk_min_count, 2)
        self.assertEqual(cfg.ingress.vad_min_rms, 0.0)
        self.assertTrue(any("CEO_START_SPEECH_CHUNKS='2' ignorato" in line for line in captured.output))

    def test_manual_overrides_are_applied_with_advanced_tuning(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CEO_INGRESS_PROFILE": "balanced",
                "CEO_ADVANCED_TUNING": "1",
                "CEO_START_SPEECH_CHUNKS": "7",
                "CEO_SPEECH_MAJORITY_RATIO": "0.55",
                "CEO_SPEECH_SUBCHUNK_MIN_COUNT": "3",
                "CEO_VAD_MIN_RMS": "0.03",
            },
            clear=True,
        ):
            cfg = CeoConfig()

        self.assertEqual(cfg.ingress.start_speech_chunks, 7)
        self.assertEqual(cfg.ingress.speech_majority_ratio, 0.55)
        self.assertEqual(cfg.ingress.speech_subchunk_min_count, 3)
        self.assertEqual(cfg.ingress.vad_min_rms, 0.03)

    def test_rejects_unknown_profile(self) -> None:
        with patch.dict(os.environ, {"CEO_INGRESS_PROFILE": "turbo"}, clear=True):
            with self.assertRaisesRegex(ValueError, "CEO_INGRESS_PROFILE"):
                CeoConfig()

    def test_rejects_out_of_range_majority_ratio(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CEO_ADVANCED_TUNING": "1",
                "CEO_SPEECH_MAJORITY_RATIO": "1.5",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "CEO_SPEECH_MAJORITY_RATIO"):
                CeoConfig()


class TestCeoOutboundAdaptivePolicyConfig(unittest.TestCase):
    def test_adaptive_policy_enabled_defaults_to_true(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = CeoConfig()

        self.assertTrue(cfg.outbound.adaptive_policy_enabled)

    def test_adaptive_policy_can_be_disabled(self) -> None:
        with patch.dict(os.environ, {"CEO_OUTBOUND_ADAPTIVE_POLICY_ENABLED": "0"}, clear=True):
            cfg = CeoConfig()

        self.assertFalse(cfg.outbound.adaptive_policy_enabled)

    def test_adaptive_start_values_are_optional(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CEO_OUTBOUND_ADAPTIVE_START_PREBUFFER_CHUNKS": "5",
                "CEO_OUTBOUND_ADAPTIVE_START_MAX_BUFFER_MS": "240",
            },
            clear=True,
        ):
            cfg = CeoConfig()

        self.assertEqual(cfg.outbound.adaptive_start_prebuffer_chunks, 5)
        self.assertEqual(cfg.outbound.adaptive_start_max_buffer_ms, 240)

    def test_outbound_adaptation_section_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = CeoConfig()

        self.assertEqual(cfg.outbound_adaptation.jitter_high_s, 0.03)
        self.assertEqual(cfg.outbound_adaptation.rtt_high_s, 0.25)
        self.assertEqual(cfg.outbound_adaptation.prebuffer_max_chunks, 8)
        self.assertEqual(cfg.outbound_adaptation.buffer_max_ms, 420)

    def test_outbound_adaptation_section_can_be_overridden(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CEO_OUTBOUND_ADAPT_JITTER_HIGH_S": "0.04",
                "CEO_OUTBOUND_ADAPT_RTT_HIGH_S": "0.3",
                "CEO_OUTBOUND_ADAPT_PREBUFFER_MAX_CHUNKS": "10",
                "CEO_OUTBOUND_ADAPT_BUFFER_MAX_MS": "500",
            },
            clear=True,
        ):
            cfg = CeoConfig()

        self.assertEqual(cfg.outbound_adaptation.jitter_high_s, 0.04)
        self.assertEqual(cfg.outbound_adaptation.rtt_high_s, 0.3)
        self.assertEqual(cfg.outbound_adaptation.prebuffer_max_chunks, 10)
        self.assertEqual(cfg.outbound_adaptation.buffer_max_ms, 500)


if __name__ == "__main__":
    unittest.main()
