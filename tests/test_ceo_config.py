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


if __name__ == "__main__":
    unittest.main()
