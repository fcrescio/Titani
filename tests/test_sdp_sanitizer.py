import unittest

from titani.common import ErmeteConfig


class TestCommon(unittest.TestCase):
    def test_auth_headers_empty_without_psk(self) -> None:
        cfg = ErmeteConfig(psk="")
        self.assertEqual(cfg.auth_headers(), {})


if __name__ == "__main__":
    unittest.main()
