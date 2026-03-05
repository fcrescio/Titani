import unittest

from titani.common import _remove_unmapped_dynamic_payload_types_from_sdp


class TestSdpSanitizer(unittest.TestCase):
    def test_removes_unmapped_dynamic_payloads_and_related_attrs(self) -> None:
        sdp = (
            "v=0\r\n"
            "m=audio 9 UDP/TLS/RTP/SAVPF 111 112\r\n"
            "a=rtpmap:111 opus/48000/2\r\n"
            "a=fmtp:111 minptime=10\r\n"
            "a=fmtp:112 apt=111\r\n"
        )

        out = _remove_unmapped_dynamic_payload_types_from_sdp(sdp)

        self.assertIn("m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n", out)
        self.assertNotIn("a=fmtp:112", out)

    def test_removes_unknown_static_payload_without_rtpmap(self) -> None:
        sdp = (
            "v=0\r\n"
            "m=audio 9 UDP/TLS/RTP/SAVPF 0 19 111\r\n"
            "a=rtpmap:111 opus/48000/2\r\n"
        )

        out = _remove_unmapped_dynamic_payload_types_from_sdp(sdp)

        self.assertIn("m=audio 9 UDP/TLS/RTP/SAVPF 0 111\r\n", out)
        self.assertNotIn(" 19", out)

    def test_keeps_known_static_payloads_without_rtpmap(self) -> None:
        sdp = (
            "v=0\r\n"
            "m=audio 9 UDP/TLS/RTP/SAVPF 0 8 111\r\n"
            "a=rtpmap:111 opus/48000/2\r\n"
        )

        out = _remove_unmapped_dynamic_payload_types_from_sdp(sdp)

        self.assertIn("m=audio 9 UDP/TLS/RTP/SAVPF 0 8 111\r\n", out)


if __name__ == "__main__":
    unittest.main()
