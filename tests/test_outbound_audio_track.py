import sys
import types

sys.modules.setdefault("webrtcvad", types.SimpleNamespace(Vad=lambda *_args, **_kwargs: None))

import unittest

import numpy as np

from titani.audio_pipeline import TtsOutboundAudioTrack
from titani.ceo_components.config import WEBRTC_CHUNK_MS, derive_outbound_buffer_watermarks


class TestTtsOutboundAudioTrack(unittest.IsolatedAsyncioTestCase):
    def test_prepare_chunks_resamples_and_pads(self) -> None:
        src = np.arange(160, dtype=np.int16)
        chunks = TtsOutboundAudioTrack._prepare_chunks(
            src,
            src_rate=16000,
            dst_rate=48000,
            chunk_samples=480,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].shape[0], 480)

    def test_prepare_chunks_downsample_length(self) -> None:
        src = np.arange(960, dtype=np.int16)
        chunks = TtsOutboundAudioTrack._prepare_chunks(
            src,
            src_rate=48000,
            dst_rate=16000,
            chunk_samples=160,
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].shape[0], 160)
        self.assertEqual(chunks[1].shape[0], 160)

    def test_derive_outbound_buffer_watermarks_ratio(self) -> None:
        low_ms, high_ms = derive_outbound_buffer_watermarks(500)

        self.assertEqual(high_ms, 500)
        self.assertEqual(low_ms, 200)

    def test_derive_outbound_buffer_watermarks_chunk_clamp(self) -> None:
        low_ms, high_ms = derive_outbound_buffer_watermarks(WEBRTC_CHUNK_MS)

        self.assertEqual(high_ms, WEBRTC_CHUNK_MS * 2)
        self.assertEqual(low_ms, WEBRTC_CHUNK_MS)

    async def test_update_buffer_policy_enforces_chunk_clamp(self) -> None:
        track = TtsOutboundAudioTrack()

        snapshot = await track.update_buffer_policy(target_buffer_ms=WEBRTC_CHUNK_MS)

        self.assertEqual(snapshot["target_buffer_ms"], WEBRTC_CHUNK_MS * 2)
        self.assertEqual(snapshot["high_watermark_ms"], WEBRTC_CHUNK_MS * 2)
        self.assertEqual(snapshot["low_watermark_ms"], WEBRTC_CHUNK_MS)

    async def test_recv_pts_monotonic_and_state_transitions(self) -> None:
        track = TtsOutboundAudioTrack()

        first = await track.recv()
        self.assertEqual(first.pts, 0)
        self.assertTrue(track._buffering)

        pcm = np.ones(track._chunk_samples * 3, dtype=np.int16)
        await track.push_pcm16(pcm, sample_rate=track.output_sample_rate)

        f2 = await track.recv()
        f3 = await track.recv()
        f4 = await track.recv()
        self.assertEqual(f2.pts, track._chunk_samples)
        self.assertEqual(f3.pts, track._chunk_samples * 2)
        self.assertEqual(f4.pts, track._chunk_samples * 3)

        _ = await track.recv()
        self.assertTrue(track._buffering)
        self.assertTrue(track._playback_idle.is_set())


if __name__ == "__main__":
    unittest.main()
