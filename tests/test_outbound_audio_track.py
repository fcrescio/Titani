import sys
import types

sys.modules.setdefault("webrtcvad", types.SimpleNamespace(Vad=lambda *_args, **_kwargs: None))

import unittest

import numpy as np

from titani.audio_pipeline import TtsOutboundAudioTrack


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

    async def test_push_pcm16_enforces_max_buffer(self) -> None:
        track = TtsOutboundAudioTrack()
        track._max_buffer_ms = 40
        chunk = np.ones(track._chunk_samples, dtype=np.int16)
        audio = np.tile(chunk, 10)

        await track.push_pcm16(audio, sample_rate=track.output_sample_rate)

        self.assertLessEqual(track._pending_chunks, 2)
        self.assertLessEqual(track._queue.qsize(), 2)

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
