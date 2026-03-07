import asyncio
import logging
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import AudioFrame
from mlx_audio.tts.utils import load_model as load_tts

from .audio_utils import resample_pcm16
from .config import (
    DEFAULT_WEBRTC_SAMPLE_RATE,
    OUTBOUND_MAX_BUFFER_MS,
    OUTBOUND_PREBUFFER_CHUNKS,
    TARGET_SAMPLE_RATE,
    WEBRTC_CHUNK_MS,
    CeoConfig,
)

logger = logging.getLogger(__name__)


class TtsOutboundAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=256)
        self._sample_rate = DEFAULT_WEBRTC_SAMPLE_RATE
        self._chunk_samples = max(1, self._sample_rate * WEBRTC_CHUNK_MS // 1000)
        self._pts = 0
        self._next_pts = 0
        self._started_at: float | None = None
        self._pending_chunks = 0
        self._pending_lock = asyncio.Lock()
        self._playback_idle = asyncio.Event()
        self._playback_idle.set()
        self._prebuffer_chunks = OUTBOUND_PREBUFFER_CHUNKS
        self._max_buffer_ms = OUTBOUND_MAX_BUFFER_MS
        self._min_buffered_chunks_for_playback = self._prebuffer_chunks
        self._buffering = True
        self._consumer_started = asyncio.Event()

    def set_output_sample_rate(self, sample_rate: int) -> None:
        if sample_rate <= 0 or sample_rate == self._sample_rate:
            return

        self._sample_rate = sample_rate
        self._chunk_samples = max(1, self._sample_rate * WEBRTC_CHUNK_MS // 1000)
        self._pts = 0
        self._next_pts = 0
        self._started_at: float | None = None
        logger.info("[ceo] outbound sample rate aggiornato a %sHz", self._sample_rate)

    @property
    def output_sample_rate(self) -> int:
        return self._sample_rate

    async def recv(self) -> AudioFrame:
        if not self._consumer_started.is_set():
            self._consumer_started.set()

        chunk_samples = int(self._chunk_samples)
        sample_rate = int(self._sample_rate)

        def _silence():
            return np.zeros(chunk_samples, dtype=np.int16)

        def _make_frame(pcm16: np.ndarray) -> AudioFrame:
            frame = AudioFrame.from_ndarray(pcm16.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = sample_rate
            frame.time_base = Fraction(1, sample_rate)
            frame.pts = self._next_pts
            return frame

        frame_to_return = None
        sleep_s = 0.0

        async with self._pending_lock:
            if self._buffering and self._queue.qsize() < int(self._min_buffered_chunks_for_playback):
                if self._started_at is None:
                    self._started_at = time.monotonic()

                frame_to_return = _make_frame(_silence())
                self._next_pts += chunk_samples

                target_t = self._started_at + (frame_to_return.pts / sample_rate)
                sleep_s = target_t - time.monotonic()

        if frame_to_return is not None:
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            return frame_to_return

        async with self._pending_lock:
            if self._buffering:
                self._buffering = False
                if self._started_at is None:
                    self._started_at = time.monotonic()
                self._playback_idle.clear()

        if self._started_at is None:
            self._started_at = time.monotonic()

        frame_pts = self._next_pts
        target_t = self._started_at + (frame_pts / sample_rate)
        remaining = target_t - time.monotonic()

        dequeued = False
        if remaining > 0:
            try:
                pcm16 = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                dequeued = True
            except asyncio.TimeoutError:
                pcm16 = _silence()
        else:
            try:
                pcm16 = self._queue.get_nowait()
                dequeued = True
            except asyncio.QueueEmpty:
                pcm16 = _silence()

        pcm16 = np.asarray(pcm16, dtype=np.int16).reshape(-1)
        if pcm16.shape[0] != chunk_samples:
            pcm16 = np.pad(pcm16[:chunk_samples], (0, max(0, chunk_samples - pcm16.shape[0])))

        frame = _make_frame(pcm16)
        self._next_pts += chunk_samples

        wait_s = target_t - time.monotonic()
        if wait_s > 0:
            await asyncio.sleep(wait_s)

        async with self._pending_lock:
            if dequeued:
                self._pending_chunks = max(0, int(self._pending_chunks) - 1)
            if self._queue.qsize() == 0:
                self._buffering = True
                self._playback_idle.set()

        return frame

    @staticmethod
    def _prepare_chunks(audio_pcm16: np.ndarray, src_rate: int, dst_rate: int, chunk_samples: int) -> list[np.ndarray]:
        adapted = resample_pcm16(audio_pcm16, src_rate, dst_rate)
        chunks: list[np.ndarray] = []
        for start in range(0, adapted.size, chunk_samples):
            pcm = np.ascontiguousarray(adapted[start : start + chunk_samples], dtype=np.int16)
            if pcm.size < chunk_samples:
                pcm = np.pad(pcm, (0, chunk_samples - pcm.size))
            chunks.append(pcm)
        return chunks

    async def push_pcm16(self, audio_pcm16: np.ndarray, sample_rate: int) -> None:
        if audio_pcm16.size == 0:
            return

        chunks = await asyncio.to_thread(
            type(self)._prepare_chunks,
            np.asarray(audio_pcm16, dtype=np.int16),
            int(sample_rate),
            int(self._sample_rate),
            int(self._chunk_samples),
        )
        if not chunks:
            return

        dropped_old = 0
        dropped_new = 0

        async with self._pending_lock:
            self._playback_idle.clear()
            max_buffer_chunks = max(1, self._max_buffer_ms // WEBRTC_CHUNK_MS)

            for pcm in chunks:
                while self._pending_chunks >= max_buffer_chunks and not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                        self._pending_chunks = max(0, self._pending_chunks - 1)
                        dropped_old += 1
                    except asyncio.QueueEmpty:
                        break

                while self._queue.full():
                    try:
                        self._queue.get_nowait()
                        self._pending_chunks = max(0, self._pending_chunks - 1)
                        dropped_old += 1
                    except asyncio.QueueEmpty:
                        break

                try:
                    self._queue.put_nowait(pcm)
                    self._pending_chunks += 1
                except asyncio.QueueFull:
                    dropped_new += 1

            if self._pending_chunks == 0 and self._queue.qsize() == 0:
                self._playback_idle.set()

        if dropped_old or dropped_new:
            logger.warning(
                "[ceo] outbound queue under backpressure: dropped_old=%s dropped_new=%s pending=%s",
                dropped_old,
                dropped_new,
                self._pending_chunks,
            )

    async def wait_until_idle(self) -> None:
        await self._playback_idle.wait()

    async def wait_consumer_started(self, timeout: float | None = None) -> bool:
        try:
            if timeout is None:
                await self._consumer_started.wait()
                return True
            await asyncio.wait_for(self._consumer_started.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


class TtsPipeline:
    def __init__(self, cfg: CeoConfig):
        self._cfg = cfg
        self._model = load_tts(cfg.tts_model)
        self._ref_audio_path = Path(cfg.tts_ref_audio).expanduser() if cfg.tts_ref_audio.strip() else None
        self._ref_text = cfg.tts_ref_text.strip()
        if self._ref_audio_path is None or not self._ref_text:
            raise ValueError(
                "Voice cloning richiede CEO_TTS_REF_AUDIO e CEO_TTS_REF_TEXT valorizzati. "
                "Configura un file wav di riferimento e la sua trascrizione."
            )
        if not self._ref_audio_path.is_file():
            raise ValueError(f"File reference audio non trovato: {self._ref_audio_path}")
        logger.info("[ceo] TTS attivo (%s, language=%s)", cfg.tts_model, cfg.tts_language)

    @property
    def model(self):
        return self._model

    def stream_voice_clone_pcm16(self, text: str):
        if not text.strip():
            return

        for result in self._model.generate(
            text=text,
            ref_audio=str(self._ref_audio_path),
            ref_text=self._ref_text,
            stream=True,
            streaming_interval=self._cfg.tts_streaming_interval,
        ):
            audio = np.asarray(result.audio, dtype=np.float32)
            if audio.size == 0:
                continue
            clipped = np.clip(audio, -1.0, 1.0)
            sample_rate = int(getattr(result, "sample_rate", TARGET_SAMPLE_RATE))
            yield (clipped * 32767.0).astype(np.int16), sample_rate
