import asyncio
import logging
import time
import wave
from collections import deque
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import webrtcvad
from aiortc import MediaStreamTrack
from aiortc.mediastreams import AudioFrame
from av.audio.resampler import AudioResampler

from titani.ceo_components.audio_utils import resample_pcm16
from titani.ceo_components.config import (
    DEFAULT_WEBRTC_SAMPLE_RATE,
    OUTBOUND_HIGH_WATERMARK_MS,
    OUTBOUND_LOW_WATERMARK_MS,
    OUTBOUND_PREBUFFER_CHUNKS,
    TARGET_SAMPLE_RATE,
    WEBRTC_CHUNK_MS,
    WEBRTC_CHUNK_SAMPLES,
)

logger = logging.getLogger(__name__)


class SmartTurnPipeline:
    """Simple VAD + silence-based turn segmentation for ASR."""

    def __init__(self, cfg: Any, debug: Any | None = None):
        self._cfg = cfg
        self._debug = debug
        self._vad = webrtcvad.Vad(2)
        self._pre_roll_chunks: deque[np.ndarray] = deque()
        self._pre_roll_samples = 0
        self._pre_roll_max_samples = max(0, TARGET_SAMPLE_RATE * max(0, cfg.pre_roll_ms) // 1000)
        self._turn_audio_chunks: deque[np.ndarray] = deque()
        self._turn_audio_samples = 0
        self._state = _TurnState.IDLE
        self._speech_streak = 0
        self._silence_streak = 0
        self._start_speech_chunks = max(1, cfg.start_speech_chunks)
        self._min_turn_samples = int(max(0.0, cfg.smart_turn_min_segment_seconds) * TARGET_SAMPLE_RATE)
        self._last_speech_ms = 0.0
        self._silence_started_ms: float | None = None
        self._end_candidate_silence_samples = 0
        self._session_id = uuid4().hex[:8]
        self._turn_id = 0
        self._debug_dump_wav_enabled = cfg.debug_dump_wav_enabled
        self._debug_dump_wav_dir = Path(cfg.debug_dump_wav_dir)
        if self._debug_dump_wav_enabled:
            self._debug_dump_wav_dir.mkdir(parents=True, exist_ok=True)
        self._audio_resampler = AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
        self._pending_input_frames_16k: list[np.ndarray] = []
        self._pending_input_frames_target = 3
        if cfg.speech_subchunk_min_count > self._pending_input_frames_target:
            logger.warning(
                "[ceo][config] speech_subchunk_min_count=%s maggiore dei subchunk nominali per frame=%s; "
                "durante il runtime verra' clampato al massimo disponibile.",
                cfg.speech_subchunk_min_count,
                self._pending_input_frames_target,
            )
        self._last_vad_stats = {
            "speech_subchunks": 0,
            "total_subchunks": 0,
            "required_subchunks": 0,
            "speech_ratio": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "is_speech_vad": False,
        }
        logger.info("[ceo] inbound VAD semplice attivo (start=%s chunk, silence_end=%sms)", self._start_speech_chunks, cfg.silence_ms_before_endpoint)

    def _transition_to(self, new_state: "_TurnState", reason: str, now_ms: float) -> None:
        if self._state == new_state:
            return

        silence_ms = max(0.0, now_ms - self._last_speech_ms) if self._last_speech_ms > 0 else 0.0
        logger.info(
            "[ceo] turn-state %s -> %s reason=%s speech_streak=%s silence_streak=%s silence_ms=%.1f",
            self._state.value,
            new_state.value,
            reason,
            self._speech_streak,
            self._silence_streak,
            silence_ms,
        )
        self._state = new_state

    def _append_turn_frame(self, frame_16k: np.ndarray) -> None:
        if frame_16k.size == 0:
            return

        frame_chunk = np.ascontiguousarray(frame_16k)
        self._turn_audio_chunks.append(frame_chunk)
        self._turn_audio_samples += int(frame_chunk.size)

    def _append_pre_roll_frame(self, frame_16k: np.ndarray) -> None:
        if frame_16k.size == 0 or self._pre_roll_max_samples <= 0:
            return

        frame_chunk = np.ascontiguousarray(frame_16k)
        self._pre_roll_chunks.append(frame_chunk)
        self._pre_roll_samples += int(frame_chunk.size)

        while self._pre_roll_samples > self._pre_roll_max_samples and self._pre_roll_chunks:
            overflow = self._pre_roll_samples - self._pre_roll_max_samples
            oldest = self._pre_roll_chunks[0]
            if oldest.size <= overflow:
                self._pre_roll_chunks.popleft()
                self._pre_roll_samples -= int(oldest.size)
                continue
            self._pre_roll_chunks[0] = oldest[overflow:]
            self._pre_roll_samples -= overflow

    def _seed_turn_audio_with_pre_roll(self) -> None:
        for pre_roll_chunk in self._pre_roll_chunks:
            self._append_turn_frame(pre_roll_chunk)

    def _clear_turn_only_state(self) -> None:
        self._turn_audio_chunks.clear()
        self._turn_audio_samples = 0
        self._speech_streak = 0
        self._silence_streak = 0
        self._last_speech_ms = 0.0
        self._silence_started_ms = None
        self._end_candidate_silence_samples = 0
        self._transition_to(_TurnState.IDLE, reason="clear-turn-state", now_ms=time.monotonic() * 1000.0)

    def _save_turn_debug_wav(self, audio_16k: np.ndarray, reason: str) -> None:
        if not self._debug_dump_wav_enabled or audio_16k.size == 0:
            return

        ts_ms = int(time.time() * 1000)
        out_path = self._debug_dump_wav_dir / f"inbound_{self._session_id}_turn{self._turn_id:04d}_{reason}_{ts_ms}.wav"
        clipped = np.clip(audio_16k, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())
        logger.info("[ceo][debug] inbound wav salvato: %s", out_path)

    def _commit_turn(self, reason: str) -> np.ndarray | None:
        completed_turn = self._drain_turn_audio()
        if completed_turn.size == 0:
            self._clear_turn_only_state()
            return None

        self._turn_id += 1
        logger.info("[ceo] turn committed reason=%s samples=%s", reason, completed_turn.size)
        self._save_turn_debug_wav(completed_turn, reason=reason)
        self._clear_turn_only_state()
        return completed_turn

    def _drain_turn_audio(self) -> np.ndarray:
        if not self._turn_audio_chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self._turn_audio_chunks) == 1:
            completed_turn = self._turn_audio_chunks[0]
        else:
            completed_turn = np.concatenate(self._turn_audio_chunks)
        self._turn_audio_chunks.clear()
        self._turn_audio_samples = 0
        return completed_turn

    def _frame_to_mono_16k(self, frame: AudioFrame) -> np.ndarray:
        resampled_frames = self._audio_resampler.resample(frame)
        if not resampled_frames:
            return np.zeros(0, dtype=np.float32)

        chunks: list[np.ndarray] = []
        for resampled in resampled_frames:
            pcm = resampled.to_ndarray()
            mono = pcm.reshape(-1) if pcm.ndim > 1 else pcm
            chunks.append(mono.astype(np.float32, copy=False) / 32768.0)

        mono_16k = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        if mono_16k.size == 0:
            return mono_16k

        self._pending_input_frames_16k.append(np.ascontiguousarray(mono_16k))
        if len(self._pending_input_frames_16k) < self._pending_input_frames_target:
            return np.zeros(0, dtype=np.float32)

        merged = np.concatenate(self._pending_input_frames_16k)
        self._pending_input_frames_16k.clear()
        return np.ascontiguousarray(merged, dtype=np.float32)

    def _is_speech(self, frame_16k: np.ndarray) -> bool:
        if frame_16k.size < WEBRTC_CHUNK_SAMPLES:
            self._last_vad_stats = {
                "speech_subchunks": 0,
                "total_subchunks": 0,
                "required_subchunks": 0,
                "speech_ratio": 0.0,
                "rms": float(np.sqrt(np.mean(np.square(frame_16k)))) if frame_16k.size else 0.0,
                "peak": float(np.max(np.abs(frame_16k))) if frame_16k.size else 0.0,
                "is_speech_vad": False,
            }
            return False

        clipped = np.clip(frame_16k, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)

        usable = (pcm16.size // WEBRTC_CHUNK_SAMPLES) * WEBRTC_CHUNK_SAMPLES
        if usable == 0:
            self._last_vad_stats = {
                "speech_subchunks": 0,
                "total_subchunks": 0,
                "required_subchunks": 0,
                "speech_ratio": 0.0,
                "rms": 0.0,
                "peak": float(np.max(np.abs(frame_16k))) if frame_16k.size else 0.0,
                "is_speech_vad": False,
            }
            return False

        total_subchunks = 0
        speech_subchunks = 0
        for start in range(0, usable, WEBRTC_CHUNK_SAMPLES):
            chunk = pcm16[start : start + WEBRTC_CHUNK_SAMPLES]
            total_subchunks += 1
            if self._vad.is_speech(chunk.tobytes(), TARGET_SAMPLE_RATE):
                speech_subchunks += 1

        if total_subchunks == 0:
            self._last_vad_stats = {
                "speech_subchunks": 0,
                "total_subchunks": 0,
                "required_subchunks": 0,
                "speech_ratio": 0.0,
                "rms": 0.0,
                "peak": float(np.max(np.abs(frame_16k))) if frame_16k.size else 0.0,
                "is_speech_vad": False,
            }
            return False

        speech_ratio = speech_subchunks / total_subchunks
        cfg_min_subchunks = max(1, self._cfg.speech_subchunk_min_count)
        required_subchunks = min(cfg_min_subchunks, total_subchunks)
        is_speech_vad = speech_subchunks >= required_subchunks and speech_ratio >= self._cfg.speech_majority_ratio

        rms = float(np.sqrt(np.mean(np.square(frame_16k)))) if frame_16k.size else 0.0
        peak = float(np.max(np.abs(frame_16k))) if frame_16k.size else 0.0
        self._last_vad_stats = {
            "speech_subchunks": speech_subchunks,
            "total_subchunks": total_subchunks,
            "required_subchunks": required_subchunks,
            "speech_ratio": speech_ratio,
            "rms": rms,
            "peak": peak,
            "is_speech_vad": is_speech_vad,
        }

        logger.debug(
            "[ceo][debug][vad] speech_subchunks=%s required_subchunks=%s total_subchunks=%s ratio=%.3f threshold=%.3f",
            speech_subchunks,
            required_subchunks,
            total_subchunks,
            speech_ratio,
            self._cfg.speech_majority_ratio,
        )

        if not is_speech_vad:
            return False

        if self._cfg.vad_min_rms <= 0.0:
            return True

        return rms >= self._cfg.vad_min_rms

    def process(self, frame: AudioFrame) -> np.ndarray | None:
        frame_16k = self._frame_to_mono_16k(frame)
        if self._debug is not None:
            self._debug.observe_frame(frame.sample_rate, frame_16k)
        self._append_pre_roll_frame(frame_16k)

        now_ms = time.monotonic() * 1000.0
        speaking = self._is_speech(frame_16k)

        if speaking:
            self._speech_streak += 1
            self._silence_streak = 0
        else:
            self._silence_streak += 1
            self._speech_streak = 0

        if self._debug is not None:
            silence_ms = max(0.0, now_ms - self._last_speech_ms) if self._last_speech_ms > 0 else 0.0
            self._debug.trace_vad_frame(
                state=self._state.value,
                speaking=speaking,
                speech_subchunks=int(self._last_vad_stats.get("speech_subchunks", 0)),
                total_subchunks=int(self._last_vad_stats.get("total_subchunks", 0)),
                required_subchunks=int(self._last_vad_stats.get("required_subchunks", 0)),
                speech_ratio=float(self._last_vad_stats.get("speech_ratio", 0.0)),
                threshold_ratio=self._cfg.speech_majority_ratio,
                rms=float(self._last_vad_stats.get("rms", 0.0)),
                peak=float(self._last_vad_stats.get("peak", 0.0)),
                rms_threshold=self._cfg.vad_min_rms,
                speech_streak=self._speech_streak,
                silence_streak=self._silence_streak,
                turn_seconds=self._turn_audio_samples / TARGET_SAMPLE_RATE,
                silence_ms=silence_ms,
            )

        if self._state == _TurnState.IDLE and speaking:
            self._transition_to(_TurnState.PRE_SPEECH, reason="speech-detected", now_ms=now_ms)

        if self._state == _TurnState.PRE_SPEECH:
            if speaking and self._speech_streak >= self._start_speech_chunks:
                self._transition_to(_TurnState.IN_TURN, reason="start-threshold-reached", now_ms=now_ms)
                self._seed_turn_audio_with_pre_roll()
                self._last_speech_ms = now_ms
                self._silence_started_ms = None
                self._end_candidate_silence_samples = 0
                logger.info("[ceo] user turn started (pre-roll=%sms)", self._cfg.pre_roll_ms)
            elif not speaking:
                self._transition_to(_TurnState.IDLE, reason="speech-faded-before-start", now_ms=now_ms)

        if self._state in {_TurnState.IN_TURN, _TurnState.END_CANDIDATE} and frame_16k.size:
            if self._state == _TurnState.END_CANDIDATE and not speaking:
                max_trailing_samples = max(0, TARGET_SAMPLE_RATE * self._cfg.trailing_silence_pad_ms // 1000)
                remaining = max_trailing_samples - self._end_candidate_silence_samples
                if remaining > 0:
                    append_chunk = frame_16k[:remaining]
                    self._append_turn_frame(append_chunk)
                    self._end_candidate_silence_samples += int(append_chunk.size)
            else:
                self._append_turn_frame(frame_16k)

        if self._state in {_TurnState.IN_TURN, _TurnState.END_CANDIDATE} and speaking:
            self._last_speech_ms = now_ms
            self._silence_started_ms = None
            self._end_candidate_silence_samples = 0
            if self._state == _TurnState.END_CANDIDATE:
                self._transition_to(_TurnState.IN_TURN, reason="speech-resumed", now_ms=now_ms)
                logger.info("[ceo] speech resumed during END_CANDIDATE")

        if self._state in {_TurnState.IN_TURN, _TurnState.END_CANDIDATE} and not speaking:
            if self._silence_started_ms is None:
                self._silence_started_ms = now_ms
            silence_ms = now_ms - self._last_speech_ms

            if self._state == _TurnState.IN_TURN and silence_ms >= self._cfg.silence_ms_before_endpoint:
                self._transition_to(_TurnState.END_CANDIDATE, reason="silence-threshold-reached", now_ms=now_ms)
                logger.info("[ceo] entered END_CANDIDATE silence_ms=%.1f", silence_ms)

            if self._state == _TurnState.END_CANDIDATE:
                if silence_ms >= self._cfg.max_silence_ms_force_commit:
                    logger.info("[ceo] turn closed by hard timeout silence_ms=%.1f", silence_ms)
                    return self._commit_turn(reason="hard-timeout")
                if silence_ms >= self._cfg.silence_ms_before_endpoint:
                    if self._turn_audio_samples < self._min_turn_samples:
                        logger.info(
                            "[ceo] turno troppo corto (%.2fs < %.2fs), ignoro",
                            self._turn_audio_samples / TARGET_SAMPLE_RATE,
                            self._min_turn_samples / TARGET_SAMPLE_RATE,
                        )
                        self._clear_turn_only_state()
                        return None

                    logger.info("[ceo] turn closed by silence silence_ms=%.1f", silence_ms)
                    return self._commit_turn(reason="silence-endpoint")

        return None

    def reset_turn_state(self) -> None:
        self._pre_roll_chunks.clear()
        self._pre_roll_samples = 0
        self._turn_audio_chunks.clear()
        self._turn_audio_samples = 0
        self._pending_input_frames_16k.clear()
        self._state = _TurnState.IDLE
        self._last_speech_ms = 0.0
        self._speech_streak = 0
        self._silence_streak = 0
        self._silence_started_ms = None
        self._end_candidate_silence_samples = 0


class _TurnState(Enum):
    IDLE = "IDLE"
    PRE_SPEECH = "PRE_SPEECH"
    IN_TURN = "IN_TURN"
    END_CANDIDATE = "END_CANDIDATE"


class TtsOutboundAudioTrack(MediaStreamTrack):
    kind = "audio"

    _SILENCE_CHUNK_PEAK_THRESHOLD = 96

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
        self._high_watermark_ms = max(OUTBOUND_HIGH_WATERMARK_MS, WEBRTC_CHUNK_MS)
        self._low_watermark_ms = min(
            max(OUTBOUND_LOW_WATERMARK_MS, WEBRTC_CHUNK_MS),
            self._high_watermark_ms - WEBRTC_CHUNK_MS,
        )
        self._min_buffered_chunks_for_playback = self._prebuffer_chunks
        self._buffering = True
        self._consumer_started = asyncio.Event()

        self.underflow_count = 0
        self.overflow_count = 0
        self.dropped_ms = 0.0

        self._last_metrics_log_ts = time.monotonic()
        self._metrics_log_interval_s = 5.0

    def _watermark_chunks(self) -> tuple[int, int]:
        low_chunks = max(1, int(self._low_watermark_ms // WEBRTC_CHUNK_MS))
        high_chunks = max(low_chunks + 1, int(self._high_watermark_ms // WEBRTC_CHUNK_MS))
        return low_chunks, high_chunks

    @staticmethod
    def _is_silence_chunk(chunk: np.ndarray) -> bool:
        if chunk.size == 0:
            return True
        return int(np.max(np.abs(chunk))) <= TtsOutboundAudioTrack._SILENCE_CHUNK_PEAK_THRESHOLD

    def _maybe_log_runtime_metrics(self) -> None:
        now = time.monotonic()
        if now - self._last_metrics_log_ts < self._metrics_log_interval_s:
            return
        self._last_metrics_log_ts = now
        logger.info(
            "[ceo][outbound-metrics] underflow_count=%s overflow_count=%s dropped_ms=%.1f pending_chunks=%s queue_size=%s",
            self.underflow_count,
            self.overflow_count,
            self.dropped_ms,
            self._pending_chunks,
            self._queue.qsize(),
        )

    def set_output_sample_rate(self, sample_rate: int) -> None:
        if sample_rate <= 0 or sample_rate == self._sample_rate:
            return

        self._sample_rate = sample_rate
        self._chunk_samples = max(1, self._sample_rate * WEBRTC_CHUNK_MS // 1000)
        self._pts = 0
        self._next_pts = 0
        self._started_at = None
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

                self.underflow_count += 1
                frame_to_return = _make_frame(_silence())
                self._next_pts += chunk_samples

                target_t = self._started_at + (frame_to_return.pts / sample_rate)
                sleep_s = target_t - time.monotonic()
                self._maybe_log_runtime_metrics()

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
            else:
                self.underflow_count += 1
            if self._queue.qsize() == 0:
                self._buffering = True
                self._playback_idle.set()
            self._maybe_log_runtime_metrics()

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
        trimmed_silence = 0

        async with self._pending_lock:
            self._playback_idle.clear()
            low_chunks, high_chunks = self._watermark_chunks()

            if self._pending_chunks > high_chunks:
                self._min_buffered_chunks_for_playback = max(1, min(self._min_buffered_chunks_for_playback, low_chunks))

            projected_pending = self._pending_chunks + len(chunks)
            overflow_chunks = max(0, projected_pending - high_chunks)

            if overflow_chunks > 0:
                trimmed = 0
                while trimmed < overflow_chunks and chunks and self._is_silence_chunk(chunks[0]):
                    chunks.pop(0)
                    trimmed += 1
                while trimmed < overflow_chunks and chunks and self._is_silence_chunk(chunks[-1]):
                    chunks.pop()
                    trimmed += 1
                trimmed_silence += trimmed

            while self._pending_chunks > high_chunks and self._pending_chunks > low_chunks and not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._pending_chunks = max(0, self._pending_chunks - 1)
                    dropped_old += 1
                except asyncio.QueueEmpty:
                    break

            for pcm in chunks:
                if self._pending_chunks >= high_chunks:
                    dropped_new += 1
                    continue

                while self._queue.full():
                    if self._pending_chunks <= low_chunks:
                        break
                    try:
                        self._queue.get_nowait()
                        self._pending_chunks = max(0, self._pending_chunks - 1)
                        dropped_old += 1
                    except asyncio.QueueEmpty:
                        break

                if self._queue.full():
                    dropped_new += 1
                    continue

                self._queue.put_nowait(pcm)
                self._pending_chunks += 1

            dropped_total = dropped_old + dropped_new + trimmed_silence
            if dropped_total:
                self.overflow_count += 1
                self.dropped_ms += dropped_total * WEBRTC_CHUNK_MS
            if self._pending_chunks == 0 and self._queue.qsize() == 0:
                self._playback_idle.set()
            self._maybe_log_runtime_metrics()

        total_in_chunks = len(chunks) + trimmed_silence
        if dropped_old or dropped_new or trimmed_silence:
            drop_ratio = (dropped_old + dropped_new) / max(1, total_in_chunks)
            logger.warning(
                "[ceo] outbound queue backpressure: dropped_old=%s dropped_new=%s trimmed_silence=%s drop_ratio=%.3f pending=%s low_wm_ms=%s high_wm_ms=%s",
                dropped_old,
                dropped_new,
                trimmed_silence,
                drop_ratio,
                self._pending_chunks,
                self._low_watermark_ms,
                self._high_watermark_ms,
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

    async def update_buffer_policy(
        self,
        *,
        prebuffer_chunks: int | None = None,
        max_buffer_ms: int | None = None,
        low_watermark_ms: int | None = None,
        high_watermark_ms: int | None = None,
        reason: str = "runtime-adaptation",
    ) -> dict[str, int]:
        async with self._pending_lock:
            if prebuffer_chunks is not None:
                self._prebuffer_chunks = max(1, int(prebuffer_chunks))
                self._min_buffered_chunks_for_playback = self._prebuffer_chunks

            if max_buffer_ms is not None and high_watermark_ms is None:
                high_watermark_ms = int(max_buffer_ms)

            if high_watermark_ms is not None:
                self._high_watermark_ms = max(WEBRTC_CHUNK_MS * 2, int(high_watermark_ms))
            if low_watermark_ms is not None:
                self._low_watermark_ms = max(WEBRTC_CHUNK_MS, int(low_watermark_ms))

            self._low_watermark_ms = min(self._low_watermark_ms, self._high_watermark_ms - WEBRTC_CHUNK_MS)

            snapshot = {
                "prebuffer_chunks": int(self._prebuffer_chunks),
                "min_buffered_chunks_for_playback": int(self._min_buffered_chunks_for_playback),
                "max_buffer_ms": int(self._high_watermark_ms),
                "high_watermark_ms": int(self._high_watermark_ms),
                "low_watermark_ms": int(self._low_watermark_ms),
                "pending_chunks": int(self._pending_chunks),
                "queue_size": int(self._queue.qsize()),
            }

        logger.info(
            "[ceo][outbound-policy] event=updated reason=%s prebuffer_chunks=%s min_chunks=%s low_wm_ms=%s high_wm_ms=%s pending_chunks=%s queue_size=%s",
            reason,
            snapshot["prebuffer_chunks"],
            snapshot["min_buffered_chunks_for_playback"],
            snapshot["low_watermark_ms"],
            snapshot["high_watermark_ms"],
            snapshot["pending_chunks"],
            snapshot["queue_size"],
        )
        return snapshot
