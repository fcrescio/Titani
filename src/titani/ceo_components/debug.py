import json
import logging
import time
import wave
from pathlib import Path

import numpy as np

from .config import TARGET_SAMPLE_RATE, CeoConfig

logger = logging.getLogger(__name__)


class CeoDebug:
    def __init__(self, cfg: CeoConfig):
        self._enabled = cfg.debug_mode
        self._out_dir = Path(cfg.debug_out_dir)
        self._heartbeat_ms = max(250, cfg.debug_heartbeat_ms)
        self._last_heartbeat_ms = 0.0
        self._seen_sample_rates: set[int] = set()
        self._frame_count = 0
        self._received_samples_16k = 0
        self._saved_segments = 0
        self._vad_trace_enabled = self._enabled and cfg.debug_vad_trace
        self._vad_trace_every_chunks = max(1, cfg.debug_vad_trace_every_chunks)
        self._vad_trace_jsonl_enabled = self._vad_trace_enabled and cfg.debug_vad_trace_jsonl
        self._vad_trace_count = 0
        self._vad_trace_path = self._out_dir / "vad_trace.jsonl"
        if self._enabled:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[ceo][debug] modalità debug attiva, directory output: %s", self._out_dir.resolve())
            if self._vad_trace_enabled:
                logger.info(
                    "[ceo][debug] VAD trace attivo (every_chunks=%s, jsonl=%s, file=%s)",
                    self._vad_trace_every_chunks,
                    self._vad_trace_jsonl_enabled,
                    self._vad_trace_path,
                )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def observe_frame(self, input_sample_rate: int, frame_16k: np.ndarray) -> None:
        if not self._enabled:
            return

        if input_sample_rate not in self._seen_sample_rates:
            self._seen_sample_rates.add(input_sample_rate)
            logger.info(
                "[ceo][debug] nuovo sample rate in ingresso %sHz (target pipeline: %sHz)",
                input_sample_rate,
                TARGET_SAMPLE_RATE,
            )

        self._frame_count += 1
        self._received_samples_16k += int(frame_16k.size)
        now_ms = time.monotonic() * 1000.0
        if now_ms - self._last_heartbeat_ms < self._heartbeat_ms:
            return

        self._last_heartbeat_ms = now_ms
        if frame_16k.size:
            rms = float(np.sqrt(np.mean(np.square(frame_16k))))
            peak = float(np.max(np.abs(frame_16k)))
        else:
            rms = 0.0
            peak = 0.0

        buffered_seconds = self._received_samples_16k / TARGET_SAMPLE_RATE
        logger.info(
            "[ceo][debug] heartbeat audio: frame=%s buffered_16k=%.2fs rms=%.6f peak=%.6f",
            self._frame_count,
            buffered_seconds,
            rms,
            peak,
        )

    def trace_vad_frame(
        self,
        *,
        state: str,
        speaking: bool,
        speech_subchunks: int,
        total_subchunks: int,
        required_subchunks: int,
        speech_ratio: float,
        threshold_ratio: float,
        rms: float,
        peak: float,
        rms_threshold: float,
        speech_streak: int,
        silence_streak: int,
        turn_seconds: float,
        silence_ms: float,
    ) -> None:
        if not self._vad_trace_enabled:
            return

        self._vad_trace_count += 1
        should_log = speaking or self._vad_trace_count % self._vad_trace_every_chunks == 0
        if not should_log:
            return

        logger.info(
            "[ceo][debug][vad] i=%s state=%s speaking=%s sub=%s/%s (required=%s) ratio=%.2f>=%.2f rms=%.5f>=%.5f peak=%.5f streak(s=%s,si=%s) turn=%.2fs silence=%.1fms",
            self._vad_trace_count,
            state,
            speaking,
            speech_subchunks,
            total_subchunks,
            required_subchunks,
            speech_ratio,
            threshold_ratio,
            rms,
            rms_threshold,
            peak,
            speech_streak,
            silence_streak,
            turn_seconds,
            silence_ms,
        )

        if self._vad_trace_jsonl_enabled:
            payload = {
                "ts_ms": int(time.time() * 1000),
                "i": self._vad_trace_count,
                "state": state,
                "speaking": speaking,
                "speech_subchunks": speech_subchunks,
                "total_subchunks": total_subchunks,
                "required_subchunks": required_subchunks,
                "speech_ratio": speech_ratio,
                "threshold_ratio": threshold_ratio,
                "rms": rms,
                "rms_threshold": rms_threshold,
                "peak": peak,
                "speech_streak": speech_streak,
                "silence_streak": silence_streak,
                "turn_seconds": turn_seconds,
                "silence_ms": silence_ms,
            }
            with self._vad_trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save_segment_for_asr(self, audio_16k: np.ndarray) -> Path | None:
        if not self._enabled:
            return None

        self._saved_segments += 1
        segment_name = f"segment_{self._saved_segments:04d}_{int(time.time() * 1000)}.wav"
        out_path = self._out_dir / segment_name

        clipped = np.clip(audio_16k, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())

        duration = (audio_16k.size / TARGET_SAMPLE_RATE) if audio_16k.size else 0.0
        logger.info("[ceo][debug] salvato segmento ASR: %s (%.2fs)", out_path, duration)
        return out_path

    def save_tts_wav(self, pcm16_audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> Path | None:
        if not self._enabled:
            return None

        self._saved_segments += 1
        out_path = self._out_dir / f"tts_{self._saved_segments:04d}_{int(time.time() * 1000)}.wav"
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16_audio.astype(np.int16, copy=False).tobytes())
        logger.info("[ceo][debug] salvato TTS wav: %s", out_path)
        return out_path
