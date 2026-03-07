import json
import logging
import time
import wave
from collections import deque
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import mlx.core as mx
import numpy as np
import webrtcvad
from aiortc.mediastreams import AudioFrame
from av.audio.resampler import AudioResampler
from mlx_audio.stt.utils import load_model as load_stt

from .audio_utils import cosine_similarity, resample_float32
from .config import TARGET_SAMPLE_RATE, WEBRTC_CHUNK_SAMPLES, CeoConfig
from .debug import CeoDebug

logger = logging.getLogger(__name__)


class SmartTurnPipeline:
    """Simple VAD + silence-based turn segmentation for ASR."""

    def __init__(self, cfg: CeoConfig, debug: CeoDebug | None = None):
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


class AsrPipeline:
    def __init__(self, cfg: CeoConfig):
        self._cfg = cfg
        self._model = load_stt(cfg.asr_model)
        logger.info("[ceo] ASR attivo (%s, language=%s)", cfg.asr_model, cfg.asr_language)

    def transcribe(self, audio_16k: np.ndarray) -> str:
        if audio_16k.size == 0:
            return ""

        clipped = np.clip(audio_16k, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        with NamedTemporaryFile(suffix=".wav") as wav_file:
            with wave.open(wav_file.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(TARGET_SAMPLE_RATE)
                wf.writeframes(pcm16.tobytes())
            result = self._model.generate(wav_file.name, language=self._cfg.asr_language)
        return str(getattr(result, "text", "")).strip()


class SpeakerEmbeddingPipeline:
    def __init__(self, cfg: CeoConfig, tts_model):
        self._cfg = cfg
        self._tts_model = tts_model
        self._target_sample_rate = 24_000
        self._threshold = float(np.clip(cfg.speaker_embedding_threshold, 0.0, 1.0))
        self._embeddings_dir = Path(cfg.speaker_embeddings_dir)
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._last_embedding: np.ndarray | None = None
        self._last_embedding_id: str | None = None
        logger.info(
            "[ceo] speaker embedding attivo (dir=%s, threshold=%.3f)",
            self._embeddings_dir.resolve(),
            self._threshold,
        )

    def _load_known_embeddings(self) -> list[tuple[str, np.ndarray]]:
        known_embeddings: list[tuple[str, np.ndarray]] = []
        for embedding_path in sorted(self._embeddings_dir.glob("*.npy")):
            try:
                embedding = np.load(embedding_path)
            except Exception:
                logger.exception("[ceo] impossibile caricare embedding speaker noto da %s", embedding_path)
                continue

            embedding_np = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32).reshape(-1), dtype=np.float32)
            if embedding_np.size == 0:
                logger.warning("[ceo] embedding speaker noto vuoto ignorato: %s", embedding_path)
                continue

            known_embeddings.append((embedding_path.stem, embedding_np))
        return known_embeddings

    def _extract_embedding(self, audio_16k: np.ndarray) -> np.ndarray:
        audio_24k = resample_float32(audio_16k, src_rate=TARGET_SAMPLE_RATE, dst_rate=self._target_sample_rate)
        audio_mx = mx.array(np.ascontiguousarray(audio_24k, dtype=np.float32))
        embedding = self._tts_model.extract_speaker_embedding(audio_mx, sr=self._target_sample_rate)
        if hasattr(embedding, "tolist"):
            embedding_np = np.asarray(embedding.tolist(), dtype=np.float32)
        else:
            embedding_np = np.asarray(embedding, dtype=np.float32)
        return np.ascontiguousarray(embedding_np.reshape(-1), dtype=np.float32)

    def process_transcribed_segment(self, audio_16k: np.ndarray) -> None:
        if audio_16k.size == 0:
            return

        try:
            current_embedding = self._extract_embedding(audio_16k)
        except Exception:
            logger.exception("[ceo] estrazione speaker embedding fallita")
            return

        if self._last_embedding is not None:
            similarity = cosine_similarity(self._last_embedding, current_embedding)
            probability_same_speaker = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            logger.debug(
                "[ceo][debug] prob stesso speaker=%.4f (sim=%.4f, prev_id=%s)",
                probability_same_speaker,
                similarity,
                self._last_embedding_id,
            )
            if probability_same_speaker >= self._threshold:
                logger.info(
                    "[ceo] speaker invariato (prob=%.4f >= threshold=%.3f)",
                    probability_same_speaker,
                    self._threshold,
                )
                return

        embedding_id = f"spk_{uuid4().hex}"
        embedding_path = self._embeddings_dir / f"{embedding_id}.npy"
        metadata_path = self._embeddings_dir / f"{embedding_id}.json"
        np.save(embedding_path, current_embedding)
        metadata_path.write_text(
            json.dumps(
                {
                    "id": embedding_id,
                    "sample_rate": self._target_sample_rate,
                    "threshold": self._threshold,
                    "created_at_ms": int(time.time() * 1000),
                },
                indent=2,
            )
        )
        self._last_embedding = current_embedding
        self._last_embedding_id = embedding_id
        logger.info("[ceo] nuovo speaker embedding salvato: %s", embedding_path)

    def recognize_known_speaker(self, audio_16k: np.ndarray) -> tuple[bool, str | None, float]:
        if audio_16k.size == 0:
            logger.info("[ceo][speaker-guard] segmento vuoto: speaker non riconosciuto")
            return False, None, 0.0

        try:
            current_embedding = self._extract_embedding(audio_16k)
        except Exception:
            logger.exception("[ceo][speaker-guard] estrazione embedding fallita")
            return False, None, 0.0

        known_embeddings = self._load_known_embeddings()
        if not known_embeddings:
            logger.info(
                "[ceo][speaker-guard] nessuno speaker noto disponibile in %s",
                self._embeddings_dir.resolve(),
            )
            return False, None, 0.0

        best_id: str | None = None
        best_probability = 0.0
        for speaker_id, known_embedding in known_embeddings:
            similarity = cosine_similarity(known_embedding, current_embedding)
            probability_same_speaker = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            if probability_same_speaker > best_probability:
                best_probability = probability_same_speaker
                best_id = speaker_id

        recognized = best_probability >= self._threshold
        if recognized:
            logger.info(
                "[ceo][speaker-guard] speaker riconosciuto: id=%s prob=%.4f threshold=%.3f",
                best_id,
                best_probability,
                self._threshold,
            )
        else:
            logger.info(
                "[ceo][speaker-guard] speaker NON riconosciuto: best_id=%s prob=%.4f threshold=%.3f",
                best_id,
                best_probability,
                self._threshold,
            )
        return recognized, best_id, best_probability
