import json
import logging
import time
import wave
from collections import deque
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import mlx.core as mx
import numpy as np
import webrtcvad
from aiortc.mediastreams import AudioFrame
from av.audio.resampler import AudioResampler
from mlx_audio.stt.utils import load_model as load_stt
from mlx_audio.vad.utils import load_model as load_vad

from .audio_utils import cosine_similarity, resample_float32
from .config import MAX_CONTEXT_SAMPLES, TARGET_SAMPLE_RATE, WEBRTC_CHUNK_SAMPLES, CeoConfig
from .debug import CeoDebug

logger = logging.getLogger(__name__)


class SmartTurnPipeline:
    """Smart Turn v3 endpoint detection with 8s rolling context."""

    def __init__(self, cfg: CeoConfig, debug: CeoDebug | None = None):
        self._cfg = cfg
        self._debug = debug
        self._vad = webrtcvad.Vad(2)
        self._model = load_vad("mlx-community/smart-turn-v3", strict=True)
        self._audio_context_chunks: deque[np.ndarray] = deque()
        self._audio_context_samples = 0
        self._turn_audio_chunks: deque[np.ndarray] = deque()
        self._turn_audio_samples = 0
        self._in_user_turn = False
        self._speech_streak = 0
        self._silence_streak = 0
        self._start_speech_chunks = 10
        self._last_speech_ms = 0.0
        self._checked_during_current_silence = False
        self._audio_resampler = AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
        logger.info("[ceo] Smart Turn v3 attivo (mlx-community/smart-turn-v3)")

    def _append_context_frame(self, frame_16k: np.ndarray) -> None:
        if frame_16k.size == 0:
            return

        if frame_16k.size >= MAX_CONTEXT_SAMPLES:
            self._audio_context_chunks.clear()
            self._audio_context_chunks.append(np.ascontiguousarray(frame_16k[-MAX_CONTEXT_SAMPLES:]))
            self._audio_context_samples = int(self._audio_context_chunks[0].size)
            return

        frame_chunk = np.ascontiguousarray(frame_16k)
        self._audio_context_chunks.append(frame_chunk)
        self._audio_context_samples += int(frame_chunk.size)

        while self._audio_context_samples > MAX_CONTEXT_SAMPLES and self._audio_context_chunks:
            overflow = self._audio_context_samples - MAX_CONTEXT_SAMPLES
            oldest = self._audio_context_chunks[0]

            if oldest.size <= overflow:
                self._audio_context_chunks.popleft()
                self._audio_context_samples -= int(oldest.size)
                continue

            self._audio_context_chunks[0] = oldest[overflow:]
            self._audio_context_samples -= overflow

    def _append_turn_frame(self, frame_16k: np.ndarray) -> None:
        if frame_16k.size == 0:
            return

        frame_chunk = np.ascontiguousarray(frame_16k)
        self._turn_audio_chunks.append(frame_chunk)
        self._turn_audio_samples += int(frame_chunk.size)

    def _build_audio_context(self) -> np.ndarray:
        if not self._audio_context_chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self._audio_context_chunks) == 1:
            return self._audio_context_chunks[0]
        return np.concatenate(self._audio_context_chunks)

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

        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def _is_speech(self, frame_16k: np.ndarray) -> bool:
        if frame_16k.size < WEBRTC_CHUNK_SAMPLES:
            return False

        clipped = np.clip(frame_16k, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)

        usable = (pcm16.size // WEBRTC_CHUNK_SAMPLES) * WEBRTC_CHUNK_SAMPLES
        if usable == 0:
            return False

        for start in range(0, usable, WEBRTC_CHUNK_SAMPLES):
            chunk = pcm16[start : start + WEBRTC_CHUNK_SAMPLES]
            if self._vad.is_speech(chunk.tobytes(), TARGET_SAMPLE_RATE):
                return True
        return False

    def process(self, frame: AudioFrame) -> np.ndarray | None:
        frame_16k = self._frame_to_mono_16k(frame)
        if self._debug is not None:
            self._debug.observe_frame(frame.sample_rate, frame_16k)
        self._append_context_frame(frame_16k)

        now_ms = time.monotonic() * 1000.0
        speaking = self._is_speech(frame_16k)

        if speaking:
            self._speech_streak += 1
            self._silence_streak = 0
        else:
            self._silence_streak += 1
            self._speech_streak = 0

        if not self._in_user_turn and self._speech_streak >= self._start_speech_chunks:
            self._in_user_turn = True
            self._last_speech_ms = now_ms
            self._checked_during_current_silence = False

        if speaking and self._in_user_turn:
            self._last_speech_ms = now_ms

        if self._in_user_turn and frame_16k.size:
            self._append_turn_frame(frame_16k)

        if not speaking and not self._in_user_turn:
            return None

        if not speaking and self._in_user_turn:
            silence_ms = now_ms - self._last_speech_ms
            if silence_ms < self._cfg.silence_ms_before_endpoint:
                return None

            if self._checked_during_current_silence:
                return None

            turn_duration_s = self._turn_audio_samples / TARGET_SAMPLE_RATE
            min_turn_s = max(0.0, self._cfg.smart_turn_min_segment_seconds)
            if turn_duration_s < min_turn_s:
                logger.info(
                    "[ceo] smart-turn rimandato: turno=%.2fs < minimo=%.2fs",
                    turn_duration_s,
                    min_turn_s,
                )
                return None
            self._checked_during_current_silence = True
            audio_context = self._build_audio_context()
            result = self._model.predict_endpoint(
                audio_context,
                sample_rate=TARGET_SAMPLE_RATE,
                threshold=self._cfg.smart_turn_threshold,
            )

            prediction = int(getattr(result, "prediction", 0))
            probability = float(getattr(result, "probability", 0.0))
            logger.info("[ceo] smart-turn prediction=%s probability=%.3f", prediction, probability)

            if prediction == 1:
                self._in_user_turn = False
                self._audio_context_chunks.clear()
                self._audio_context_samples = 0
                completed_turn = self._drain_turn_audio()
                return completed_turn

        return None

    def reset_turn_state(self) -> None:
        self._audio_context_chunks.clear()
        self._audio_context_samples = 0
        self._turn_audio_chunks.clear()
        self._turn_audio_samples = 0
        self._in_user_turn = False
        self._last_speech_ms = 0.0
        self._checked_during_current_silence = False


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
