import json
import logging
import time
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import mlx.core as mx
import numpy as np
from mlx_audio.stt.utils import load_model as load_stt

from .audio_utils import cosine_similarity, resample_float32
from .config import TARGET_SAMPLE_RATE, CeoConfig

logger = logging.getLogger(__name__)

from titani.audio_pipeline import SmartTurnPipeline


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
