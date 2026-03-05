import asyncio
import logging
import math
import json
from contextlib import suppress
from fractions import Fraction
import os
from pathlib import Path
import time
import wave
from dataclasses import dataclass
from uuid import uuid4
from tempfile import NamedTemporaryFile

import numpy as np
import webrtcvad
import websockets
from av.audio.resampler import AudioResampler
from aiortc import MediaStreamTrack, RTCPeerConnection
from aiortc.mediastreams import AudioFrame
from mlx_audio.stt.utils import load_model as load_stt
from mlx_audio.tts.utils import load_model as load_tts
from mlx_audio.vad.utils import load_model as load_vad

from titani.common import ErmeteConfig, WebRTCCommandChannel, maybe_handle_offer, run, setup_logging

TARGET_SAMPLE_RATE = 16_000
MAX_CONTEXT_SECONDS = 8
MAX_CONTEXT_SAMPLES = TARGET_SAMPLE_RATE * MAX_CONTEXT_SECONDS
WEBRTC_CHUNK_MS = int(os.getenv("CEO_WEBRTC_CHUNK_MS", "20"))
WEBRTC_CHUNK_SAMPLES = TARGET_SAMPLE_RATE * WEBRTC_CHUNK_MS // 1000
DEFAULT_WEBRTC_SAMPLE_RATE = 48_000
OUTBOUND_PREBUFFER_CHUNKS = max(1, int(os.getenv("CEO_OUTBOUND_PREBUFFER_CHUNKS", "3")))
OUTBOUND_MAX_BUFFER_MS = max(100, int(os.getenv("CEO_OUTBOUND_MAX_BUFFER_MS", "400")))

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class CeoConfig(ErmeteConfig):
    silence_ms_before_endpoint: int = int(os.getenv("CEO_SILENCE_MS_BEFORE_ENDPOINT", "300"))
    smart_turn_threshold: float = float(os.getenv("CEO_SMART_TURN_THRESHOLD", "0.5"))
    smart_turn_min_segment_seconds: float = float(os.getenv("CEO_SMART_TURN_MIN_SEGMENT_SECONDS", "3.0"))
    asr_model: str = os.getenv("CEO_ASR_MODEL", "mlx-community/Qwen3-ASR-0.6B-8bit")
    asr_language: str = os.getenv("CEO_ASR_LANGUAGE", "Italian")
    debug_mode: bool = _env_bool("CEO_DEBUG_MODE", False)
    debug_out_dir: str = os.getenv("CEO_DEBUG_OUT_DIR", "./ceo_debug")
    debug_heartbeat_ms: int = int(os.getenv("CEO_DEBUG_HEARTBEAT_MS", "2000"))
    tts_model: str = os.getenv(
        "CEO_TTS_MODEL",
        "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    )
    tts_language: str = os.getenv("CEO_TTS_LANGUAGE", "Italian")
    tts_instruct: str = os.getenv(
        "CEO_TTS_INSTRUCT",
        "Una voce femminile adulta, calda e naturale, con tono colloquiale e ritmo medio.",
    )
    tts_streaming_interval: float = float(os.getenv("CEO_TTS_STREAMING_INTERVAL", "0.32"))
    speaker_embedding_threshold: float = float(os.getenv("CEO_SPEAKER_EMBEDDING_THRESHOLD", "0.8"))
    speaker_embeddings_dir: str = os.getenv("CEO_SPEAKER_EMBEDDINGS_DIR", "./ceo_speakers")
    speaker_embedding_model: str = os.getenv(
        "CEO_SPEAKER_EMBEDDING_MODEL",
        "marksverdhei/Qwen3-Voice-Embedding-12Hz-0.6B",
    )
    speaker_embedding_use_tts_fallback: bool = _env_bool("CEO_SPEAKER_EMBEDDING_USE_TTS_FALLBACK", True)


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
        if self._enabled:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[ceo][debug] modalità debug attiva, directory output: %s", self._out_dir.resolve())

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


class SmartTurnPipeline:
    """Smart Turn v3 endpoint detection with 8s rolling context."""

    def __init__(self, cfg: CeoConfig, debug: CeoDebug | None = None):
        self._cfg = cfg
        self._debug = debug
        self._vad = webrtcvad.Vad(2)
        self._model = load_vad("mlx-community/smart-turn-v3", strict=True)
        self._audio_context = np.zeros(0, dtype=np.float32)
        self._turn_audio_chunks: list[np.ndarray] = []
        self._in_user_turn = False
        self._last_speech_ms = 0.0
        self._checked_during_current_silence = False
        self._audio_resampler = AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
        logger.info("[ceo] Smart Turn v3 attivo (mlx-community/smart-turn-v3)")

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
        if frame_16k.size:
            self._audio_context = np.concatenate((self._audio_context, frame_16k))[-MAX_CONTEXT_SAMPLES:]

        now_ms = time.monotonic() * 1000.0
        speaking = self._is_speech(frame_16k)

        if speaking:
            self._in_user_turn = True
            self._last_speech_ms = now_ms
            self._checked_during_current_silence = False
        if self._in_user_turn and frame_16k.size:
            self._turn_audio_chunks.append(frame_16k)

        if not speaking and not self._in_user_turn:
            return None

        if not speaking and self._in_user_turn:
            silence_ms = now_ms - self._last_speech_ms
            if silence_ms < self._cfg.silence_ms_before_endpoint:
                return None

            if self._checked_during_current_silence:
                return None

            segment_duration_s = self._audio_context.size / TARGET_SAMPLE_RATE
            min_segment_duration_s = max(0.0, self._cfg.smart_turn_min_segment_seconds)
            if segment_duration_s < min_segment_duration_s:
                logger.info(
                    "[ceo] smart-turn rimandato: contesto=%.2fs < minimo=%.2fs",
                    segment_duration_s,
                    min_segment_duration_s,
                )
                return None

            self._checked_during_current_silence = True
            result = self._model.predict_endpoint(
                self._audio_context,
                sample_rate=TARGET_SAMPLE_RATE,
                threshold=self._cfg.smart_turn_threshold,
            )

            prediction = int(getattr(result, "prediction", 0))
            probability = float(getattr(result, "probability", 0.0))
            logger.info("[ceo] smart-turn prediction=%s probability=%.3f", prediction, probability)

            if prediction == 1:
                self._in_user_turn = False
                self._audio_context = np.zeros(0, dtype=np.float32)
                completed_turn = (
                    np.concatenate(self._turn_audio_chunks)
                    if self._turn_audio_chunks
                    else np.zeros(0, dtype=np.float32)
                )
                self._turn_audio_chunks = []
                return completed_turn

        return None

    def reset_turn_state(self) -> None:
        self._audio_context = np.zeros(0, dtype=np.float32)
        self._turn_audio_chunks = []
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


def _resample_pcm16(audio_pcm16: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if audio_pcm16.size == 0 or src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return np.ascontiguousarray(audio_pcm16, dtype=np.int16)

    src = audio_pcm16.astype(np.float32)
    src_len = src.size
    dst_len = max(1, int(round(src_len * dst_rate / src_rate)))
    src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    dst = np.interp(dst_x, src_x, src)
    return np.ascontiguousarray(np.clip(dst, -32768, 32767).astype(np.int16))


def _resample_float32(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if audio.size == 0 or src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)

    src = np.ascontiguousarray(audio, dtype=np.float32)
    src_len = src.size
    dst_len = max(1, int(round(src_len * dst_rate / src_rate)))
    src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    dst = np.interp(dst_x, src_x, src)
    return np.ascontiguousarray(np.clip(dst, -1.0, 1.0).astype(np.float32))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a_flat = a.reshape(-1).astype(np.float32, copy=False)
    b_flat = b.reshape(-1).astype(np.float32, copy=False)
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    if denom <= 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


class SpeakerEmbeddingPipeline:
    def __init__(self, cfg: CeoConfig, tts_model):
        self._cfg = cfg
        self._tts_model = tts_model
        self._speaker_embedding_model = self._load_standalone_speaker_encoder()
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

    def _load_standalone_speaker_encoder(self):
        model_name = self._cfg.speaker_embedding_model.strip()
        if not model_name:
            return None

        try:
            import mlx.core as mx
            from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSSpeakerEncoderConfig
            from mlx_audio.tts.models.qwen3_tts.speaker_encoder import Qwen3TTSSpeakerEncoder
            from mlx_audio.utils import get_model_path, load_config, load_weights

            model_path = get_model_path(model_name)
            config = load_config(model_path)
            cfg_fields = getattr(Qwen3TTSSpeakerEncoderConfig, "__dataclass_fields__", {})
            root_cfg = {name: config[name] for name in cfg_fields if name in config}
            speaker_cfg_values = dict(config.get("speaker_encoder_config", {}))
            if not speaker_cfg_values:
                speaker_cfg_values = root_cfg
            else:
                for name, value in root_cfg.items():
                    speaker_cfg_values.setdefault(name, value)
            speaker_cfg = Qwen3TTSSpeakerEncoderConfig(
                **speaker_cfg_values,
            )
            speaker_model = Qwen3TTSSpeakerEncoder(speaker_cfg)

            raw_weights = load_weights(model_path)
            candidate_weights: list[tuple[str, dict]] = []
            seen_signatures: set[tuple[str, ...]] = set()

            def _register_candidate(tag: str, values: dict):
                if not values:
                    return
                signature = tuple(sorted(values.keys()))
                if signature in seen_signatures:
                    return
                seen_signatures.add(signature)
                candidate_weights.append((tag, values))

            _register_candidate("sanitize", Qwen3TTSSpeakerEncoder.sanitize(dict(raw_weights)))
            _register_candidate("raw", dict(raw_weights))

            for prefix in ("speaker_encoder.", "model.speaker_encoder.", "module.speaker_encoder."):
                trimmed = {
                    key[len(prefix) :]: value
                    for key, value in raw_weights.items()
                    if key.startswith(prefix)
                }
                _register_candidate(f"trim:{prefix}", trimmed)

            attempted_tags: list[str] = []
            selected_strategy: str | None = None
            selected_strict = True
            for tag, weights in candidate_weights:
                attempted_tags.append(tag)
                try:
                    speaker_model.load_weights(list(weights.items()), strict=True)
                    selected_strategy = tag
                    break
                except ValueError as exc:
                    logger.warning(
                        "[ceo] speaker encoder strict load fallito (%s, strategy=%s): %s",
                        model_name,
                        tag,
                        exc,
                    )

            if selected_strategy is None:
                if not candidate_weights:
                    raise ValueError("nessun set di pesi valido trovato nel checkpoint")
                tag, weights = candidate_weights[-1]
                attempted_tags.append(f"{tag}:strict=False")
                speaker_model.load_weights(list(weights.items()), strict=False)
                selected_strategy = tag
                selected_strict = False

            log_level = logger.info if selected_strict else logger.warning
            log_level(
                "[ceo] speaker encoder dedicato caricato (%s, strategy=%s, strict=%s, attempted=%s)",
                model_name,
                selected_strategy,
                selected_strict,
                ",".join(attempted_tags),
            )

            mx.eval(speaker_model.parameters())
            speaker_model.eval()
            return speaker_model
        except Exception:
            logger.exception("[ceo] caricamento speaker encoder dedicato fallito (%s)", model_name)
            return None

    def _extract_embedding(self, audio_16k: np.ndarray) -> np.ndarray:
        if self._speaker_embedding_model is not None:
            import mlx.core as mx
            from mlx_audio.tts.models.qwen3_tts.qwen3_tts import mel_spectrogram

            audio_24k = _resample_float32(audio_16k, src_rate=TARGET_SAMPLE_RATE, dst_rate=self._target_sample_rate)
            mels = mel_spectrogram(
                mx.array(audio_24k),
                n_fft=1024,
                num_mels=128,
                sample_rate=self._target_sample_rate,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=12_000,
            )
            embedding = self._speaker_embedding_model(mels)
        elif self._cfg.speaker_embedding_use_tts_fallback:
            audio_24k = _resample_float32(audio_16k, src_rate=TARGET_SAMPLE_RATE, dst_rate=self._target_sample_rate)
            embedding = self._tts_model.extract_speaker_embedding(audio_24k, sr=self._target_sample_rate)
        else:
            raise RuntimeError("speaker encoder non disponibile e fallback TTS disattivato")

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
            similarity = _cosine_similarity(self._last_embedding, current_embedding)
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


class TtsOutboundAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=256)
        self._sample_rate = DEFAULT_WEBRTC_SAMPLE_RATE
        self._chunk_samples = max(1, self._sample_rate * WEBRTC_CHUNK_MS // 1000)
        self._pts = 0
        self._started_at: float | None = None
        self._pending_chunks = 0
        self._pending_lock = asyncio.Lock()
        self._playback_idle = asyncio.Event()
        self._playback_idle.set()
        self._prebuffer_chunks = OUTBOUND_PREBUFFER_CHUNKS
        self._max_buffer_ms = OUTBOUND_MAX_BUFFER_MS
        self._min_buffered_chunks_for_playback = self._prebuffer_chunks
        self._buffering = True

    def set_output_sample_rate(self, sample_rate: int) -> None:
        if sample_rate <= 0 or sample_rate == self._sample_rate:
            return

        self._sample_rate = sample_rate
        self._chunk_samples = max(1, self._sample_rate * WEBRTC_CHUNK_MS // 1000)
        self._pts = 0
        logger.info("[ceo] outbound sample rate aggiornato a %sHz", self._sample_rate)

    @property
    def output_sample_rate(self) -> int:
        return self._sample_rate

    async def recv(self) -> AudioFrame:
        dequeued = False
        try:
            pcm16 = self._queue.get_nowait()
            dequeued = True
        except asyncio.QueueEmpty:
            pcm16 = np.zeros(self._chunk_samples, dtype=np.int16)

        async with self._pending_lock:
            if self._buffering and self._pending_chunks < self._min_buffered_chunks_for_playback:
                pcm16 = np.zeros(self._chunk_samples, dtype=np.int16)
            else:
                self._buffering = False

        if self._started_at is None:
            self._started_at = time.monotonic()
        else:
            expected_elapsed = self._pts / self._sample_rate
            wait_s = (self._started_at + expected_elapsed) - time.monotonic()
            if wait_s > 0:
                await asyncio.sleep(wait_s)

        frame = AudioFrame.from_ndarray(pcm16.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self._sample_rate
        frame.time_base = Fraction(1, self._sample_rate)
        frame.pts = self._pts
        self._pts += pcm16.size

        async with self._pending_lock:
            if dequeued:
                self._pending_chunks = max(0, self._pending_chunks - 1)
            if self._pending_chunks == 0:
                self._buffering = True
                self._playback_idle.set()

        return frame

    async def push_pcm16(self, audio_pcm16: np.ndarray, sample_rate: int) -> None:
        if audio_pcm16.size == 0:
            return

        adapted = _resample_pcm16(audio_pcm16, sample_rate, self._sample_rate)
        chunk_count = int(math.ceil(adapted.size / self._chunk_samples))
        if chunk_count <= 0:
            return

        async with self._pending_lock:
            self._pending_chunks += chunk_count
            self._playback_idle.clear()
            max_buffer_chunks = max(1, self._max_buffer_ms // WEBRTC_CHUNK_MS)
            chunks_to_drop = max(0, self._pending_chunks - max_buffer_chunks)
            while chunks_to_drop > 0 and not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._pending_chunks = max(0, self._pending_chunks - 1)
                    chunks_to_drop -= 1
                except asyncio.QueueEmpty:
                    break
            if chunks_to_drop > 0:
                logger.debug("[ceo] impossibile svuotare backlog completo (residuo=%s)", chunks_to_drop)

        for start in range(0, adapted.size, self._chunk_samples):
            pcm = np.ascontiguousarray(adapted[start : start + self._chunk_samples], dtype=np.int16)
            if pcm.size < self._chunk_samples:
                pcm = np.pad(pcm, (0, self._chunk_samples - pcm.size))
            await self._queue.put(pcm)

    async def wait_until_idle(self) -> None:
        await self._playback_idle.wait()


class TtsPipeline:
    def __init__(self, cfg: CeoConfig):
        self._cfg = cfg
        self._model = load_tts(cfg.tts_model)
        logger.info("[ceo] TTS attivo (%s, language=%s)", cfg.tts_model, cfg.tts_language)

    @property
    def model(self):
        return self._model

    def stream_voice_design_pcm16(self, text: str):
        if not text.strip():
            return

        for result in self._model.generate_voice_design(
            text=text,
            language=self._cfg.tts_language,
            instruct=self._cfg.tts_instruct,
            stream=True,
            streaming_interval=self._cfg.tts_streaming_interval,
        ):
            audio = np.asarray(result.audio, dtype=np.float32)
            if audio.size == 0:
                continue
            clipped = np.clip(audio, -1.0, 1.0)
            sample_rate = int(getattr(result, "sample_rate", TARGET_SAMPLE_RATE))
            yield (clipped * 32767.0).astype(np.int16), sample_rate


async def ceo_consumer(cfg: CeoConfig) -> None:
    pc = RTCPeerConnection()
    cmd_channel = WebRTCCommandChannel(pc)
    debug = CeoDebug(cfg)
    turn_pipeline = SmartTurnPipeline(cfg, debug=debug)
    asr_pipeline = AsrPipeline(cfg)
    tts_pipeline = TtsPipeline(cfg)
    speaker_pipeline = SpeakerEmbeddingPipeline(cfg, tts_pipeline.model)
    outbound_track = TtsOutboundAudioTrack()
    pc.addTrack(outbound_track)
    logger.info("[webrtc] traccia audio outbound TTS aggiunta")
    cmd_send_lock = asyncio.Lock()
    tts_idle = asyncio.Event()
    tts_idle.set()
    say_to_user_queue: asyncio.Queue[str] = asyncio.Queue()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info("[webrtc] connection state -> %s", pc.connectionState)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        logger.info("[webrtc] ice connection state -> %s", pc.iceConnectionState)

    async def handle_say_to_user_text(text: str) -> None:
        if not text:
            return

        logger.info("[ceo] say_to_user ricevuto -> %r", text)
        tts_idle.clear()
        try:
            stream_iter = iter(tts_pipeline.stream_voice_design_pcm16(text))
            streamed_samples = 0
            tts_sample_rate: int | None = None
            accumulated_chunks: list[np.ndarray] = []

            while True:
                chunk = await asyncio.to_thread(lambda: next(stream_iter, None))
                if chunk is None:
                    break
                pcm16_chunk, chunk_sample_rate = chunk
                if pcm16_chunk.size == 0:
                    continue
                if tts_sample_rate is None:
                    tts_sample_rate = chunk_sample_rate
                streamed_samples += pcm16_chunk.size
                accumulated_chunks.append(pcm16_chunk)
                await outbound_track.push_pcm16(pcm16_chunk, sample_rate=chunk_sample_rate)

            if streamed_samples == 0 or tts_sample_rate is None:
                logger.warning("[ceo] TTS non ha generato audio")
                return

            if debug.enabled:
                full_audio = np.concatenate(accumulated_chunks)
                debug.save_tts_wav(full_audio, sample_rate=tts_sample_rate)

            await outbound_track.wait_until_idle()
            logger.info(
                "[ceo] TTS inviato su WebRTC (%.2fs, src_sr=%sHz, dst_sr=%sHz)",
                streamed_samples / max(1, tts_sample_rate),
                tts_sample_rate,
                outbound_track.output_sample_rate,
            )
        finally:
            tts_idle.set()

    async def say_to_user_worker() -> None:
        while True:
            text = await say_to_user_queue.get()
            try:
                await handle_say_to_user_text(text)
            finally:
                say_to_user_queue.task_done()

    say_to_user_worker_task = asyncio.create_task(say_to_user_worker())

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        logger.info("[webrtc] traccia in ingresso aperta: kind=%s id=%s", track.kind, getattr(track, "id", "-"))
        if track.kind != "audio":
            return

        async def pump() -> None:
            while True:
                frame = await track.recv()
                outbound_track.set_output_sample_rate(int(frame.sample_rate or DEFAULT_WEBRTC_SAMPLE_RATE))
                if not tts_idle.is_set():
                    turn_pipeline.reset_turn_state()
                    continue
                completed_audio = turn_pipeline.process(frame)
                if completed_audio is not None:
                    debug.save_segment_for_asr(completed_audio)
                    transcript = await asyncio.to_thread(asr_pipeline.transcribe, completed_audio)
                    if transcript:
                        await asyncio.to_thread(speaker_pipeline.process_transcribed_segment, completed_audio)
                    message = {
                        "type": "speaker_turn_completed",
                        "producer": "ceo",
                        "ts": frame.time,
                        "transcript": transcript,
                        "asr_model": cfg.asr_model,
                    }
                    try:
                        async with cmd_send_lock:
                            await cmd_channel.send_json(message)
                    except Exception:
                        logger.exception("[ceo] invio speaker_turn_completed fallito")
                        continue
                    logger.info("[ceo] turn-end -> %s", message)

        asyncio.create_task(pump())

    logger.info("[ceo] connessione websocket verso backend: %s", cfg.ermete_ws)
    async with websockets.connect(cfg.ermete_ws, additional_headers=cfg.auth_headers()) as ws:
        await maybe_handle_offer(ws, pc)

        async def consume_cmd_messages() -> None:
            async for data in cmd_channel.iter_json():
                t = data.get("type")
                if t == "ping":
                    async with cmd_send_lock:
                        await cmd_channel.send_json({"type": "pong"})
                elif t == "say_to_user" and data.get("producer") == "teia":
                    text = str(data.get("text", "")).strip()
                    if text:
                        await say_to_user_queue.put(text)
                else:
                    logger.info("[dc] msg: %s", data)

        try:
            await consume_cmd_messages()
        finally:
            say_to_user_worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await say_to_user_worker_task


def main() -> None:
    setup_logging()
    cfg = CeoConfig()
    logger.info("[ceo] ERMETE_WS=%s", cfg.ermete_ws)
    run(ceo_consumer(cfg))


if __name__ == "__main__":
    main()
