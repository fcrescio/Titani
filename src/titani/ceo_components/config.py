import os
from dataclasses import dataclass

from titani.common import ErmeteConfig


TARGET_SAMPLE_RATE = 16_000
MAX_CONTEXT_SECONDS = 8
MAX_CONTEXT_SAMPLES = TARGET_SAMPLE_RATE * MAX_CONTEXT_SECONDS
WEBRTC_CHUNK_MS = int(os.getenv("CEO_WEBRTC_CHUNK_MS", "20"))
WEBRTC_CHUNK_SAMPLES = TARGET_SAMPLE_RATE * WEBRTC_CHUNK_MS // 1000
DEFAULT_WEBRTC_SAMPLE_RATE = 48_000
OUTBOUND_PREBUFFER_CHUNKS = max(1, int(os.getenv("CEO_OUTBOUND_PREBUFFER_CHUNKS", "3")))
OUTBOUND_MAX_BUFFER_MS = max(100, int(os.getenv("CEO_OUTBOUND_MAX_BUFFER_MS", "400")))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class CeoConfig(ErmeteConfig):
    pre_roll_ms: int = int(os.getenv("CEO_PRE_ROLL_MS", "250"))
    start_speech_chunks: int = int(os.getenv("CEO_START_SPEECH_CHUNKS", "10"))
    speech_majority_ratio: float = float(os.getenv("CEO_SPEECH_MAJORITY_RATIO", "0.5"))
    speech_subchunk_min_count: int = int(os.getenv("CEO_SPEECH_SUBCHUNK_MIN_COUNT", "2"))
    vad_min_rms: float = float(os.getenv("CEO_VAD_MIN_RMS", "0.0"))
    silence_ms_before_endpoint: int = int(os.getenv("CEO_SILENCE_MS_BEFORE_ENDPOINT", "300"))
    endpoint_retry_ms: int = int(os.getenv("CEO_ENDPOINT_RETRY_MS", "150"))
    max_silence_ms_force_commit: int = int(os.getenv("CEO_MAX_SILENCE_MS_FORCE_COMMIT", "1500"))
    trailing_silence_pad_ms: int = int(os.getenv("CEO_TRAILING_SILENCE_PAD_MS", "200"))
    smart_turn_threshold: float = float(os.getenv("CEO_SMART_TURN_THRESHOLD", "0.5"))
    smart_turn_min_segment_seconds: float = float(os.getenv("CEO_SMART_TURN_MIN_SEGMENT_SECONDS", "3.0"))
    debug_dump_wav_enabled: bool = _env_bool("CEO_DEBUG_DUMP_WAV_ENABLED", False)
    debug_dump_wav_dir: str = os.getenv("CEO_DEBUG_DUMP_WAV_DIR", "./ceo_debug/smart_turn")
    asr_model: str = os.getenv("CEO_ASR_MODEL", "mlx-community/Qwen3-ASR-0.6B-8bit")
    asr_language: str = os.getenv("CEO_ASR_LANGUAGE", "Italian")
    debug_mode: bool = _env_bool("CEO_DEBUG_MODE", False)
    debug_out_dir: str = os.getenv("CEO_DEBUG_OUT_DIR", "./ceo_debug")
    debug_heartbeat_ms: int = int(os.getenv("CEO_DEBUG_HEARTBEAT_MS", "2000"))
    tts_model: str = os.getenv(
        "CEO_TTS_MODEL",
        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    )
    tts_language: str = os.getenv("CEO_TTS_LANGUAGE", "Italian")
    tts_ref_audio: str = os.getenv("CEO_TTS_REF_AUDIO", "")
    tts_ref_text: str = os.getenv("CEO_TTS_REF_TEXT", "")
    tts_streaming_interval: float = float(os.getenv("CEO_TTS_STREAMING_INTERVAL", "0.04"))
    speaker_embedding_threshold: float = float(os.getenv("CEO_SPEAKER_EMBEDDING_THRESHOLD", "0.8"))
    speaker_embeddings_dir: str = os.getenv("CEO_SPEAKER_EMBEDDINGS_DIR", "./ceo_speakers")
    require_known_speaker_for_transcript: bool = _env_bool("CEO_REQUIRE_KNOWN_SPEAKER_FOR_TRANSCRIPT", False)
