import logging
import os
from dataclasses import dataclass, field

from titani.common import ErmeteConfig


logger = logging.getLogger(__name__)


TARGET_SAMPLE_RATE = 16_000
MAX_CONTEXT_SECONDS = 8
MAX_CONTEXT_SAMPLES = TARGET_SAMPLE_RATE * MAX_CONTEXT_SECONDS
WEBRTC_CHUNK_MS = int(os.getenv("CEO_WEBRTC_CHUNK_MS", "20"))
WEBRTC_CHUNK_SAMPLES = TARGET_SAMPLE_RATE * WEBRTC_CHUNK_MS // 1000
DEFAULT_WEBRTC_SAMPLE_RATE = 48_000
OUTBOUND_PREBUFFER_CHUNKS = max(1, int(os.getenv("CEO_OUTBOUND_PREBUFFER_CHUNKS", "3")))
OUTBOUND_TARGET_BUFFER_MS = max(
    WEBRTC_CHUNK_MS * 2,
    int(os.getenv("CEO_OUTBOUND_TARGET_BUFFER_MS", "400")),
)
OUTBOUND_LOW_WATERMARK_RATIO = 0.4


def derive_outbound_buffer_watermarks(target_buffer_ms: int) -> tuple[int, int]:
    high_watermark_ms = max(WEBRTC_CHUNK_MS * 2, int(target_buffer_ms))
    low_watermark_ms = max(
        WEBRTC_CHUNK_MS,
        int(high_watermark_ms * OUTBOUND_LOW_WATERMARK_RATIO),
    )
    low_watermark_ms = min(low_watermark_ms, high_watermark_ms - WEBRTC_CHUNK_MS)
    return low_watermark_ms, high_watermark_ms

UNSUPPORTED_NOOP_ENV_VARS: dict[str, str] = {
    "CEO_TTS_LANGUAGE": "Il backend TTS corrente (mlx_audio.tts) non supporta la selezione lingua in generate().",
}

INGRESS_PROFILE_DEFAULT = "balanced"
INGRESS_PROFILE_PRESETS: dict[str, dict[str, int | float]] = {
    "balanced": {
        "start_speech_chunks": 10,
        "speech_majority_ratio": 0.5,
        "speech_subchunk_min_count": 2,
        "vad_min_rms": 0.0,
    },
    "noisy": {
        "start_speech_chunks": 12,
        "speech_majority_ratio": 0.65,
        "speech_subchunk_min_count": 3,
        "vad_min_rms": 0.02,
    },
    "fast": {
        "start_speech_chunks": 6,
        "speech_majority_ratio": 0.4,
        "speech_subchunk_min_count": 1,
        "vad_min_rms": 0.0,
    },
}

INGRESS_ADVANCED_OVERRIDE_ENV_VARS: tuple[str, ...] = (
    "CEO_START_SPEECH_CHUNKS",
    "CEO_SPEECH_MAJORITY_RATIO",
    "CEO_SPEECH_SUBCHUNK_MIN_COUNT",
    "CEO_VAD_MIN_RMS",
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class IngressConfig:
    pre_roll_ms: int = int(os.getenv("CEO_PRE_ROLL_MS", "250"))
    ingress_profile: str = os.getenv("CEO_INGRESS_PROFILE", INGRESS_PROFILE_DEFAULT).strip().lower()
    advanced_tuning: bool = _env_bool("CEO_ADVANCED_TUNING", False)
    start_speech_chunks: int = int(INGRESS_PROFILE_PRESETS[INGRESS_PROFILE_DEFAULT]["start_speech_chunks"])
    speech_majority_ratio: float = float(INGRESS_PROFILE_PRESETS[INGRESS_PROFILE_DEFAULT]["speech_majority_ratio"])
    speech_subchunk_min_count: int = int(INGRESS_PROFILE_PRESETS[INGRESS_PROFILE_DEFAULT]["speech_subchunk_min_count"])
    vad_min_rms: float = float(INGRESS_PROFILE_PRESETS[INGRESS_PROFILE_DEFAULT]["vad_min_rms"])
    silence_ms_before_endpoint: int = int(os.getenv("CEO_SILENCE_MS_BEFORE_ENDPOINT", "300"))
    max_silence_ms_force_commit: int = int(os.getenv("CEO_MAX_SILENCE_MS_FORCE_COMMIT", "1500"))
    trailing_silence_pad_ms: int = int(os.getenv("CEO_TRAILING_SILENCE_PAD_MS", "200"))
    smart_turn_min_segment_seconds: float = float(os.getenv("CEO_SMART_TURN_MIN_SEGMENT_SECONDS", "3.0"))
    debug_dump_wav_enabled: bool = _env_bool("CEO_DEBUG_DUMP_WAV_ENABLED", False)
    debug_dump_wav_dir: str = os.getenv("CEO_DEBUG_DUMP_WAV_DIR", "./ceo_debug/smart_turn")

    def __post_init__(self) -> None:
        self.ingress_profile = os.getenv("CEO_INGRESS_PROFILE", INGRESS_PROFILE_DEFAULT).strip().lower()
        self.advanced_tuning = _env_bool("CEO_ADVANCED_TUNING", False)

        if self.ingress_profile not in INGRESS_PROFILE_PRESETS:
            valid_profiles = ", ".join(sorted(INGRESS_PROFILE_PRESETS))
            raise ValueError(
                f"CEO_INGRESS_PROFILE={self.ingress_profile!r} non valido. Valori ammessi: {valid_profiles}."
            )

        profile_values = INGRESS_PROFILE_PRESETS[self.ingress_profile]
        self.start_speech_chunks = int(profile_values["start_speech_chunks"])
        self.speech_majority_ratio = float(profile_values["speech_majority_ratio"])
        self.speech_subchunk_min_count = int(profile_values["speech_subchunk_min_count"])
        self.vad_min_rms = float(profile_values["vad_min_rms"])

        if self.advanced_tuning:
            self.start_speech_chunks = max(1, int(os.getenv("CEO_START_SPEECH_CHUNKS", str(self.start_speech_chunks))))
            self.speech_majority_ratio = float(os.getenv("CEO_SPEECH_MAJORITY_RATIO", str(self.speech_majority_ratio)))
            self.speech_subchunk_min_count = max(
                1,
                int(os.getenv("CEO_SPEECH_SUBCHUNK_MIN_COUNT", str(self.speech_subchunk_min_count))),
            )
            self.vad_min_rms = max(0.0, float(os.getenv("CEO_VAD_MIN_RMS", str(self.vad_min_rms))))
        else:
            for env_name in INGRESS_ADVANCED_OVERRIDE_ENV_VARS:
                raw_value = os.getenv(env_name)
                if raw_value is None:
                    continue
                logger.warning(
                    "[ceo][config] %s=%r ignorato: richiede CEO_ADVANCED_TUNING=1 (profilo ingress=%s)",
                    env_name,
                    raw_value,
                    self.ingress_profile,
                )

        if not 0.0 <= self.speech_majority_ratio <= 1.0:
            raise ValueError(
                f"CEO_SPEECH_MAJORITY_RATIO fuori range [0.0, 1.0]: {self.speech_majority_ratio!r}."
            )

        if self.start_speech_chunks < 1:
            raise ValueError(f"CEO_START_SPEECH_CHUNKS deve essere >= 1, ricevuto: {self.start_speech_chunks!r}.")

        if self.speech_subchunk_min_count < 1:
            raise ValueError(
                f"CEO_SPEECH_SUBCHUNK_MIN_COUNT deve essere >= 1, ricevuto: {self.speech_subchunk_min_count!r}."
            )


@dataclass(slots=True)
class AsrConfig:
    asr_model: str = os.getenv("CEO_ASR_MODEL", "mlx-community/Qwen3-ASR-0.6B-8bit")
    asr_language: str = os.getenv("CEO_ASR_LANGUAGE", "Italian")


@dataclass(slots=True)
class SpeakerConfig:
    speaker_embedding_threshold: float = float(os.getenv("CEO_SPEAKER_EMBEDDING_THRESHOLD", "0.8"))
    speaker_embeddings_dir: str = os.getenv("CEO_SPEAKER_EMBEDDINGS_DIR", "./ceo_speakers")
    require_known_speaker_for_transcript: bool = _env_bool("CEO_REQUIRE_KNOWN_SPEAKER_FOR_TRANSCRIPT", False)


@dataclass(slots=True)
class OutboundConfig:
    tts_model: str = os.getenv(
        "CEO_TTS_MODEL",
        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    )
    tts_ref_audio: str = os.getenv("CEO_TTS_REF_AUDIO", "")
    tts_ref_text: str = os.getenv("CEO_TTS_REF_TEXT", "")
    tts_streaming_interval: float = float(os.getenv("CEO_TTS_STREAMING_INTERVAL", "0.04"))
    say_to_user_queue_maxsize: int = max(1, int(os.getenv("CEO_SAY_TO_USER_QUEUE_MAXSIZE", "32")))
    say_to_user_queue_overflow_policy: str = os.getenv("CEO_SAY_TO_USER_QUEUE_OVERFLOW_POLICY", "drop_oldest")
    say_to_user_max_retries: int = max(0, int(os.getenv("CEO_SAY_TO_USER_MAX_RETRIES", "2")))
    say_to_user_retry_delay_s: float = max(0.0, float(os.getenv("CEO_SAY_TO_USER_RETRY_DELAY_S", "0.1")))


@dataclass(slots=True)
class DebugConfig:
    debug_mode: bool = _env_bool("CEO_DEBUG_MODE", False)
    debug_out_dir: str = os.getenv("CEO_DEBUG_OUT_DIR", "./ceo_debug")
    debug_heartbeat_ms: int = int(os.getenv("CEO_DEBUG_HEARTBEAT_MS", "2000"))
    debug_vad_trace: bool = _env_bool("CEO_DEBUG_VAD_TRACE", False)
    debug_vad_trace_every_chunks: int = int(os.getenv("CEO_DEBUG_VAD_TRACE_EVERY_CHUNKS", "25"))
    debug_vad_trace_jsonl: bool = _env_bool("CEO_DEBUG_VAD_TRACE_JSONL", False)


@dataclass(slots=True)
class CeoConfig(ErmeteConfig):
    ingress: IngressConfig = field(default_factory=IngressConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    outbound: OutboundConfig = field(default_factory=OutboundConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def __post_init__(self) -> None:
        for env_name, reason in UNSUPPORTED_NOOP_ENV_VARS.items():
            raw_value = os.getenv(env_name)
            if raw_value is None:
                continue
            logger.warning(
                "[ceo][config] %s=%r ignorato: %s",
                env_name,
                raw_value,
                reason,
            )
