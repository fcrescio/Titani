"""Componenti del consumer CEO separati per responsabilità."""

from .config import (
    DEFAULT_WEBRTC_SAMPLE_RATE,
    TARGET_SAMPLE_RATE,
    AsrConfig,
    CeoConfig,
    DebugConfig,
    IngressConfig,
    OutboundAdaptationConfig,
    OutboundConfig,
    SpeakerConfig,
)
from .debug import CeoDebug

try:
    from .inbound import AsrPipeline, SmartTurnPipeline, SpeakerEmbeddingPipeline
except ModuleNotFoundError:  # optional runtime deps for CEO mode
    AsrPipeline = SmartTurnPipeline = SpeakerEmbeddingPipeline = None  # type: ignore[assignment]

try:
    from .outbound import TtsOutboundAudioTrack, TtsPipeline
except ModuleNotFoundError:  # optional runtime deps for CEO mode
    TtsOutboundAudioTrack = TtsPipeline = None  # type: ignore[assignment]

__all__ = [
    "AsrConfig",
    "AsrPipeline",
    "CeoConfig",
    "CeoDebug",
    "DebugConfig",
    "DEFAULT_WEBRTC_SAMPLE_RATE",
    "IngressConfig",
    "OutboundAdaptationConfig",
    "OutboundConfig",
    "SmartTurnPipeline",
    "SpeakerConfig",
    "SpeakerEmbeddingPipeline",
    "TARGET_SAMPLE_RATE",
    "TtsOutboundAudioTrack",
    "TtsPipeline",
]
