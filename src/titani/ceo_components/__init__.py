"""Componenti del consumer CEO separati per responsabilità."""

from .config import (
    DEFAULT_WEBRTC_SAMPLE_RATE,
    TARGET_SAMPLE_RATE,
    CeoConfig,
)
from .debug import CeoDebug
from .inbound import AsrPipeline, SmartTurnPipeline, SpeakerEmbeddingPipeline
from .outbound import TtsOutboundAudioTrack, TtsPipeline

__all__ = [
    "AsrPipeline",
    "CeoConfig",
    "CeoDebug",
    "DEFAULT_WEBRTC_SAMPLE_RATE",
    "SmartTurnPipeline",
    "SpeakerEmbeddingPipeline",
    "TARGET_SAMPLE_RATE",
    "TtsOutboundAudioTrack",
    "TtsPipeline",
]
