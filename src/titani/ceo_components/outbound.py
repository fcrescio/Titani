import logging
from pathlib import Path

import numpy as np
from mlx_audio.tts.utils import load_model as load_tts

from titani.audio_pipeline import TtsOutboundAudioTrack

from .config import TARGET_SAMPLE_RATE, CeoConfig

logger = logging.getLogger(__name__)




class TtsPipeline:

    def __init__(self, cfg: CeoConfig):
        self._cfg = cfg
        self._model = load_tts(cfg.tts_model)
        self._ref_audio_path = Path(cfg.tts_ref_audio).expanduser() if cfg.tts_ref_audio.strip() else None
        self._ref_text = cfg.tts_ref_text.strip()
        if self._ref_audio_path is None or not self._ref_text:
            raise ValueError(
                "Voice cloning richiede CEO_TTS_REF_AUDIO e CEO_TTS_REF_TEXT valorizzati. "
                "Configura un file wav di riferimento e la sua trascrizione."
            )
        if not self._ref_audio_path.is_file():
            raise ValueError(f"File reference audio non trovato: {self._ref_audio_path}")
        logger.info("[ceo] TTS attivo (%s)", cfg.tts_model)

    @property
    def model(self):
        return self._model

    def stream_voice_clone_pcm16(self, text: str):
        if not text.strip():
            return

        for result in self._model.generate(
            text=text,
            ref_audio=str(self._ref_audio_path),
            ref_text=self._ref_text,
            stream=True,
            streaming_interval=self._cfg.tts_streaming_interval,
        ):
            audio = np.asarray(result.audio, dtype=np.float32)
            if audio.size == 0:
                continue
            clipped = np.clip(audio, -1.0, 1.0)
            sample_rate = int(getattr(result, "sample_rate", TARGET_SAMPLE_RATE))
            yield (clipped * 32767.0).astype(np.int16), sample_rate
