import asyncio
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import websockets
from aiortc import MediaStreamTrack, RTCPeerConnection
from aiortc.mediastreams import AudioFrame
from mlx_audio.vad import VoiceActivityDetector, load

from titani.common import ErmeteConfig, iter_ws_json, maybe_handle_offer, run

TARGET_SAMPLE_RATE = 16_000
MAX_CONTEXT_SECONDS = 8
MAX_CONTEXT_SAMPLES = TARGET_SAMPLE_RATE * MAX_CONTEXT_SECONDS


@dataclass(slots=True)
class CeoConfig(ErmeteConfig):
    silence_ms_before_endpoint: int = int(os.getenv("CEO_SILENCE_MS_BEFORE_ENDPOINT", "300"))
    smart_turn_threshold: float = float(os.getenv("CEO_SMART_TURN_THRESHOLD", "0.5"))


class SmartTurnPipeline:
    """Smart Turn v3 endpoint detection with 8s rolling context."""

    def __init__(self, cfg: CeoConfig):
        self._cfg = cfg
        self._vad = VoiceActivityDetector()
        self._model = load("mlx-community/smart-turn-v3", strict=True)
        self._audio_context = np.zeros(0, dtype=np.float32)
        self._in_user_turn = False
        self._last_speech_ms = 0.0
        self._checked_during_current_silence = False
        print("[ceo] Smart Turn v3 attivo (mlx-community/smart-turn-v3)")

    def _frame_to_mono(self, frame: AudioFrame) -> np.ndarray:
        pcm = frame.to_ndarray()
        mono = pcm.mean(axis=0) if pcm.ndim == 2 else pcm
        mono = mono.astype(np.float32)
        if np.max(np.abs(mono), initial=0.0) > 1.5:
            mono = mono / 32768.0
        return mono

    def _resample_to_16k(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == TARGET_SAMPLE_RATE:
            return audio
        if sample_rate <= 0 or audio.size == 0:
            return np.zeros(0, dtype=np.float32)

        duration_s = audio.size / sample_rate
        target_size = int(round(duration_s * TARGET_SAMPLE_RATE))
        if target_size <= 0:
            return np.zeros(0, dtype=np.float32)

        src_x = np.linspace(0.0, duration_s, num=audio.size, endpoint=False)
        dst_x = np.linspace(0.0, duration_s, num=target_size, endpoint=False)
        return np.interp(dst_x, src_x, audio).astype(np.float32)

    def _append_context(self, frame: AudioFrame) -> np.ndarray:
        mono = self._frame_to_mono(frame)
        mono_16k = self._resample_to_16k(mono, frame.sample_rate)
        if mono_16k.size == 0:
            return self._audio_context

        self._audio_context = np.concatenate((self._audio_context, mono_16k))[-MAX_CONTEXT_SAMPLES:]
        return self._audio_context

    def _is_speech(self, frame: AudioFrame) -> bool:
        mono = self._frame_to_mono(frame)
        return bool(self._vad(mono, sample_rate=frame.sample_rate))

    def process(self, frame: AudioFrame) -> bool:
        context = self._append_context(frame)
        now_ms = time.monotonic() * 1000.0
        speaking = self._is_speech(frame)

        if speaking:
            self._in_user_turn = True
            self._last_speech_ms = now_ms
            self._checked_during_current_silence = False
            return False

        if not self._in_user_turn:
            return False

        silence_ms = now_ms - self._last_speech_ms
        if silence_ms < self._cfg.silence_ms_before_endpoint:
            return False

        if self._checked_during_current_silence:
            return False

        self._checked_during_current_silence = True
        result = self._model.predict_endpoint(
            context,
            sample_rate=TARGET_SAMPLE_RATE,
            threshold=self._cfg.smart_turn_threshold,
        )

        prediction = int(getattr(result, "prediction", 0))
        probability = float(getattr(result, "probability", 0.0))
        print(f"[ceo] smart-turn prediction={prediction} probability={probability:.3f}")

        if prediction == 1:
            self._in_user_turn = False
            self._audio_context = np.zeros(0, dtype=np.float32)
            return True

        return False


async def ceo_consumer(cfg: CeoConfig) -> None:
    pc = RTCPeerConnection()
    turn_pipeline = SmartTurnPipeline(cfg)
    ws_send_lock = asyncio.Lock()

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        if track.kind != "audio":
            return

        async def pump() -> None:
            while True:
                frame = await track.recv()
                if turn_pipeline.process(frame):
                    message = {
                        "type": "speaker_turn_completed",
                        "producer": "ceo",
                        "ts": frame.time,
                    }
                    async with ws_send_lock:
                        await ws.send(json.dumps(message))
                    print(f"[ceo] turn-end -> {message}")

        asyncio.create_task(pump())

    async with websockets.connect(cfg.ermete_ws, additional_headers=cfg.auth_headers()) as ws:
        await maybe_handle_offer(ws, pc)
        async for data in iter_ws_json(ws):
            t = data.get("type")
            if t == "ping":
                async with ws_send_lock:
                    await ws.send(json.dumps({"type": "pong"}))
            else:
                print(f"[ws] msg: {data}")


def main() -> None:
    cfg = CeoConfig()
    print(f"[ceo] ERMETE_WS={cfg.ermete_ws}")
    run(ceo_consumer(cfg))


if __name__ == "__main__":
    main()
