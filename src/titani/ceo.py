import asyncio
import json
from dataclasses import dataclass

import numpy as np
import websockets
from aiortc import MediaStreamTrack, RTCPeerConnection
from aiortc.mediastreams import AudioFrame

from titani.common import ErmeteConfig, iter_ws_json, maybe_handle_offer, run


@dataclass(slots=True)
class CeoConfig(ErmeteConfig):
    vad_threshold: float = 0.01


class MlxVadPipeline:
    """Minimal VAD pipeline designed for macOS + mlx-audio.

    If mlx-audio is not available, falls back to a simple RMS threshold.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self._mlx_vad = None
        try:
            from mlx_audio.vad import VoiceActivityDetector  # type: ignore

            self._mlx_vad = VoiceActivityDetector()
            print("[ceo] mlx-audio VAD attivo")
        except Exception:
            print("[ceo] mlx-audio non disponibile, fallback su RMS")

    def is_speech(self, frame: AudioFrame) -> bool:
        pcm = frame.to_ndarray()
        mono = pcm.mean(axis=0) if pcm.ndim == 2 else pcm
        mono = mono.astype(np.float32)
        if np.max(np.abs(mono)) > 1.5:
            mono = mono / 32768.0

        if self._mlx_vad is not None:
            # API indicative: adaptarala se cambia nella versione reale di mlx-audio.
            return bool(self._mlx_vad(mono, sample_rate=frame.sample_rate))

        rms = float(np.sqrt(np.mean(mono ** 2)))
        return rms >= self.threshold


async def ceo_consumer(cfg: CeoConfig) -> None:
    pc = RTCPeerConnection()
    pipeline = MlxVadPipeline(threshold=cfg.vad_threshold)

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        if track.kind != "audio":
            return

        async def pump() -> None:
            while True:
                frame = await track.recv()
                speaking = pipeline.is_speech(frame)
                print(f"[ceo] speech={speaking} ts={frame.time}")

        asyncio.create_task(pump())

    async with websockets.connect(cfg.ermete_ws, additional_headers=cfg.auth_headers()) as ws:
        await maybe_handle_offer(ws, pc)
        async for data in iter_ws_json(ws):
            t = data.get("type")
            if t == "ping":
                await ws.send(json.dumps({"type": "pong"}))
            else:
                print(f"[ws] msg: {data}")


def main() -> None:
    cfg = CeoConfig()
    print(f"[ceo] ERMETE_WS={cfg.ermete_ws}")
    run(ceo_consumer(cfg))


if __name__ == "__main__":
    main()
