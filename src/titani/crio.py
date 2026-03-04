import asyncio
import logging
from dataclasses import dataclass

import websockets
from aiortc import MediaStreamTrack, RTCPeerConnection
from aiortc.mediastreams import AudioFrame

from titani.common import ErmeteConfig, WebRTCCommandChannel, maybe_handle_offer, run, setup_logging

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CrioConfig(ErmeteConfig):
    pass


class LoopbackAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=200)

    async def recv(self) -> AudioFrame:
        return await self._queue.get()

    async def push(self, frame: AudioFrame) -> None:
        if self._queue.full():
            try:
                _ = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(frame)


async def crio_consumer(cfg: CrioConfig) -> None:
    pc = RTCPeerConnection()
    cmd_channel = WebRTCCommandChannel(pc)
    loop_track = LoopbackAudioTrack()
    pc.addTrack(loop_track)
    logger.info("[webrtc] traccia audio loopback locale aggiunta")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info("[webrtc] connection state -> %s", pc.connectionState)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        logger.info("[webrtc] ice connection state -> %s", pc.iceConnectionState)

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        logger.info("[webrtc] traccia in ingresso aperta: kind=%s id=%s", track.kind, getattr(track, "id", "-"))
        if track.kind != "audio":
            return

        async def pump() -> None:
            while True:
                frame = await track.recv()
                await loop_track.push(frame)

        asyncio.create_task(pump())

    logger.info("[crio] connessione websocket verso backend: %s", cfg.ermete_ws)
    async with websockets.connect(cfg.ermete_ws, additional_headers=cfg.auth_headers()) as ws:
        await maybe_handle_offer(ws, pc)
        async for data in cmd_channel.iter_json():
            t = data.get("type")
            if t == "ping":
                await cmd_channel.send_json({"type": "pong"})
            else:
                logger.info("[dc] msg: %s", data)


def main() -> None:
    setup_logging()
    cfg = CrioConfig()
    logger.info("[crio] ERMETE_WS=%s", cfg.ermete_ws)
    run(crio_consumer(cfg))


if __name__ == "__main__":
    main()
