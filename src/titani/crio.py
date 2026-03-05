import asyncio
import logging
from dataclasses import dataclass
from contextlib import suppress

import websockets
from aiortc import MediaStreamTrack, RTCPeerConnection
from aiortc.mediastreams import AudioFrame, MediaStreamError

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
    pump_tasks: set[asyncio.Task[None]] = set()
    shutdown_lock = asyncio.Lock()
    pc.addTrack(loop_track)
    logger.info("[webrtc] traccia audio loopback locale aggiunta")

    async def shutdown_peer_connection() -> None:
        async with shutdown_lock:
            tasks = list(pump_tasks)
            for task in tasks:
                task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task

            if pc.connectionState != "closed":
                await pc.close()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info("[webrtc] connection state -> %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await shutdown_peer_connection()

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        logger.info("[webrtc] ice connection state -> %s", pc.iceConnectionState)

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        logger.info("[webrtc] traccia in ingresso aperta: kind=%s id=%s", track.kind, getattr(track, "id", "-"))
        if track.kind != "audio":
            return

        async def pump() -> None:
            try:
                while True:
                    frame = await track.recv()
                    await loop_track.push(frame)
            except asyncio.CancelledError:
                logger.info("[webrtc] pump annullato: track id=%s", getattr(track, "id", "-"))
                raise
            except MediaStreamError:
                logger.info("[webrtc] stream terminato: track id=%s", getattr(track, "id", "-"))
            except Exception:
                logger.exception("[webrtc] errore inatteso nel pump: track id=%s", getattr(track, "id", "-"))

        task = asyncio.create_task(pump())
        pump_tasks.add(task)
        task.add_done_callback(pump_tasks.discard)

    logger.info("[crio] connessione websocket verso backend: %s", cfg.ermete_ws)
    try:
        async with websockets.connect(cfg.ermete_ws, additional_headers=cfg.auth_headers()) as ws:
            await maybe_handle_offer(ws, pc)
            async for data in cmd_channel.iter_json():
                t = data.get("type")
                if t == "ping":
                    await cmd_channel.send_json({"type": "pong"})
                else:
                    logger.info("[dc] msg: %s", data)
    finally:
        await shutdown_peer_connection()


def main() -> None:
    setup_logging()
    cfg = CrioConfig()
    logger.info("[crio] ERMETE_WS=%s", cfg.ermete_ws)
    run(crio_consumer(cfg))


if __name__ == "__main__":
    main()
