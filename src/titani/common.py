import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcdatachannel import RTCDataChannel


logger = logging.getLogger(__name__)


def setup_logging(default_level: str = "INFO") -> None:
    level_name = os.getenv("TITANI_LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.debug("Logging inizializzato con livello %s", logging.getLevelName(level))


@dataclass(slots=True)
class ErmeteConfig:
    ermete_ws: str = os.getenv("ERMETE_WS", "wss://alveare.metallize.it:8080/v1/ws?role=consumer")
    ermete_http_base: str = os.getenv("ERMETE_HTTP_BASE", "https://alveare.metallize.it:8080")
    psk_header: str = os.getenv("ERMETE_PSK_HEADER", "X-Ermete-PSK")
    psk: str = os.getenv("ERMETE_PSK", "fin che la barca va")

    frames_out_dir: str = os.getenv("FRAMES_OUT_DIR", "./frames_downloaded")

    def auth_headers(self) -> dict[str, str]:
        if not self.psk:
            return {}
        return {self.psk_header: self.psk}


def _enable_opus_fec_in_sdp(sdp: str) -> str:
    lines = sdp.splitlines()
    opus_payload_type: str | None = None

    for line in lines:
        if line.startswith("a=rtpmap:") and " opus/" in line.lower():
            opus_payload_type = line.split(":", 1)[1].split()[0]
            break

    if opus_payload_type is None:
        return sdp

    updated: list[str] = []
    fmtp_found = False
    fec_key = "useinbandfec="

    for line in lines:
        if line.startswith(f"a=fmtp:{opus_payload_type} "):
            fmtp_found = True
            prefix, params = line.split(" ", 1)
            if fec_key not in params:
                params = f"{params};useinbandfec=1"
            line = f"{prefix} {params}"
        updated.append(line)

    if not fmtp_found:
        updated.append(f"a=fmtp:{opus_payload_type} useinbandfec=1")

    return "\r\n".join(updated) + "\r\n"


async def maybe_handle_offer(ws: websockets.WebSocketClientProtocol, pc: RTCPeerConnection) -> None:
    """If the first WS message is an offer, reply with an answer and return.

    If the first message is not an offer, it is ignored by this helper.
    """

    logger.info("[webrtc] in attesa del primo messaggio di segnalazione")
    raw = await ws.recv()
    logger.debug("[webrtc] primo messaggio ricevuto (%d bytes)", len(raw))
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[webrtc] primo messaggio non JSON, handshake ignorato")
        return

    if msg.get("type") != "offer":
        logger.info("[webrtc] primo messaggio non è un'offerta: type=%s", msg.get("type"))
        return

    sdp = msg.get("sdp", "")
    sdp = _enable_opus_fec_in_sdp(sdp)
    logger.info("[webrtc] offerta ricevuta, imposto remote description (sdp=%d bytes)", len(sdp))
    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
    logger.info("[webrtc] remote description impostata, creo answer")
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await ws.send(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))
    logger.info("[webrtc] answer inviato (%d bytes)", len(pc.localDescription.sdp))


async def download_frame(http: httpx.AsyncClient, cfg: ErmeteConfig, payload: dict[str, Any]) -> Path:
    Path(cfg.frames_out_dir).mkdir(parents=True, exist_ok=True)

    url = payload.get("download_url") or ""
    if not url:
        file_name = payload.get("file_name") or payload.get("frame_id") or f"{int(time.time() * 1000)}.bin"
        url = f"{cfg.ermete_http_base.rstrip('/')}/v1/frames/file/{file_name}"

    file_name = payload.get("file_name") or os.path.basename(url)
    out_path = Path(cfg.frames_out_dir) / file_name

    response = await http.get(url, headers=cfg.auth_headers(), timeout=30.0)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    logger.info("[frames] scaricato -> %s (%d bytes)", out_path, len(response.content))
    return out_path


async def iter_ws_json(ws: websockets.WebSocketClientProtocol):
    while True:
        raw = await ws.recv()
        try:
            yield json.loads(raw)
        except Exception:
            logger.warning("[ws] non-json: %r", raw)


def run(coro):
    asyncio.run(coro)


class WebRTCCommandChannel:
    """Gestisce il data channel WebRTC `cmd` per messaggi JSON applicativi."""

    def __init__(self, pc: RTCPeerConnection, label: str = "cmd"):
        self._label = label
        self._channel: RTCDataChannel | None = None
        self._open_event = asyncio.Event()
        self._incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            logger.info(
                "[webrtc] datachannel ricevuto: label=%s id=%s state=%s",
                channel.label,
                getattr(channel, "id", "-"),
                channel.readyState,
            )
            if channel.label != self._label:
                logger.debug(
                    "[webrtc] datachannel ignorato: atteso=%s ricevuto=%s",
                    self._label,
                    channel.label,
                )
                return
            self._attach(channel)

    def _attach(self, channel: RTCDataChannel) -> None:
        self._channel = channel

        @channel.on("open")
        def _on_open() -> None:
            logger.info("[webrtc] data channel aperto: %s", channel.label)
            self._open_event.set()

        @channel.on("close")
        def _on_close() -> None:
            logger.info("[webrtc] data channel chiuso: %s", channel.label)
            self._open_event.clear()

        @channel.on("error")
        def _on_error(error: Exception) -> None:
            logger.warning("[webrtc] errore data channel %s: %s", channel.label, error)

        @channel.on("message")
        def _on_message(message: str | bytes) -> None:
            payload: Any
            if isinstance(message, bytes):
                try:
                    payload = json.loads(message.decode("utf-8"))
                except Exception:
                    return
            else:
                try:
                    payload = json.loads(message)
                except Exception:
                    return

            if isinstance(payload, dict):
                self._incoming.put_nowait(payload)
                logger.debug("[webrtc] payload cmd ricevuto: keys=%s", sorted(payload.keys()))

        if channel.readyState == "open":
            self._open_event.set()

    async def send_json(self, payload: dict[str, Any], timeout_s: float = 10.0) -> None:
        try:
            await asyncio.wait_for(self._open_event.wait(), timeout=timeout_s)
        except TimeoutError as exc:
            state = self._channel.readyState if self._channel is not None else "missing"
            logger.warning(
                "[webrtc] timeout apertura data channel (%ss), impossibile inviare type=%s (state=%s)",
                timeout_s,
                payload.get("type"),
                state,
            )
            raise RuntimeError("data channel cmd non aperto") from exc
        if self._channel is None or self._channel.readyState != "open":
            raise RuntimeError("data channel cmd non disponibile")
        logger.debug("[webrtc] invio payload cmd: type=%s", payload.get("type"))
        self._channel.send(json.dumps(payload))

    async def iter_json(self):
        while True:
            yield await self._incoming.get()
