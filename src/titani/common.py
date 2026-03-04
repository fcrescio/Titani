import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription


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


async def maybe_handle_offer(ws: websockets.WebSocketClientProtocol, pc: RTCPeerConnection) -> None:
    """If the first WS message is an offer, reply with an answer and return.

    If the first message is not an offer, it is ignored by this helper.
    """

    raw = await ws.recv()
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return

    if msg.get("type") != "offer":
        return

    await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["sdp"], type="offer"))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await ws.send(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))
    print("[webrtc] answer inviato")


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
    print(f"[frames] scaricato -> {out_path} ({len(response.content)} bytes)")
    return out_path


async def iter_ws_json(ws: websockets.WebSocketClientProtocol):
    while True:
        raw = await ws.recv()
        try:
            yield json.loads(raw)
        except Exception:
            print(f"[ws] non-json: {raw!r}")


def run(coro):
    asyncio.run(coro)
