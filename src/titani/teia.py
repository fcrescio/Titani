import asyncio
import base64
import logging
import os
import string
from dataclasses import dataclass
from pathlib import Path

import httpx
import websockets
from aiortc import RTCPeerConnection

from titani.common import (
    ErmeteConfig,
    WebRTCCommandChannel,
    download_frame,
    iter_ws_json,
    maybe_handle_offer,
    run,
    setup_logging,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TeiaConfig(ErmeteConfig):
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_min_turn_chars: int = int(os.getenv("TEIA_LLM_MIN_TURN_CHARS", "8"))
    llm_history_max_turns: int = int(os.getenv("TEIA_LLM_HISTORY_MAX_TURNS", "20"))


def is_sensible_turn(text: str, min_chars: int) -> bool:
    cleaned = " ".join(text.split())
    if len(cleaned) < min_chars:
        return False

    unpunctuated = cleaned.translate(str.maketrans("", "", string.punctuation)).strip()
    return len(unpunctuated) >= min_chars


async def describe_snapshot(http: httpx.AsyncClient, cfg: TeiaConfig, image_path: Path) -> str:
    if not cfg.llm_api_key:
        return "LLM_API_KEY non impostata: impossibile descrivere lo snapshot."

    try:
        from openai import AsyncOpenAI
    except ImportError:
        return "Dipendenza 'openai' non installata: usa `uv sync --extra teia`."

    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    mime = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"

    client = AsyncOpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
    response = await client.responses.create(
        model=cfg.llm_model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Descrivi brevemente in italiano cosa vedi nello snapshot."},
                {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
            ],
        }],
        timeout=60.0,
    )
    return response.output_text or "Descrizione non disponibile."


def to_response_input(history: list[dict[str, str]]) -> list[dict[str, object]]:
    formatted_history: list[dict[str, object]] = []
    for turn in history:
        role = turn.get("role", "user")
        content = str(turn.get("content", "")).strip()
        if not content:
            continue

        text_type = "input_text" if role == "user" else "output_text"
        formatted_history.append(
            {
                "type": "message",
                "role": role,
                "content": [{"type": text_type, "text": content}],
            }
        )

    return formatted_history


async def generate_reply(client, cfg: TeiaConfig, history: list[dict[str, str]]) -> str:
    response = await client.responses.create(
        model=cfg.llm_model,
        input=to_response_input(history),
        timeout=60.0,
    )
    return (response.output_text or "").strip()


async def teia_consumer(cfg: TeiaConfig) -> None:
    pc = RTCPeerConnection()
    cmd_channel = WebRTCCommandChannel(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info("[webrtc] connection state -> %s", pc.connectionState)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        logger.info("[webrtc] ice connection state -> %s", pc.iceConnectionState)

    @pc.on("track")
    async def on_track(track) -> None:
        logger.info("[webrtc] traccia in ingresso aperta: kind=%s id=%s", track.kind, getattr(track, "id", "-"))

    llm_client = None
    llm_history: list[dict[str, str]] = []
    if cfg.llm_api_key:
        try:
            from openai import AsyncOpenAI

            llm_client = AsyncOpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
        except ImportError:
            logger.warning("[teia] openai non installato: disattivata gestione speaker_turn_completed")
    else:
        logger.warning("[teia] LLM_API_KEY non configurata: disattivata gestione speaker_turn_completed")

    async with httpx.AsyncClient() as http:
        logger.info("[teia] connessione websocket verso backend: %s", cfg.ermete_ws)
        async with websockets.connect(cfg.ermete_ws, additional_headers=cfg.auth_headers()) as ws:
            await maybe_handle_offer(ws, pc)

            async def consume_data_channel() -> None:
                async for data in cmd_channel.iter_json():
                    t = data.get("type")
                    if t == "ping":
                        await cmd_channel.send_json({"type": "pong"})
                    elif t == "speaker_turn_completed":
                        transcript = str(data.get("transcript", "")).strip()
                        if not is_sensible_turn(transcript, cfg.llm_min_turn_chars):
                            logger.info("[teia] speaker_turn_completed ignorato (trascrizione troppo corta): %r", transcript)
                            continue

                        if llm_client is None:
                            logger.warning("[teia] impossibile processare speaker_turn_completed: client LLM non disponibile")
                            continue

                        llm_history.append({"role": "user", "content": transcript})
                        llm_history[:] = llm_history[-(cfg.llm_history_max_turns * 2) :]

                        try:
                            reply = await generate_reply(llm_client, cfg, llm_history)
                        except Exception:
                            logger.exception("[teia] errore chiamata LLM")
                            continue

                        if not reply:
                            logger.warning("[teia] risposta LLM vuota")
                            continue

                        llm_history.append({"role": "assistant", "content": reply})
                        llm_history[:] = llm_history[-(cfg.llm_history_max_turns * 2) :]
                        await cmd_channel.send_json({"type": "say_to_user", "producer": "teia", "text": reply})
                        logger.info("[teia] say_to_user inviato (%s chars)", len(reply))
                    else:
                        logger.info("[dc] msg: %s", data)

            async def consume_websocket() -> None:
                async for data in iter_ws_json(ws):
                    if data.get("type") != "frame_available":
                        logger.info("[ws] msg: %s", data)
                        continue

                    out_path = await download_frame(http, cfg, data)
                    description = await describe_snapshot(http, cfg, out_path)
                    message = {
                        "type": "snapshot_description",
                        "producer": "teia",
                        "frame_id": data.get("frame_id"),
                        "file_name": data.get("file_name"),
                        "description": description,
                    }
                    await cmd_channel.send_json(message)
                    logger.info("[teia] snapshot_description inviato: frame_id=%s", data.get("frame_id"))

            await asyncio.gather(consume_data_channel(), consume_websocket())


def main() -> None:
    setup_logging()
    cfg = TeiaConfig()
    logger.info("[teia] ERMETE_WS=%s", cfg.ermete_ws)
    run(teia_consumer(cfg))


if __name__ == "__main__":
    main()
