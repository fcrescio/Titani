import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import websockets
from aiortc import RTCPeerConnection

from titani.common import ErmeteConfig, download_frame, iter_ws_json, maybe_handle_offer, run


@dataclass(slots=True)
class TeiaConfig(ErmeteConfig):
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")


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


async def teia_consumer(cfg: TeiaConfig) -> None:
    pc = RTCPeerConnection()

    async with httpx.AsyncClient() as http:
        async with websockets.connect(cfg.ermete_ws, additional_headers=cfg.auth_headers()) as ws:
            await maybe_handle_offer(ws, pc)
            async for data in iter_ws_json(ws):
                t = data.get("type")
                if t == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                elif t == "frame_available":
                    out_path = await download_frame(http, cfg, data)
                    description = await describe_snapshot(http, cfg, out_path)
                    message = {
                        "type": "snapshot_description",
                        "producer": "teia",
                        "frame_id": data.get("frame_id"),
                        "file_name": data.get("file_name"),
                        "description": description,
                    }
                    await ws.send(json.dumps(message))
                    print(f"[teia] {message}")
                else:
                    print(f"[ws] msg: {data}")


def main() -> None:
    cfg = TeiaConfig()
    print(f"[teia] ERMETE_WS={cfg.ermete_ws}")
    run(teia_consumer(cfg))


if __name__ == "__main__":
    main()
