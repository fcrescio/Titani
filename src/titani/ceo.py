import asyncio
import logging
from contextlib import suppress

import numpy as np
import websockets
from aiortc import MediaStreamTrack, RTCPeerConnection
from aiortc.mediastreams import MediaStreamError

from titani.ceo_components import (
    DEFAULT_WEBRTC_SAMPLE_RATE,
    AsrPipeline,
    CeoConfig,
    CeoDebug,
    SmartTurnPipeline,
    SpeakerEmbeddingPipeline,
    TtsOutboundAudioTrack,
    TtsPipeline,
)
from titani.common import WebRTCCommandChannel, maybe_handle_offer, run, setup_logging

logger = logging.getLogger(__name__)


async def ceo_consumer(cfg: CeoConfig) -> None:
    pc = RTCPeerConnection()
    cmd_channel = WebRTCCommandChannel(pc)
    debug = CeoDebug(cfg)
    turn_pipeline = SmartTurnPipeline(cfg, debug=debug)
    asr_pipeline = AsrPipeline(cfg)
    tts_pipeline = TtsPipeline(cfg)
    speaker_pipeline = SpeakerEmbeddingPipeline(cfg, tts_pipeline.model)
    if cfg.require_known_speaker_for_transcript:
        logger.info("[ceo][speaker-guard] modalità riconoscimento speaker nota ATTIVA")
    else:
        logger.info("[ceo][speaker-guard] modalità riconoscimento speaker nota DISATTIVATA")
    outbound_track = TtsOutboundAudioTrack()
    pc.addTrack(outbound_track)
    logger.info("[webrtc] traccia audio outbound TTS aggiunta")
    cmd_send_lock = asyncio.Lock()
    tts_idle = asyncio.Event()
    tts_idle.set()
    say_to_user_queue: asyncio.Queue[str] = asyncio.Queue()
    pump_tasks: set[asyncio.Task[None]] = set()
    shutdown_lock = asyncio.Lock()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info("[webrtc] connection state -> %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await shutdown_peer_connection()

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        logger.info("[webrtc] ice connection state -> %s", pc.iceConnectionState)

    async def handle_say_to_user_text(text: str) -> None:
        if not text:
            return

        logger.info("[ceo] say_to_user ricevuto -> %r", text)
        ok = await outbound_track.wait_consumer_started(timeout=2.0)
        if not ok:
            logger.warning("TTS consumer not started yet; dropping or delaying say_to_user")
            return

        tts_idle.clear()
        try:
            stream_iter = iter(tts_pipeline.stream_voice_clone_pcm16(text))
            streamed_samples = 0
            tts_sample_rate: int | None = None
            accumulated_chunks: list[np.ndarray] = []

            while True:
                chunk = await asyncio.to_thread(lambda: next(stream_iter, None))
                if chunk is None:
                    break
                pcm16_chunk, chunk_sample_rate = chunk
                if pcm16_chunk.size == 0:
                    continue
                if tts_sample_rate is None:
                    tts_sample_rate = chunk_sample_rate
                streamed_samples += pcm16_chunk.size
                accumulated_chunks.append(pcm16_chunk)
                await outbound_track.push_pcm16(pcm16_chunk, sample_rate=chunk_sample_rate)

            if streamed_samples == 0 or tts_sample_rate is None:
                logger.warning("[ceo] TTS non ha generato audio")
                return

            if debug.enabled:
                full_audio = np.concatenate(accumulated_chunks)
                debug.save_tts_wav(full_audio, sample_rate=tts_sample_rate)

            await outbound_track.wait_until_idle()
            logger.info(
                "[ceo] TTS inviato su WebRTC (%.2fs, src_sr=%sHz, dst_sr=%sHz)",
                streamed_samples / max(1, tts_sample_rate),
                tts_sample_rate,
                outbound_track.output_sample_rate,
            )
        finally:
            tts_idle.set()

    async def say_to_user_worker() -> None:
        while True:
            text = await say_to_user_queue.get()
            try:
                await handle_say_to_user_text(text)
            finally:
                say_to_user_queue.task_done()

    say_to_user_worker_task = asyncio.create_task(say_to_user_worker())

    async def shutdown_peer_connection() -> None:
        async with shutdown_lock:
            tasks = list(pump_tasks)
            for task in tasks:
                task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task

            say_to_user_worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await say_to_user_worker_task

            if pc.connectionState != "closed":
                await pc.close()

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        logger.info("[webrtc] traccia in ingresso aperta: kind=%s id=%s", track.kind, getattr(track, "id", "-"))
        if track.kind != "audio":
            return

        async def pump() -> None:
            try:
                while True:
                    frame = await track.recv()
                    outbound_track.set_output_sample_rate(int(frame.sample_rate or DEFAULT_WEBRTC_SAMPLE_RATE))
                    if not tts_idle.is_set():
                        turn_pipeline.reset_turn_state()
                        continue
                    completed_audio = turn_pipeline.process(frame)
                    if completed_audio is not None:
                        debug.save_segment_for_asr(completed_audio)
                        transcript = await asyncio.to_thread(asr_pipeline.transcribe, completed_audio)
                        if cfg.require_known_speaker_for_transcript and transcript:
                            recognized, speaker_id, probability = await asyncio.to_thread(
                                speaker_pipeline.recognize_known_speaker,
                                completed_audio,
                            )
                            if not recognized:
                                logger.info(
                                    "[ceo][speaker-guard] trascrizione scartata: speaker sconosciuto (best_id=%s prob=%.4f)",
                                    speaker_id,
                                    probability,
                                )
                                continue
                            logger.info(
                                "[ceo][speaker-guard] trascrizione autorizzata per speaker noto id=%s (prob=%.4f)",
                                speaker_id,
                                probability,
                            )
                        if transcript:
                            await asyncio.to_thread(speaker_pipeline.process_transcribed_segment, completed_audio)
                        message = {
                            "type": "speaker_turn_completed",
                            "producer": "ceo",
                            "ts": frame.time,
                            "transcript": transcript,
                            "asr_model": cfg.asr_model,
                        }
                        try:
                            async with cmd_send_lock:
                                await cmd_channel.send_json(message)
                        except Exception:
                            logger.exception("[ceo] invio speaker_turn_completed fallito")
                            continue
                        logger.info("[ceo] turn-end -> %s", message)
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

    logger.info("[ceo] connessione websocket verso backend: %s", cfg.ermete_ws)
    async with websockets.connect(cfg.ermete_ws, additional_headers=cfg.auth_headers()) as ws:
        await maybe_handle_offer(ws, pc)

        async def consume_cmd_messages() -> None:
            async for data in cmd_channel.iter_json():
                t = data.get("type")
                if t == "ping":
                    async with cmd_send_lock:
                        await cmd_channel.send_json({"type": "pong"})
                elif t == "say_to_user":
                    producer = str(data.get("producer", "")).strip() or "unknown"
                    if producer != "teia":
                        logger.warning(
                            "[ceo] say_to_user ricevuto da producer inatteso (%s), continuo comunque",
                            producer,
                        )
                    text = str(data.get("text", "")).strip()
                    if text:
                        await say_to_user_queue.put(text)
                else:
                    logger.info("[dc] msg: %s", data)

        try:
            await consume_cmd_messages()
        finally:
            await shutdown_peer_connection()


def main() -> None:
    setup_logging()
    cfg = CeoConfig()
    logger.info("[ceo] ERMETE_WS=%s", cfg.ermete_ws)
    run(ceo_consumer(cfg))
