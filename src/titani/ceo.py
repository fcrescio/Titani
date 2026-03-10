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
from titani.say_queue import SayToUserItem, enqueue_say_to_user, handle_say_to_user_retry

logger = logging.getLogger(__name__)

OUTBOUND_OBS_INTERVAL_S = 1.0
OUTBOUND_JITTER_HIGH_S = 0.030
OUTBOUND_RTT_HIGH_S = 0.250
OUTBOUND_JITTER_STABLE_S = 0.012
OUTBOUND_RTT_STABLE_S = 0.120
OUTBOUND_PREBUFFER_MIN_CHUNKS = 1
OUTBOUND_PREBUFFER_MAX_CHUNKS = 8
OUTBOUND_MAX_BUFFER_MIN_MS = 60
OUTBOUND_MAX_BUFFER_MAX_MS = 420
OUTBOUND_PREBUFFER_STEP_CHUNKS = 1
OUTBOUND_MAX_BUFFER_STEP_MS = 20


def _resolve_outbound_adaptation_start(
    *,
    startup_snapshot: dict[str, int],
    cfg: CeoConfig,
) -> tuple[int, int]:
    prebuffer_target = max(
        OUTBOUND_PREBUFFER_MIN_CHUNKS,
        min(
            OUTBOUND_PREBUFFER_MAX_CHUNKS,
            int(cfg.outbound.adaptive_start_prebuffer_chunks or startup_snapshot["prebuffer_chunks"]),
        ),
    )
    max_buffer_target_ms = max(
        OUTBOUND_MAX_BUFFER_MIN_MS,
        min(
            OUTBOUND_MAX_BUFFER_MAX_MS,
            int(cfg.outbound.adaptive_start_max_buffer_ms or startup_snapshot["max_buffer_ms"]),
        ),
    )
    return prebuffer_target, max_buffer_target_ms


async def ceo_consumer(cfg: CeoConfig) -> None:
    pc = RTCPeerConnection()
    cmd_channel = WebRTCCommandChannel(pc)
    debug = CeoDebug(cfg.debug)
    turn_pipeline = SmartTurnPipeline(cfg.ingress, debug=debug)
    asr_pipeline = AsrPipeline(cfg.asr)
    tts_pipeline = TtsPipeline(cfg.outbound)
    speaker_pipeline = SpeakerEmbeddingPipeline(cfg.speaker, tts_pipeline.model)
    if cfg.speaker.require_known_speaker_for_transcript:
        logger.info("[ceo][speaker-guard] modalità riconoscimento speaker nota ATTIVA")
    else:
        logger.info("[ceo][speaker-guard] modalità riconoscimento speaker nota DISATTIVATA")
    outbound_track = TtsOutboundAudioTrack()
    pc.addTrack(outbound_track)
    logger.info("[webrtc] traccia audio outbound TTS aggiunta")
    cmd_send_lock = asyncio.Lock()
    tts_idle = asyncio.Event()
    tts_idle.set()
    overflow_policy = (cfg.outbound.say_to_user_queue_overflow_policy or "drop_oldest").strip().lower()
    say_to_user_queue: asyncio.Queue[SayToUserItem] = asyncio.Queue(maxsize=cfg.outbound.say_to_user_queue_maxsize)
    pump_tasks: set[asyncio.Task[None]] = set()
    housekeeping_tasks: set[asyncio.Task[None]] = set()
    shutdown_lock = asyncio.Lock()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info("[webrtc] connection state -> %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await shutdown_peer_connection()

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        logger.info("[webrtc] ice connection state -> %s", pc.iceConnectionState)

    async def handle_say_to_user_text(item: SayToUserItem) -> bool:
        text = item.text
        if not text:
            return True

        logger.info("[ceo] say_to_user ricevuto -> %r", text)
        ready = await handle_say_to_user_retry(
            outbound_track=outbound_track,
            queue=say_to_user_queue,
            item=item,
            overflow_policy=overflow_policy,
            retry_delay_s=cfg.outbound.say_to_user_retry_delay_s,
        )
        if not ready:
            return False

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
                return True

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
            return True
        finally:
            tts_idle.set()

    async def say_to_user_worker() -> None:
        while True:
            item = await say_to_user_queue.get()
            try:
                await handle_say_to_user_text(item)
            finally:
                say_to_user_queue.task_done()

    say_to_user_worker_task = asyncio.create_task(say_to_user_worker())

    async def adapt_outbound_audio_policy() -> None:
        startup_snapshot = await outbound_track.update_buffer_policy(reason="startup-baseline")
        prebuffer_target, max_buffer_target_ms = _resolve_outbound_adaptation_start(
            startup_snapshot=startup_snapshot,
            cfg=cfg,
        )
        snapshot = await outbound_track.update_buffer_policy(
            prebuffer_chunks=prebuffer_target,
            target_buffer_ms=max_buffer_target_ms,
            reason="startup-baseline",
        )

        logger.info(
            "[ceo][net-adapt] event=startup adaptive_enabled=%s startup_prebuffer_chunks=%s startup_max_buffer_ms=%s initial_prebuffer_chunks=%s initial_max_buffer_ms=%s",
            cfg.outbound.adaptive_policy_enabled,
            startup_snapshot["prebuffer_chunks"],
            startup_snapshot["max_buffer_ms"],
            snapshot["prebuffer_chunks"],
            snapshot["max_buffer_ms"],
        )

        if not cfg.outbound.adaptive_policy_enabled:
            logger.info("[ceo][net-adapt] event=disabled reason=static-config")
            return

        while True:
            await asyncio.sleep(OUTBOUND_OBS_INTERVAL_S)
            try:
                stats_report = await pc.getStats()
                outbound_audio_stats = [
                    stat
                    for stat in stats_report.values()
                    if getattr(stat, "type", "") == "outbound-rtp"
                    and getattr(stat, "kind", None) == "audio"
                    and not bool(getattr(stat, "isRemote", False))
                ]

                if not outbound_audio_stats:
                    logger.debug("[ceo][net-adapt] event=no-audio-outbound-stats")
                    continue

                stat = outbound_audio_stats[0]
                packets_lost = int(getattr(stat, "packetsLost", 0) or 0)
                jitter = float(getattr(stat, "jitter", 0.0) or 0.0)
                round_trip_time = float(getattr(stat, "roundTripTime", 0.0) or 0.0)
                bytes_sent = int(getattr(stat, "bytesSent", 0) or 0)
                bitrate_bps = int(getattr(stat, "bitrateMean", 0) or 0)

                if bitrate_bps <= 0:
                    bitrate_bps = int(max(0.0, bytes_sent * 8.0 / OUTBOUND_OBS_INTERVAL_S))

                high_jitter = jitter >= OUTBOUND_JITTER_HIGH_S
                high_rtt = round_trip_time >= OUTBOUND_RTT_HIGH_S
                stable_link = jitter <= OUTBOUND_JITTER_STABLE_S and round_trip_time <= OUTBOUND_RTT_STABLE_S and packets_lost == 0

                reason = "hold"
                if high_jitter or high_rtt:
                    prebuffer_target = min(
                        OUTBOUND_PREBUFFER_MAX_CHUNKS,
                        prebuffer_target + OUTBOUND_PREBUFFER_STEP_CHUNKS,
                    )
                    max_buffer_target_ms = min(
                        OUTBOUND_MAX_BUFFER_MAX_MS,
                        max_buffer_target_ms + OUTBOUND_MAX_BUFFER_STEP_MS,
                    )
                    reason = "network-degraded"
                elif stable_link:
                    prebuffer_target = max(
                        OUTBOUND_PREBUFFER_MIN_CHUNKS,
                        prebuffer_target - OUTBOUND_PREBUFFER_STEP_CHUNKS,
                    )
                    max_buffer_target_ms = max(
                        OUTBOUND_MAX_BUFFER_MIN_MS,
                        max_buffer_target_ms - OUTBOUND_MAX_BUFFER_STEP_MS,
                    )
                    reason = "network-stable"

                snapshot = await outbound_track.update_buffer_policy(
                    prebuffer_chunks=prebuffer_target,
                    target_buffer_ms=max_buffer_target_ms,
                    reason=reason,
                )

                logger.info(
                    "[ceo][net-adapt] event=stats reason=%s packets_lost=%s jitter_s=%.4f rtt_s=%.4f bitrate_bps=%s prebuffer_chunks=%s max_buffer_ms=%s base_prebuffer_chunks=%s base_max_buffer_ms=%s",
                    reason,
                    packets_lost,
                    jitter,
                    round_trip_time,
                    bitrate_bps,
                    snapshot["prebuffer_chunks"],
                    snapshot["max_buffer_ms"],
                    startup_snapshot["prebuffer_chunks"],
                    startup_snapshot["max_buffer_ms"],
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[ceo][net-adapt] event=failed-to-adapt")

    net_adaptation_task = asyncio.create_task(adapt_outbound_audio_policy())
    housekeeping_tasks.add(net_adaptation_task)

    async def shutdown_peer_connection() -> None:
        async with shutdown_lock:
            tasks = [*pump_tasks, *housekeeping_tasks]
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
                        if cfg.speaker.require_known_speaker_for_transcript and transcript:
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
                            "asr_model": cfg.asr.asr_model,
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
                        queued = enqueue_say_to_user(
                            say_to_user_queue,
                            SayToUserItem(text=text, retries_left=cfg.outbound.say_to_user_max_retries),
                            overflow_policy=overflow_policy,
                        )
                        if not queued:
                            logger.warning("[ceo] say_to_user scartato per overflow queue (policy=%s)", overflow_policy)
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
