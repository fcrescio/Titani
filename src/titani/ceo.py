import asyncio
import json
import os
from pathlib import Path
import time
import wave
from dataclasses import dataclass
from tempfile import NamedTemporaryFile

import numpy as np
import webrtcvad
import websockets
from aiortc import MediaStreamTrack, RTCPeerConnection
from aiortc.mediastreams import AudioFrame
from mlx_audio.stt.utils import load_model as load_stt
from mlx_audio.vad.utils import load_model as load_vad

from titani.common import ErmeteConfig, iter_ws_json, maybe_handle_offer, run

TARGET_SAMPLE_RATE = 16_000
MAX_CONTEXT_SECONDS = 8
MAX_CONTEXT_SAMPLES = TARGET_SAMPLE_RATE * MAX_CONTEXT_SECONDS
WEBRTC_CHUNK_MS = 30
WEBRTC_CHUNK_SAMPLES = TARGET_SAMPLE_RATE * WEBRTC_CHUNK_MS // 1000


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class CeoConfig(ErmeteConfig):
    silence_ms_before_endpoint: int = int(os.getenv("CEO_SILENCE_MS_BEFORE_ENDPOINT", "300"))
    smart_turn_threshold: float = float(os.getenv("CEO_SMART_TURN_THRESHOLD", "0.5"))
    asr_model: str = os.getenv("CEO_ASR_MODEL", "mlx-community/Qwen3-ASR-0.6B-8bit")
    asr_language: str = os.getenv("CEO_ASR_LANGUAGE", "Italian")
    debug_mode: bool = _env_bool("CEO_DEBUG_MODE", False)
    debug_out_dir: str = os.getenv("CEO_DEBUG_OUT_DIR", "./ceo_debug")
    debug_heartbeat_ms: int = int(os.getenv("CEO_DEBUG_HEARTBEAT_MS", "2000"))


class CeoDebug:
    def __init__(self, cfg: CeoConfig):
        self._enabled = cfg.debug_mode
        self._out_dir = Path(cfg.debug_out_dir)
        self._heartbeat_ms = max(250, cfg.debug_heartbeat_ms)
        self._last_heartbeat_ms = 0.0
        self._seen_sample_rates: set[int] = set()
        self._frame_count = 0
        self._received_samples_16k = 0
        self._saved_segments = 0
        if self._enabled:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[ceo][debug] modalità debug attiva, directory output: {self._out_dir.resolve()}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def observe_frame(self, input_sample_rate: int, frame_16k: np.ndarray) -> None:
        if not self._enabled:
            return

        if input_sample_rate not in self._seen_sample_rates:
            self._seen_sample_rates.add(input_sample_rate)
            print(
                "[ceo][debug] nuovo sample rate in ingresso "
                f"{input_sample_rate}Hz (target pipeline: {TARGET_SAMPLE_RATE}Hz)"
            )

        self._frame_count += 1
        self._received_samples_16k += int(frame_16k.size)
        now_ms = time.monotonic() * 1000.0
        if now_ms - self._last_heartbeat_ms < self._heartbeat_ms:
            return

        self._last_heartbeat_ms = now_ms
        if frame_16k.size:
            rms = float(np.sqrt(np.mean(np.square(frame_16k))))
            peak = float(np.max(np.abs(frame_16k)))
        else:
            rms = 0.0
            peak = 0.0

        buffered_seconds = self._received_samples_16k / TARGET_SAMPLE_RATE
        print(
            "[ceo][debug] heartbeat audio: "
            f"frame={self._frame_count} "
            f"buffered_16k={buffered_seconds:.2f}s "
            f"rms={rms:.6f} peak={peak:.6f}"
        )

    def save_segment_for_asr(self, audio_16k: np.ndarray) -> Path | None:
        if not self._enabled:
            return None

        self._saved_segments += 1
        segment_name = f"segment_{self._saved_segments:04d}_{int(time.time() * 1000)}.wav"
        out_path = self._out_dir / segment_name

        clipped = np.clip(audio_16k, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())

        duration = (audio_16k.size / TARGET_SAMPLE_RATE) if audio_16k.size else 0.0
        print(f"[ceo][debug] salvato segmento ASR: {out_path} ({duration:.2f}s)")
        return out_path


class SmartTurnPipeline:
    """Smart Turn v3 endpoint detection with 8s rolling context."""

    def __init__(self, cfg: CeoConfig, debug: CeoDebug | None = None):
        self._cfg = cfg
        self._debug = debug
        self._vad = webrtcvad.Vad(2)
        self._model = load_vad("mlx-community/smart-turn-v3", strict=True)
        self._audio_context = np.zeros(0, dtype=np.float32)
        self._turn_audio_chunks: list[np.ndarray] = []
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

    def _is_speech(self, frame_16k: np.ndarray) -> bool:
        if frame_16k.size < WEBRTC_CHUNK_SAMPLES:
            return False

        clipped = np.clip(frame_16k, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)

        usable = (pcm16.size // WEBRTC_CHUNK_SAMPLES) * WEBRTC_CHUNK_SAMPLES
        if usable == 0:
            return False

        for start in range(0, usable, WEBRTC_CHUNK_SAMPLES):
            chunk = pcm16[start : start + WEBRTC_CHUNK_SAMPLES]
            if self._vad.is_speech(chunk.tobytes(), TARGET_SAMPLE_RATE):
                return True
        return False

    def process(self, frame: AudioFrame) -> np.ndarray | None:
        frame_16k = self._resample_to_16k(self._frame_to_mono(frame), frame.sample_rate)
        if self._debug is not None:
            self._debug.observe_frame(frame.sample_rate, frame_16k)
        if frame_16k.size:
            self._audio_context = np.concatenate((self._audio_context, frame_16k))[-MAX_CONTEXT_SAMPLES:]

        now_ms = time.monotonic() * 1000.0
        speaking = self._is_speech(frame_16k)

        if speaking:
            self._in_user_turn = True
            self._last_speech_ms = now_ms
            self._checked_during_current_silence = False
        if self._in_user_turn and frame_16k.size:
            self._turn_audio_chunks.append(frame_16k)

        if not speaking and not self._in_user_turn:
            return None

        if not speaking and self._in_user_turn:
            silence_ms = now_ms - self._last_speech_ms
            if silence_ms < self._cfg.silence_ms_before_endpoint:
                return None

            if self._checked_during_current_silence:
                return None

            self._checked_during_current_silence = True
            result = self._model.predict_endpoint(
                self._audio_context,
                sample_rate=TARGET_SAMPLE_RATE,
                threshold=self._cfg.smart_turn_threshold,
            )

            prediction = int(getattr(result, "prediction", 0))
            probability = float(getattr(result, "probability", 0.0))
            print(f"[ceo] smart-turn prediction={prediction} probability={probability:.3f}")

            if prediction == 1:
                self._in_user_turn = False
                self._audio_context = np.zeros(0, dtype=np.float32)
                completed_turn = (
                    np.concatenate(self._turn_audio_chunks)
                    if self._turn_audio_chunks
                    else np.zeros(0, dtype=np.float32)
                )
                self._turn_audio_chunks = []
                return completed_turn

        return None


class AsrPipeline:
    def __init__(self, cfg: CeoConfig):
        self._cfg = cfg
        self._model = load_stt(cfg.asr_model)
        print(f"[ceo] ASR attivo ({cfg.asr_model}, language={cfg.asr_language})")

    def transcribe(self, audio_16k: np.ndarray) -> str:
        if audio_16k.size == 0:
            return ""

        clipped = np.clip(audio_16k, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        with NamedTemporaryFile(suffix=".wav") as wav_file:
            with wave.open(wav_file.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(TARGET_SAMPLE_RATE)
                wf.writeframes(pcm16.tobytes())
            result = self._model.generate(wav_file.name, language=self._cfg.asr_language)
        return str(getattr(result, "text", "")).strip()


async def ceo_consumer(cfg: CeoConfig) -> None:
    pc = RTCPeerConnection()
    debug = CeoDebug(cfg)
    turn_pipeline = SmartTurnPipeline(cfg, debug=debug)
    asr_pipeline = AsrPipeline(cfg)
    ws_send_lock = asyncio.Lock()

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        if track.kind != "audio":
            return

        async def pump() -> None:
            while True:
                frame = await track.recv()
                completed_audio = turn_pipeline.process(frame)
                if completed_audio is not None:
                    debug.save_segment_for_asr(completed_audio)
                    transcript = await asyncio.to_thread(asr_pipeline.transcribe, completed_audio)
                    message = {
                        "type": "speaker_turn_completed",
                        "producer": "ceo",
                        "ts": frame.time,
                        "transcript": transcript,
                        "asr_model": cfg.asr_model,
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
