import numpy as np


def resample_pcm16(audio_pcm16: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if audio_pcm16.size == 0 or src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return np.ascontiguousarray(audio_pcm16, dtype=np.int16)

    src = audio_pcm16.astype(np.float32)
    src_len = src.size
    dst_len = max(1, int(round(src_len * dst_rate / src_rate)))
    src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    dst = np.interp(dst_x, src_x, src)
    return np.ascontiguousarray(np.clip(dst, -32768, 32767).astype(np.int16))


def resample_float32(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if audio.size == 0 or src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)

    src = np.ascontiguousarray(audio, dtype=np.float32)
    src_len = src.size
    dst_len = max(1, int(round(src_len * dst_rate / src_rate)))
    src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    dst = np.interp(dst_x, src_x, src)
    return np.ascontiguousarray(np.clip(dst, -1.0, 1.0).astype(np.float32))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a_flat = a.reshape(-1).astype(np.float32, copy=False)
    b_flat = b.reshape(-1).astype(np.float32, copy=False)
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    if denom <= 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)
