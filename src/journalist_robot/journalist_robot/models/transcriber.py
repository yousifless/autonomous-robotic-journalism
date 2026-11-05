"""Whisper speech recognition on buffered microphone audio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TranscriptChunk:
    start_s: float
    end_s: float
    text: str

    def line(self) -> str:
        return f"[{self.start_s:.1f}s - {self.end_s:.1f}s] {self.text}"


class SpeechTranscriber:
    def __init__(self, model_id: str = "openai/whisper-small", device: str = "cuda",
                 local_files_only: bool = False) -> None:
        import torch
        from transformers import pipeline

        dev = 0 if (device.startswith("cuda") and torch.cuda.is_available()) else -1
        self._asr = pipeline("automatic-speech-recognition", model=model_id, device=dev,
                             model_kwargs={"local_files_only": local_files_only})

    def transcribe(self, samples: np.ndarray, sample_rate: int, start_s: float,
                   chunk_s: float = 10.0) -> List[TranscriptChunk]:
        """Transcribe mono float32 audio in fixed-length chunks.

        ``start_s`` is the timestamp of the first sample on the segment
        timeline, so transcript chunks share the time base of the frames.
        """
        chunks: List[TranscriptChunk] = []
        step = max(1, int(chunk_s * sample_rate))
        for i in range(0, len(samples), step):
            piece = samples[i:i + step]
            if len(piece) < sample_rate * 0.5:      # skip fragments shorter than 0.5 s
                continue
            result = self._asr({"raw": piece.astype(np.float32), "sampling_rate": sample_rate})
            text = (result.get("text") or "").strip()
            if text:
                t0 = start_s + i / sample_rate
                chunks.append(TranscriptChunk(t0, t0 + len(piece) / sample_rate, text))
        return chunks


def pcm16_to_float(data: bytes, channels: int = 1) -> np.ndarray:
    """Convert interleaved little-endian 16-bit PCM to mono float32 in [-1, 1]."""
    pcm = np.frombuffer(bytes(data), dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm[: len(pcm) - len(pcm) % channels].reshape(-1, channels).mean(axis=1)
    return pcm
