"""FastAPI app: health check + WebSocket stream of FFT magnitudes (demo signal)."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

SAMPLE_RATE = 44_100
FRAME_SIZE = 1024
BINS_OUT = 256  # send first N rFFT bins (excluding DC for display)

app = FastAPI(title="Voice Analyser DSP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def synth_frame(phase: float) -> np.ndarray:
    """A short window of synthetic audio (sum of sines) for demo FFT."""
    t = np.arange(FRAME_SIZE, dtype=np.float64) / SAMPLE_RATE
    w = 2 * math.pi
    a = (
        0.35 * np.sin(w * 440 * t + phase)
        + 0.25 * np.sin(w * 880 * t + phase * 1.3)
        + 0.15 * np.sin(w * 1320 * t + phase * 0.7)
    )
    hann = np.hanning(FRAME_SIZE)
    return (a * hann).astype(np.float64)


def fft_magnitudes(frame: np.ndarray) -> np.ndarray:
    spec = np.fft.rfft(frame)
    mag = np.abs(spec[1 : 1 + BINS_OUT])
    max_v = float(np.max(mag)) + 1e-12
    return (mag / max_v).astype(np.float32)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_fft(websocket: WebSocket) -> None:
    await websocket.accept()
    phase = 0.0
    try:
        while True:
            frame = synth_frame(phase)
            bins_list = fft_magnitudes(frame).tolist()
            payload: dict[str, Any] = {
                "type": "fft",
                "sampleRate": SAMPLE_RATE,
                "frameSize": FRAME_SIZE,
                "bins": bins_list,
            }
            await websocket.send_text(json.dumps(payload))
            phase += 0.08
            await asyncio.sleep(1 / 30)
    except WebSocketDisconnect:
        return
