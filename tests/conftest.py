"""
conftest.py — Shared test fixtures for VoiceGuard test suite.
=============================================================

Provides deterministic audio fixtures for:
    - Digital silence
    - Empty audio
    - Very short audio
    - Normal speech-like audio (sine + noise)
    - High-energy audio (spoof-like)
    - Clipped audio
    - Background noise only
    - Corrupted audio bytes
"""

from __future__ import annotations

import io
import os
import struct
import sys

import numpy as np
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voiceguard_config import reset_config_cache


SAMPLE_RATE = 16000


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset config cache between tests."""
    reset_config_cache()
    yield
    reset_config_cache()


def make_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Wrap a float32 audio array as a 16-bit PCM WAV file in memory."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = len(pcm) * (bits_per_sample // 8)

    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Audio fixtures (deterministic, reproducible)
# ---------------------------------------------------------------------------

@pytest.fixture
def silence_audio() -> np.ndarray:
    """2 seconds of digital silence."""
    return np.zeros(SAMPLE_RATE * 2, dtype=np.float32)


@pytest.fixture
def silence_wav(silence_audio) -> bytes:
    """2 seconds of digital silence as WAV bytes."""
    return make_wav_bytes(silence_audio)


@pytest.fixture
def empty_audio() -> np.ndarray:
    """Empty audio array (0 samples)."""
    return np.array([], dtype=np.float32)


@pytest.fixture
def short_audio() -> np.ndarray:
    """0.1 seconds of audio — too short for analysis."""
    t = np.linspace(0, 0.1, int(SAMPLE_RATE * 0.1), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def short_wav(short_audio) -> bytes:
    """0.1 seconds of audio as WAV bytes."""
    return make_wav_bytes(short_audio)


@pytest.fixture
def normal_audio() -> np.ndarray:
    """3 seconds of speech-like audio (440 Hz sine + noise)."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, 3.0, SAMPLE_RATE * 3, endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.randn(len(t))
    return signal.astype(np.float32)


@pytest.fixture
def normal_wav(normal_audio) -> bytes:
    """3 seconds of speech-like audio as WAV bytes."""
    return make_wav_bytes(normal_audio)


@pytest.fixture
def live_normal_audio() -> np.ndarray:
    """4 seconds of speech-like audio for live-analysis tests.

    Uses exactly 4000 ms (FULL_WINDOW_MS) to satisfy the live-analysis
    complete-window contract.  Upload tests should continue to use normal_audio
    (3 seconds) since /predict has no minimum-duration guard beyond the
    quality check (0.5 s).
    """
    rng = np.random.RandomState(42)
    t = np.linspace(0, 4.0, SAMPLE_RATE * 4, endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.randn(len(t))
    return signal.astype(np.float32)


@pytest.fixture
def live_normal_wav(live_normal_audio) -> bytes:
    """4 seconds of speech-like audio as WAV bytes (for /live/analyze tests)."""
    return make_wav_bytes(live_normal_audio)


@pytest.fixture
def high_energy_audio() -> np.ndarray:
    """3 seconds of high-energy audio."""
    rng = np.random.RandomState(123)
    t = np.linspace(0, 3.0, SAMPLE_RATE * 3, endpoint=False)
    signal = 0.7 * np.sin(2 * np.pi * 880 * t) + 0.2 * rng.randn(len(t))
    return signal.astype(np.float32)


@pytest.fixture
def clipped_audio() -> np.ndarray:
    """3 seconds of severely clipped audio."""
    rng = np.random.RandomState(77)
    t = np.linspace(0, 3.0, SAMPLE_RATE * 3, endpoint=False)
    signal = 2.0 * np.sin(2 * np.pi * 440 * t) + 0.3 * rng.randn(len(t))
    # Clip to [-1, 1] — many samples will be at the boundary
    return np.clip(signal, -1.0, 1.0).astype(np.float32)


@pytest.fixture
def noise_only_audio() -> np.ndarray:
    """3 seconds of pure noise (no tonal content)."""
    rng = np.random.RandomState(99)
    return (0.05 * rng.randn(SAMPLE_RATE * 3)).astype(np.float32)


@pytest.fixture
def corrupted_wav_bytes() -> bytes:
    """Invalid/corrupted WAV file bytes."""
    return b"NOT_A_VALID_WAV_FILE_HEADER_" + os.urandom(100)
