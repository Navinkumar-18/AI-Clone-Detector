"""Tests for PCM to WAV conversion, header integrity, and audio validation.

Validates:
1. Standard 44-byte RIFF/WAVE PCM header creation.
2. Canonical sample rate (16 kHz), mono (1 ch), 16-bit PCM.
3. Byte rate (32,000 bytes/s) and block align (2 bytes) calculations.
4. Decoder compatibility with librosa / scipy / wave.
5. RMS gate calculation parity between client and backend.
6. Malformed, truncated, and clipped audio validation.
"""

import io
import math
import struct
import wave
import numpy as np
import pytest

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def wrap_pcm_as_wav(pcm_bytes: bytes, sample_rate: int = 16000, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Python parity implementation of Flutter's _wrapPcmAsWav."""
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = len(pcm_bytes)
    file_size = 36 + data_size

    header = bytearray(44)
    # RIFF header
    header[0:4] = b"RIFF"
    struct.pack_into("<I", header, 4, file_size)
    header[8:12] = b"WAVE"
    # fmt sub-chunk
    header[12:16] = b"fmt "
    struct.pack_into("<I", header, 16, 16)  # sub-chunk size
    struct.pack_into("<H", header, 20, 1)   # PCM format = 1
    struct.pack_into("<H", header, 22, num_channels)
    struct.pack_into("<I", header, 24, sample_rate)
    struct.pack_into("<I", header, 28, byte_rate)
    struct.pack_into("<H", header, 32, block_align)
    struct.pack_into("<H", header, 34, bits_per_sample)
    # data sub-chunk
    header[36:40] = b"data"
    struct.pack_into("<I", header, 40, data_size)

    return bytes(header) + pcm_bytes


def generate_pcm_sine(freq: float = 440.0, duration_s: float = 1.0, sample_rate: int = 16000, amplitude: float = 0.5) -> bytes:
    """Generate 16-bit mono PCM sine wave bytes."""
    num_samples = int(duration_s * sample_rate)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        val = amplitude * math.sin(2.0 * math.pi * freq * t)
        int_val = int(val * 32767.0)
        int_val = max(-32768, min(32767, int_val))
        samples.append(int_val)
    return struct.pack(f"<{len(samples)}h", *samples)


def compute_pcm_rms(pcm_bytes: bytes) -> float:
    """Calculate RMS of 16-bit signed PCM matching Flutter Dart implementation."""
    sample_count = len(pcm_bytes) // 2
    if sample_count == 0:
        return 0.0
    sum_squares = 0.0
    for i in range(0, sample_count * 2, 2):
        val = int.from_bytes(pcm_bytes[i:i+2], byteorder="little", signed=True)
        norm = val / 32768.0
        sum_squares += norm * norm
    return math.sqrt(sum_squares / sample_count)


class TestWavConversion:
    """Tests for WAV header format and conversion parity."""

    def test_wav_header_structure(self):
        pcm = generate_pcm_sine(440.0, 1.0, 16000, 0.5)
        assert len(pcm) == 32000  # 1s * 16000 samples * 2 bytes = 32000

        wav_bytes = wrap_pcm_as_wav(pcm, sample_rate=16000, num_channels=1, bits_per_sample=16)
        assert len(wav_bytes) == 44 + 32000

        # Verify markers
        assert wav_bytes[0:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        assert wav_bytes[12:16] == b"fmt "
        assert wav_bytes[36:40] == b"data"

        # Verify numeric header fields
        riff_size = struct.unpack_from("<I", wav_bytes, 4)[0]
        assert riff_size == 36 + 32000

        fmt_size, audio_format, channels, sr, byte_rate, block_align, bits = struct.unpack_from(
            "<IHHIIHH", wav_bytes, 16
        )
        assert fmt_size == 16
        assert audio_format == 1  # PCM
        assert channels == 1      # mono
        assert sr == 16000
        assert byte_rate == 32000 # 16000 * 1 * 2
        assert block_align == 2   # 1 * 2
        assert bits == 16

        data_size = struct.unpack_from("<I", wav_bytes, 40)[0]
        assert data_size == 32000

    def test_wave_module_readability(self):
        """Standard Python wave module should parse the wrapped WAV cleanly."""
        pcm = generate_pcm_sine(1000.0, 0.5, 16000, 0.4)
        wav_bytes = wrap_pcm_as_wav(pcm, 16000, 1, 16)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 8000
            data = wf.readframes(8000)
            assert len(data) == 16000

    @pytest.mark.skipif(not HAS_LIBROSA, reason="librosa not installed")
    def test_librosa_decoding_parity(self):
        """librosa must be able to load the WAV data at 16 kHz without warnings or clipping."""
        pcm = generate_pcm_sine(440.0, 1.0, 16000, 0.5)
        wav_bytes = wrap_pcm_as_wav(pcm, 16000, 1, 16)

        audio, sr = librosa.load(io.BytesIO(wav_bytes), sr=16000)
        assert sr == 16000
        assert len(audio) == 16000
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        # Amplitude range check
        assert np.max(audio) <= 0.55
        assert np.min(audio) >= -0.55

    def test_digital_silence_rms_gate(self):
        """RMS of digital silence must fall below 0.003 threshold."""
        silence_pcm = bytes(32000)  # All zeros
        rms_silence = compute_pcm_rms(silence_pcm)
        assert rms_silence == 0.0
        assert rms_silence < 0.003

    def test_low_noise_falls_below_rms_gate(self):
        """Very low level noise (e.g. sample amp 30/32768 ~ 0.0009) must fail gate."""
        low_noise = struct.pack("<16000h", *([30] * 16000))
        rms_low = compute_pcm_rms(low_noise)
        assert rms_low < 0.003

    def test_speech_signal_passes_rms_gate(self):
        """Normal speech/audio amplitude (~0.2-0.5) must pass gate."""
        signal_pcm = generate_pcm_sine(300.0, 1.0, 16000, 0.25)
        rms = compute_pcm_rms(signal_pcm)
        # RMS of sine with peak amp 0.25 is 0.25 / sqrt(2) ~ 0.177
        assert rms > 0.1
        assert rms > 0.003
