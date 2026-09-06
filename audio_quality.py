"""
audio_quality.py — Audio quality analysis for VoiceGuard.
=========================================================

Runs BEFORE model inference to determine whether the audio is suitable
for classification. If quality is insufficient, the backend returns
INSUFFICIENT_EVIDENCE instead of running the classifier.

Quality checks:
    1. Duration (minimum / maximum)
    2. RMS energy (silence detection)
    3. Clipping ratio
    4. Voiced-frame ratio (via zero-crossing rate heuristic)
    5. Basic SNR estimate
    6. Decode status

Returns a structured AudioQualityReport with status and reason codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class AudioQualityReport:
    """Structured audio quality assessment."""
    status: str  # "acceptable" | "poor" | "silent" | "invalid" | "unknown"
    duration_seconds: float = 0.0
    rms: float = 0.0
    snr_db: Optional[float] = None
    clipping_ratio: float = 0.0
    voiced_ratio: float = 0.0
    reason_codes: List[str] = field(default_factory=list)

    def is_usable(self) -> bool:
        """Whether the audio is suitable for classifier inference."""
        return self.status == "acceptable"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 3),
            "rms": round(self.rms, 6),
            "snr_db": round(self.snr_db, 1) if self.snr_db is not None else None,
            "clipping_ratio": round(self.clipping_ratio, 4),
            "voiced_ratio": round(self.voiced_ratio, 4),
            "reason_codes": list(self.reason_codes),
        }


def analyze_audio_quality(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_duration: float = 0.5,
    max_duration: float = 60.0,
    rms_silence_threshold: float = 0.003,
    min_voiced_ratio: float = 0.10,
    max_clipping_ratio: float = 0.05,
    poor_snr_db: float = 5.0,
) -> AudioQualityReport:
    """
    Analyze audio quality and return a structured report.

    Parameters
    ----------
    audio : np.ndarray
        1-D float array of audio samples (expected range [-1.0, 1.0]).
    sample_rate : int
        Sample rate in Hz.
    min_duration : float
        Minimum acceptable duration in seconds.
    max_duration : float
        Maximum acceptable duration in seconds.
    rms_silence_threshold : float
        RMS below this value is considered silence.
    min_voiced_ratio : float
        Minimum ratio of voiced frames for acceptable audio.
    max_clipping_ratio : float
        Maximum ratio of clipped samples before quality is poor.
    poor_snr_db : float
        SNR below this threshold (dB) is considered poor.

    Returns
    -------
    AudioQualityReport
        Structured quality assessment with status and reason codes.
    """
    reason_codes: List[str] = []

    # --- Handle None / empty audio ---
    if audio is None or len(audio) == 0:
        return AudioQualityReport(
            status="invalid",
            reason_codes=["empty_audio"],
        )

    # --- Duration ---
    duration = len(audio) / sample_rate

    if duration < min_duration:
        reason_codes.append("audio_too_short")

    if duration > max_duration:
        reason_codes.append("audio_too_long")

    # --- RMS energy ---
    rms = float(np.sqrt(np.mean(audio ** 2)))

    if rms < rms_silence_threshold:
        return AudioQualityReport(
            status="silent",
            duration_seconds=duration,
            rms=rms,
            reason_codes=["no_speech"],
        )

    # --- Clipping ratio ---
    clip_threshold = 0.99
    n_clipped = int(np.sum(np.abs(audio) >= clip_threshold))
    clipping_ratio = n_clipped / len(audio) if len(audio) > 0 else 0.0

    if clipping_ratio > max_clipping_ratio:
        reason_codes.append("clipping_detected")

    # --- Voiced ratio (zero-crossing rate heuristic) ---
    # Speech typically has lower ZCR than noise. We use a frame-based
    # approach: frames with ZCR below a threshold are considered "voiced".
    voiced_ratio = _estimate_voiced_ratio(audio, sample_rate)

    if voiced_ratio < min_voiced_ratio and "audio_too_short" not in reason_codes:
        reason_codes.append("low_voiced_ratio")

    # --- SNR estimate ---
    snr_db = _estimate_snr(audio, sample_rate)

    if snr_db is not None and snr_db < poor_snr_db:
        reason_codes.append("excessive_noise")

    # --- Determine overall status ---
    if "audio_too_short" in reason_codes:
        status = "poor"
    elif "clipping_detected" in reason_codes and clipping_ratio > max_clipping_ratio * 2:
        status = "poor"
    elif len(reason_codes) >= 2:
        status = "poor"
    elif len(reason_codes) == 1:
        # Single minor issue — still usable but noted
        status = "acceptable"
    else:
        status = "acceptable"

    return AudioQualityReport(
        status=status,
        duration_seconds=duration,
        rms=rms,
        snr_db=snr_db,
        clipping_ratio=clipping_ratio,
        voiced_ratio=voiced_ratio,
        reason_codes=reason_codes,
    )


def _estimate_voiced_ratio(audio: np.ndarray, sample_rate: int) -> float:
    """
    Estimate the ratio of voiced frames using zero-crossing rate.

    Speech frames typically have ZCR < 0.1, while noise/silence has
    higher ZCR. This is a rough heuristic, not a VAD.
    """
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)     # 10ms hop

    if len(audio) < frame_length:
        return 0.0

    n_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
    voiced_count = 0

    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_length, len(audio))
        frame = audio[start:end]

        if len(frame) < 2:
            continue

        # Zero-crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / (2 * len(frame))

        # Frame energy
        frame_rms = float(np.sqrt(np.mean(frame ** 2)))

        # A frame is "voiced" if it has moderate energy and low-ish ZCR
        if frame_rms > 0.01 and zcr < 0.15:
            voiced_count += 1

    return voiced_count / n_frames if n_frames > 0 else 0.0


def _estimate_snr(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """
    Estimate signal-to-noise ratio using a simple energy-based approach.

    Splits audio into frames, uses the top-quartile frames as "signal"
    and bottom-quartile as "noise". Returns SNR in dB.

    Returns None if estimation is not possible.
    """
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)

    if len(audio) < frame_length * 4:
        return None

    n_frames = (len(audio) - frame_length) // hop_length + 1
    if n_frames < 4:
        return None

    frame_energies = []
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_length, len(audio))
        frame = audio[start:end]
        energy = float(np.mean(frame ** 2))
        frame_energies.append(energy)

    frame_energies = sorted(frame_energies)

    # Bottom 25% as noise, top 25% as signal
    n_quarter = max(1, len(frame_energies) // 4)
    noise_energy = np.mean(frame_energies[:n_quarter])
    signal_energy = np.mean(frame_energies[-n_quarter:])

    if noise_energy <= 0 or signal_energy <= 0:
        return None

    snr = 10.0 * np.log10(signal_energy / noise_energy)
    return float(snr)
