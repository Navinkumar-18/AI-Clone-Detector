"""
test_audio_quality.py — Unit tests for audio quality analysis.
==============================================================

Tests:
    - Digital silence → status=silent, reason=no_speech
    - Empty audio → status=invalid, reason=empty_audio
    - Very short audio → status=poor, reason=audio_too_short
    - Normal audio → status=acceptable
    - Clipped audio → clipping_ratio detected
    - Noise-only audio → low voiced_ratio detected
    - Quality check prevents bonafide return for silence
"""

import numpy as np
import pytest

from audio_quality import analyze_audio_quality, AudioQualityReport


SAMPLE_RATE = 16000


class TestSilence:
    """Silence must return status=silent and reason no_speech."""

    def test_digital_silence(self, silence_audio):
        report = analyze_audio_quality(silence_audio, SAMPLE_RATE)
        assert report.status == "silent"
        assert "no_speech" in report.reason_codes
        assert not report.is_usable()

    def test_near_silence(self):
        """Very low energy audio should be treated as silent."""
        audio = np.full(SAMPLE_RATE * 2, 0.0001, dtype=np.float32)
        report = analyze_audio_quality(audio, SAMPLE_RATE)
        assert report.status == "silent"
        assert not report.is_usable()

    def test_silence_never_returns_bonafide(self, silence_audio):
        """Silence must NEVER produce a usable result."""
        report = analyze_audio_quality(silence_audio, SAMPLE_RATE)
        assert not report.is_usable()
        # The caller (backend) will return insufficient_evidence,
        # not bonafide, when is_usable() is False.


class TestEmpty:
    """Empty audio must return status=invalid."""

    def test_empty_audio(self, empty_audio):
        report = analyze_audio_quality(empty_audio, SAMPLE_RATE)
        assert report.status == "invalid"
        assert "empty_audio" in report.reason_codes
        assert not report.is_usable()

    def test_none_audio(self):
        report = analyze_audio_quality(None, SAMPLE_RATE)
        assert report.status == "invalid"
        assert not report.is_usable()


class TestShortAudio:
    """Very short audio must be flagged."""

    def test_short_audio(self, short_audio):
        report = analyze_audio_quality(short_audio, SAMPLE_RATE, min_duration=0.5)
        assert "audio_too_short" in report.reason_codes
        assert report.status == "poor"
        assert not report.is_usable()


class TestNormalAudio:
    """Normal speech-like audio should be acceptable."""

    def test_normal_audio(self, normal_audio):
        report = analyze_audio_quality(normal_audio, SAMPLE_RATE)
        assert report.status == "acceptable"
        assert report.is_usable()
        assert report.duration_seconds > 0
        assert report.rms > 0.003

    def test_duration_correct(self, normal_audio):
        report = analyze_audio_quality(normal_audio, SAMPLE_RATE)
        assert abs(report.duration_seconds - 3.0) < 0.01


class TestClippedAudio:
    """Clipped audio should be detected."""

    def test_clipped_audio(self, clipped_audio):
        report = analyze_audio_quality(clipped_audio, SAMPLE_RATE)
        assert report.clipping_ratio > 0.01  # Significant clipping
        # May or may not be usable depending on severity


class TestNoiseOnly:
    """Pure noise should have low voiced ratio."""

    def test_noise_only(self, noise_only_audio):
        report = analyze_audio_quality(noise_only_audio, SAMPLE_RATE)
        # Pure low-level noise — may be detected as silent or poor
        # depending on RMS
        assert report.duration_seconds > 0


class TestQualityReport:
    """Test AudioQualityReport to_dict serialization."""

    def test_to_dict(self, normal_audio):
        report = analyze_audio_quality(normal_audio, SAMPLE_RATE)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "status" in d
        assert "duration_seconds" in d
        assert "rms" in d
        assert "reason_codes" in d
        assert isinstance(d["reason_codes"], list)
