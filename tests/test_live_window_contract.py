"""
test_live_window_contract.py — Live-window contract tests for VoiceGuard.
=========================================================================

Tests that enforce the FULL_WINDOW_MS contract on /live/analyze.

Required tests (per implementation spec):
    test_partial_window_is_not_inferred
    test_partial_window_does_not_update_ema
    test_partial_window_does_not_count_for_persistence
    test_partial_window_does_not_clear_action_hold
    test_first_live_window_is_insufficient_evidence
    test_complete_window_updates_aggregator
    test_complete_live_window_matches_upload
    test_silence_returns_insufficient_evidence
    test_invalid_audio_returns_insufficient_evidence
    test_backend_unavailable_fails_closed
    test_model_unavailable_fails_closed
    test_final_partial_buffer_is_not_classified

/predict is exercised only for parity comparison — its behavior is not
changed by this fix and no upload tests are modified.

IMPORTANT: This module tests ONLY the /live/analyze path safety contract.
The upload (/predict) path is tested separately in test_backend.py and
must not be regressed.
"""

from __future__ import annotations

import io
import time
import unittest.mock as mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend
from backend import app, _state
from risk_aggregator import RiskAggregator, CallDecision, CallAction, ClipEvidence
from voiceguard_config import load_config
from tests.conftest import make_wav_bytes


# ---------------------------------------------------------------------------
# Constants — derived from model_config.yaml live_analysis section
# ---------------------------------------------------------------------------
cfg = load_config()
FULL_WINDOW_MS = cfg.live_analysis.full_window_ms   # 4000
STRIDE_MS = cfg.live_analysis.stride_ms              # 2000
SAMPLE_RATE = cfg.audio.sample_rate                  # 16000

FULL_WINDOW_SAMPLES = int(SAMPLE_RATE * FULL_WINDOW_MS / 1000)
PARTIAL_SAMPLES = int(SAMPLE_RATE * 1.84)  # ~1.84 s — the problematic short window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pcm_wav(n_samples: int, amplitude: float = 0.4, freq: float = 440.0) -> bytes:
    """Create a deterministic sine-wave WAV of exactly n_samples."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * freq * t) + 0.02 * rng.randn(n_samples)
    return make_wav_bytes(signal.astype(np.float32), sample_rate=SAMPLE_RATE)


def make_silence_wav(n_samples: int) -> bytes:
    """Digital silence WAV."""
    return make_wav_bytes(np.zeros(n_samples, dtype=np.float32), sample_rate=SAMPLE_RATE)


def make_evidence(
    spoof_score: float = 0.1,
    speech_detected: bool = True,
    quality: str = "acceptable",
    label: str = "bonafide",
) -> ClipEvidence:
    return ClipEvidence(
        spoof_score=spoof_score,
        speech_detected=speech_detected,
        audio_quality_status=quality,
        label=label,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """FastAPI test client with mocked model load (matches test_backend.py pattern)."""
    backend._rate_limit_store.clear()
    with mock.patch("transformers.AutoFeatureExtractor.from_pretrained") as mock_fe, \
         mock.patch("transformers.AutoModelForAudioClassification.from_pretrained") as mock_m:
        mock_model_obj = mock.MagicMock()
        mock_model_obj.to.return_value = mock_model_obj
        mock_model_obj.config.id2label = {0: "bonafide", 1: "spoof"}
        mock_m.return_value = mock_model_obj
        mock_fe.return_value = mock.MagicMock()

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    backend._rate_limit_store.clear()


@pytest.fixture
def mock_model():
    """Mock model state without loading real HuggingFace weights."""
    original = {k: _state[k] for k in _state}
    _state["model_loaded"] = True
    mb = mock.MagicMock()
    mb.is_loaded.return_value = True
    mb.backend_name = "wav2vec2"
    mb.score_type = "uncalibrated_softmax_score"
    _state["backend"] = mb
    _state["model"] = mock.MagicMock()
    _state["feature_extractor"] = mock.MagicMock()
    _state["device"] = "cpu"

    if backend._inference_executor is None:
        from concurrent.futures import ThreadPoolExecutor
        backend._inference_executor = ThreadPoolExecutor(max_workers=1)
    yield
    for k, v in original.items():
        _state[k] = v


@pytest.fixture
def full_window_wav() -> bytes:
    return make_pcm_wav(FULL_WINDOW_SAMPLES)


@pytest.fixture
def partial_window_wav() -> bytes:
    return make_pcm_wav(PARTIAL_SAMPLES)


@pytest.fixture
def silence_full_window_wav() -> bytes:
    return make_silence_wav(FULL_WINDOW_SAMPLES)


# ---------------------------------------------------------------------------
# 1. Partial window must not reach inference
# ---------------------------------------------------------------------------

class TestPartialWindowRejected:
    def test_partial_window_is_not_inferred(self, client, mock_model, partial_window_wav):
        """A chunk shorter than FULL_WINDOW_MS must return insufficient_evidence."""
        with mock.patch("backend._run_inference_sync") as mock_inf:
            files = {"file": ("chunk.wav", io.BytesIO(partial_window_wav), "audio/wav")}
            resp = client.post("/live/analyze", files=files)
            assert resp.status_code == 200
            data = resp.json()

            # Inference must NOT have been called
            mock_inf.assert_not_called()

            assert data["decision"] == "insufficient_evidence"
            assert "partial_window" in data["reason_codes"]
            assert data["speech_detected"] is False

    def test_partial_window_reason_code(self, client, mock_model, partial_window_wav):
        """partial_window must be in reason_codes for sub-FULL_WINDOW_MS chunks."""
        files = {"file": ("chunk.wav", io.BytesIO(partial_window_wav), "audio/wav")}
        resp = client.post("/live/analyze", files=files)
        data = resp.json()
        assert "partial_window" in data["reason_codes"]

    def test_very_short_chunk_rejected(self, client, mock_model):
        """Even a 0.5-second chunk (above old 0.3s threshold) must be rejected."""
        short_wav = make_pcm_wav(int(SAMPLE_RATE * 0.5))
        files = {"file": ("short.wav", io.BytesIO(short_wav), "audio/wav")}
        resp = client.post("/live/analyze", files=files)
        data = resp.json()
        assert data["decision"] == "insufficient_evidence"
        assert "partial_window" in data["reason_codes"]

    def test_three_second_chunk_rejected(self, client, mock_model):
        """A 3-second chunk (below 4s threshold) must be rejected."""
        three_s_wav = make_pcm_wav(int(SAMPLE_RATE * 3.0))
        files = {"file": ("3s.wav", io.BytesIO(three_s_wav), "audio/wav")}
        resp = client.post("/live/analyze", files=files)
        data = resp.json()
        assert data["decision"] == "insufficient_evidence"
        assert "partial_window" in data["reason_codes"]


# ---------------------------------------------------------------------------
# 2. Partial window must not update aggregator state
# ---------------------------------------------------------------------------

class TestPartialWindowDoesNotUpdateAggregator:
    def test_partial_window_does_not_update_ema(self):
        """Partial windows (rejected by backend) must not reach aggregator EMA."""
        agg = RiskAggregator()
        ev = make_evidence(spoof_score=0.0, speech_detected=False, quality="unknown")
        state = agg.update(ev)
        assert state.total_windows == 0
        assert state.ema_score == 0.0
        assert state.decision != CallDecision.LOW_RISK

    def test_partial_window_does_not_count_for_persistence(self):
        """Partial window evidence must not count toward persistence counter."""
        agg = RiskAggregator(persistent_high_windows=2)
        ev = make_evidence(spoof_score=0.95, speech_detected=False, quality="unknown")
        agg.update(ev)
        assert agg.state.consecutive_high_windows == 0

    def test_partial_window_does_not_clear_action_hold(self):
        """A partial window result must NOT clear an existing ACTION_HELD state."""
        agg = RiskAggregator(persistent_high_windows=2)
        agg.update(make_evidence(spoof_score=0.92, speech_detected=True, quality="acceptable", label="spoof"))
        agg.update(make_evidence(spoof_score=0.94, speech_detected=True, quality="acceptable", label="spoof"))
        assert agg.state.decision == CallDecision.ACTION_HELD

        partial_ev = make_evidence(spoof_score=0.0, speech_detected=False, quality="unknown")
        state = agg.update(partial_ev)

        assert state.decision == CallDecision.INSUFFICIENT_EVIDENCE
        assert state.action == CallAction.HOLD
        assert "action_remains_held" in state.reason_codes

    def test_partial_window_does_not_produce_low_risk(self):
        """Partial window evidence must never produce LOW_RISK."""
        agg = RiskAggregator()
        ev = make_evidence(spoof_score=0.0, speech_detected=False, quality="unknown")
        state = agg.update(ev)
        assert state.decision != CallDecision.LOW_RISK


# ---------------------------------------------------------------------------
# 3. First live window must show ANALYZING / INSUFFICIENT_EVIDENCE
# ---------------------------------------------------------------------------

class TestFirstLiveWindow:
    def test_first_live_window_is_insufficient_evidence(self, client, mock_model):
        """The very first partial chunk from startup must return insufficient_evidence."""
        first_chunk = make_pcm_wav(int(SAMPLE_RATE * 1.84))
        files = {"file": ("first.wav", io.BytesIO(first_chunk), "audio/wav")}
        resp = client.post("/live/analyze", files=files)
        data = resp.json()
        assert data["decision"] == "insufficient_evidence"
        assert data["action"] in ("verify", "unavailable")
        assert data["decision"] != "low_risk"
        assert data["label"] == "unknown"


# ---------------------------------------------------------------------------
# 4. Complete window must pass through and update aggregator
# ---------------------------------------------------------------------------

class TestCompleteWindowAccepted:
    def test_complete_window_updates_aggregator(self, client, mock_model, full_window_wav):
        """A complete 4-second window must reach inference and update the aggregator."""
        with mock.patch("backend._run_inference_sync") as mock_inf:
            mock_inf.return_value = {
                "label": "bonafide",
                "confidence": 0.90,
                "risk_level": "low",
                "spoof_score": 0.10,
                "prob_real": 0.90,
                "model_backend": "wav2vec2",
                "score_type": "uncalibrated_softmax_score",
            }
            files = {"file": ("full.wav", io.BytesIO(full_window_wav), "audio/wav")}
            resp = client.post("/live/analyze", files=files)
            assert resp.status_code == 200
            data = resp.json()
            mock_inf.assert_called_once()
            assert data["spoof_score"] == 0.10
            assert data["speech_detected"] is True

    def test_complete_window_returns_valid_decision(self, client, mock_model, full_window_wav):
        """A complete window with low spoof score must produce low_risk decision."""
        with mock.patch("backend._run_inference_sync") as mock_inf:
            mock_inf.return_value = {
                "label": "bonafide",
                "confidence": 0.92,
                "risk_level": "low",
                "spoof_score": 0.08,
                "prob_real": 0.92,
                "model_backend": "wav2vec2",
                "score_type": "uncalibrated_softmax_score",
            }
            files = {"file": ("full.wav", io.BytesIO(full_window_wav), "audio/wav")}
            resp = client.post("/live/analyze", files=files)
            data = resp.json()
            assert data["decision"] == "low_risk"
            assert data["action"] == "allow_with_caution"


# ---------------------------------------------------------------------------
# 5. Parity — complete live window vs direct upload
# ---------------------------------------------------------------------------

class TestCompleteWindowParityWithUpload:
    def test_complete_window_matches_upload(self, client, mock_model, full_window_wav):
        """
        The same 4-second waveform sent via /live/analyze and /predict must
        produce the same spoof_score and label (within mocked inference parity).
        """
        fixed_result = {
            "label": "bonafide",
            "confidence": 0.91,
            "risk_level": "low",
            "spoof_score": 0.09,
            "prob_real": 0.91,
            "model_backend": "wav2vec2",
            "score_type": "uncalibrated_softmax_score",
        }

        with mock.patch("backend._run_inference_sync", return_value=fixed_result):
            files_a = {"file": ("upload.wav", io.BytesIO(full_window_wav), "audio/wav")}
            resp_a = client.post("/predict", files=files_a)
            assert resp_a.status_code == 200
            data_a = resp_a.json()

        with mock.patch("backend._run_inference_sync", return_value=fixed_result):
            files_b = {"file": ("live.wav", io.BytesIO(full_window_wav), "audio/wav")}
            resp_b = client.post("/live/analyze", files=files_b)
            assert resp_b.status_code == 200
            data_b = resp_b.json()

        assert data_a["spoof_score"] == data_b["spoof_score"], (
            f"Parity failure: upload={data_a['spoof_score']}, live={data_b['spoof_score']}"
        )
        assert data_a["label"] == data_b["label"]
        assert data_a["risk_level"] == data_b["risk_level"]


# ---------------------------------------------------------------------------
# 6. Silence must return insufficient_evidence
# ---------------------------------------------------------------------------

class TestSilenceFailing:
    def test_silence_returns_insufficient_evidence(self, client, mock_model, silence_full_window_wav):
        """A complete silence window must return insufficient_evidence, not LOW_RISK."""
        files = {"file": ("silence.wav", io.BytesIO(silence_full_window_wav), "audio/wav")}
        resp = client.post("/live/analyze", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "insufficient_evidence"
        assert data["speech_detected"] is False
        assert data["decision"] != "low_risk"

    def test_silence_aggregator_does_not_allow(self):
        """Silence evidence must not produce ALLOW_WITH_CAUTION in the aggregator."""
        agg = RiskAggregator()
        ev = make_evidence(spoof_score=0.0, speech_detected=False, quality="silent")
        state = agg.update(ev)
        assert state.action != CallAction.ALLOW_WITH_CAUTION
        assert state.decision == CallDecision.INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# 7. Invalid audio must return insufficient_evidence
# ---------------------------------------------------------------------------

class TestInvalidAudioFailing:
    def test_invalid_audio_returns_insufficient_evidence(self, client, mock_model):
        """Corrupted/non-WAV bytes must fail closed."""
        corrupted = b"NOT_A_VALID_WAV_FILE_HEADER_" + b"\x00" * 100
        files = {"file": ("bad.wav", io.BytesIO(corrupted), "audio/wav")}
        resp = client.post("/live/analyze", files=files)
        assert resp.status_code in (200, 422)
        if resp.status_code == 200:
            data = resp.json()
            assert data["decision"] == "insufficient_evidence"


# ---------------------------------------------------------------------------
# 8. Backend/model unavailability must fail closed
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_model_unavailable_fails_closed(self, client, mock_model, full_window_wav):
        """If the model backend is not loaded, /live/analyze must return 503."""
        _state["model_loaded"] = False
        _state["backend"] = None
        try:
            files = {"file": ("full.wav", io.BytesIO(full_window_wav), "audio/wav")}
            resp = client.post("/live/analyze", files=files)
            assert resp.status_code in (200, 503)
            if resp.status_code == 200:
                data = resp.json()
                assert data["decision"] == "insufficient_evidence"
                assert data["action"] in ("verify", "unavailable")
                assert data["decision"] != "low_risk"
        finally:
            _state["model_loaded"] = True

    def test_backend_unavailable_fails_closed(self):
        """Backend failure must not produce ALLOW_WITH_CAUTION."""
        agg = RiskAggregator()
        state = agg.record_backend_failure()
        assert state.action != CallAction.ALLOW_WITH_CAUTION
        assert state.decision == CallDecision.INSUFFICIENT_EVIDENCE
        assert "backend_unavailable" in state.reason_codes

    def test_backend_unavailable_after_action_held_stays_held(self):
        """Backend failure after ACTION_HELD must not clear the hold."""
        agg = RiskAggregator(persistent_high_windows=2)
        agg.update(make_evidence(spoof_score=0.92, speech_detected=True, quality="acceptable", label="spoof"))
        agg.update(make_evidence(spoof_score=0.94, speech_detected=True, quality="acceptable", label="spoof"))
        assert agg.state.decision == CallDecision.ACTION_HELD

        state = agg.record_backend_failure()
        assert state.action == CallAction.HOLD
        assert state.decision == CallDecision.INSUFFICIENT_EVIDENCE
        assert "action_remains_held" in state.reason_codes


# ---------------------------------------------------------------------------
# 9. Final partial buffer must not be classified
# ---------------------------------------------------------------------------

class TestFinalPartialBufferNotClassified:
    def test_final_partial_buffer_not_classified(self, client, mock_model):
        """A final short buffer when capture stops must return insufficient_evidence."""
        final_chunk = make_pcm_wav(int(SAMPLE_RATE * 2.1))
        with mock.patch("backend._run_inference_sync") as mock_inf:
            files = {"file": ("final.wav", io.BytesIO(final_chunk), "audio/wav")}
            resp = client.post("/live/analyze", files=files)
            data = resp.json()
            mock_inf.assert_not_called()
            assert data["decision"] == "insufficient_evidence"
            assert "partial_window" in data["reason_codes"]


# ---------------------------------------------------------------------------
# 10. Upload path must be unaffected (regression guard)
# ---------------------------------------------------------------------------

class TestUploadPathUnchanged:
    def test_predict_still_accepts_short_audio(self, client, mock_model):
        """
        /predict must still accept audio shorter than FULL_WINDOW_MS.
        The 4-second guard must NOT be applied to the upload path.
        """
        short_audio = make_pcm_wav(SAMPLE_RATE * 1)
        with mock.patch("backend._run_inference_sync") as mock_inf:
            mock_inf.return_value = {
                "label": "bonafide",
                "confidence": 0.80,
                "risk_level": "low",
                "spoof_score": 0.20,
                "prob_real": 0.80,
                "model_backend": "wav2vec2",
                "score_type": "uncalibrated_softmax_score",
            }
            files = {"file": ("short_upload.wav", io.BytesIO(short_audio), "audio/wav")}
            resp = client.post("/predict", files=files)
            assert resp.status_code == 200
            mock_inf.assert_called_once()

    def test_live_rejects_same_short_audio(self, client, mock_model):
        """
        The same 1-second audio that /predict accepts must be rejected by /live/analyze.
        """
        short_audio = make_pcm_wav(SAMPLE_RATE * 1)
        with mock.patch("backend._run_inference_sync") as mock_inf:
            files = {"file": ("short_live.wav", io.BytesIO(short_audio), "audio/wav")}
            resp = client.post("/live/analyze", files=files)
            data = resp.json()
            mock_inf.assert_not_called()
            assert data["decision"] == "insufficient_evidence"
            assert "partial_window" in data["reason_codes"]
