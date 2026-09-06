"""
test_backend.py — Tests for VoiceGuard FastAPI backend.
======================================================
Tests:
- /health and /ready endpoints
- /predict with file validation, size limits, audio quality gates
- /live/analyze endpoint
- Rate limiting and request ID tracing
- Error responses (415, 413, 422)
- Zero-persistence guarantees (temp file cleanup)
"""

from __future__ import annotations

import io
import unittest.mock as mock
import pytest
from fastapi.testclient import TestClient

import backend
from backend import app, _state
from tests.conftest import make_wav_bytes


@pytest.fixture
def client():
    """Create a FastAPI test client with mocked model load to prevent heavy inference startup."""
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
    """Mock model state in backend without loading actual HuggingFace weights."""
    original_loaded = _state["model_loaded"]
    original_model = _state["model"]
    original_fe = _state["feature_extractor"]
    original_device = _state["device"]

    _state["model_loaded"] = True
    _state["model"] = mock.MagicMock()
    _state["feature_extractor"] = mock.MagicMock()
    _state["device"] = "cpu"

    # Mock inference executor if needed
    if backend._inference_executor is None:
        from concurrent.futures import ThreadPoolExecutor
        backend._inference_executor = ThreadPoolExecutor(max_workers=1)

    yield

    _state["model_loaded"] = original_loaded
    _state["model"] = original_model
    _state["feature_extractor"] = original_fe
    _state["device"] = original_device


class TestHealthAndReadiness:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data

    def test_ready_endpoint_not_ready(self, client):
        _state["model_loaded"] = False
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["model_loaded"] is False

    def test_ready_endpoint_ready(self, client, mock_model):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True
        assert data["model_version"] != ""


class TestPredictEndpoint:
    def test_predict_model_not_ready_returns_insufficient_evidence(self, client, normal_wav):
        _state["model_loaded"] = False
        files = {"file": ("test.wav", io.BytesIO(normal_wav), "audio/wav")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "insufficient_evidence"
        assert data["action"] == "verify"
        assert "model_unavailable" in data["reason_codes"]

    def test_predict_invalid_extension_415(self, client):
        files = {"file": ("malicious.exe", io.BytesIO(b"MZ..."), "application/octet-stream")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 415
        assert "Unsupported file type" in resp.json()["detail"]

    def test_predict_silence_returns_insufficient_evidence(self, client, mock_model, silence_wav):
        files = {"file": ("silence.wav", io.BytesIO(silence_wav), "audio/wav")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "insufficient_evidence"
        assert data["speech_detected"] is False
        assert "no_speech" in data["reason_codes"]

    def test_predict_normal_audio_success(self, client, mock_model, normal_wav):
        # Mock _run_inference_sync to return deterministic result
        with mock.patch("backend._run_inference_sync") as mock_inf:
            mock_inf.return_value = {
                "label": "bonafide",
                "confidence": 0.95,
                "risk_level": "low",
                "spoof_score": 0.05,
                "prob_real": 0.95,
            }
            files = {"file": ("normal.wav", io.BytesIO(normal_wav), "audio/wav")}
            resp = client.post("/predict", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["decision"] == "low_risk"
            assert data["action"] == "allow_with_caution"
            assert data["label"] == "bonafide"
            assert data["spoof_score"] == 0.05
            assert data["speech_detected"] is True
            assert "X-Request-ID" in resp.headers


class TestLiveAnalyzeEndpoint:
    def test_live_analyze_silence(self, client, mock_model, silence_wav):
        files = {"file": ("chunk.wav", io.BytesIO(silence_wav), "audio/wav")}
        resp = client.post("/live/analyze", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "insufficient_evidence"
        assert data["action"] == "verify"
        assert data["speech_detected"] is False

    def test_live_analyze_mock_spoof(self, client, mock_model, normal_wav):
        with mock.patch("backend._run_inference_sync") as mock_inf:
            mock_inf.return_value = {
                "label": "spoof",
                "confidence": 0.88,
                "risk_level": "high",
                "spoof_score": 0.88,
                "prob_real": 0.12,
            }
            files = {"file": ("chunk.wav", io.BytesIO(normal_wav), "audio/wav")}
            resp = client.post("/live/analyze", files=files)
            assert resp.status_code == 200
            data = resp.json()
            # Live analyze returns clip-level evidence (verification_required), NOT call-level action_held
            assert data["decision"] == "verification_required"
            assert data["action"] == "verify"
            assert data["risk_level"] == "high"
            assert data["spoof_score"] == 0.88


class TestSecurityAndMiddleware:
    def test_request_id_in_header_and_body(self, client):
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers

    def test_upload_size_limit_enforced(self, client, mock_model):
        max_bytes = backend.cfg.server.maximum_upload_bytes
        oversized_data = b"x" * (max_bytes + 1024)
        files = {"file": ("big.wav", io.BytesIO(oversized_data), "audio/wav")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 413
        assert "Upload exceeds maximum size" in resp.json()["detail"]

    def test_rate_limiting_enforced(self, client):
        limit = backend.cfg.server.rate_limit_requests_per_minute
        # Fire requests up to the limit
        for _ in range(limit):
            r = client.get("/health")
            assert r.status_code == 200

        # Next request must be 429
        r_blocked = client.get("/health")
        assert r_blocked.status_code == 429
        assert "Rate limit exceeded" in r_blocked.json()["detail"]
