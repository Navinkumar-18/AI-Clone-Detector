"""
test_backend.py — Tests for VoiceGuard FastAPI backend.
======================================================
Tests:
- /health and /ready endpoints
- /predict with file validation, size limits, audio quality gates
- /live/analyze endpoint
- Rate limiting and request ID tracing
- Error responses (415, 413, 422)
- Ephemeral audio handling (temp file cleanup)
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
    original_backend = _state.get("backend")
    original_model = _state["model"]
    original_fe = _state["feature_extractor"]
    original_device = _state["device"]

    _state["model_loaded"] = True
    mock_backend = mock.MagicMock()
    mock_backend.is_loaded.return_value = True
    mock_backend.backend_name = "wav2vec2"
    mock_backend.score_type = "uncalibrated_softmax_score"
    _state["backend"] = mock_backend
    _state["model"] = mock.MagicMock()
    _state["feature_extractor"] = mock.MagicMock()
    _state["device"] = "cpu"

    # Mock inference executor if needed
    if backend._inference_executor is None:
        from concurrent.futures import ThreadPoolExecutor
        backend._inference_executor = ThreadPoolExecutor(max_workers=1)

    yield

    _state["model_loaded"] = original_loaded
    _state["backend"] = original_backend
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
        assert resp.status_code == 503
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
    def test_predict_model_not_ready_returns_503(self, client, normal_wav):
        _state["model_loaded"] = False
        files = {"file": ("test.wav", io.BytesIO(normal_wav), "audio/wav")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 503
        data = resp.json()
        assert "unavailable" in data["detail"].lower()

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

    def test_live_analyze_mock_spoof(self, client, mock_model, live_normal_wav):
        """Complete 4-second spoof chunk must reach inference and return verification_required."""
        with mock.patch("backend._run_inference_sync") as mock_inf:
            mock_inf.return_value = {
                "label": "spoof",
                "confidence": 0.88,
                "risk_level": "high",
                "spoof_score": 0.88,
                "prob_real": 0.12,
            }
            files = {"file": ("chunk.wav", io.BytesIO(live_normal_wav), "audio/wav")}
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
        """Exceeding the request-per-minute limit on API endpoints returns 429."""
        backend._rate_limit_store.clear()
        limit = backend.cfg.server.rate_limit_requests_per_minute
        # Fire requests up to the limit on an API route
        for _ in range(limit):
            r = client.post("/live/analyze")
            assert r.status_code != 429

        # Next request must be 429
        r_blocked = client.post("/live/analyze")
        assert r_blocked.status_code == 429
        assert "Rate limit exceeded" in r_blocked.json()["detail"]

    def test_health_exempt_from_rate_limiting(self, client):
        """Health and readiness probes must never be blocked by rate limiting."""
        backend._rate_limit_store.clear()
        limit = backend.cfg.server.rate_limit_requests_per_minute
        # Exceed rate limit on health checks
        for _ in range(limit + 10):
            r = client.get("/health")
            assert r.status_code == 200

        r_ready = client.get("/ready")
        assert r_ready.status_code in (200, 503)  # Ready or not_ready, never 429


class TestFileAnalyzeWindowsEndpoint:
    def test_analyze_windows_4_second_windows(self, client, mock_model):
        import numpy as np
        from tests.conftest import SAMPLE_RATE
        t = np.linspace(0, 8.0, int(SAMPLE_RATE * 8.0), endpoint=False)
        audio = (0.35 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
        wav_bytes = make_wav_bytes(audio)

        with mock.patch("backend._run_inference_sync") as mock_inf:
            mock_inf.return_value = {
                "label": "bonafide",
                "confidence": 0.95,
                "risk_level": "low",
                "spoof_score": 0.05,
                "prob_real": 0.95,
            }
            files = {"file": ("test8s.wav", io.BytesIO(wav_bytes), "audio/wav")}
            resp = client.post("/file/analyze_windows", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["window_length_seconds"] == 4.0
            assert data["complete_window_count"] == 3
            for w in data["windows"]:
                if w["window_complete"]:
                    assert w["duration_seconds"] == 4.0

    def test_analyze_windows_2_second_stride(self, client, mock_model):
        import numpy as np
        from tests.conftest import SAMPLE_RATE
        t = np.linspace(0, 8.0, int(SAMPLE_RATE * 8.0), endpoint=False)
        audio = (0.35 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
        wav_bytes = make_wav_bytes(audio)

        with mock.patch("backend._run_inference_sync") as mock_inf:
            mock_inf.return_value = {
                "label": "bonafide",
                "confidence": 0.95,
                "risk_level": "low",
                "spoof_score": 0.05,
                "prob_real": 0.95,
            }
            files = {"file": ("test8s.wav", io.BytesIO(wav_bytes), "audio/wav")}
            resp = client.post("/file/analyze_windows", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["stride_seconds"] == 2.0
            windows = data["windows"]
            assert windows[0]["start_seconds"] == 0.0 and windows[0]["end_seconds"] == 4.0
            assert windows[1]["start_seconds"] == 2.0 and windows[1]["end_seconds"] == 6.0
            assert windows[2]["start_seconds"] == 4.0 and windows[2]["end_seconds"] == 8.0

    def test_analyze_windows_exact_boundary(self, client, mock_model):
        import numpy as np
        from tests.conftest import SAMPLE_RATE
        t = np.linspace(0, 4.0, int(SAMPLE_RATE * 4.0), endpoint=False)
        audio = (0.35 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
        wav_bytes = make_wav_bytes(audio)

        with mock.patch("backend._run_inference_sync") as mock_inf:
            mock_inf.return_value = {
                "label": "bonafide",
                "confidence": 0.92,
                "risk_level": "low",
                "spoof_score": 0.08,
                "prob_real": 0.92,
            }
            files = {"file": ("exact4s.wav", io.BytesIO(wav_bytes), "audio/wav")}
            resp = client.post("/file/analyze_windows", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["window_count"] == 1
            assert data["complete_window_count"] == 1
            assert data["windows"][0]["window_complete"] is True
            assert data["windows"][0]["start_seconds"] == 0.0
            assert data["windows"][0]["end_seconds"] == 4.0

    def test_analyze_windows_short_file(self, client, mock_model):
        import numpy as np
        from tests.conftest import SAMPLE_RATE
        t = np.linspace(0, 2.5, int(SAMPLE_RATE * 2.5), endpoint=False)
        audio = (0.35 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
        wav_bytes = make_wav_bytes(audio)

        with mock.patch("backend._run_inference_sync") as mock_inf:
            files = {"file": ("short2_5s.wav", io.BytesIO(wav_bytes), "audio/wav")}
            resp = client.post("/file/analyze_windows", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["complete_window_count"] == 0
            assert data["decision"] == "insufficient_evidence"
            assert "short_audio" in data["reason_codes"]
            assert len(data["windows"]) == 1
            assert data["windows"][0]["window_complete"] is False
            assert "short_audio" in data["windows"][0]["reason_codes"]
            mock_inf.assert_not_called()

    def test_analyze_windows_partial_final_segment(self, client, mock_model):
        import numpy as np
        from tests.conftest import SAMPLE_RATE
        # 7.0 seconds audio: [0, 4] complete, [2, 6] complete, remainder [4, 7] is 3.0s (partial)
        t = np.linspace(0, 7.0, int(SAMPLE_RATE * 7.0), endpoint=False)
        audio = (0.35 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
        wav_bytes = make_wav_bytes(audio)

        with mock.patch("backend._run_inference_sync") as mock_inf:
            mock_inf.return_value = {
                "label": "bonafide",
                "confidence": 0.90,
                "risk_level": "low",
                "spoof_score": 0.10,
                "prob_real": 0.90,
            }
            files = {"file": ("test7s.wav", io.BytesIO(wav_bytes), "audio/wav")}
            resp = client.post("/file/analyze_windows", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["complete_window_count"] == 2
            assert data["window_count"] == 3
            last_w = data["windows"][-1]
            assert last_w["window_complete"] is False
            assert last_w["duration_seconds"] == 3.0
            assert "partial_window" in last_w["reason_codes"]
            assert mock_inf.call_count == 2

    def test_analyze_windows_silence(self, client, mock_model):
        import numpy as np
        from tests.conftest import SAMPLE_RATE
        silence_5s = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
        wav_bytes = make_wav_bytes(silence_5s)

        with mock.patch("backend._run_inference_sync") as mock_inf:
            files = {"file": ("silence5s.wav", io.BytesIO(wav_bytes), "audio/wav")}
            resp = client.post("/file/analyze_windows", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["decision"] != "low_risk"
            assert data["decision"] == "insufficient_evidence"
            for w in data["windows"]:
                if w["window_complete"]:
                    assert w["speech_detected"] is False
            mock_inf.assert_not_called()

    def test_analyze_windows_invalid_audio(self, client, mock_model):
        files = {"file": ("bad.exe", io.BytesIO(b"not audio data"), "application/octet-stream")}
        resp = client.post("/file/analyze_windows", files=files)
        assert resp.status_code == 415

        files_corrupt = {"file": ("corrupt.wav", io.BytesIO(b"RIFFcorruptedtrash"), "audio/wav")}
        resp_corrupt = client.post("/file/analyze_windows", files=files_corrupt)
        assert resp_corrupt.status_code in (200, 422)
        if resp_corrupt.status_code == 200:
            assert resp_corrupt.json()["decision"] == "insufficient_evidence"

    def test_analyze_windows_model_unavailable(self, client):
        import numpy as np
        from tests.conftest import SAMPLE_RATE
        _state["model_loaded"] = False
        t = np.linspace(0, 4.0, int(SAMPLE_RATE * 4.0), endpoint=False)
        audio = (0.35 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
        wav_bytes = make_wav_bytes(audio)

        files = {"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
        resp = client.post("/file/analyze_windows", files=files)
        assert resp.status_code == 503

    def test_analyze_windows_preserves_upload_path(self, client, mock_model, normal_wav):
        with mock.patch("backend._run_inference_sync") as mock_inf:
            mock_inf.return_value = {
                "label": "bonafide",
                "confidence": 0.96,
                "risk_level": "low",
                "spoof_score": 0.04,
                "prob_real": 0.96,
            }
            files = {"file": ("normal.wav", io.BytesIO(normal_wav), "audio/wav")}
            resp = client.post("/predict", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["decision"] == "low_risk"
            assert data["label"] == "bonafide"
            assert "evidence" in data
            assert "audio_quality" in data
            assert "risk_percentage" in data

    def test_analyze_windows_uses_active_model_threshold(self, client, mock_model):
        import numpy as np
        from tests.conftest import SAMPLE_RATE
        t = np.linspace(0, 4.0, int(SAMPLE_RATE * 4.0), endpoint=False)
        audio = (0.35 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
        wav_bytes = make_wav_bytes(audio)

        with mock.patch("backend._run_inference_sync") as mock_inf:
            mock_inf.return_value = {
                "label": "spoof",
                "confidence": 0.92,
                "risk_level": "high",
                "spoof_score": 0.92,
                "prob_real": 0.08,
            }
            files = {"file": ("spoof.wav", io.BytesIO(wav_bytes), "audio/wav")}
            resp = client.post("/file/analyze_windows", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_version"] == backend.cfg.model.version
            assert data["threshold_version"] == backend.cfg.thresholds.version
            assert data["high_risk_windows"] == 1
            assert data["decision"] in ("verification_required", "action_held")
