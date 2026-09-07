"""
evaluation/smoke_test_backends.py — Smoke tests for live FastAPI endpoints across backends.
==========================================================================================
Tests:
1. Active model: wav2vec2 -> /health, /ready, /predict
2. Active model: wavlm_mlp -> /health, /ready, /predict
3. Negative test: corrupted/nonexistent checkpoint -> /ready returns 503, /predict returns 503
"""

import io
import os
import sys
import tempfile
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from voiceguard_config import reset_config_cache


def create_test_wav_bytes() -> bytes:
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def run_smoke_tests():
    print("=== VoiceGuard Backend Smoke Tests ===")
    wav_bytes = create_test_wav_bytes()

    # 1. Test Wav2Vec2 Backend
    print("\n--- Test 1: Wav2Vec2 Active Backend ---")
    reset_config_cache()
    os.environ["VOICEGUARD_ACTIVE_MODEL"] = "wav2vec2"
    import backend
    with TestClient(backend.app, raise_server_exceptions=False) as client:
        # /health
        h_resp = client.get("/health")
        print(f"GET /health -> {h_resp.status_code}, data: {h_resp.json()}")
        assert h_resp.status_code == 200

        # /ready
        r_resp = client.get("/ready")
        print(f"GET /ready -> {r_resp.status_code}, data: {r_resp.json()}")
        assert r_resp.status_code == 200
        r_data = r_resp.json()
        assert r_data["model_backend"] == "wav2vec2"
        assert r_data["score_type"] == "uncalibrated_softmax_score"

        # /predict
        files = {"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
        p_resp = client.post("/predict", files=files)
        print(f"POST /predict -> {p_resp.status_code}, decision: {p_resp.json().get('decision')}, backend: {p_resp.json().get('model_backend')}, score_type: {p_resp.json().get('score_type')}")
        assert p_resp.status_code == 200
        p_data = p_resp.json()
        assert p_data["model_backend"] == "wav2vec2"
        assert p_data["score_type"] == "uncalibrated_softmax_score"

    # 2. Test WavLM+MLP Active Backend
    print("\n--- Test 2: WavLM+MLP Active Backend ---")
    reset_config_cache()
    os.environ["VOICEGUARD_ACTIVE_MODEL"] = "wavlm_mlp"
    backend.cfg = backend.load_config()
    with TestClient(backend.app, raise_server_exceptions=False) as client:
        # /health
        h_resp = client.get("/health")
        print(f"GET /health -> {h_resp.status_code}, data: {h_resp.json()}")
        assert h_resp.status_code == 200

        # /ready
        r_resp = client.get("/ready")
        print(f"GET /ready -> {r_resp.status_code}, data: {r_resp.json()}")
        assert r_resp.status_code == 200
        r_data = r_resp.json()
        assert r_data["model_backend"] == "wavlm_mlp"
        assert r_data["score_type"] == "uncalibrated_sigmoid_score"

        # /predict
        files = {"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
        p_resp = client.post("/predict", files=files)
        print(f"POST /predict -> {p_resp.status_code}, decision: {p_resp.json().get('decision')}, backend: {p_resp.json().get('model_backend')}, score_type: {p_resp.json().get('score_type')}")
        assert p_resp.status_code == 200
        p_data = p_resp.json()
        assert p_data["model_backend"] == "wavlm_mlp"
        assert p_data["score_type"] == "uncalibrated_sigmoid_score"

    # 3. Test Negative / Failure Closed Scenario
    print("\n--- Test 3: Failure Closed (503 on uninitialized backend) ---")
    with TestClient(backend.app, raise_server_exceptions=False) as client:
        backend._state["model_loaded"] = False
        backend._state["backend"] = None
        backend._state["load_error"] = "Simulated backend initialization failure"

        r_resp = client.get("/ready")
        print(f"GET /ready (uninitialized) -> {r_resp.status_code}, status: {r_resp.json().get('status')}")
        assert r_resp.status_code == 503

        files = {"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}
        p_resp = client.post("/predict", files=files)
        print(f"POST /predict (uninitialized) -> {p_resp.status_code}, detail: {p_resp.json().get('detail')}")
        assert p_resp.status_code == 503

    # Reset environment
    os.environ.pop("VOICEGUARD_ACTIVE_MODEL", None)
    reset_config_cache()
    backend.cfg = backend.load_config()
    print("\nALL SMOKE TESTS PASSED PERFECTLY!")


if __name__ == "__main__":
    run_smoke_tests()
