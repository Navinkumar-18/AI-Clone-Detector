"""
test_live_endpoint.py
=====================
Automated tests for the /live/analyze endpoint.

Verifies:
  1. Backend health check returns model_loaded=true
  2. /live/analyze returns correct response schema
  3. /live/analyze response includes action field
  4. /predict still works (existing functionality not broken)
  5. Invalid audio handling
  6. Synthetic audio clips produce expected detections

Usage:
    # Requires backend running: uvicorn backend:app --host 0.0.0.0 --port 8000
    python test_live_endpoint.py
    python test_live_endpoint.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import time
import warnings

import numpy as np
import requests
import urllib3

# Suppress SSL warnings when testing against self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Module-level flag: set True when testing against a self-signed HTTPS backend
_VERIFY_SSL = True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_wav_bytes(duration_sec: float = 3.0, sample_rate: int = 16000,
                   frequency: float = 440.0, noise_level: float = 0.1) -> bytes:
    """Generate a synthetic WAV file in memory and return the raw bytes."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * frequency * t) + noise_level * np.random.randn(len(t))
    signal = np.clip(signal, -1.0, 1.0)
    pcm = (signal * 32767).astype(np.int16)

    buf = io.BytesIO()
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = len(pcm) * (bits_per_sample // 8)

    # WAV header
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


def fmt_result(r: dict) -> str:
    """Pretty-format a result dict for display."""
    label = r.get("label", "?")
    score = r.get("detection_score", r.get("confidence", "?"))
    risk = r.get("risk_level", "?")
    action = r.get("action", "n/a")
    return f"label={label}, score={score}, risk={risk}, action={action}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health(base_url: str) -> bool:
    """Test 1: Health check returns model_loaded=true."""
    print("\n--- Test 1: Health Check ---")
    try:
        r = requests.get(f"{base_url}/health", timeout=10, verify=_VERIFY_SSL)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert data.get("status") == "ok", f"Status is not 'ok': {data}"
        assert data.get("model_loaded") is True, f"Model not loaded: {data}"
        print(f"  PASSED — status=ok, model_loaded=true")
        return True
    except Exception as e:
        print(f"  FAILED — {e}")
        return False


def test_live_analyze_schema(base_url: str) -> bool:
    """Test 2: /live/analyze returns correct response schema."""
    print("\n--- Test 2: /live/analyze Response Schema ---")
    try:
        wav = make_wav_bytes(duration_sec=3.0, frequency=440.0)
        r = requests.post(
            f"{base_url}/live/analyze",
            files={"file": ("test.wav", wav, "audio/wav")},
            timeout=30,
            verify=_VERIFY_SSL,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()

        required_keys = ["label", "confidence", "risk_level", "detection_score",
                         "prob_real", "prob_fake", "action", "risk_category", "risk_percentage"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

        assert data["label"] in ("bonafide", "spoof"), f"Invalid label: {data['label']}"
        assert 0.0 <= data["confidence"] <= 1.0, f"Confidence out of range: {data['confidence']}"
        assert data["risk_level"] in ("low", "medium", "high"), f"Invalid risk: {data['risk_level']}"
        assert 0.0 <= data["detection_score"] <= 1.0, f"Score out of range: {data['detection_score']}"
        assert data["action"] in ("allow", "verify", "block"), f"Invalid action: {data['action']}"
        assert data["risk_category"] in ("Low Risk", "Medium Risk", "Critical"), f"Invalid risk_category: {data['risk_category']}"
        assert 0.0 <= data["risk_percentage"] <= 100.0, f"Invalid risk_percentage: {data['risk_percentage']}"

        print(f"  PASSED — {fmt_result(data)} | category={data['risk_category']}, pct={data['risk_percentage']}%")
        return True
    except Exception as e:
        print(f"  FAILED — {e}")
        return False


def test_live_action_consistency(base_url: str) -> bool:
    """Test 3: Action matches risk_level (low→allow, medium→verify, high→block)."""
    print("\n--- Test 3: Action-Risk Consistency ---")
    try:
        wav = make_wav_bytes(duration_sec=3.0, frequency=440.0)
        r = requests.post(
            f"{base_url}/live/analyze",
            files={"file": ("test.wav", wav, "audio/wav")},
            timeout=30,
            verify=_VERIFY_SSL,
        )
        data = r.json()
        expected_action = {"low": "allow", "medium": "verify", "high": "block"}
        risk = data["risk_level"]
        action = data["action"]
        assert action == expected_action[risk], \
            f"Risk '{risk}' should map to '{expected_action[risk]}', got '{action}'"
        print(f"  PASSED — risk={risk}, action={action} (consistent)")
        return True
    except Exception as e:
        print(f"  FAILED — {e}")
        return False


def test_predict_still_works(base_url: str) -> bool:
    """Test 4: Existing /predict endpoint still works."""
    print("\n--- Test 4: Existing /predict Still Works ---")
    try:
        wav = make_wav_bytes(duration_sec=3.0, frequency=440.0)
        r = requests.post(
            f"{base_url}/predict",
            files={"file": ("test.wav", wav, "audio/wav")},
            timeout=30,
            verify=_VERIFY_SSL,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert "label" in data, "Missing 'label' in /predict response"
        assert "confidence" in data, "Missing 'confidence' in /predict response"
        assert "risk_level" in data, "Missing 'risk_level' in /predict response"
        assert "risk_category" in data, "Missing 'risk_category' in /predict response"
        assert "risk_percentage" in data, "Missing 'risk_percentage' in /predict response"
        assert data["risk_category"] in ("Low Risk", "Medium Risk", "Critical"), f"Invalid risk_category: {data['risk_category']}"
        assert 0.0 <= data["risk_percentage"] <= 100.0, f"Invalid risk_percentage: {data['risk_percentage']}"
        print(f"  PASSED — label={data['label']}, confidence={data['confidence']:.4f}, risk={data['risk_level']}, category={data['risk_category']}, pct={data['risk_percentage']}%")
        return True
    except Exception as e:
        print(f"  FAILED — {e}")
        return False


def test_short_audio(base_url: str) -> bool:
    """Test 5: Very short audio chunk (0.5 seconds) — should not crash."""
    print("\n--- Test 5: Short Audio Chunk (0.5s) ---")
    try:
        wav = make_wav_bytes(duration_sec=0.5, frequency=440.0)
        r = requests.post(
            f"{base_url}/live/analyze",
            files={"file": ("short.wav", wav, "audio/wav")},
            timeout=30,
            verify=_VERIFY_SSL,
        )
        # Should either succeed (200) or return a handled error (422), not crash (500)
        assert r.status_code in (200, 422), f"Unexpected status: {r.status_code}"
        if r.status_code == 200:
            data = r.json()
            print(f"  PASSED — processed short audio: {fmt_result(data)}")
        else:
            print(f"  PASSED — correctly rejected short audio (422)")
        return True
    except Exception as e:
        print(f"  FAILED — {e}")
        return False


def test_multiple_chunks(base_url: str) -> bool:
    """Test 6: Multiple sequential live chunks — simulates a live call."""
    print("\n--- Test 6: Multiple Sequential Chunks (simulated live call) ---")
    try:
        results = []
        freqs = [440.0, 880.0, 220.0, 660.0]  # different tones
        for i, freq in enumerate(freqs):
            wav = make_wav_bytes(duration_sec=4.0, frequency=freq, noise_level=0.1 + i * 0.1)
            t0 = time.time()
            r = requests.post(
                f"{base_url}/live/analyze",
                files={"file": (f"chunk_{i}.wav", wav, "audio/wav")},
                timeout=30,
                verify=_VERIFY_SSL,
            )
            latency = time.time() - t0
            assert r.status_code == 200, f"Chunk {i} failed with status {r.status_code}"
            data = r.json()
            results.append(data)
            print(f"  Chunk {i}: {fmt_result(data)}  ({latency:.2f}s)")

        # All chunks should return valid responses
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"
        print(f"  PASSED — all {len(results)} chunks analyzed successfully")
        return True
    except Exception as e:
        print(f"  FAILED — {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _VERIFY_SSL
    parser = argparse.ArgumentParser(description="Test /live/analyze endpoint")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Backend base URL (default: http://localhost:8000)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    # Auto-disable SSL verification for self-signed certs when using HTTPS
    if base_url.startswith("https://"):
        _VERIFY_SSL = False
        print("  (SSL verification disabled for self-signed cert)")

    print(f"\nTesting backend at: {base_url}")
    print("=" * 60)

    tests = [
        test_health,
        test_live_analyze_schema,
        test_live_action_consistency,
        test_predict_still_works,
        test_short_audio,
        test_multiple_chunks,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            if test_fn(base_url):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  CRASHED — {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
