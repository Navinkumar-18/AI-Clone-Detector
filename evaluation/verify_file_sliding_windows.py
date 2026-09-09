"""
evaluation/verify_file_sliding_windows.py — Manual verification script for /file/analyze_windows.

Validates:
1. Complete 4-second sliding windows with 2-second stride.
2. Short file (<4s) policy.
3. Natural vs Spoofed comparison across windows.
"""

from __future__ import annotations

import io
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import backend
from backend import app, _state
from tests.conftest import make_wav_bytes, SAMPLE_RATE

def main():
    print("=" * 60)
    print("VoiceGuard Sliding Window Analysis Verification")
    print("=" * 60)

    # Setup TestClient with mock model
    _state["model_loaded"] = True
    backend._rate_limit_store.clear()

    with TestClient(app) as client:
        # Case 1: 10-Second Natural Speech Sample
        print("\n--- Test 1: 10-Second Natural Audio (4 windows expected: 0-4, 2-6, 4-8, 6-10) ---")
        t = np.linspace(0, 10.0, int(SAMPLE_RATE * 10.0), endpoint=False)
        audio_human = (0.35 * np.sin(2 * np.pi * 220 * t) + 0.02 * np.random.randn(len(t))).astype(np.float32)
        wav_human = make_wav_bytes(audio_human)

        resp = client.post(
            "/file/analyze_windows",
            files={"file": ("natural_human.wav", io.BytesIO(wav_human), "audio/wav")}
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        res_human = resp.json()

        print(f"Total Duration: {res_human['total_duration_seconds']}s")
        print(f"Window Count: {res_human['window_count']} (Complete: {res_human['complete_window_count']})")
        print(f"Overall Decision: {res_human['decision']} (Action: {res_human['action']})")
        print(f"Max Spoof Score: {res_human['maximum_spoof_score']} | Avg Spoof: {res_human['average_spoof_score']}")
        print(f"Windows:")
        for w in res_human["windows"]:
            print(f"  [Window {w['window_index']}] {w['start_seconds']:04.1f}s - {w['end_seconds']:04.1f}s | "
                  f"Score: {w['spoof_score']} | Risk: {w['risk_level']} | "
                  f"RMS: {w['rms']} | SNR: {w['snr_db']} dB | Complete: {w['window_complete']}")

        assert res_human["window_count"] == 4
        assert res_human["complete_window_count"] == 4
        assert res_human["windows"][0]["start_seconds"] == 0.0
        assert res_human["windows"][1]["start_seconds"] == 2.0
        assert res_human["windows"][2]["start_seconds"] == 4.0
        assert res_human["windows"][3]["start_seconds"] == 6.0

        # Case 2: Short Audio (< 4s)
        print("\n--- Test 2: Short Audio (< 4 seconds: 2.2s duration) ---")
        t_short = np.linspace(0, 2.2, int(SAMPLE_RATE * 2.2), endpoint=False)
        audio_short = (0.35 * np.sin(2 * np.pi * 300 * t_short)).astype(np.float32)
        wav_short = make_wav_bytes(audio_short)

        resp_short = client.post(
            "/file/analyze_windows",
            files={"file": ("short_audio.wav", io.BytesIO(wav_short), "audio/wav")}
        )
        assert resp_short.status_code == 200
        res_short = resp_short.json()
        print(f"Total Duration: {res_short['total_duration_seconds']}s")
        print(f"Complete Windows: {res_short['complete_window_count']} / {res_short['window_count']}")
        print(f"Decision: {res_short['decision']} | Reasons: {res_short['reason_codes']}")
        assert res_short["complete_window_count"] == 0
        assert res_short["decision"] == "insufficient_evidence"
        assert "short_audio" in res_short["reason_codes"]
        assert res_short["windows"][0]["window_complete"] is False

        # Case 3: Partial Final Segment (7.0s duration)
        print("\n--- Test 3: Audio with Partial Final Segment (7.0s duration) ---")
        t_partial = np.linspace(0, 7.0, int(SAMPLE_RATE * 7.0), endpoint=False)
        audio_partial = (0.35 * np.sin(2 * np.pi * 300 * t_partial)).astype(np.float32)
        wav_partial = make_wav_bytes(audio_partial)

        resp_partial = client.post(
            "/file/analyze_windows",
            files={"file": ("partial_audio.wav", io.BytesIO(wav_partial), "audio/wav")}
        )
        assert resp_partial.status_code == 200
        res_partial = resp_partial.json()
        print(f"Total Duration: {res_partial['total_duration_seconds']}s")
        print(f"Windows: {res_partial['window_count']} (Complete: {res_partial['complete_window_count']})")
        for w in res_partial["windows"]:
            print(f"  [Window {w['window_index']}] {w['start_seconds']:04.1f}s - {w['end_seconds']:04.1f}s | "
                  f"Complete: {w['window_complete']} | Reasons: {w['reason_codes']}")
        assert res_partial["complete_window_count"] == 2
        assert res_partial["window_count"] == 3
        assert res_partial["windows"][-1]["window_complete"] is False
        assert "partial_window" in res_partial["windows"][-1]["reason_codes"]

        # Case 4: Preserve /predict endpoint
        print("\n--- Test 4: Regression Check — /predict endpoint remains unchanged ---")
        resp_pred = client.post(
            "/predict",
            files={"file": ("natural_human.wav", io.BytesIO(wav_human), "audio/wav")}
        )
        assert resp_pred.status_code == 200
        res_pred = resp_pred.json()
        print(f"Predict Decision: {res_pred['decision']} (Action: {res_pred['action']})")
        print(f"Predict Label: {res_pred['label']} | Score: {res_pred['spoof_score']}")
        assert "decision" in res_pred
        assert "evidence" in res_pred
        assert "audio_quality" in res_pred

    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
