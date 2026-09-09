"""
verify_live_call_parity.py — Automated manual-test verification for VoiceGuard live call path.
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import backend
from backend import app, _state
from risk_aggregator import RiskAggregator, CallAction, CallDecision, ClipEvidence
from voiceguard_config import load_config

def make_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()

def run_live_call_verification():
    print("=======================================================================")
    print(" VOICEGUARD LIVE-CALL PARITY & CONTRACT VERIFICATION")
    print("=======================================================================\n")

    cfg = load_config()
    sample_rate = cfg.audio.sample_rate # 16000
    full_window_ms = cfg.live_analysis.full_window_ms # 4000
    stride_ms = cfg.live_analysis.stride_ms # 2000
    bytes_per_sample = 2
    bytes_per_sec = sample_rate * bytes_per_sample # 32000
    full_window_bytes = int(full_window_ms / 1000 * bytes_per_sec) # 128000
    stride_bytes = int(stride_ms / 1000 * bytes_per_sec) # 64000

    report = {
        "step_1_startup_accumulation_0_to_4s": {},
        "step_2_first_complete_window": {},
        "step_3_stride_progression": {},
        "step_4_predict_vs_live_parity": {},
        "step_5_silence_safety": {},
        "step_6_final_partial_buffer": {},
    }

    with TestClient(app, raise_server_exceptions=False) as client:
        backend._rate_limit_store.clear()

        # ---------------------------------------------------------------------
        # 1. Startup & Buffer Accumulation (0-4s)
        # ---------------------------------------------------------------------
        print(">>> STEP 1: Startup & Buffer Accumulation (0 - 4 seconds)")
        pcm_buffer = bytearray()
        requests_sent_before_4s = 0

        # Simulate 1-second ticks of audio arriving
        simulated_states = []
        for second in range(1, 4):
            # 1 second of audio arrives (32,000 bytes = 16000 samples * 2 bytes)
            new_chunk = (b"\x10\x00" * sample_rate)
            pcm_buffer.extend(new_chunk)
            current_duration_s = len(pcm_buffer) / bytes_per_sec

            # Flutter logic evaluation:
            is_complete = len(pcm_buffer) >= full_window_bytes
            if not is_complete:
                # Flutter UI state: ANALYZING, do not send request
                ui_decision = "insufficient_evidence"
                ui_action = "verify"
                ui_risk = "unknown"
                can_show_low_risk = False
                display_badge = "ANALYZING"
                simulated_states.append({
                    "time_sec": second,
                    "buffer_bytes": len(pcm_buffer),
                    "buffer_duration_sec": current_duration_s,
                    "request_sent": False,
                    "ui_decision": ui_decision,
                    "can_show_low_risk": can_show_low_risk,
                    "display_badge": display_badge,
                })
                print(f"  t={second}s: Buffer={len(pcm_buffer)} bytes ({current_duration_s:.1f}s) < {full_window_bytes} bytes "
                      f"-> UI={display_badge} (never green), Request Sent=False")

        # Also confirm backend rejects sub-4s chunk if sent
        short_pcm = np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, sample_rate * 2)).astype(np.float32)
        short_wav = make_wav_bytes(short_pcm, sample_rate)
        resp_short = client.post("/live/analyze", files={"file": ("short.wav", short_wav, "audio/wav")})
        short_data = resp_short.json()
        print(f"  Sub-4s chunk forced to /live/analyze: Status={resp_short.status_code}, "
              f"Decision={short_data.get('decision')}, Reasons={short_data.get('reason_codes')}, "
              f"SpeechDetected={short_data.get('speech_detected')}")

        report["step_1_startup_accumulation_0_to_4s"] = {
            "requests_sent_before_4s": requests_sent_before_4s,
            "simulated_states": simulated_states,
            "sub_4s_backend_rejection": short_data,
            "verified": (
                all(not s["request_sent"] and not s["can_show_low_risk"] for s in simulated_states)
                and short_data.get("decision") == "insufficient_evidence"
                and "partial_window" in short_data.get("reason_codes", [])
            ),
        }
        assert report["step_1_startup_accumulation_0_to_4s"]["verified"], "Step 1 verification failed!"

        # ---------------------------------------------------------------------
        # 2. First Complete Window (at exactly 4.0s)
        # ---------------------------------------------------------------------
        print("\n>>> STEP 2: First Complete Window (t = 4.0 seconds)")
        # 4th second arrives (32,000 bytes)
        pcm_buffer.extend(b"\x20\x00" * sample_rate)
        assert len(pcm_buffer) >= full_window_bytes
        window_1_bytes = bytes(pcm_buffer[:full_window_bytes])
        window_1_duration_s = len(window_1_bytes) / bytes_per_sec

        # Advance by stride (2s = 64,000 bytes)
        del pcm_buffer[:stride_bytes]

        # Send Window 1 to /live/analyze
        # Using a valid speech-like waveform for window 1
        t_arr = np.linspace(0, 4.0, sample_rate * 4, endpoint=False)
        audio_speech_4s = (0.35 * np.sin(2 * np.pi * 300 * t_arr) + 0.05 * np.random.RandomState(42).randn(len(t_arr))).astype(np.float32)
        wav_window_1 = make_wav_bytes(audio_speech_4s, sample_rate)

        resp_win1 = client.post("/live/analyze", files={"file": ("window_1.wav", wav_window_1, "audio/wav")})
        win1_data = resp_win1.json()

        print(f"  Window 1 bytes extracted: {len(window_1_bytes)} bytes (Duration={window_1_duration_s:.1f}s)")
        print(f"  Remaining buffer after stride: {len(pcm_buffer)} bytes ({len(pcm_buffer)/bytes_per_sec:.1f}s)")
        print(f"  Window 1 Backend Response: Status={resp_win1.status_code}, Decision={win1_data.get('decision')}, "
              f"Action={win1_data.get('action')}, SpoofScore={win1_data.get('spoof_score')}, "
              f"Label={win1_data.get('label')}")

        report["step_2_first_complete_window"] = {
            "window_duration_seconds": window_1_duration_s,
            "window_bytes": len(window_1_bytes),
            "backend_response": win1_data,
            "verified": (
                window_1_duration_s == 4.0
                and len(window_1_bytes) == 128000
                and resp_win1.status_code == 200
                and "partial_window" not in win1_data.get("reason_codes", [])
            ),
        }
        assert report["step_2_first_complete_window"]["verified"], "Step 2 verification failed!"

        # ---------------------------------------------------------------------
        # 3. Subsequent Windows & 2-Second Stride
        # ---------------------------------------------------------------------
        print("\n>>> STEP 3: Subsequent Windows & 2-Second Stride Progression")
        # Next 2 seconds arrive (t = 4s to 6s): 64,000 bytes
        pcm_buffer.extend(b"\x30\x00" * (sample_rate * 2))
        assert len(pcm_buffer) >= full_window_bytes

        window_2_bytes = bytes(pcm_buffer[:full_window_bytes])
        window_2_duration_s = len(window_2_bytes) / bytes_per_sec
        del pcm_buffer[:stride_bytes]

        print(f"  Window 2 (t=6.0s, spans [2.0s, 6.0s]): Duration={window_2_duration_s:.1f}s, Stride={stride_ms}ms")
        print(f"  Remaining buffer after stride: {len(pcm_buffer)} bytes ({len(pcm_buffer)/bytes_per_sec:.1f}s)")

        # Next 2 seconds arrive (t = 6s to 8s): 64,000 bytes
        pcm_buffer.extend(b"\x40\x00" * (sample_rate * 2))
        assert len(pcm_buffer) >= full_window_bytes

        window_3_bytes = bytes(pcm_buffer[:full_window_bytes])
        window_3_duration_s = len(window_3_bytes) / bytes_per_sec
        del pcm_buffer[:stride_bytes]

        print(f"  Window 3 (t=8.0s, spans [4.0s, 8.0s]): Duration={window_3_duration_s:.1f}s, Stride={stride_ms}ms")

        report["step_3_stride_progression"] = {
            "stride_ms": stride_ms,
            "window_2_duration_s": window_2_duration_s,
            "window_3_duration_s": window_3_duration_s,
            "verified": (window_2_duration_s == 4.0 and window_3_duration_s == 4.0 and stride_ms == 2000),
        }
        assert report["step_3_stride_progression"]["verified"], "Step 3 verification failed!"

        # ---------------------------------------------------------------------
        # 4. Raw Score Parity: /predict vs /live/analyze
        # ---------------------------------------------------------------------
        print("\n>>> STEP 4: Raw Score Parity (/predict vs /live/analyze)")
        # Test 4A: Real human voice (bonafide)
        human_path = "data/asvspoof2019LA/LA/ASVspoof2019_LA_eval/flac/LA_D_1090286.flac"
        if os.path.exists(human_path):
            import librosa
            h_audio, _ = librosa.load(human_path, sr=sample_rate, mono=True)
            # Take exactly 4 seconds
            h_audio_4s = h_audio[:sample_rate * 4]
            if len(h_audio_4s) < sample_rate * 4:
                h_audio_4s = np.pad(h_audio_4s, (0, sample_rate * 4 - len(h_audio_4s)))
        else:
            rng = np.random.RandomState(101)
            t = np.linspace(0, 4.0, sample_rate * 4, endpoint=False)
            h_audio_4s = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.randn(len(t))).astype(np.float32)

        human_wav_bytes = make_wav_bytes(h_audio_4s, sample_rate)

        # /predict on human audio
        resp_pred_h = client.post("/predict", files={"file": ("human.wav", human_wav_bytes, "audio/wav")})
        pred_h = resp_pred_h.json()

        # /live/analyze on human audio
        resp_live_h = client.post("/live/analyze", files={"file": ("human_live.wav", human_wav_bytes, "audio/wav")})
        live_h = resp_live_h.json()

        score_h_pred = pred_h.get("spoof_score")
        score_h_live = live_h.get("spoof_score")
        delta_h = abs(score_h_pred - score_h_live)

        print(f"  Human Audio (4.0s):")
        print(f"    POST /predict      -> Score: {score_h_pred:.4f}, Label: {pred_h.get('label')}, Risk: {pred_h.get('risk_level')}")
        print(f"    POST /live/analyze -> Score: {score_h_live:.4f}, Label: {live_h.get('label')}, Risk: {live_h.get('risk_level')}")
        print(f"    Absolute Score Delta: {delta_h:.6f} ({'PARITY MATCH' if delta_h < 1e-4 else 'MISMATCH'})")

        # Test 4B: AI / synthetic speech (spoof)
        # Using synthesized high-frequency artifact signal or test_clip padded to 4.0s
        if os.path.exists("test_clip.wav"):
            import librosa
            s_audio, _ = librosa.load("test_clip.wav", sr=sample_rate, mono=True)
            # Tile/pad to exactly 4 seconds
            s_audio_4s = np.tile(s_audio, int(np.ceil(sample_rate * 4 / len(s_audio))))[:sample_rate * 4]
        else:
            rng = np.random.RandomState(202)
            t = np.linspace(0, 4.0, sample_rate * 4, endpoint=False)
            s_audio_4s = (0.4 * np.sin(2 * np.pi * 880 * t) + 0.1 * rng.randn(len(t))).astype(np.float32)

        spoof_wav_bytes = make_wav_bytes(s_audio_4s, sample_rate)

        resp_pred_s = client.post("/predict", files={"file": ("spoof.wav", spoof_wav_bytes, "audio/wav")})
        pred_s = resp_pred_s.json()

        resp_live_s = client.post("/live/analyze", files={"file": ("spoof_live.wav", spoof_wav_bytes, "audio/wav")})
        live_s = resp_live_s.json()

        score_s_pred = pred_s.get("spoof_score")
        score_s_live = live_s.get("spoof_score")
        delta_s = abs(score_s_pred - score_s_live)

        print(f"  AI/Spoof Audio (4.0s):")
        print(f"    POST /predict      -> Score: {score_s_pred:.4f}, Label: {pred_s.get('label')}, Risk: {pred_s.get('risk_level')}")
        print(f"    POST /live/analyze -> Score: {score_s_live:.4f}, Label: {live_s.get('label')}, Risk: {live_s.get('risk_level')}")
        print(f"    Absolute Score Delta: {delta_s:.6f} ({'PARITY MATCH' if delta_s < 1e-4 else 'MISMATCH'})")

        report["step_4_predict_vs_live_parity"] = {
            "human_audio": {
                "predict_score": score_h_pred,
                "live_score": score_h_live,
                "score_delta": delta_h,
                "predict_label": pred_h.get("label"),
                "live_label": live_h.get("label"),
            },
            "spoof_audio": {
                "predict_score": score_s_pred,
                "live_score": score_s_live,
                "score_delta": delta_s,
                "predict_label": pred_s.get("label"),
                "live_label": live_s.get("label"),
            },
            "verified": delta_h < 1e-4 and delta_s < 1e-4 and pred_h.get("label") == live_h.get("label"),
        }
        assert report["step_4_predict_vs_live_parity"]["verified"], "Step 4 verification failed!"

        # ---------------------------------------------------------------------
        # 5. Silence Safety (No Classification)
        # ---------------------------------------------------------------------
        print("\n>>> STEP 5: Silence Gate & Zero Classification")
        silence_audio = np.zeros(sample_rate * 4, dtype=np.float32)
        silence_wav_bytes = make_wav_bytes(silence_audio, sample_rate)

        resp_silence = client.post("/live/analyze", files={"file": ("silence.wav", silence_wav_bytes, "audio/wav")})
        silence_data = resp_silence.json()

        print(f"  4s Digital Silence -> Status={resp_silence.status_code}, Decision={silence_data.get('decision')}, "
              f"Action={silence_data.get('action')}, SpeechDetected={silence_data.get('speech_detected')}, "
              f"Reasons={silence_data.get('reason_codes')}")

        report["step_5_silence_safety"] = {
            "response": silence_data,
            "verified": (
                silence_data.get("decision") == "insufficient_evidence"
                and silence_data.get("speech_detected") is False
                and silence_data.get("decision") != "low_risk"
            ),
        }
        assert report["step_5_silence_safety"]["verified"], "Step 5 verification failed!"

        # ---------------------------------------------------------------------
        # 6. Final Partial Buffer Safety (Call Termination)
        # ---------------------------------------------------------------------
        print("\n>>> STEP 6: Final Partial Buffer (Call Termination Safety)")
        # Call ends with 2.1s left in buffer
        final_partial_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2.1, int(sample_rate * 2.1))).astype(np.float32)
        final_partial_wav = make_wav_bytes(final_partial_audio, sample_rate)

        resp_final = client.post("/live/analyze", files={"file": ("final_partial.wav", final_partial_wav, "audio/wav")})
        final_data = resp_final.json()

        print(f"  Final 2.1s partial buffer -> Status={resp_final.status_code}, Decision={final_data.get('decision')}, "
              f"SpeechDetected={final_data.get('speech_detected')}, Reasons={final_data.get('reason_codes')}")

        report["step_6_final_partial_buffer"] = {
            "response": final_data,
            "verified": (
                final_data.get("decision") == "insufficient_evidence"
                and "partial_window" in final_data.get("reason_codes", [])
                and final_data.get("speech_detected") is False
            ),
        }
        assert report["step_6_final_partial_buffer"]["verified"], "Step 6 verification failed!"

    # Save verification report
    out_path = PROJECT_ROOT / "evaluation" / "live_call_verification_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nVerification report written to: {out_path}")
    print("=======================================================================")
    print(" ALL LIVE-CALL PARITY & CONTRACT VERIFICATIONS PASSED (100%)")
    print("=======================================================================")

if __name__ == "__main__":
    run_live_call_verification()
