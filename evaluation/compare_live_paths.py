"""compare_live_paths.py — 3-way live pipeline comparison script.

Executes and compares:
1. POST /predict with test_clip.wav
2. POST /live/analyze with sequential chunks
3. Simulated physical speakerphone acoustic capture over /live/analyze

Outputs raw JSON responses, scores, decisions, and actions to evaluation/live_path_comparison.json.
"""

from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import wave
import numpy as np
import scipy.signal as signal
from fastapi.testclient import TestClient

import backend
from backend import app, _state
from risk_aggregator import RiskAggregator, CallAction, CallDecision, ClipEvidence


def simulate_speakerphone_acoustics(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulate mobile speakerphone transducer and over-the-air transmission.

    1. Bandpass filter (300 Hz - 3500 Hz typical phone loudspeaker).
    2. Subtle transducer soft-clipping / saturation.
    3. Ambient acoustic room noise (-32 dB SNR).
    """
    sos = signal.butter(4, [300, 3500], btype="bandpass", fs=sr, output="sos")
    filtered = signal.sosfilt(sos, audio)

    # Transducer saturation (tanh soft clip)
    saturated = np.tanh(filtered * 1.3) / 1.3

    # Ambient room acoustic noise
    noise = np.random.normal(0, 0.008, size=len(saturated))
    degraded = saturated + noise

    # Normalize to prevent overflow
    peak = np.max(np.abs(degraded))
    if peak > 1.0:
        degraded = degraded / peak

    return degraded.astype(np.float32)


def audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Convert float32 numpy array to 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        int_samples = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


def run_comparison(audio_path: str = "test_clip.wav") -> dict:
    print(f"[Comparison] Loading audio: {audio_path}")
    with wave.open(audio_path, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_frames = wf.readframes(n_frames)
        samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0

    print(f"[Comparison] Audio loaded: {len(samples)} samples ({len(samples)/sr:.2f}s) at {sr}Hz")

    with TestClient(app, raise_server_exceptions=False) as client:
        backend._rate_limit_store.clear()

        active_backend = _state["backend"].backend_name if _state.get("backend") else "unknown"
        results = {
            "audio_source": audio_path,
            "sample_rate": sr,
            "duration_seconds": len(samples) / sr,
            "active_backend": active_backend,
            "paths": {},
        }

        # =====================================================================
        # Path A: POST /predict (Full file upload)
        # =====================================================================
        print("\n--- Running Path A: POST /predict ---")
        with open(audio_path, "rb") as f:
            file_bytes = f.read()

        resp_predict = client.post(
            "/predict",
            files={"file": ("test_clip.wav", file_bytes, "audio/wav")},
        )
        raw_predict = resp_predict.json() if resp_predict.status_code == 200 else {"error": resp_predict.text}
        results["paths"]["path_a_predict"] = {
            "status_code": resp_predict.status_code,
            "response": raw_predict,
        }
        print(f"Path A: Status={resp_predict.status_code}, Score={raw_predict.get('spoof_score')}, Decision={raw_predict.get('decision')}, Action={raw_predict.get('action')}")

        # =====================================================================
        # Path B: POST /live/analyze (Clean sequential chunks)
        # =====================================================================
        print("\n--- Running Path B: POST /live/analyze (Clean Chunks) ---")
        # Use 4-second windows to match the live-analysis contract
        # (model_config.yaml live_analysis.full_window_ms = 4000)
        chunk_size = sr * 4   # FULL_WINDOW_MS = 4000 ms
        step_size = sr * 2    # STRIDE_MS = 2000 ms
        aggregator_b = RiskAggregator()

        chunks_b = []
        chunk_idx = 0
        for start in range(0, max(1, len(samples) - chunk_size + 1), step_size):
            chunk_samples = samples[start:start + chunk_size]
            chunk_wav = audio_to_wav_bytes(chunk_samples, sr)

            resp_live = client.post(
                "/live/analyze",
                files={"file": (f"chunk_{chunk_idx}.wav", chunk_wav, "audio/wav")},
            )
            data = resp_live.json()
            evidence = ClipEvidence(
                spoof_score=data.get("spoof_score", 0.0),
                speech_detected=data.get("speech_detected", True),
                audio_quality_status=data.get("audio_quality", {}).get("status", "acceptable"),
                label=data.get("label", "unknown"),
            )
            agg_state = aggregator_b.update(evidence)

            chunks_b.append({
                "chunk_index": chunk_idx,
                "window_start_sec": start / sr,
                "status_code": resp_live.status_code,
                "live_response": data,
                "aggregator_state": {
                    "decision": agg_state.decision.value,
                    "action": agg_state.action.value,
                    "spoof_score": agg_state.current_score,
                    "ema_score": agg_state.ema_score,
                    "consecutive_high_windows": agg_state.consecutive_high_windows,
                    "reason_codes": agg_state.reason_codes,
                },
            })
            print(f"Chunk {chunk_idx}: Score={data.get('spoof_score', 0):.4f}, RawDecision={data.get('decision')}, AggAction={agg_state.action.value}, EMA={agg_state.ema_score:.4f}")
            chunk_idx += 1

        results["paths"]["path_b_live_chunks"] = {
            "total_chunks": len(chunks_b),
            "chunks": chunks_b,
            "final_aggregator_decision": aggregator_b.state.decision.value,
            "final_aggregator_action": aggregator_b.state.action.value,
        }

        # =====================================================================
        # Path C: POST /live/analyze (Simulated physical speakerphone capture)
        # =====================================================================
        print("\n--- Running Path C: POST /live/analyze (Speakerphone Simulation) ---")
        speakerphone_samples = simulate_speakerphone_acoustics(samples, sr)
        aggregator_c = RiskAggregator()

        chunks_c = []
        chunk_idx = 0
        for start in range(0, max(1, len(speakerphone_samples) - chunk_size + 1), step_size):
            chunk_samples = speakerphone_samples[start:start + chunk_size]
            chunk_wav = audio_to_wav_bytes(chunk_samples, sr)

            resp_live = client.post(
                "/live/analyze",
                files={"file": (f"speakerphone_chunk_{chunk_idx}.wav", chunk_wav, "audio/wav")},
            )
            data = resp_live.json()
            evidence = ClipEvidence(
                spoof_score=data.get("spoof_score", 0.0),
                speech_detected=data.get("speech_detected", True),
                audio_quality_status=data.get("audio_quality", {}).get("status", "acceptable"),
                label=data.get("label", "unknown"),
            )
            agg_state = aggregator_c.update(evidence)

            chunks_c.append({
                "chunk_index": chunk_idx,
                "window_start_sec": start / sr,
                "status_code": resp_live.status_code,
                "live_response": data,
                "aggregator_state": {
                    "decision": agg_state.decision.value,
                    "action": agg_state.action.value,
                    "spoof_score": agg_state.current_score,
                    "ema_score": agg_state.ema_score,
                    "consecutive_high_windows": agg_state.consecutive_high_windows,
                    "reason_codes": agg_state.reason_codes,
                },
            })
            print(f"Speakerphone Chunk {chunk_idx}: Score={data.get('spoof_score', 0):.4f}, RawDecision={data.get('decision')}, AggAction={agg_state.action.value}, EMA={agg_state.ema_score:.4f}")
            chunk_idx += 1

        results["paths"]["path_c_speakerphone_live"] = {
            "total_chunks": len(chunks_c),
            "chunks": chunks_c,
            "final_aggregator_decision": aggregator_c.state.decision.value,
            "final_aggregator_action": aggregator_c.state.action.value,
        }

        # Save to json
        os.makedirs("evaluation", exist_ok=True)
        out_path = "evaluation/live_path_comparison.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\n[Comparison] Results saved to {out_path}")
        return results


if __name__ == "__main__":
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_clip.wav"
    run_comparison(audio_file)
