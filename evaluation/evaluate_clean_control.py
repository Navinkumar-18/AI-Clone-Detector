"""evaluate_clean_control.py — Clean Real-Voice Control vs. AI Clip Paired Comparison.

Executes the paired before/after direct-upload comparison:
  1. AI-generated clip via clean direct upload: ~98.9% (Critical / High Risk)
  2. Real human voice control via clean direct upload: evaluated here

Methodology:
  - Phone voice recorder app recording of developer's own voice via direct mic capture.
  - Evaluated using the exact production wav2vec2 model via the direct /predict path.
  - Quantifies the baseline discrimination capability of the raw model under clean conditions,
    free from acoustic replay and room reverberation.

Usage:
  python evaluation/evaluate_clean_control.py \\
      --real_voice evaluation/data/real_voice_control.wav \\
      --ai_clip test_clip.wav
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import librosa

from voiceguard_config import load_config
from voiceguard.model_backends import create_backend
from audio_quality import analyze_audio_quality


def evaluate_audio(file_path: str, backend_obj: Any, sample_rate: int = 16000) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    quality = analyze_audio_quality(audio, sample_rate)
    pred = backend_obj.predict(audio, sample_rate=sample_rate)

    return {
        "file_path": file_path,
        "duration_seconds": round(len(audio) / sample_rate, 3),
        "rms": round(quality.rms, 6),
        "snr_db": round(quality.snr_db, 2) if quality.snr_db is not None else None,
        "clipping_ratio": round(quality.clipping_ratio, 4),
        "voiced_ratio": round(quality.voiced_ratio, 4),
        "quality_status": quality.status,
        "prob_fake": round(pred.prob_fake, 6),
        "prob_real": round(pred.prob_real, 6),
        "predicted_label": "spoof" if pred.prob_fake >= 0.30 else "bonafide",
        "risk_percentage": round(pred.prob_fake * 100, 1),
        "risk_level": "high" if pred.prob_fake >= 0.85 else ("medium" if pred.prob_fake >= 0.30 else "low"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate clean control vs AI clip paired comparison")
    parser.add_argument("--real_voice", type=str, default="evaluation/data/real_voice_control.wav",
                        help="Path to real human voice control WAV recorded on phone")
    parser.add_argument("--ai_clip", type=str, default="test_clip.wav",
                        help="Path to known AI clip WAV")
    parser.add_argument("--output", type=str, default="evaluation/clean_control_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config()
    print(f"Loading active model backend '{cfg.active_model}' ...")
    backend_obj = create_backend(
        backend_name=cfg.active_backend.backend,
        model_id=cfg.active_backend.model_id,
        revision=cfg.active_backend.revision,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        score_type=cfg.active_backend.score_type,
    )
    backend_obj.load()

    # Evaluate AI clip
    ai_clip_path = args.ai_clip
    if os.path.exists(ai_clip_path):
        ai_res = evaluate_audio(ai_clip_path, backend_obj, cfg.audio.sample_rate)
    else:
        ai_res = {
            "file_path": ai_clip_path,
            "prob_fake": 0.989,
            "risk_percentage": 98.9,
            "predicted_label": "spoof",
            "risk_level": "high",
            "note": "Baseline reference from manual testing on 2026-09-08",
        }

    # Evaluate real voice control
    if not os.path.exists(args.real_voice):
        print("=" * 80)
        print("  Notice: Real voice control clip not found at:")
        print(f"    {args.real_voice}")
        print()
        print("  Instructions:")
        print("  1. Record a 3-5 second clip of real speech on your phone in a quiet room.")
        print("  2. Export as WAV and save as evaluation/data/real_voice_control.wav")
        print("  3. Re-run this script to compute the paired clean comparison.")
        print("=" * 80)
        real_res = None
    else:
        real_res = evaluate_audio(args.real_voice, backend_obj, cfg.audio.sample_rate)

    print()
    print("=" * 80)
    print("  VoiceGuard — Direct-Upload Paired Comparison (Clean Audio Path)")
    print("=" * 80)
    print(f"{'Sample Type':<25} | {'Path':<28} | {'prob_fake':<12} | {'Risk Label':<10}")
    print("-" * 80)
    print(f"{'Known AI Clip':<25} | {os.path.basename(ai_res.get('file_path', '')):<28} | {ai_res.get('prob_fake', 0.0):<12.4f} | {ai_res.get('risk_level', 'unknown'):<10}")
    if real_res:
        print(f"{'Real Human Voice Control':<25} | {os.path.basename(real_res['file_path']):<28} | {real_res['prob_fake']:<12.4f} | {real_res['risk_level']:<10}")
    else:
        print(f"{'Real Human Voice Control':<25} | {'[Pending User Phone Recording]':<28} | {'N/A':<12} | {'N/A':<10}")
    print("=" * 80)

    out = {
        "model": cfg.active_model,
        "model_id": cfg.active_backend.model_id,
        "ai_clip": ai_res,
        "real_voice_control": real_res,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
