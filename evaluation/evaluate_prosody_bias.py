"""evaluate_prosody_bias.py — 2x2 Matrix Prosody vs Content Evaluation for VoiceGuard.

Evaluates whether model score elevations are correlated with:
  1. Delivery/prosody (calm vs. urgent/demanding)
  2. Spoken content (neutral script vs. money-transfer demand)
  3. Both factors interactively
  4. Inconclusive / dominated by noise or RMS energy variance

Methodology & Consistency Constraints:
  All 4 audio files MUST be recorded:
  - Using the exact same device (e.g. phone voice recorder app)
  - At the same microphone distance (~15–20 cm)
  - In the same room and acoustic environment
  - Within the same recording session
  - Varying ONLY vocal delivery (calm vs. urgent) and script text (neutral vs. transfer)

Usage:
  python evaluation/evaluate_prosody_bias.py \\
      --calm_neutral path/to/calm_neutral.wav \\
      --calm_transfer path/to/calm_transfer.wav \\
      --urgent_neutral path/to/urgent_neutral.wav \\
      --urgent_transfer path/to/urgent_transfer.wav
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import librosa

from voiceguard_config import load_config
from voiceguard.model_backends import create_backend
from audio_quality import analyze_audio_quality


SCRIPTS = {
    "neutral": "The weather is pleasant today and the sun is shining brightly.",
    "transfer": "Please transfer two lakh rupees to this account immediately.",
}


def evaluate_clip(
    file_path: str,
    backend_obj: Any,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """Load an audio file, assess quality, and compute raw model scores."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Load and resample to 16kHz mono
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)

    # Audio quality metrics
    quality = analyze_audio_quality(audio, sample_rate)

    # Model inference
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
        "risk_level": "high" if pred.prob_fake >= 0.85 else ("medium" if pred.prob_fake >= 0.30 else "low"),
    }


def analyze_matrix(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute deltas and statistical tendencies across delivery vs content."""
    cn = results["calm_neutral"]["prob_fake"]
    ct = results["calm_transfer"]["prob_fake"]
    un = results["urgent_neutral"]["prob_fake"]
    ut = results["urgent_transfer"]["prob_fake"]

    # Delivery effect (holding content constant)
    delivery_effect_neutral = un - cn
    delivery_effect_transfer = ut - ct
    mean_delivery_effect = (delivery_effect_neutral + delivery_effect_transfer) / 2.0

    # Content effect (holding delivery constant)
    content_effect_calm = ct - cn
    content_effect_urgent = ut - un
    mean_content_effect = (content_effect_calm + content_effect_urgent) / 2.0

    # Classify outcome without presupposition
    # Threshold for noticeable shift: 0.15 (15% prob_fake delta)
    threshold = 0.15
    is_delivery_dominant = mean_delivery_effect >= threshold and mean_delivery_effect > (mean_content_effect + 0.10)
    is_content_dominant = mean_content_effect >= threshold and mean_content_effect > (mean_delivery_effect + 0.10)
    is_both = (mean_delivery_effect >= threshold) and (mean_content_effect >= threshold)
    is_inconclusive = not (is_delivery_dominant or is_content_dominant or is_both)

    if is_delivery_dominant:
        conclusion = "Score elevation correlates primarily with delivery/prosody (vocal strain/urgency), not semantic content."
    elif is_content_dominant:
        conclusion = "Score elevation correlates primarily with semantic phrasing/content, not delivery."
    elif is_both:
        conclusion = "Score elevation correlates with both delivery urgency and semantic content."
    else:
        conclusion = "Inconclusive or noise-dominated: Neither factor produced a consistent >=15% score separation."

    return {
        "mean_delivery_effect": round(mean_delivery_effect, 4),
        "mean_content_effect": round(mean_content_effect, 4),
        "delivery_effect_neutral": round(delivery_effect_neutral, 4),
        "delivery_effect_transfer": round(delivery_effect_transfer, 4),
        "content_effect_calm": round(content_effect_calm, 4),
        "content_effect_urgent": round(content_effect_urgent, 4),
        "conclusion": conclusion,
    }


def print_report(results: Dict[str, Dict[str, Any]], analysis: Dict[str, Any]) -> None:
    """Print formatted 2x2 table and findings."""
    print("=" * 80)
    print("  VoiceGuard — 2x2 Prosody vs. Content Sensitivity Matrix")
    print("=" * 80)
    print()
    print(f"{'Condition':<25} | {'Score (prob_fake)':<18} | {'Risk Level':<10} | {'RMS Energy':<10} | {'Voiced Ratio'}")
    print("-" * 80)

    order = [
        ("calm_neutral", "Calm + Neutral Script"),
        ("calm_transfer", "Calm + Transfer Script"),
        ("urgent_neutral", "Urgent + Neutral Script"),
        ("urgent_transfer", "Urgent + Transfer Script"),
    ]

    for key, label in order:
        r = results[key]
        print(f"{label:<25} | {r['prob_fake']:<18.4f} | {r['risk_level']:<10} | {r['rms']:<10.4f} | {r['voiced_ratio']:.2f}")

    print("-" * 80)
    print()
    print("2x2 Summary Matrix (prob_fake):")
    print(f"                      | Neutral Script        | Transfer Script")
    print(f"  Calm Delivery       | {results['calm_neutral']['prob_fake']:.4f}                | {results['calm_transfer']['prob_fake']:.4f}")
    print(f"  Urgent Delivery     | {results['urgent_neutral']['prob_fake']:.4f}                | {results['urgent_transfer']['prob_fake']:.4f}")
    print()
    print("Acoustic / Prosodic vs. Content Effects:")
    print(f"  Mean Delivery Effect (Urgent - Calm):  {analysis['mean_delivery_effect']:+.4f}")
    print(f"  Mean Content Effect (Transfer - Neut): {analysis['mean_content_effect']:+.4f}")
    print()
    print(f"Conclusion: {analysis['conclusion']}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate 2x2 prosody vs content matrix")
    parser.add_argument("--calm_neutral", type=str, default="evaluation/data/calm_neutral.wav",
                        help="Path to calm delivery + neutral script WAV")
    parser.add_argument("--calm_transfer", type=str, default="evaluation/data/calm_transfer.wav",
                        help="Path to calm delivery + money-transfer script WAV")
    parser.add_argument("--urgent_neutral", type=str, default="evaluation/data/urgent_neutral.wav",
                        help="Path to urgent delivery + neutral script WAV")
    parser.add_argument("--urgent_transfer", type=str, default="evaluation/data/urgent_transfer.wav",
                        help="Path to urgent delivery + money-transfer script WAV")
    parser.add_argument("--output", type=str, default="evaluation/prosody_bias_results.json",
                        help="Path to save output JSON")
    args = parser.parse_args()

    files = {
        "calm_neutral": args.calm_neutral,
        "calm_transfer": args.calm_transfer,
        "urgent_neutral": args.urgent_neutral,
        "urgent_transfer": args.urgent_transfer,
    }

    # Check if files exist
    missing = [k for k, p in files.items() if not os.path.exists(p)]
    if missing:
        print("=" * 80)
        print("  Notice: Pre-recorded audio files not yet found for:")
        for m in missing:
            print(f"    - {m}: {files[m]}")
        print()
        print("  Instructions for developer (Record on phone voice recorder app):")
        print("  1. Device: Same phone, in the same room, ~15-20cm mic distance.")
        print("  2. Record 4 clips in one session:")
        print(f"     a. calm_neutral.wav   : Read calmly: \"{SCRIPTS['neutral']}\"")
        print(f"     b. calm_transfer.wav  : Read calmly: \"{SCRIPTS['transfer']}\"")
        print(f"     c. urgent_neutral.wav : Read urgently/loudly: \"{SCRIPTS['neutral']}\"")
        print(f"     d. urgent_transfer.wav: Read urgently/loudly: \"{SCRIPTS['transfer']}\"")
        print("  3. Place them in evaluation/data/ or provide their file paths via CLI.")
        print("=" * 80)
        sys.exit(2)

    # Load active configuration & model
    cfg = load_config()
    print(f"Loading active backend '{cfg.active_model}' ...")
    backend_obj = create_backend(
        backend_name=cfg.active_backend.backend,
        model_id=cfg.active_backend.model_id,
        revision=cfg.active_backend.revision,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        score_type=cfg.active_backend.score_type,
    )
    backend_obj.load()

    # Evaluate each clip
    results = {}
    for key, path in files.items():
        results[key] = evaluate_clip(path, backend_obj, cfg.audio.sample_rate)

    analysis = analyze_matrix(results)
    print_report(results, analysis)

    # Save results
    output_data = {
        "model": cfg.active_model,
        "model_id": cfg.active_backend.model_id,
        "scripts": SCRIPTS,
        "results": results,
        "analysis": analysis,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
