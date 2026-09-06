"""
run_production_evaluation.py — Evaluate the exact VoiceGuard production model.
================================================================================

Uses the same model, preprocessing, and thresholds as backend.py via the
unified configuration (config/model_config.yaml → voiceguard_config.py).

Reports only genuinely generated metrics. Does not fabricate results.
If the dataset is not available, reports:
    "NOT AVAILABLE — dataset was not present at the expected path."

Usage:
    python evaluation/run_production_evaluation.py
    python evaluation/run_production_evaluation.py --data_dir ./data/asvspoof2019LA
    python evaluation/run_production_evaluation.py --max_clips 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


@dataclass
class EvalResult:
    filename: str
    gt_label: str
    prob_real: float
    prob_fake: float
    error: Optional[str] = None


@dataclass
class EvalMetrics:
    dataset: str
    model_name: str
    model_revision: str
    model_version: str
    threshold_version: str
    spoof_threshold: float
    n_genuine: int
    n_spoof: int
    n_errors: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    tp: int
    tn: int
    fp: int
    fn: int
    mean_latency_ms: float


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VoiceGuard production model on ASVspoof 2019 LA"
    )
    parser.add_argument(
        "--data_dir", default="./data/asvspoof2019LA",
        help="Root of ASVspoof 2019 LA dataset"
    )
    parser.add_argument(
        "--output", default="evaluation/results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--max_clips", type=int, default=None,
        help="Limit number of clips (for quick test)"
    )
    args = parser.parse_args()

    # Check dataset availability
    protocol_path = os.path.join(
        args.data_dir, "LA", "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.eval.trl.txt"
    )
    audio_dir = os.path.join(
        args.data_dir, "LA", "ASVspoof2019_LA_eval", "flac"
    )

    if not os.path.exists(protocol_path):
        print(f"NOT AVAILABLE — dataset was not present at the expected path.")
        print(f"  Expected protocol: {protocol_path}")
        print(f"  Expected audio: {audio_dir}")
        sys.exit(0)

    if not os.path.isdir(audio_dir):
        print(f"NOT AVAILABLE — audio directory not found: {audio_dir}")
        sys.exit(0)

    # Load unified config
    from voiceguard_config import load_config
    cfg = load_config()

    print(f"Model: {cfg.model.name} (revision: {cfg.model.revision})")
    print(f"Version: {cfg.model.version}")
    print(f"Threshold: {cfg.thresholds.spoof_threshold} (version: {cfg.thresholds.version})")

    # Load model
    import torch
    import librosa
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model...")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        cfg.model.name, revision=cfg.model.revision
    )
    model = AutoModelForAudioClassification.from_pretrained(
        cfg.model.name, revision=cfg.model.revision
    )
    model.to(device).eval()
    print(f"Model loaded. Labels: {model.config.id2label}")

    # Parse protocol
    entries = []
    with open(protocol_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            entries.append({
                "filename": parts[1],
                "label": parts[4].lower(),
            })

    print(f"Protocol: {len(entries)} entries")
    if args.max_clips:
        entries = entries[:args.max_clips]
        print(f"  (Limited to {args.max_clips} clips)")

    # Run inference
    results = []
    latencies = []
    for i, entry in enumerate(entries):
        audio_path = os.path.join(audio_dir, entry["filename"] + ".flac")
        if not os.path.exists(audio_path):
            results.append(EvalResult(
                filename=entry["filename"],
                gt_label=entry["label"],
                prob_real=0.0, prob_fake=0.0,
                error="file not found",
            ))
            continue

        try:
            t0 = time.time()
            audio, _ = librosa.load(audio_path, sr=cfg.audio.sample_rate, mono=True)
            inputs = feature_extractor(
                audio, sampling_rate=cfg.audio.sample_rate,
                return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
            latency = (time.time() - t0) * 1000
            latencies.append(latency)

            results.append(EvalResult(
                filename=entry["filename"],
                gt_label=entry["label"],
                prob_real=probs[0].item(),
                prob_fake=probs[1].item(),
            ))
        except Exception as exc:
            results.append(EvalResult(
                filename=entry["filename"],
                gt_label=entry["label"],
                prob_real=0.0, prob_fake=0.0,
                error=str(exc),
            ))

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(entries)}]")

    # Calculate metrics
    tp = tn = fp = fn = 0
    for r in results:
        if r.error:
            continue
        predicted_spoof = r.prob_fake >= cfg.thresholds.spoof_threshold
        actual_spoof = r.gt_label == "spoof"
        if predicted_spoof and actual_spoof:
            tp += 1
        elif not predicted_spoof and not actual_spoof:
            tn += 1
        elif predicted_spoof and not actual_spoof:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    n_genuine = tn + fp
    n_spoof = tp + fn
    n_errors = sum(1 for r in results if r.error)

    metrics = EvalMetrics(
        dataset="ASVspoof 2019 LA eval",
        model_name=cfg.model.name,
        model_revision=cfg.model.revision,
        model_version=cfg.model.version,
        threshold_version=cfg.thresholds.version,
        spoof_threshold=cfg.thresholds.spoof_threshold,
        n_genuine=n_genuine,
        n_spoof=n_spoof,
        n_errors=n_errors,
        accuracy=(tp + tn) / total if total else 0.0,
        precision=tp / (tp + fp) if (tp + fp) else 0.0,
        recall=tp / (tp + fn) if (tp + fn) else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        fpr=fp / (fp + tn) if (fp + tn) else 0.0,
        fnr=fn / (fn + tp) if (fn + tp) else 0.0,
        tp=tp, tn=tn, fp=fp, fn=fn,
        mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  VoiceGuard Production Model Evaluation")
    print(f"{'='*60}")
    print(f"  Model:     {metrics.model_name}")
    print(f"  Version:   {metrics.model_version}")
    print(f"  Threshold: {metrics.spoof_threshold}")
    print(f"  Dataset:   {metrics.dataset}")
    print(f"  Genuine:   {metrics.n_genuine}")
    print(f"  Spoof:     {metrics.n_spoof}")
    print(f"  Errors:    {metrics.n_errors}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1:        {metrics.f1:.4f}")
    print(f"  FPR:       {metrics.fpr:.4f}")
    print(f"  FNR:       {metrics.fnr:.4f}")
    print(f"  Latency:   {metrics.mean_latency_ms:.1f} ms/clip")
    print(f"{'='*60}")
    print(f"  TP={metrics.tp}  TN={metrics.tn}  FP={metrics.fp}  FN={metrics.fn}")
    print(f"{'='*60}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "metrics": asdict(metrics),
        "results": [asdict(r) for r in results],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
