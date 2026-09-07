"""
evaluation/run_experiment_evaluation.py
========================================
Executes a controlled, reproducible evaluation comparing:
  1. Wav2Vec2 Deepfake Detector (off-the-shelf checkpoint)
  2. WavLM-base + MLP Classifier (research pipeline checkpoint)

Uses the exact 250-clip fixed evaluation manifest (evaluation/manifest_250.json).
Outputs:
  - evaluation/results_wav2vec2.json
  - evaluation/results_wavlm_mlp.json
  - evaluation/results_wavlm_mlp.txt
  - evaluation/threshold_sweep_wav2vec2.csv
  - evaluation/threshold_sweep_wavlm_mlp.csv
  - evaluation/model_comparison.json
  - docs/MODEL_COMPARISON.md
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from sklearn.metrics import average_precision_score, roc_auc_score

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voiceguard.model_backends import Wav2Vec2Backend, WavLMMLPBackend, ModelPrediction


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the optimal threshold.
    y_true: 1 for spoof (positive class), 0 for bonafide.
    y_score: probability of spoof.
    """
    thresholds = np.linspace(0.0, 1.0, 1001)
    fpr_list = []
    fnr_list = []
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.5
        
    for th in thresholds:
        pred_pos = (y_score >= th)
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        fn = np.sum((pred_pos == 0) & (y_true == 1))
        fpr = fp / n_neg
        fnr = fn / n_pos
        fpr_list.append(fpr)
        fnr_list.append(fnr)
        
    fpr_arr = np.array(fpr_list)
    fnr_arr = np.array(fnr_list)
    diff = np.abs(fpr_arr - fnr_arr)
    idx = np.argmin(diff)
    eer = (fpr_arr[idx] + fnr_arr[idx]) / 2.0
    eer_th = thresholds[idx]
    return float(eer), float(eer_th)


def compute_distribution_stats(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    arr = np.array(scores)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def run_threshold_sweep(
    results: List[Dict[str, Any]],
    output_csv_path: str,
) -> List[Dict[str, Any]]:
    thresholds = [round(t, 2) for t in np.arange(0.01, 1.00, 0.01)]
    sweep_rows = []

    y_true = np.array([1 if r["label"] == "spoof" else 0 for r in results])
    y_scores = np.array([r["prob_fake"] for r in results])
    n_spoof = int(np.sum(y_true == 1))
    n_bonafide = int(np.sum(y_true == 0))

    for th in thresholds:
        pred_spoof = (y_scores >= th)
        tp = int(np.sum((pred_spoof == 1) & (y_true == 1)))
        tn = int(np.sum((pred_spoof == 0) & (y_true == 0)))
        fp = int(np.sum((pred_spoof == 1) & (y_true == 0)))
        fn = int(np.sum((pred_spoof == 0) & (y_true == 1)))

        fpr = fp / n_bonafide if n_bonafide > 0 else 0.0
        fnr = fn / n_spoof if n_spoof > 0 else 0.0
        accuracy = (tp + tn) / len(results) if results else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        balanced_acc = ((1.0 - fpr) + (1.0 - fnr)) / 2.0

        sweep_rows.append({
            "threshold": th,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(balanced_acc, 4),
        })

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    return sweep_rows


def evaluate_backend(
    backend,
    manifest: List[Dict[str, Any]],
    output_prefix: str,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Evaluating Backend: {backend.backend_name} ({backend.model_id})")
    print(f"=======================================================")

    t0_load = time.perf_counter()
    backend.load()
    load_time_s = time.perf_counter() - t0_load
    print(f"Backend loaded in {load_time_s:.2f}s")

    clip_results: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for idx, item in enumerate(manifest):
        flac_path = item["flac_relpath"]
        audio, sr = sf.read(flac_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        t0_inf = time.perf_counter()
        pred = backend.predict(audio, sample_rate=sr)
        inf_ms = (time.perf_counter() - t0_inf) * 1000.0

        latencies.append(inf_ms)
        clip_results.append({
            "filename": item["filename"],
            "speaker_id": item["speaker_id"],
            "system_id": item["system_id"],
            "label": item["label"],
            "target": item["target"],
            "prob_real": pred.prob_real,
            "prob_fake": pred.prob_fake,
            "score_type": pred.score_type,
            "latency_ms": round(inf_ms, 2),
            "error": None,
        })

        if (idx + 1) % 25 == 0 or idx == 0 or (idx + 1) == len(manifest):
            print(f"  [{idx + 1:>3}/{len(manifest)}] {item['filename']} -> prob_fake={pred.prob_fake:.4f} ({pred.inference_time_ms:.1f}ms)", flush=True)

    # Separate distributions
    bonafide_scores = [r["prob_fake"] for r in clip_results if r["label"] == "bonafide"]
    spoof_scores = [r["prob_fake"] for r in clip_results if r["label"] == "spoof"]

    bona_dist = compute_distribution_stats(bonafide_scores)
    spoof_dist = compute_distribution_stats(spoof_scores)

    # Threshold sweeps
    sweep_csv_path = f"evaluation/threshold_sweep_{backend.backend_name}.csv"
    sweep_data = run_threshold_sweep(clip_results, sweep_csv_path)

    # Metrics at inherited threshold 0.30
    th_030_match = [r for r in sweep_data if r["threshold"] == 0.30][0]

    # Global discrimination metrics
    y_true = np.array([1 if r["label"] == "spoof" else 0 for r in clip_results])
    y_scores = np.array([r["prob_fake"] for r in clip_results])

    roc_auc = float(roc_auc_score(y_true, y_scores))
    pr_auc = float(average_precision_score(y_true, y_scores))
    eer, eer_th = compute_eer(y_true, y_scores)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    metrics = {
        "backend_name": backend.backend_name,
        "model_id": backend.model_id,
        "revision": backend.revision,
        "score_type": backend.score_type,
        "inherited_threshold": 0.30,
        "metrics_at_030": th_030_match,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "eer": round(eer, 4),
        "eer_threshold": round(eer_th, 4),
        "latency_ms": {
            "mean": round(mean_latency, 2),
            "p95": round(p95_latency, 2),
        },
        "score_distributions": {
            "bonafide": bona_dist,
            "spoof": spoof_dist,
        },
    }

    # Save results JSON
    json_path = f"evaluation/results_{backend.backend_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": clip_results}, f, indent=2)

    # Save summary report text
    txt_path = f"evaluation/results_{backend.backend_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Report: {backend.backend_name}\n")
        f.write(f"Model ID: {backend.model_id}\n")
        f.write(f"Revision: {backend.revision}\n")
        f.write(f"Score Type: {backend.score_type}\n")
        f.write(f"Clips: {len(clip_results)} (50 bonafide, 200 spoof)\n\n")
        f.write(f"Metrics at Threshold 0.30:\n")
        f.write(f"  FPR (False Positive Rate): {th_030_match['fpr'] * 100:.1f}%\n")
        f.write(f"  FNR (False Negative Rate): {th_030_match['fnr'] * 100:.1f}%\n")
        f.write(f"  Accuracy: {th_030_match['accuracy'] * 100:.1f}%\n")
        f.write(f"  Balanced Accuracy: {th_030_match['balanced_accuracy'] * 100:.1f}%\n")
        f.write(f"  F1 Score: {th_030_match['f1']:.4f}\n")
        f.write(f"  Precision: {th_030_match['precision']:.4f}\n")
        f.write(f"  Recall: {th_030_match['recall']:.4f}\n\n")
        f.write(f"Discrimination:\n")
        f.write(f"  ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"  PR-AUC: {pr_auc:.4f}\n")
        f.write(f"  EER: {eer * 100:.2f}% at threshold {eer_th:.4f}\n\n")
        f.write(f"Score Distributions:\n")
        f.write(f"  Bona-fide (N=50) : min={bona_dist['min']:.4f}, mean={bona_dist['mean']:.4f}, median={bona_dist['median']:.4f}, max={bona_dist['max']:.4f}\n")
        f.write(f"  Spoof     (N=200): min={spoof_dist['min']:.4f}, mean={spoof_dist['mean']:.4f}, median={spoof_dist['median']:.4f}, max={spoof_dist['max']:.4f}\n\n")
        f.write(f"Latency:\n")
        f.write(f"  Mean: {mean_latency:.2f} ms\n")
        f.write(f"  P95 : {p95_latency:.2f} ms\n")

    return metrics


def main():
    manifest_path = "evaluation/manifest_250.json"
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Loaded manifest with {len(manifest)} clips.")

    # 1. Evaluate Wav2Vec2
    wav2vec2_backend = Wav2Vec2Backend(device="cpu")
    w2v2_metrics = evaluate_backend(wav2vec2_backend, manifest, "wav2vec2")

    # 2. Evaluate WavLM + MLP
    wavlm_backend = WavLMMLPBackend(checkpoint_path="models/best_mlp_wavlm_base.pt", device="cpu")
    wavlm_metrics = evaluate_backend(wavlm_backend, manifest, "wavlm_mlp")

    # 3. Build Comparison JSON
    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_path": manifest_path,
        "sample_count": len(manifest),
        "bonafide_count": 50,
        "spoof_count": 200,
        "models": {
            "wav2vec2": w2v2_metrics,
            "wavlm_mlp": wavlm_metrics,
        },
        "delta": {
            "fpr_030": round(wavlm_metrics["metrics_at_030"]["fpr"] - w2v2_metrics["metrics_at_030"]["fpr"], 4),
            "fnr_030": round(wavlm_metrics["metrics_at_030"]["fnr"] - w2v2_metrics["metrics_at_030"]["fnr"], 4),
            "roc_auc": round(wavlm_metrics["roc_auc"] - w2v2_metrics["roc_auc"], 4),
            "eer": round(wavlm_metrics["eer"] - w2v2_metrics["eer"], 4),
            "latency_mean_ms": round(wavlm_metrics["latency_ms"]["mean"] - w2v2_metrics["latency_ms"]["mean"], 2),
        }
    }

    comp_path = "evaluation/model_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nWrote comparison to {comp_path}")

    # 4. Generate docs/MODEL_COMPARISON.md
    md_path = "docs/MODEL_COMPARISON.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison: Wav2Vec2 vs. WavLM-base + MLP Classifier\n\n")
        f.write("## Overview\n\n")
        f.write("Controlled empirical comparison evaluated on the fixed 250-clip ASVspoof 2019 LA evaluation split (`evaluation/manifest_250.json`).\n\n")
        f.write("| Attribute | Wav2Vec2 (Current Baseline) | WavLM-base + MLP (Candidate) |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Backend identifier | `wav2vec2` | `wavlm_mlp` |\n")
        f.write(f"| Model / Backbone ID | `{w2v2_metrics['model_id']}` | `{wavlm_metrics['model_id']}` |\n")
        f.write(f"| Revision SHA | `{w2v2_metrics['revision'][:12]}` | `{wavlm_metrics['revision'][:12]}` |\n")
        f.write(f"| Score type | `{w2v2_metrics['score_type']}` | `{wavlm_metrics['score_type']}` |\n")
        f.write(f"| Checkpoint path | Pretrained weights | `models/best_mlp_wavlm_base.pt` |\n\n")

        f.write("## Performance at Inherited Threshold 0.30\n\n")
        f.write("| Metric | Wav2Vec2 | WavLM + MLP | Difference (WavLM - Wav2Vec2) |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| False Positive Rate (FPR) | {w2v2_metrics['metrics_at_030']['fpr']*100:.1f}% | {wavlm_metrics['metrics_at_030']['fpr']*100:.1f}% | {comparison['delta']['fpr_030']*100:+.1f}% |\n")
        f.write(f"| False Negative Rate (FNR) | {w2v2_metrics['metrics_at_030']['fnr']*100:.1f}% | {wavlm_metrics['metrics_at_030']['fnr']*100:.1f}% | {comparison['delta']['fnr_030']*100:+.1f}% |\n")
        f.write(f"| Accuracy | {w2v2_metrics['metrics_at_030']['accuracy']*100:.1f}% | {wavlm_metrics['metrics_at_030']['accuracy']*100:.1f}% | {(wavlm_metrics['metrics_at_030']['accuracy'] - w2v2_metrics['metrics_at_030']['accuracy'])*100:+.1f}% |\n")
        f.write(f"| Balanced Accuracy | {w2v2_metrics['metrics_at_030']['balanced_accuracy']*100:.1f}% | {wavlm_metrics['metrics_at_030']['balanced_accuracy']*100:.1f}% | {(wavlm_metrics['metrics_at_030']['balanced_accuracy'] - w2v2_metrics['metrics_at_030']['balanced_accuracy'])*100:+.1f}% |\n")
        f.write(f"| Precision | {w2v2_metrics['metrics_at_030']['precision']:.4f} | {wavlm_metrics['metrics_at_030']['precision']:.4f} | {wavlm_metrics['metrics_at_030']['precision'] - w2v2_metrics['metrics_at_030']['precision']:+.4f} |\n")
        f.write(f"| Recall | {w2v2_metrics['metrics_at_030']['recall']:.4f} | {wavlm_metrics['metrics_at_030']['recall']:.4f} | {wavlm_metrics['metrics_at_030']['recall'] - w2v2_metrics['metrics_at_030']['recall']:+.4f} |\n")
        f.write(f"| F1 Score | {w2v2_metrics['metrics_at_030']['f1']:.4f} | {wavlm_metrics['metrics_at_030']['f1']:.4f} | {wavlm_metrics['metrics_at_030']['f1'] - w2v2_metrics['metrics_at_030']['f1']:+.4f} |\n\n")

        f.write("## Global Discrimination & Latency\n\n")
        f.write("| Metric | Wav2Vec2 | WavLM + MLP | Difference |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| ROC-AUC | {w2v2_metrics['roc_auc']:.4f} | {wavlm_metrics['roc_auc']:.4f} | {comparison['delta']['roc_auc']:+.4f} |\n")
        f.write(f"| PR-AUC | {w2v2_metrics['pr_auc']:.4f} | {wavlm_metrics['pr_auc']:.4f} | {wavlm_metrics['pr_auc'] - w2v2_metrics['pr_auc']:+.4f} |\n")
        f.write(f"| EER | {w2v2_metrics['eer']*100:.2f}% | {wavlm_metrics['eer']*100:.2f}% | {comparison['delta']['eer']*100:+.2f}% |\n")
        f.write(f"| Optimal Threshold (EER) | {w2v2_metrics['eer_threshold']:.4f} | {wavlm_metrics['eer_threshold']:.4f} | - |\n")
        f.write(f"| Mean Latency (ms) | {w2v2_metrics['latency_ms']['mean']:.2f} ms | {wavlm_metrics['latency_ms']['mean']:.2f} ms | {comparison['delta']['latency_mean_ms']:+.2f} ms |\n")
        f.write(f"| P95 Latency (ms) | {w2v2_metrics['latency_ms']['p95']:.2f} ms | {wavlm_metrics['latency_ms']['p95']:.2f} ms | {wavlm_metrics['latency_ms']['p95'] - w2v2_metrics['latency_ms']['p95']:+.2f} ms |\n\n")

        f.write("## Raw Score Distributions (prob_fake)\n\n")
        f.write("### Bona-fide Samples (N=50)\n\n")
        w_b = w2v2_metrics["score_distributions"]["bonafide"]
        l_b = wavlm_metrics["score_distributions"]["bonafide"]
        f.write("| Model | Min | P25 | Median | P75 | Max | Mean | Std |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        f.write(f"| Wav2Vec2 | {w_b['min']:.4f} | {w_b['p25']:.4f} | {w_b['median']:.4f} | {w_b['p75']:.4f} | {w_b['max']:.4f} | {w_b['mean']:.4f} | {w_b['std']:.4f} |\n")
        f.write(f"| WavLM + MLP | {l_b['min']:.4f} | {l_b['p25']:.4f} | {l_b['median']:.4f} | {l_b['p75']:.4f} | {l_b['max']:.4f} | {l_b['mean']:.4f} | {l_b['std']:.4f} |\n\n")

        f.write("### Spoof Samples (N=200)\n\n")
        w_s = w2v2_metrics["score_distributions"]["spoof"]
        l_s = wavlm_metrics["score_distributions"]["spoof"]
        f.write("| Model | Min | P25 | Median | P75 | Max | Mean | Std |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        f.write(f"| Wav2Vec2 | {w_s['min']:.4f} | {w_s['p25']:.4f} | {w_s['median']:.4f} | {w_s['p75']:.4f} | {w_s['max']:.4f} | {w_s['mean']:.4f} | {w_s['std']:.4f} |\n")
        f.write(f"| WavLM + MLP | {l_s['min']:.4f} | {l_s['p25']:.4f} | {l_s['median']:.4f} | {l_s['p75']:.4f} | {l_s['max']:.4f} | {l_s['mean']:.4f} | {l_s['std']:.4f} |\n\n")

        f.write("## Dense Threshold Sweep Summary\n\n")
        f.write("> [!NOTE]\n")
        f.write("> All threshold sweep findings on the evaluation split are exploratory only. No separate development partition was available to lock decision boundaries independently.\n\n")
        f.write("Detailed sweep tables are preserved in CSV artifacts:\n")
        f.write("- `evaluation/threshold_sweep_wav2vec2.csv`\n")
        f.write("- `evaluation/threshold_sweep_wavlm_mlp.csv`\n")

    print(f"Wrote markdown report to {md_path}")


if __name__ == "__main__":
    main()
