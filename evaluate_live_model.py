# -*- coding: utf-8 -*-
"""
evaluate_live_model.py
======================
Phase 2 -- Threshold evaluation for the LIVE production model.

Model  : garystafford/wav2vec2-deepfake-voice-detector
Dataset: ASVspoof 2019 LA -- eval split (250 clips: 50 bonafide, 200 spoof)

What this script does
---------------------
1. Loads the live production model (same as backend.py) ONCE.
2. Parses the ASVspoof 2019 LA eval protocol for ground-truth labels.
3. Runs inference on every eval audio file (.flac).
4. Sweeps thresholds: 0.30 -> 0.80 in steps of 0.05.
5. Computes per-threshold: Accuracy, Precision, Recall, F1, FPR, FNR, Confusion Matrix.
6. Prints a formatted report to stdout AND writes threshold_report.txt.

IMPORTANT -- What this is NOT
-----------------------------
- This is NOT a re-training.
- This does NOT use the cached WavLM-base .npy embeddings.
- This evaluates the wav2vec2-deepfake-voice-detector model only.
- The WavLM + MLP pipeline is a separate offline research model.
- Do NOT change backend.py's FAKE_THRESHOLD before reviewing this report.

Usage
-----
    python evaluate_live_model.py
    python evaluate_live_model.py --data_dir ./data/asvspoof2019LA
    python evaluate_live_model.py --output threshold_report.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ---------------------------------------------------------------------------
# Constants -- mirror backend.py exactly
# ---------------------------------------------------------------------------
MODEL_NAME      = "garystafford/wav2vec2-deepfake-voice-detector"
SAMPLE_RATE     = 16_000
CURRENT_THRESHOLD = 0.40           # existing backend.py value -- for reference only

SWEEP_THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.85, 0.05)]

PROTOCOL_RELATIVE = os.path.join(
    "LA", "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019.LA.cm.eval.trl.txt",
)
AUDIO_RELATIVE = os.path.join(
    "LA", "ASVspoof2019_LA_eval", "flac",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ClipResult:
    filename:   str
    gt_label:   str          # "bonafide" or "spoof"
    gt_target:  int          # 1 = bonafide, 0 = spoof
    prob_real:  float        # softmax index 0
    prob_fake:  float        # softmax index 1 -- this is the detection score
    error:      Optional[str] = None


@dataclass
class ThresholdMetrics:
    threshold:  float
    tp: int = 0   # true  positives  (spoof correctly caught)
    tn: int = 0   # true  negatives  (bonafide correctly passed)
    fp: int = 0   # false positives  (bonafide wrongly flagged as spoof)
    fn: int = 0   # false negatives  (spoof wrongly passed as bonafide)

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        """Of clips flagged as spoof, fraction that are actually spoof."""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        """Of all real spoof clips, fraction correctly flagged. (= 1 - FNR)"""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fpr(self) -> float:
        """False Positive Rate: bonafide wrongly flagged / total bonafide."""
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    @property
    def fnr(self) -> float:
        """False Negative Rate: spoof missed / total spoof.  (= 1 - Recall)
        In security terms: probability of MISSING a cloned voice -- the dangerous error.
        """
        return self.fn / (self.fn + self.tp) if (self.fn + self.tp) else 0.0

    @property
    def bonafide_acc(self) -> float:
        total_b = self.tn + self.fp
        return self.tn / total_b if total_b else 0.0

    @property
    def spoof_acc(self) -> float:
        total_s = self.tp + self.fn
        return self.tp / total_s if total_s else 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(device: str):
    print(f"\nLoading production model: {MODEL_NAME}")
    print(f"Device: {device}")
    t0 = time.time()
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    model.to(device).eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"id2label: {model.config.id2label}")
    print(f"  -> Verifying: index 0 should be 'real'/'bonafide', index 1 should be 'fake'/'spoof'")
    return model, feature_extractor


# ---------------------------------------------------------------------------
# Protocol parsing -- inline, no dependency on utils.py to keep this standalone
# ---------------------------------------------------------------------------
def parse_eval_protocol(protocol_path: str) -> list[dict]:
    """
    Parse ASVspoof 2019 LA eval protocol.
    Format: SPEAKER_ID  AUDIO_FILENAME  SYSTEM_ID  -  LABEL
    Returns list of {filename, label, target}
    """
    entries = []
    with open(protocol_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            audio_name = parts[1]
            label = parts[4].lower()
            target = 1 if label == "bonafide" else 0
            entries.append({"filename": audio_name, "label": label, "target": target})
    return entries


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    audio_path: str,
    model,
    feature_extractor,
    device: str,
) -> tuple[float, float]:
    """Returns (prob_real, prob_fake). Raises on any audio/model error."""
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if len(audio) == 0:
        raise ValueError("Empty audio file")
    inputs = feature_extractor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    return probs[0].item(), probs[1].item()


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------
def calculate_metrics(
    results: list[ClipResult],
    threshold: float,
) -> ThresholdMetrics:
    m = ThresholdMetrics(threshold=threshold)
    for r in results:
        if r.error:
            continue
        predicted_spoof = r.prob_fake >= threshold
        actual_spoof    = r.gt_target == 0   # target 0 = spoof

        if predicted_spoof and actual_spoof:
            m.tp += 1
        elif not predicted_spoof and not actual_spoof:
            m.tn += 1
        elif predicted_spoof and not actual_spoof:
            m.fp += 1
        else:
            m.fn += 1
    return m


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
DIV = "=" * 100
DIV2 = "-" * 100

def fmt_pct(v: float) -> str:
    return f"{v * 100:6.2f}%"

def format_report(
    results: list[ClipResult],
    metrics_list: list[ThresholdMetrics],
    n_errors: int,
) -> str:
    lines: list[str] = []

    lines.append(DIV)
    lines.append("  VOICEGUARD -- LIVE MODEL THRESHOLD EVALUATION REPORT")
    lines.append(f"  Model : {MODEL_NAME}")
    lines.append(f"  Dataset: ASVspoof 2019 LA -- eval split")
    n_ok = len(results) - n_errors
    n_bona  = sum(1 for r in results if r.gt_label == "bonafide" and not r.error)
    n_spoof = sum(1 for r in results if r.gt_label == "spoof"    and not r.error)
    lines.append(f"  Clips evaluated: {n_ok} / {len(results)} "
                 f"({n_bona} bonafide, {n_spoof} spoof, {n_errors} errors)")
    lines.append(f"  Current backend FAKE_THRESHOLD: {CURRENT_THRESHOLD}")
    lines.append(DIV)
    lines.append("")

    # ------------------------------------------------------------------
    # TABLE 1 -- per-threshold summary
    # ------------------------------------------------------------------
    lines.append("  TABLE 1: Per-threshold metrics")
    lines.append(DIV2)
    hdr = (
        f"  {'Threshold':>9}  {'Accuracy':>9}  "
        f"{'Precision':>9}  {'Recall':>9}  {'F1':>9}  "
        f"{'FPR':>9}  {'FNR':>9}  "
        f"{'Spoof_Acc':>10}  {'Bona_Acc':>9}  "
        f"{'TP':>5}  {'TN':>5}  {'FP':>5}  {'FN':>5}  Note"
    )
    lines.append(hdr)
    lines.append(DIV2)

    # Find best thresholds by different criteria
    best_f1_idx   = max(range(len(metrics_list)), key=lambda i: metrics_list[i].f1)
    best_acc_idx  = max(range(len(metrics_list)), key=lambda i: metrics_list[i].accuracy)
    best_fnr_idx  = min(range(len(metrics_list)), key=lambda i: metrics_list[i].fnr)

    for i, m in enumerate(metrics_list):
        notes = []
        if m.threshold == CURRENT_THRESHOLD:
            notes.append("<- CURRENT")
        if i == best_f1_idx:
            notes.append("<- BEST F1")
        if i == best_acc_idx and i != best_f1_idx:
            notes.append("<- BEST ACC")
        if i == best_fnr_idx:
            notes.append("<- LOWEST FNR (fewest missed spoofs)")
        note_str = "  ".join(notes)

        row = (
            f"  {m.threshold:>9.2f}  {fmt_pct(m.accuracy)}  "
            f"{fmt_pct(m.precision)}  {fmt_pct(m.recall)}  {fmt_pct(m.f1)}  "
            f"{fmt_pct(m.fpr)}  {fmt_pct(m.fnr)}  "
            f"{fmt_pct(m.spoof_acc):>10}  {fmt_pct(m.bonafide_acc):>9}  "
            f"{m.tp:>5}  {m.tn:>5}  {m.fp:>5}  {m.fn:>5}  {note_str}"
        )
        lines.append(row)

    lines.append(DIV2)
    lines.append("")

    # ------------------------------------------------------------------
    # TABLE 2 -- confusion matrices for a few key thresholds
    # ------------------------------------------------------------------
    key_thresholds = sorted({
        CURRENT_THRESHOLD,
        metrics_list[best_f1_idx].threshold,
        metrics_list[best_acc_idx].threshold,
        metrics_list[best_fnr_idx].threshold,
    })
    lines.append("  TABLE 2: Confusion matrices for key thresholds")
    lines.append(DIV2)
    for m in [mx for mx in metrics_list if mx.threshold in key_thresholds]:
        lines.append(f"  Threshold = {m.threshold:.2f}")
        lines.append(f"    Predicted ->  SPOOF      BONAFIDE")
        lines.append(f"    Actual SPOOF   TP={m.tp:>4}   FN={m.fn:>4}   (FNR={fmt_pct(m.fnr).strip()}, missed spoofs)")
        lines.append(f"    Actual BONA    FP={m.fp:>4}   TN={m.tn:>4}   (FPR={fmt_pct(m.fpr).strip()}, false alarms)")
        lines.append("")

    # ------------------------------------------------------------------
    # TABLE 3 -- per-file score distribution
    # ------------------------------------------------------------------
    bona_scores  = sorted([r.prob_fake for r in results if r.gt_label == "bonafide" and not r.error])
    spoof_scores = sorted([r.prob_fake for r in results if r.gt_label == "spoof"    and not r.error])

    lines.append("  TABLE 3: Detection score (prob_fake) distribution")
    lines.append(DIV2)
    lines.append(f"  {'Class':>10}   {'N':>5}   {'Min':>8}   {'p25':>8}   {'Median':>8}   {'p75':>8}   {'Max':>8}   {'Mean':>8}")
    lines.append(DIV2)
    for label, scores in [("bonafide", bona_scores), ("spoof", spoof_scores)]:
        if not scores:
            continue
        arr = np.array(scores)
        lines.append(
            f"  {label:>10}   {len(arr):>5}   {arr.min():>8.4f}   "
            f"{np.percentile(arr, 25):>8.4f}   {np.median(arr):>8.4f}   "
            f"{np.percentile(arr, 75):>8.4f}   {arr.max():>8.4f}   {arr.mean():>8.4f}"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # TABLE 4 -- per-file detail (abbreviated: first 20 + any errors)
    # ------------------------------------------------------------------
    lines.append("  TABLE 4: Per-file scores (sorted by prob_fake, first 40 shown)")
    lines.append(DIV2)
    lines.append(f"  {'filename':<26}  {'gt':>9}  {'prob_fake':>10}  {'prob_real':>10}  {'pred@curr':>10}  {'match?':>7}")
    lines.append(DIV2)
    sorted_results = sorted(results, key=lambda r: (r.error is not None, r.prob_fake), reverse=True)
    for r in sorted_results[:40]:
        if r.error:
            lines.append(f"  {r.filename:<26}  ERROR: {r.error}")
            continue
        pred = "spoof" if r.prob_fake >= CURRENT_THRESHOLD else "bonafide"
        match = "OK" if pred == r.gt_label else "XX"
        lines.append(
            f"  {r.filename:<26}  {r.gt_label:>9}  {r.prob_fake:>10.6f}  "
            f"{r.prob_real:>10.6f}  {pred:>10}  {match:>7}"
        )
    if len(results) > 40:
        lines.append(f"  ... ({len(results) - 40} more clips not shown -- see full data above)")
    lines.append("")

    # ------------------------------------------------------------------
    # RECOMMENDATION
    # ------------------------------------------------------------------
    lines.append(DIV)
    lines.append("  THRESHOLD ANALYSIS -- SECURITY CONTEXT")
    lines.append(DIV)
    lines.append("")
    lines.append("  Security objective: Missing a cloned voice (FN) is MORE dangerous")
    lines.append("  than triggering extra verification for a genuine voice (FP).")
    lines.append("  Therefore: prefer lower FNR over lower FPR, with a reasonable F1/Accuracy trade-off.")
    lines.append("")
    cur_m = next(m for m in metrics_list if m.threshold == CURRENT_THRESHOLD)
    lines.append(f"  Current threshold ({CURRENT_THRESHOLD}):")
    lines.append(f"    Accuracy={fmt_pct(cur_m.accuracy).strip()}, F1={fmt_pct(cur_m.f1).strip()}, "
                 f"FNR={fmt_pct(cur_m.fnr).strip()} (missed spoofs), "
                 f"FPR={fmt_pct(cur_m.fpr).strip()} (false alarms)")
    lines.append("")
    lines.append("  Candidate thresholds to consider:")
    for m in metrics_list:
        if m.fnr < 0.10 or m.threshold == CURRENT_THRESHOLD:
            lines.append(
                f"    th={m.threshold:.2f}  F1={fmt_pct(m.f1).strip():7}  "
                f"FNR={fmt_pct(m.fnr).strip():7} (missed spoofs)  "
                f"FPR={fmt_pct(m.fpr).strip():7} (false alarms)  "
                f"Accuracy={fmt_pct(m.accuracy).strip()}"
            )
    lines.append("")
    lines.append("  >>> DO NOT CHANGE backend.py FAKE_THRESHOLD until this report is reviewed.")
    lines.append("  >>> Present this report to the team before locking the threshold.")
    lines.append(DIV)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 -- Evaluate live wav2vec2 deepfake detector on ASVspoof 2019 LA eval set"
    )
    parser.add_argument(
        "--data_dir",
        default="./data/asvspoof2019LA",
        help="Root of the ASVspoof 2019 LA dataset (default: ./data/asvspoof2019LA)",
    )
    parser.add_argument(
        "--output",
        default="threshold_report.txt",
        help="Output file for the threshold report (default: threshold_report.txt)",
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=None,
        help="Limit number of clips (for quick smoke test; omit for full evaluation)",
    )
    args = parser.parse_args()

    # Resolve paths
    protocol_path = os.path.join(args.data_dir, PROTOCOL_RELATIVE)
    audio_dir     = os.path.join(args.data_dir, AUDIO_RELATIVE)

    if not os.path.exists(protocol_path):
        print(f"ERROR: Protocol file not found: {protocol_path}")
        sys.exit(1)
    if not os.path.isdir(audio_dir):
        print(f"ERROR: Audio directory not found: {audio_dir}")
        sys.exit(1)

    # Parse protocol
    protocol_entries = parse_eval_protocol(protocol_path)
    print(f"\nProtocol loaded: {len(protocol_entries)} entries")
    n_bona  = sum(1 for e in protocol_entries if e["label"] == "bonafide")
    n_spoof = sum(1 for e in protocol_entries if e["label"] == "spoof")
    print(f"  bonafide: {n_bona}, spoof: {n_spoof}")

    if args.max_clips:
        protocol_entries = protocol_entries[:args.max_clips]
        print(f"  (Limited to {args.max_clips} clips for quick test)")

    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, feature_extractor = load_model(device)

    # Verify label mapping
    id2label = model.config.id2label
    print(f"\nLabel mapping check:")
    for idx, lbl in id2label.items():
        print(f"  index {idx} -> '{lbl}'")
    # Warn if mapping is unexpected
    lbl0 = str(id2label.get(0, "")).lower()
    lbl1 = str(id2label.get(1, "")).lower()
    real_is_0 = any(k in lbl0 for k in ("real", "bonafide", "genuine"))
    fake_is_1 = any(k in lbl1 for k in ("fake", "spoof", "synth"))
    if not (real_is_0 or fake_is_1):
        print("\n  WARNING: Label mapping may be inverted -- review prob_real/prob_fake assignments!")
    else:
        print("  OK: Label mapping looks correct (index 0 = real/bonafide, index 1 = fake/spoof)")

    # Run inference
    print(f"\nRunning inference on {len(protocol_entries)} clips...")
    print(f"  Audio dir: {audio_dir}")
    print(f"  {'Progress':>10}  {'Filename':26}  {'GT':9}  {'prob_fake':>10}  {'Status'}")
    print(f"  {'-'*80}")

    results: list[ClipResult] = []
    t_start = time.time()
    for i, entry in enumerate(protocol_entries):
        audio_path = os.path.join(audio_dir, entry["filename"] + ".flac")
        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_start
            eta = (elapsed / (i + 1)) * (len(protocol_entries) - i - 1)
            print(f"  [{i+1:>4}/{len(protocol_entries)}]  ETA: {eta:.0f}s")

        if not os.path.exists(audio_path):
            results.append(ClipResult(
                filename=entry["filename"],
                gt_label=entry["label"],
                gt_target=entry["target"],
                prob_real=0.0,
                prob_fake=0.0,
                error=f"file not found: {audio_path}",
            ))
            continue

        try:
            prob_real, prob_fake = run_inference(audio_path, model, feature_extractor, device)
            results.append(ClipResult(
                filename=entry["filename"],
                gt_label=entry["label"],
                gt_target=entry["target"],
                prob_real=prob_real,
                prob_fake=prob_fake,
            ))
        except Exception as exc:
            results.append(ClipResult(
                filename=entry["filename"],
                gt_label=entry["label"],
                gt_target=entry["target"],
                prob_real=0.0,
                prob_fake=0.0,
                error=str(exc),
            ))

    total_time = time.time() - t_start
    n_errors = sum(1 for r in results if r.error)
    n_ok     = len(results) - n_errors
    print(f"\nInference complete: {n_ok} succeeded, {n_errors} errors. Total time: {total_time:.1f}s")

    # Calculate metrics for each threshold
    metrics_list = [calculate_metrics(results, t) for t in SWEEP_THRESHOLDS]

    # Format and print report
    report = format_report(results, metrics_list, n_errors)
    print("\n" + report)

    # Write to file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n>>> Report saved to: {args.output}")
    print(">>> Review the report before changing FAKE_THRESHOLD in backend.py")


if __name__ == "__main__":
    main()
