"""
test_multiple_clips.py
======================
Batch inference + threshold sweep for the wav2vec2 deepfake detector.

Usage
-----
    # Run against every audio file in ./test_clips/
    python test_multiple_clips.py

    # Override the clips folder
    python test_multiple_clips.py --folder path/to/clips

    # Skip the sweep (just show per-file scores)
    python test_multiple_clips.py --no-sweep

Filename convention for the accuracy summary
---------------------------------------------
Files whose name starts with "real_"   are treated as ground-truth BONAFIDE.
Files whose name starts with "ai_",
  "fake_", "spoof_", or "synth_"       are treated as ground-truth SPOOF.
Any other prefix is labelled "unknown" and excluded from the accuracy tally.

Example filenames
-----------------
    real_my_voice.wav
    real_friend_clip.flac
    ai_gemini_1.wav
    fake_elevenlabs.wav
    spoof_bark_tts.mp3
"""

from __future__ import annotations

import argparse
import os
import sys

import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ─────────────────────────────────────────────────────────────────────────────
# Config – mirror the same constants used in backend.py
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
SAMPLE_RATE = 16_000
DEFAULT_THRESHOLD = 0.4      # backend.py default
SWEEP_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".webm"}

BONAFIDE_PREFIXES = ("real_",)
SPOOF_PREFIXES    = ("ai_", "fake_", "spoof_", "synth_")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    print(f"\nLoading model: {MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device       : {device}")
    fe   = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    mdl  = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    mdl.to(device).eval()
    print(f"\n>>> model.config.id2label = {mdl.config.id2label}")
    print("    (triple-check: 0 should be 'real'/bonafide, 1 should be 'fake'/spoof)")
    return mdl, fe, device


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(audio_path: str, model, feature_extractor, device: str) -> dict:
    """
    Returns:
        prob_real  : float  (softmax index 0 – 'real')
        prob_fake  : float  (softmax index 1 – 'fake')
    """
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    inputs = feature_extractor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    return {
        "prob_real": round(probs[0].item(), 6),
        "prob_fake": round(probs[1].item(), 6),
    }


def classify(prob_fake: float, threshold: float) -> tuple[str, float]:
    """Return (label, confidence) for a given threshold."""
    is_fake = prob_fake >= threshold
    label      = "spoof"    if is_fake else "bonafide"
    confidence = prob_fake  if is_fake else (1.0 - prob_fake)
    return label, round(confidence, 6)


def ground_truth_label(filename: str) -> str:
    name = filename.lower()
    if any(name.startswith(p) for p in BONAFIDE_PREFIXES):
        return "bonafide"
    if any(name.startswith(p) for p in SPOOF_PREFIXES):
        return "spoof"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────
COL = {
    "file":       32,
    "gt":         9,
    "label":      9,
    "conf":       10,
    "prob_real":  10,
    "prob_fake":  10,
}

SEP  = "─"
TICK = "✓"
CROSS = "✗"

def hdr_row(columns: list[tuple[str, int]]) -> str:
    return "  ".join(h.ljust(w) for h, w in columns)

def sep_row(columns: list[tuple[str, int]]) -> str:
    return "  ".join(SEP * w for _, w in columns)

def cell(val: str, width: int, ok: bool | None = None) -> str:
    s = str(val).ljust(width)
    if ok is True:
        return s   # could add ANSI green here if desired
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Batch deepfake inference + threshold sweep")
    parser.add_argument("--folder",   default="./test_clips", help="Folder with audio clips")
    parser.add_argument("--no-sweep", action="store_true",    help="Skip threshold sweep table")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"ERROR: folder '{folder}' does not exist.")
        sys.exit(1)

    audio_files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        print(f"No audio files found in '{folder}'.")
        sys.exit(0)

    model, feature_extractor, device = load_model()

    # ── collect raw scores for every file ────────────────────────────────────
    print(f"\nFound {len(audio_files)} audio file(s) in '{folder}'\n")
    results: list[dict] = []
    for fname in audio_files:
        fpath = os.path.join(folder, fname)
        try:
            raw = run_inference(fpath, model, feature_extractor, device)
            gt  = ground_truth_label(fname)
            label, conf = classify(raw["prob_fake"], DEFAULT_THRESHOLD)
            results.append({
                "filename":  fname,
                "gt":        gt,
                "prob_real": raw["prob_real"],
                "prob_fake": raw["prob_fake"],
                "label":     label,
                "conf":      conf,
            })
        except Exception as exc:
            print(f"  [SKIP] {fname}: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE 1: per-file scores at default threshold
    # ─────────────────────────────────────────────────────────────────────────
    cols = [
        ("filename",  COL["file"]),
        ("gt_label",  COL["gt"]),
        ("predicted", COL["label"]),
        ("confidence",COL["conf"]),
        ("prob_real", COL["prob_real"]),
        ("prob_fake", COL["prob_fake"]),
        ("correct?",  8),
    ]
    print("=" * 90)
    print(f"  TABLE 1: Per-file scores  (threshold = {DEFAULT_THRESHOLD})")
    print("=" * 90)
    print(hdr_row(cols))
    print(sep_row(cols))
    for r in results:
        correct_sym = ""
        if r["gt"] != "unknown":
            correct_sym = TICK if r["label"] == r["gt"] else CROSS
        row = [
            r["filename"][:COL["file"]],
            r["gt"],
            r["label"],
            f"{r['conf']:.4f}",
            f"{r['prob_real']:.6f}",
            f"{r['prob_fake']:.6f}",
            correct_sym,
        ]
        print("  ".join(str(v).ljust(w) for v, (_, w) in zip(row, cols)))

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE 2: threshold sweep per file
    # ─────────────────────────────────────────────────────────────────────────
    if not args.no_sweep:
        print()
        print("=" * 90)
        print("  TABLE 2: Threshold sweep — label at each threshold")
        print("=" * 90)
        th_cols = [("filename", COL["file"])] + [
            (f"th={t}", 12) for t in SWEEP_THRESHOLDS
        ]
        print(hdr_row(th_cols))
        print(sep_row(th_cols))
        for r in results:
            row_vals = [r["filename"][:COL["file"]]]
            for t in SWEEP_THRESHOLDS:
                lbl, conf = classify(r["prob_fake"], t)
                marker = f"{'*' if lbl=='spoof' else ' '}{lbl[:4]}({conf:.2f})"
                row_vals.append(marker)
            print("  ".join(str(v).ljust(w) for v, (_, w) in zip(row_vals, th_cols)))
        print("\n  Legend: *spoo = spoof  | bona = bonafide  | (conf) = confidence at that threshold")

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE 3: accuracy summary per threshold
    # ─────────────────────────────────────────────────────────────────────────
    bonafide_clips = [r for r in results if r["gt"] == "bonafide"]
    spoof_clips    = [r for r in results if r["gt"] == "spoof"]

    print()
    print("=" * 90)
    print("  TABLE 3: Accuracy summary per threshold")
    print("=" * 90)

    if not bonafide_clips and not spoof_clips:
        print("  (No clips with recognised prefixes — rename files to real_*.wav or ai_*.wav)")
    else:
        sum_cols = [
            ("threshold", 10),
            ("bonafide_correct", 18),
            ("spoof_correct",    16),
            ("overall",          10),
        ]
        print(hdr_row(sum_cols))
        print(sep_row(sum_cols))

        for t in SWEEP_THRESHOLDS:
            b_correct = sum(1 for r in bonafide_clips if classify(r["prob_fake"], t)[0] == "bonafide")
            s_correct = sum(1 for r in spoof_clips    if classify(r["prob_fake"], t)[0] == "spoof")
            total_known = len(bonafide_clips) + len(spoof_clips)
            overall = (b_correct + s_correct) / total_known if total_known else 0.0

            marker = " <-- current default" if t == DEFAULT_THRESHOLD else ""
            row = [
                f"{t}",
                f"{b_correct}/{len(bonafide_clips)}",
                f"{s_correct}/{len(spoof_clips)}",
                f"{overall:.0%}{marker}",
            ]
            print("  ".join(str(v).ljust(w) for v, (_, w) in zip(row, sum_cols)))

    print()
    print("  Recommendation: pick the threshold row with the best 'overall' score.")
    print("  Then update FAKE_THRESHOLD in backend.py when you're ready to commit.")
    print()


if __name__ == "__main__":
    main()
