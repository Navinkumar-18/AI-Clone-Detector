# Model Comparison: Wav2Vec2 vs. WavLM-base + MLP Classifier

## Overview

Controlled empirical comparison evaluated on the fixed 250-clip ASVspoof 2019 LA evaluation split (`evaluation/manifest_250.json`).

| Attribute | Wav2Vec2 (Current Baseline) | WavLM-base + MLP (Candidate) |
|---|---|---|
| Backend identifier | `wav2vec2` | `wavlm_mlp` |
| Model / Backbone ID | `garystafford/wav2vec2-deepfake-voice-detector` | `microsoft/wavlm-base` |
| Revision SHA | `c66306024a7e` | `efa81aae7ff7` |
| Score type | `uncalibrated_softmax_score` | `uncalibrated_sigmoid_score` |
| Checkpoint path | Pretrained weights | `models/best_mlp_wavlm_base.pt` |

## Performance at Inherited Threshold 0.30

| Metric | Wav2Vec2 | WavLM + MLP | Difference (WavLM - Wav2Vec2) |
|---|---|---|---|
| False Positive Rate (FPR) | 100.0% | 72.0% | -28.0% |
| False Negative Rate (FNR) | 22.5% | 18.0% | -4.5% |
| Accuracy | 62.0% | 71.2% | +9.2% |
| Balanced Accuracy | 38.8% | 55.0% | +16.3% |
| Precision | 0.7561 | 0.8200 | +0.0639 |
| Recall | 0.7750 | 0.8200 | +0.0450 |
| F1 Score | 0.7654 | 0.8200 | +0.0546 |

## Global Discrimination & Latency

| Metric | Wav2Vec2 | WavLM + MLP | Difference |
|---|---|---|---|
| ROC-AUC | 0.3087 | 0.4879 | +0.1792 |
| PR-AUC | 0.7349 | 0.7903 | +0.0554 |
| EER | 61.50% | 52.50% | -9.00% |
| Optimal Threshold (EER) | 0.8790 | 1.0000 | - |
| Mean Latency (ms) | 1044.24 ms | 473.76 ms | -570.48 ms |
| P95 Latency (ms) | 1787.56 ms | 836.00 ms | -951.56 ms |

## Raw Score Distributions (prob_fake)

### Bona-fide Samples (N=50)

| Model | Min | P25 | Median | P75 | Max | Mean | Std |
|---|---|---|---|---|---|---|---|
| Wav2Vec2 | 0.8107 | 0.8699 | 0.8976 | 0.9877 | 0.9952 | 0.9080 | 0.0589 |
| WavLM + MLP | 0.0003 | 0.0062 | 1.0000 | 1.0000 | 1.0000 | 0.7207 | 0.4479 |

### Spoof Samples (N=200)

| Model | Min | P25 | Median | P75 | Max | Mean | Std |
|---|---|---|---|---|---|---|---|
| Wav2Vec2 | 0.0316 | 0.3756 | 0.8620 | 0.8932 | 0.9953 | 0.7050 | 0.3189 |
| WavLM + MLP | 0.0004 | 0.9998 | 1.0000 | 1.0000 | 1.0000 | 0.8201 | 0.3832 |

## Dense Threshold Sweep Summary

> [!NOTE]
> All threshold sweep findings on the evaluation split are exploratory only. No separate development partition was available to lock decision boundaries independently.

Detailed sweep tables are preserved in CSV artifacts:
- `evaluation/threshold_sweep_wav2vec2.csv`
- `evaluation/threshold_sweep_wavlm_mlp.csv`
