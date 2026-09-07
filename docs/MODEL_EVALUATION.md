# VoiceGuard — Model Evaluation Report

```text
Model source: garystafford/wav2vec2-deepfake-voice-detector
Application model version: voiceguard-v1
Exact upstream model revision: c66306024a7ede0be291e9c4558b37634782dc4e
```

**Threshold Version**: `threshold-v2`  
**Operating Threshold (`spoof_threshold`)**: `0.30` (from `config/model_config.yaml`)  
**Evaluation Script**: `evaluation/run_production_evaluation.py`  
**Results Artifact**: `evaluation/results.json`  

---

## 1. Evaluation Methodology & Configuration

The evaluation script uses the same model and preprocessing path configured for the live backend (`backend.py`), loading audio at 16 kHz mono via `librosa` and applying the Hugging Face `AutoFeatureExtractor`.

### Benchmark Setup:
- **Dataset**: ASVspoof 2019 Logical Access (LA) Evaluation Split
- **Dataset Path**: `./data/asvspoof2019LA`
- **Subset Evaluated**: Evaluated a 250-clip subset of the ASVspoof 2019 LA evaluation set: 50 bona-fide and 200 spoof samples.
- **Scope Qualification**: This subset result must not be interpreted as full-dataset performance or real-world call performance.
- **Attack Algorithms**: A07 through A19 (neural vocoders, waveform concatenation, transfer learning TTS/VC)
- **Speaker-Disjoint Status**: Yes (speakers in evaluation set do not appear in training or development sets)
- **Generator-Disjoint Status**: Yes (unseen generation methods included in eval set)
- **Preprocessing**: `librosa.load(sr=16000, mono=True)` + `feature_extractor(sampling_rate=16000, padding=True)`

---

## 2. Empirical Benchmark Results

| Metric | Measured Value | Meaning & Context |
| :--- | :--- | :--- |
| **Total Evaluated Clips** | 250 | 50 genuine, 200 spoof |
| **True Positives (TP)** | 155 | Cloned audio correctly flagged as spoof |
| **True Negatives (TN)** | 0 | Genuine human audio classified as bonafide |
| **False Positives (FP)** | 50 | Genuine human audio classified as spoof |
| **False Negatives (FN)** | 45 | Cloned audio misclassified as bonafide |
| **False Positive Rate (FPR)** | **1.0000 (100%)** | Fraction of genuine clips scoring above threshold |
| **False Negative Rate (FNR)** | **0.2250 (22.5%)** | Fraction of spoof clips scoring below threshold |
| **Precision** | 0.7561 | $\text{TP} / (\text{TP} + \text{FP})$ |
| **Recall** | 0.7750 | $\text{TP} / (\text{TP} + \text{FN})$ |
| **F1 Score** | 0.7654 | Harmonic mean of precision and recall |
| **Observed Latency** | ~554.1 ms / clip | CPU inference latency per audio window |

---

## 3. Scientific Analysis & Claims Disclosure

> [!IMPORTANT]
> **Exact Scientific Statement**:  
> On the evaluated subset of 50 bona-fide and 200 spoof clips, using the production model and threshold 0.30, all 50 bona-fide samples were classified above the spoof threshold, producing an observed FPR of 100% on this subset. The committed threshold report records a 100% false-positive rate at the reviewed threshold. A complete threshold sweep is required before making any claim about all thresholds.  
> The score overlap is consistent with poor class separation under this evaluation, although the exact cause requires further investigation.

### Raw Score Distribution on Evaluated Subset:
| Class | N | Min | p25 | Median | p75 | Max | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **bonafide** | 50 | **0.8107** | 0.8699 | **0.8976** | 0.9877 | **0.9952** | **0.9080** |
| **spoof** | 200 | **0.0316** | 0.3756 | **0.8620** | 0.8932 | **0.9953** | **0.7050** |

### Detailed Diagnostic Findings:

1. **Softmax Output Label Indexing**:
   - `model.config.id2label` maps index 0 to `'real'` and index 1 to `'fake'`.
   - In `backend.py`, `prob_real = probs[0].item()` and `prob_fake = probs[1].item()`.
   - The label indexing in `backend.py` matches the model configuration. The 100% FPR is not caused by an inverted label index mapping.

2. **Threshold Selection History**:
   - Threshold 0.30 was derived in `evaluate_live_model.py` against `wav2vec2-deepfake-voice-detector` by maximizing the aggregate F1 score (76.54%) across the imbalanced 250-clip subset (200 spoof vs. 50 bonafide).
   - Because the 155 true positives masked the 50 false positives in harmonic mean calculations, F1 appeared optimal at 0.30 despite zero true negatives (TN = 0, FPR = 100%).
   - The WavLM research pipeline (`predict.py`) used an offline MLP architecture with bonafide-probability outputs evaluated around 0.5.

3. **Preprocessing Alignment**:
   - Both `backend.py` and the evaluation scripts load audio with `librosa.load(..., sr=16000, mono=True)` and invoke `feature_extractor(audio, sampling_rate=16000, padding=True)`.
   - The audio preprocessing path is identical; the high FPR is not attributable to a preprocessing discrepancy.

4. **Score Overlap & Domain Shift Assessment**:
   - Minimum bona-fide `prob_fake` is 0.8107; every bona-fide sample in the subset scored $\ge 0.8107$.
   - The bona-fide mean score (0.9080) is higher than the spoof mean score (0.7050).
   - The score distributions show severe overlap, with bona-fide scores higher on average than spoof scores. This indicates poor class separation and is consistent with substantial domain shift; calibration and model behavior require further investigation before any definitive root-cause attribution. Shifting thresholds alone cannot cleanly separate classes on this evaluation set without substantial retraining or calibration.

---

## 4. Reproducibility

To re-run the benchmark locally:
```bash
python evaluation/run_production_evaluation.py --data_dir ./data/asvspoof2019LA
```
If the dataset is absent, the script outputs:
```text
NOT AVAILABLE — dataset was not present at the expected path.
```
No fabricated metrics were identified in the reviewed evaluation artifacts. The reported metrics are traceable to the recorded evaluation results and available raw per-clip outputs.

---

## 5. Controlled Model-Backend Experiment (Wav2Vec2 vs. WavLM-base + MLP)

A controlled model-backend experiment was conducted to evaluate whether the fine-tuned WavLM-base + MLP classifier from the research pipeline (`predict.py`) could address the baseline Wav2Vec2 limitations for live serving.

### 5.1 Benchmark Comparison Summary (250-clip eval split)

| Metric | Wav2Vec2 (Baseline) | WavLM + MLP (Candidate) | Absolute Delta |
|---|---|---|---|
| **Model Backend ID** | `wav2vec2` | `wavlm_mlp` | — |
| **Score Type** | `uncalibrated_softmax_score` | `uncalibrated_sigmoid_score` | — |
| **FPR at 0.30 (Bona-fide)** | **100.0%** (50/50 FP) | **72.0%** (36/50 FP) | **-28.0%** |
| **FNR at 0.30 (Spoof)** | **22.5%** (45/200 FN) | **18.0%** (36/200 FN) | **-4.5%** |
| **Accuracy at 0.30** | 62.0% | 71.2% | +9.2% |
| **Balanced Accuracy** | 38.8% | 55.0% | +16.2% |
| **F1 Score** | 0.7654 | 0.8200 | +0.0546 |
| **ROC-AUC** | 0.3087 (Inverted) | 0.4879 (Near-chance) | +0.1792 |
| **PR-AUC** | 0.7349 | 0.7903 | +0.0554 |
| **Equal Error Rate (EER)** | 61.50% | 52.50% | -9.00% |
| **Mean Inference Latency** | 1044.24 ms | 473.76 ms | **-570.48 ms (55% faster)** |
| **P95 Inference Latency** | 1787.56 ms | 836.00 ms | **-951.56 ms** |

### 5.2 Key Findings & Acceptance Outcome

1. **Latency and Throughput**: WavLM-base + MLP is ~55% faster on CPU (473.8 ms vs 1044.2 ms mean latency), well within real-time streaming constraints.
2. **False Positive Reduction**: WavLM reduced the false positive rate on bona-fide speech from 100% to 72% at threshold 0.30. However, a 72% FPR remains unacceptably high for production serving.
3. **Score Saturation & Near-Chance Discrimination**: WavLM outputs saturate heavily near 1.0 (median bona-fide and spoof scores are both $\approx 1.0$). ROC-AUC is 0.4879 (near-chance discrimination), consistent with the checkpoint's internal recorded training EER of 53.5%.
4. **Production Selection**: In adherence to the pre-established acceptance gates, `active_model: wav2vec2` remains the production default. The WavLM backend is preserved in `voiceguard/model_backends/wavlm_mlp_backend.py` as an available alternative that can be selected via configuration.

For complete artifact hashes, raw distributions, and gate evaluation, refer to:
- [Model Comparison Report](docs/MODEL_COMPARISON.md)
- [Model-Backend Experiment Audit](docs/MODEL_EXPERIMENT_AUDIT.md)

