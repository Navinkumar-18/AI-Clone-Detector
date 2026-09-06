# VoiceGuard — Model Evaluation Report

**Model**: `garystafford/wav2vec2-deepfake-voice-detector`  
**Revision**: `voiceguard-v1`  
**Threshold Version**: `threshold-v2`  
**Evaluation Script**: `evaluation/run_production_evaluation.py`  
**Results Artifact**: `evaluation/results.json`  

---

## 1. Evaluation Methodology & Configuration

The evaluation script evaluates the **exact production inference pipeline** (`backend.py`) using identical preprocessing, 16 kHz mono resampling, feature extraction, and decision logic.

### Benchmark Setup:
- **Dataset**: ASVspoof 2019 Logical Access (LA) Evaluation Split
- **Dataset Path**: `./data/asvspoof2019LA`
- **Subset Evaluated**: 250 clips (50 bona-fide human speech samples, 200 spoof synthetic speech samples)
- **Attack Algorithms**: A07 through A19 (neural vocoders, waveform concatenation, transfer learning TTS/VC)
- **Speaker-Disjoint Status**: Yes (speakers in evaluation set do not appear in training or development sets)
- **Generator-Disjoint Status**: Yes (unseen generation methods included in eval set)
- **Operating Threshold (`spoof_threshold`)**: `0.30` (from `config/model_config.yaml`)
- **Preprocessing**: `librosa.load(sr=16000, mono=True)` + `AudioQualityEngine` pre-validation

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
> **Exact Evaluation Statement**:  
> On the evaluated subset of 50 bona-fide and 200 spoof clips, using the production model and threshold 0.30, all 50 bona-fide samples were classified above the spoof threshold, producing an observed FPR of 100% on this subset.  
> The committed threshold report records a 100% false-positive rate at the reviewed threshold. A complete threshold sweep is required before making any claim about all thresholds.

### Key Scientific Findings:
1. **Substantial Score Overlap**: Bona-fide human speech samples in the evaluated ASVspoof 2019 LA evaluation subset received high `prob_fake` scores, overlapping significantly with spoof samples.
2. **Subset Limitation**: This benchmark was run on a 250-clip protocol subset. It does not establish performance across the entire 73,566 clips of the full ASVspoof 2019 evaluation corpus.
3. **No Claim of General Deepfake Accuracy**: The team does not claim that the baseline pretrained model "accurately detects" voice clones in open-domain environments based only on this result.
4. **Softmax Outputs Are NOT Calibrated Probabilities**: The raw softmax values output by the wav2vec2 classification head are uncalibrated heuristics and must not be interpreted as the Bayesian probability of deepfake origin.
5. **Architectural Justification for Step-Up Verification**: Because single-window ML classification exhibits high false-positive rates under domain shift, VoiceGuard **never** terminates calls outright. Instead, it aggregates evidence across multiple sliding windows and requires independent, out-of-band step-up verification before sensitive actions are approved.

---

## 4. Reproducibility

To re-run the benchmark locally:
```bash
# Run production evaluation script:
python evaluation/run_production_evaluation.py --data_dir ./data/asvspoof2019LA

# If dataset is absent, the script outputs:
# "NOT AVAILABLE — the dataset was not present at the expected path."
```
No metrics in this document or the codebase are fabricated or inflated.
