# VoiceGuard SIH 2026 — Model-Backend Experiment Audit

## 1. Purpose and Decision Principle

This document records the audit evidence, provenance, and experimental findings for the VoiceGuard SIH 2026 model-backend evaluation.

The goal of this experiment is to evaluate whether the fine-tuned WavLM-base + MLP classifier from the offline research pipeline is better suited for live serving than the current off-the-shelf Wav2Vec2 checkpoint, while preserving the existing Wav2Vec2 implementation as a production fallback.

### Decision Principle
> **Rule of Evidence**: Do not switch the production default merely because WavLM has a lower false-positive rate at one inherited threshold. Select a model only after evaluating:
> 1. False-positive rate (FPR) on bona-fide speech.
> 2. False-negative rate (FNR) on spoof attacks.
> 3. Score separation between classes.
> 4. Threshold robustness across sweeps.
> 5. Serving latency.
> 6. Deterministic checkpoint loading and integrity.
> 7. Live API compatibility.
> 8. Preservation of all safety and privacy contracts.
>
> If WavLM does not decisively outperform the baseline across these criteria, Wav2Vec2 remains the default.

---

## 2. Baseline & Experiment Provenance

| Item | Identifier / Value | Verification Command / Source | Status |
|---|---|---|---|
| Baseline Branch | `voiceguard/sih2026-hardening` | `git branch --show-current` | Verified |
| Baseline Commit SHA | `3c228da111dc81d5ee8f210f84b098e1dbfe3d58` | `git rev-parse HEAD` | Verified |
| Experiment Branch | `experiment/wavlm-mlp-backend` | `git checkout -b experiment/wavlm-mlp-backend` | Verified |
| Fixed Evaluation Split | 250 clips (50 bona-fide, 200 spoof) | `data/asvspoof2019LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt` | Verified |
| Manifest File | `evaluation/manifest_250.json` | 250 clips with per-clip SHA-256 | Verified |
| Manifest SHA-256 | `2d18c7124ee6495ed1078635f7e3c51b0ca18fa87140f4a15cdedb0491dad7fb` | `evaluation/manifest_250.sha256` | Verified |
| Environment Spec | `evaluation/environment.json` | Python 3.11.9, PyTorch 2.10.0+cpu, Transformers 5.2.0 | Verified |

---

## 3. Upstream Model & Checkpoint Provenance

| Model Name | Upstream Repository / Source | Exact Commit / Checkpoint SHA-256 | File Status in Git | Score Type Classification |
|---|---|---|---|---|
| **Wav2Vec2 Deepfake Detector** | `garystafford/wav2vec2-deepfake-voice-detector` | `c66306024a7ede0be291e9c4558b37634782dc4e` | Upstream HuggingFace model | `uncalibrated_softmax_score` |
| **WavLM-base Backbone** | `microsoft/wavlm-base` | `efa81aae7ff777e464159e0f877d54eac5b84f81` | Upstream HuggingFace backbone | Frozen representation |
| **MLP Classifier Head** | `models/best_mlp_wavlm_base.pt` | `f7e518d57e8f96ec6c5c34b8ebb2ecd78c91ba67e7c260287a367aeb53816d2f` | Tracked in Git (397,077 bytes) | `uncalibrated_sigmoid_score` |

### Checkpoint Inspection Findings: `models/best_mlp_wavlm_base.pt`
- **File size**: 397,077 bytes
- **Git tracking**: Tracked in git (`git ls-files -s models/best_mlp_wavlm_base.pt` -> mode 100644, blob `839ee716...`)
- **Clean clone availability**: Pre-packaged in repository clone; no separate download required for demo.
- **Internal structure**:
  - `input_dim`: 768
  - `hidden_dim`: 128
  - `model_name`: `microsoft/wavlm-base`
  - `eer`: 53.5%
  - `threshold`: 1.35e-06
  - `model_state_dict` keys:
    - `net.0.weight`: `torch.Size([128, 768])`
    - `net.0.bias`: `torch.Size([128])`
    - `net.3.weight`: `torch.Size([1, 128])`
    - `net.3.bias`: `torch.Size([1])`
- **Trained State Note**: The checkpoint's recorded EER (53.5%) indicates near-chance performance on its training split, consistent with training on limited synthetic demo data rather than the full ASVspoof 2019 dataset. Live empirical evaluation on the 250-clip eval split determines its true behavior.

---

## 4. Score Type and Inversion Verification

1. **Label Convention**:
   - In ASVspoof 2019 protocol: `target = 1` for bona-fide, `target = 0` for spoof.
   - The MLP head outputs a scalar logit: $\text{logit} \in \mathbb{R}$.
   - $\text{prob\_real} = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$.
   - $\text{prob\_fake} = 1.0 - \text{prob\_real}$.
   - Higher $\text{prob\_fake}$ indicates higher probability of spoof voice.
2. **Decision Threshold**:
   - `is_fake = prob_fake >= spoof_threshold`.
   - Positive class: `spoof` (attack detected).
   - False Positive (FP): Bona-fide voice flagged as spoof (`prob_fake >= spoof_threshold` on bona-fide).
   - False Negative (FN): Spoof voice passed as bona-fide (`prob_fake < spoof_threshold` on spoof).
3. **Score Type Distinction**:
   - Wav2Vec2: `uncalibrated_softmax_score` (HuggingFace AutoModelForAudioClassification softmax output over 2 classes).
   - WavLM + MLP: `uncalibrated_sigmoid_score` (Sigmoid of 2-layer MLP head over mean-pooled 768-dim embeddings).
   - Neither score represents a calibrated Bayesian probability.

---

## 5. Architectural Isolation & Safety

The implementation uses an abstract model backend pattern in `voiceguard/model_backends/`:
- `voiceguard/model_backends/base.py`: Defines `ModelBackend` ABC and `ModelPrediction` dataclass.
- `voiceguard/model_backends/wav2vec2_backend.py`: Encapsulates Wav2Vec2 HuggingFace classification.
- `voiceguard/model_backends/wavlm_mlp_backend.py`: Encapsulates WavLM feature extraction and MLP head inference.
- `voiceguard/model_backends/__init__.py`: Factory `create_backend(backend_name, ...)`.

### Safety Contracts Preserved:
1. **Silence Input Contract**: Audio containing only silence continues to trigger `insufficient_evidence` with reason code `no_speech` via the pre-inference audio quality gate. It is never marked as `bonafide` or `low_risk`.
2. **Fail-Closed Readiness**: When a model cannot be loaded (missing checkpoint, network failure), `GET /ready` returns `HTTP 503 Service Unavailable` with `status: "not_ready"`. Calls to `POST /predict` and `POST /live/analyze` also return `HTTP 503 Service Unavailable`.
3. **No Implicit Fallback**: A failure in the active backend raises `ModelLoadError` and halts loading; it does not silently instantiate random weights or fall back to an unvalidated architecture.

---

## 6. Empirical Evaluation & Performance Comparison

The benchmark was executed across all 250 clips specified in `evaluation/manifest_250.json` (50 bona-fide, 200 spoof) under identical acoustic preprocessing (16 kHz mono) using `evaluation/run_experiment_evaluation.py`.

### 6.1 Metric Comparison at Inherited Operating Threshold (0.30)

| Metric | Wav2Vec2 (Baseline) | WavLM + MLP (Candidate) | Absolute Delta (WavLM - Wav2Vec2) | Interpretation |
|---|---|---|---|---|
| **True Positives (TP)** | 155 | 164 | +9 | Cloned audio correctly flagged |
| **True Negatives (TN)** | 0 | 14 | +14 | Genuine audio correctly passed |
| **False Positives (FP)** | 50 | 36 | -14 | Genuine audio falsely flagged |
| **False Negatives (FN)** | 45 | 36 | -9 | Cloned audio missed |
| **False Positive Rate (FPR)** | **100.0%** (50/50) | **72.0%** (36/50) | **-28.0%** | Lower FPR, but 72% remains unacceptable |
| **False Negative Rate (FNR)** | **22.5%** (45/200) | **18.0%** (36/200) | **-4.5%** | Modest improvement in recall |
| **Precision** | 0.7561 | 0.8200 | +0.0639 | Higher precision |
| **Recall** | 0.7750 | 0.8200 | +0.0450 | Higher recall |
| **F1 Score** | 0.7654 | 0.8200 | +0.0546 | Macro F1 higher |
| **Accuracy** | 62.0% | 71.2% | +9.2% | Raw accuracy higher |
| **Balanced Accuracy** | 38.8% | 55.0% | +16.2% | Balanced accuracy marginally above chance |

### 6.2 Global Discrimination & Latency Metrics

| Metric | Wav2Vec2 (Baseline) | WavLM + MLP (Candidate) | Delta | Notes |
|---|---|---|---|---|
| **ROC-AUC** | 0.3087 | 0.4879 | +0.1792 | Wav2Vec2 inverted (<0.50); WavLM is near-chance (~0.50) |
| **PR-AUC** | 0.7349 | 0.7903 | +0.0554 | Baseline prevalence is 0.80 (200/250) |
| **Equal Error Rate (EER)** | 61.50% | 52.50% | -9.00% | Both exhibit poor discriminative power |
| **EER Threshold** | 0.8790 | 1.0000 | - | WavLM threshold saturates at upper boundary |
| **Mean Inference Latency** | 1044.24 ms | 473.76 ms | **-570.48 ms** | WavLM is ~55% faster (474 ms vs 1044 ms) |
| **P95 Inference Latency** | 1787.56 ms | 836.00 ms | **-951.56 ms** | WavLM stays well under 1.0 s SLA on CPU |

### 6.3 Raw Score Distribution Analysis (`prob_fake`)

#### Bona-fide Samples ($N = 50$)
| Model | Min | P25 | Median | P75 | Max | Mean | Std Dev |
|---|---|---|---|---|---|---|---|
| **Wav2Vec2** | 0.8107 | 0.8699 | 0.8976 | 0.9877 | 0.9952 | 0.9080 | 0.0589 |
| **WavLM + MLP** | 0.0003 | 0.0062 | 1.0000 | 1.0000 | 1.0000 | 0.7207 | 0.4479 |

#### Spoof Samples ($N = 200$)
| Model | Min | P25 | Median | P75 | Max | Mean | Std Dev |
|---|---|---|---|---|---|---|---|
| **Wav2Vec2** | 0.0316 | 0.3756 | 0.8620 | 0.8932 | 0.9953 | 0.7050 | 0.3189 |
| **WavLM + MLP** | 0.0004 | 0.9998 | 1.0000 | 1.0000 | 1.0000 | 0.8201 | 0.3832 |

### 6.4 Distribution Observations
1. **Wav2Vec2 Score Inversion**: Bona-fide speech consistently scores higher (`mean = 0.9080`) than spoof attacks (`mean = 0.7050`), driving ROC-AUC down to 0.3087.
2. **WavLM Score Saturation**: Over 50% of bona-fide clips and over 75% of spoof clips produce $\text{prob\_fake} = 1.0000$. The median score for both classes is $\approx 1.0000$. The classifier head produces extreme sigmoid outputs with little gradient in between, reflecting the checkpoint's internal 53.5% training EER.

### 6.5 Exploratory Threshold Sweeps
> [!NOTE]
> All threshold sweep findings on the evaluation split are exploratory only. No separate development partition was available to lock decision boundaries independently.

Complete sweep profiles across thresholds $0.05$ to $0.95$ (step $0.05$) are preserved in:
- `evaluation/threshold_sweep_wav2vec2.csv`
- `evaluation/threshold_sweep_wavlm_mlp.csv`

---

## 7. Acceptance Gates Evaluation

Each gate defined in the implementation plan is evaluated against empirical evidence:

| Gate | Criterion | Target Requirement | Measured Value / Finding | Gate Status |
|---|---|---|---|:---:|
| **Gate 1** | Deterministic Checkpoint Loading | SHA-256 matches; loads without warning or weight init drift | SHA-256 `f7e518d5...` verified; loaded via strict `torch.load` | **PASS** |
| **Gate 2** | Exact Numerical Parity | Abs diff between research `predict.py` and `WavLMMLPBackend` $< 10^{-4}$ | Measured max diff: $0.000000$ across all tested samples | **PASS** |
| **Gate 3** | Latency Acceptance | CPU inference latency within SLA (mean $< 1500$ ms) | WavLM: mean 473.8 ms, P95 836.0 ms | **PASS** |
| **Gate 4** | Fail-Closed Error Handling | Uninitialized or corrupted model returns HTTP 503 on `/ready` and `/predict` | Tested with uninitialized backend: `/ready` $\to$ 503, `/predict` $\to$ 503 | **PASS** |
| **Gate 5** | Silence Input Contract | Digital silence returns `insufficient_evidence` with `no_speech` code | Quality gate intercepts silence; never returns `bonafide` or `low_risk` | **PASS** |
| **Gate 6** | Live API Compatibility | Client receives schema-compliant JSON with `model_backend` and `score_type` | Validated in smoke tests (`backend_name="wavlm_mlp"`, `score_type="uncalibrated_sigmoid_score"`) | **PASS** |
| **Gate 7** | Centralized Configuration | Thresholds loaded dynamically from config; no hardcoded constructor values | `risk_aggregator.py` and backend refactored to read from `voiceguard_config.py` | **PASS** |
| **Gate 8** | Exploratory Sweep Labeling | All threshold sweeps explicitly documented as exploratory | Disclaimers embedded in all documentation and evaluation reports | **PASS** |
| **Gate 9** | Bona-fide FPR Reduction | Substantial, statistically meaningful reduction in FPR on genuine speech | FPR dropped from 100% to 72% (-28%), but 72% remains unacceptably high | **FAIL** |
| **Gate 10** | Spoof FNR Control | FNR does not significantly degrade | FNR improved from 22.5% to 18.0% (-4.5%) | **PASS** |
| **Gate 11** | Score Separation & ROC-AUC | Clear class separation; ROC-AUC $> 0.70$ without score saturation | ROC-AUC is 0.4879 (near-chance); severe saturation at 1.0000 | **FAIL** |
| **Gate 12** | Production Default Replacement | All gates pass before switching `active_model` default | Gates 9 and 11 failed; candidate is not ready for production replacement | **FAIL** |

---

## 8. Model Selection Decision & Recommendations

### 8.1 Decision: Maintain Wav2Vec2 as Production Default
In strict adherence to the decision principle:
- **`active_model: wav2vec2` remains the production default in `config/model_config.yaml`**.
- Even though WavLM-base + MLP achieved lower latency (474 ms vs 1044 ms) and a lower FPR at 0.30 (72% vs 100%), its ROC-AUC of 0.4879 and extreme score saturation at 1.0 demonstrate that the candidate MLP checkpoint (`best_mlp_wavlm_base.pt`) lacks discriminative capability on the evaluation split.
- Neither model is suitable for real-world production deployment without extensive retraining or recalibration.

### 8.2 Preservation of the Candidate Backend
The WavLM-base + MLP backend is preserved as a fully tested, verified alternative backend:
- Switching between backends requires only editing `active_model` in `config/model_config.yaml` (or setting an environment variable `VOICEGUARD_ACTIVE_MODEL=wavlm_mlp`).
- The client and server dynamically adapt their responses, reporting `model_backend: "wavlm_mlp"` and `score_type: "uncalibrated_sigmoid_score"`.

### 8.3 Recommended Roadmap for Future Work
1. **Retrain the Classifier Head**: Train a projection head on the full ASVspoof 2019 train set using frozen WavLM representations with proper data augmentation (additive noise, RIR reverberation, codec simulation).
2. **Temperature & Platt Scaling**: Implement post-hoc probability calibration on an independent development set to convert raw logits into well-calibrated confidence scores.
3. **Multi-Domain & Indian Accent Evaluation**: Evaluate on multilingual Indian speech corpora to measure domain generalization before any real-world pilot.

