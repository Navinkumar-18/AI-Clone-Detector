# Audio Deepfake / Voice-Clone Detector (MVP)

A lightweight, research-grade MVP codebase for classifying short audio clips as **Bonafide** (real human speech) or **Spoof** (AI-generated/cloned synthetic speech).

Built using **frozen self-supervised speech representations (WavLM-base)**, fixed-length **sequence mean-pooling**, a lightweight **PyTorch 2-layer MLP classifier head**, and evaluated using **Equal Error Rate (EER)**.

---

## 📌 Technical Pipeline Overview

```
 [Input Audio (.flac / .wav)]
              │
              ▼
  [Resample to 16kHz Mono]
              │
              ▼
  [Frozen WavLM-Base Backbone] ────► Pretrained SSL representations (95M params)
              │
              ▼
    [Sequence Mean-Pooling]    ────► Condenses (Seq_Len, 768) to (768,) embedding vector
              │
              ▼
  [Model-Versioned Disk Cache] ────► Saves X_train_wavlm_base.npy & y_train_wavlm_base.npy
              │
              ▼
 [PyTorch 2-Layer MLP Head]   ────► Linear(768->128) -> ReLU -> Dropout(0.3) -> Linear(128->1)
              │
              ▼
 [Label & Confidence Score]   ────► Output: "bonafide" / "spoof" + confidence %
```

---

## 🛠️ Environment Setup & Installation

### 1. Requirements
Ensure Python 3.9+ and PyTorch are installed. Install dependencies via:

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchaudio`, `transformers`, `scikit-learn`, `librosa`, `soundfile`, `matplotlib`, `scipy`, `tqdm`.

---

## 📂 Dataset Setup (ASVspoof 2019 LA)

Download the **ASVspoof 2019 Logical Access (LA)** dataset from official sources or Kaggle/Zenodo and extract it into the `./data/asvspoof2019LA` folder.

Expected directory structure:
```
data/
└── asvspoof2019LA/
    ├── LA/
    │   ├── ASVspoof2019_LA_cm_protocols/
    │   │   ├── ASVspoof2019.LA.cm.train.trn.txt
    │   │   ├── ASVspoof2019.LA.cm.dev.trl.txt
    │   │   └── ASVspoof2019.LA.cm.eval.trl.txt
    │   ├── ASVspoof2019_LA_train/flac/
    │   ├── ASVspoof2019_LA_dev/flac/
    │   └── ASVspoof2019_LA_eval/flac/
```

*(Note: The codebase automatically detects layout variations and supports both `.flac` and `.wav` formats).*

---

## 🚀 Execution Guide (Step-by-Step)

### Step 1: Explore Data & Sanity-Check Split Sizes
Parse dataset protocols, check class balances, verify physical audio files on disk against standard benchmark counts, and plot comparative waveforms + Mel-Spectrograms:

```bash
python explore_data.py --data_dir ./data/asvspoof2019LA
```
*Outputs: Printed split sanity report & `data_visualization.png`.*

### Step 2: Extract & Cache SSL Embeddings
Pass audio clips through frozen `microsoft/wavlm-base` and save mean-pooled 768-dim embeddings:

```bash
python extract_features.py --data_dir ./data/asvspoof2019LA --model_name microsoft/wavlm-base
```
*Outputs: Versioned disk cache files in `./data/features/` (`X_train_wavlm_base.npy`, `y_train_wavlm_base.npy`, etc.).*

### Step 3: Train Classifier Head & Evaluate EER
Train Scikit-Learn `LogisticRegression` baseline and PyTorch 2-layer `DeepfakeMLPClassifier`. Early stopping is monitored on Dev EER, and final metrics (Accuracy, F1, AUC, EER) are reported on Eval:

```bash
python train_classifier.py --model_name microsoft/wavlm-base --epochs 25
```
*Outputs: Saved model checkpoints in `./models/best_mlp_wavlm_base.pt` and `logistic_regression_wavlm_base.joblib`.*

### Step 4: Run Real Automated Smoke Test Suite
Verify end-to-end audio loading, 16kHz resampling, mono conversion, feature extraction, and prediction on real audio files:

```bash
python predict.py --test
```

### Step 5: Run Single-Clip Inference
Classify any custom `.wav` or `.flac` audio clip:

```bash
python predict.py --audio /path/to/sample.wav
```

---

## 🎓 Hackathon Review & Defense Guide (Q&A Cheat Sheet)

This section explains key architectural decisions in plain language so non-ML teammates can confidently defend every choice during code reviews and judge evaluations.

### Q1: Why freeze the WavLM backbone instead of fine-tuning it?
- **Compute Efficiency**: WavLM-base has ~95 million parameters. Fine-tuning 95M parameters requires massive VRAM and hours of training. Freezing the backbone lets our model train in under 2 minutes on a free Google Colab T4 GPU.
- **Preventing Overfitting**: Pretrained SSL models (trained on 940+ hours of speech) already possess rich acoustic and phase representations. Fine-tuning on a small dataset risks "memorizing" specific channel noise or speakers rather than learning general deepfake artifacts.

### Q2: Why use Mean-Pooling across time frames?
- Audio clips have variable durations (e.g. 2s vs 5s). WavLM outputs sequence hidden states of shape `(Batch, Time_Frames, 768)`.
- Mean-pooling computes the average across time frames $\frac{1}{T} \sum_{t=1}^{T} H_t$, producing a single fixed-length 768-dimensional vector per clip. This captures overall spectral and synthetic acoustic traits without needing dynamic padding or complex sequence models.

### Q3: Why version cached feature files (e.g. `X_train_wavlm_base.npy`)?
- If we test different backbones later (e.g. switching from `microsoft/wavlm-base` to `facebook/wav2vec2-base`), model-versioned filenames ensure that stale, incompatible embeddings from a previous backbone are never silently reused.

### Q4: Why is Equal Error Rate (EER) used instead of Accuracy alone?
- **Class Imbalance**: In security datasets, spoof clips heavily outnumber bonafide clips (e.g., 90% spoof vs 10% bonafide). A dummy model predicting "spoof" 100% of the time gets 90% Accuracy but fails completely in real-world security.
- **Biometric Standard**: Equal Error Rate (EER) is the threshold point where False Acceptance Rate (FAR, accepting fake audio) equals False Rejection Rate (FRR, rejecting real audio). EER measures true biometric security capability independent of class imbalance or threshold choice.

### Q5: What is the "Suspiciously Good EER" warning?
- In ASVspoof 2019 LA, speaker IDs in train, dev, and eval sets are strictly **disjoint** (no speaker overlap). If custom data splits leak the same speaker's voice into both train and test sets, the model learns speaker recognition instead of spoof detection, producing artificially near-0% EER. Our `train_classifier.py` automatically flags a warning if Eval EER drops below 2.0% to catch data leakage before presenting to judges.
