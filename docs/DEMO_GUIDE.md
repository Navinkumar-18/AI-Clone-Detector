# VoiceGuard — Hackathon Demonstration Guide (SIH 2026)

**Single Source of Truth for Live Demonstrations**  
*Covers environment setup, pre-flight readiness checks, complete demo flow, and judge Q&A.*

---

## 1. Complete Environment Setup

### Step 1: Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate environment:
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Step 3: Verify Configuration Integrity
```bash
python -c "from voiceguard_config import get_config; print(get_config())"
```
*Expected: Prints `VoiceGuardConfig(...)` with `demo_mode=False` and `spoof_threshold=0.3`.*

### Step 4: Start the Backend Server
```bash
python backend.py
```
- **TLS Certificate Handling**: If `cert.pem` and `key.pem` are not present, `generate_cert.py` is invoked automatically to generate an ephemeral self-signed certificate for `localhost:8443`.
- **Model Download**: On the first start, Hugging Face downloads `garystafford/wav2vec2-deepfake-voice-detector` (~360 MB) into local cache (`~/.cache/huggingface/hub`). Subsequent launches load instantly from cache.
- **HTTP Dev Mode (Optional)**: If testing in plain HTTP environments:
  ```bash
  VOICEGUARD_DEV_MODE=1 python backend.py
  ```

### Step 5: Check Local Readiness Probe
In a separate terminal:
```bash
# Check liveness:
curl -k https://localhost:8443/health

# Check model readiness:
curl -k https://localhost:8443/ready
```
*Expected output:*
```json
{"status":"ready","model_loaded":true,"model_version":"voiceguard-v1","threshold_version":"threshold-v2","device":"cpu","demo_mode":false}
```
> [!NOTE]
> `curl -k` (insecure TLS flag) is used **only** for local loopback self-signed certificate testing. It is **not** a secure public-deployment command.

### Step 6: Launch the Flutter Application
```bash
flutter pub get

# For demo mode (accepts local self-signed cert on VoiceGuard host):
flutter run -d windows --dart-define=DEMO_MODE=true

# To run in strict secure mode (production-ready verification):
flutter run -d windows
```

---

## 2. Pre-Flight Troubleshooting Checklist

| Symptom | Cause | Remediation |
| :--- | :--- | :--- |
| `GET /ready` returns `{"status":"not_ready"}` | Model weights are still downloading or initializing into memory. | Wait 15–30 seconds for initial model download to complete. |
| Flutter shows "AI Engine Offline" | Backend is not started or port 8443 is blocked. | Open Settings in the app bar, verify `https://127.0.0.1:8443`, and tap Save. |
| Audio capture permission error | Microphone access denied by Windows/Android. | Grant microphone permissions in system settings. |

---

## 3. Five-Minute Live Demonstration Walkthrough

### Scenario 1: File Screening & Quality Diagnostics (1.5 min)
1. From the VoiceGuard home screen, tap **Analyze Audio File**.
2. Select `audio_samples/genuine_sample.wav`.
3. Tap **Analyze Voice Clip**:
   - Status: `LOW SPOOF EVIDENCE` / `Continue with caution`
   - Spoof Detection Score: $< 0.30$
   - Audio Quality Diagnostics: RMS energy, SNR estimate, clipping ratio, and voiced frame ratio.
4. Select `audio_samples/spoof_sample.wav`:
   - Status: `VERIFICATION REQUIRED`
   - Spoof Detection Score: Elevated ($\ge 0.85$)
   - Reason Code: `elevated_spoof_score`

### Scenario 2: Live Call Simulation & Step-Up Hold (2 min)
1. Tap **Live Call Protection**.
2. A UX dialog explains the acoustic air-gap:
   *"Please enable speakerphone so VoiceGuard can monitor both conversation sides. Audio properties are analyzed ephemerally without storage."*
3. Tap **Continue** to start streaming.
4. Normal Conversation:
   - Green visualizer, status `ALLOW WITH CAUTION`.
5. Cloned Voice Injection:
   - Play synthetic speech clip near microphone.
   - Chunk 1: Spoof score spikes $\ge 0.85$ $\rightarrow$ Status: `VERIFICATION REQUIRED`.
   - Chunk 2: Persistent elevated evidence ($N=2$) $\rightarrow$ Status: **`ACTION HELD`**.
   - The simulated transaction card ("₹2,00,000 to ABC Suppliers") disables immediately with label: **`TRANSACTION HELD`**.
6. Step-Up Challenge Demonstration:
   - Tap **Verify Caller First**: Explains that out-of-band verification (official banking app push or phone callback) is required.
7. Cooldown Recovery:
   - Resume genuine speech for 3 consecutive clean windows ($M=3$).
   - System recovers to `ALLOW WITH CAUTION`.

### Scenario 3: Silence Resilience & Privacy Guarantee (1.5 min)
1. During active call, mute the microphone or remain silent:
   - State transitions to **`INSUFFICIENT_EVIDENCE`**.
   - Emphasize: *Silence never returns bonafide, and never clears an active security hold.*
2. Switch to the backend server terminal console:
   - Point out the real-time privacy audit log:
     ```text
     PRIVACY | Temp audio purged — no persistent storage of raw audio
     ```
   - Emphasize DPDP Act 2023 compliance: zero voice biometric templates stored.

---

## 4. Five-Minute Judge Presentation Script

> **[Minute 1: Problem Statement & Positioning]**  
> *"Respected evaluators, generative voice cloning enables fraudsters to impersonate family members and corporate officers in live phone calls using just seconds of cloned audio.  
> Existing approaches fail because they either attempt to store biometric voiceprints—violating user privacy—or use naive classifiers that falsely terminate legitimate calls on the first network glitch.  
> **VoiceGuard** is a privacy-aware voice-clone risk detection and step-up verification prototype. It analyzes synthetic-speech characteristics, aggregates evidence across multiple audio windows, handles uncertainty safely, and holds simulated sensitive actions until independent verification is completed."*

> **[Minute 2: Architecture & Pre-Inference Quality]**  
> *"Our system does not claim to verify caller identity; it evaluates acoustic generation artifacts using a fine-tuned wav2vec2 model.  
> Crucially, we introduce an **Audio Quality Engine** before inference: digital silence, clipping, or poor SNR produce `INSUFFICIENT_EVIDENCE`. Silence never registers as genuine human speech."*

> **[Minute 3: Multi-Window Aggregation in Action]**  
> *"In our live call monitor, an isolated score spike generates `VERIFICATION_REQUIRED`. Only persistent high-risk evidence across multiple windows triggers `ACTION_HELD`.  
> As you see on the screen, when persistent spoofing is detected, the sensitive financial transfer freezes. VoiceGuard guides the user to out-of-band verification rather than abruptly terminating the call."*

> **[Minute 4: Zero-Persistence Privacy Guarantee]**  
> *"VoiceGuard complies with India's DPDP Act: no caller voiceprints, transcripts, or raw audio files are ever stored in databases. As visible in our terminal, every temporary chunk is purged immediately after inference with an audited log."*

> **[Minute 5: Honest Evaluation & Technical Integrity]**  
> *"In our evaluation on the ASVspoof 2019 dataset, our reproduction confirms a 100% false-positive rate at the reviewed 0.30 threshold on out-of-domain data due to score overlap. Rather than hiding this limitation with fabricated metrics, our entire architecture—from temporal smoothing to step-up verification—is designed specifically to operate safely in real-world conditions.  
> We have 54 automated tests passing and clean static analysis. Thank you, and we welcome your questions."*

---

## 5. Claims Cheat Sheet for Judges

### ✅ Safe Claims
- *"VoiceGuard detects acoustic indicators of synthetic speech and voice-cloning algorithms."*
- *"VoiceGuard enforces zero-persistence privacy: raw audio is purged immediately after inference."*
- *"Signal quality checks ensure silence returns INSUFFICIENT_EVIDENCE and cannot clear security holds."*
- *"Multi-window temporal aggregation prevents single-spike false positives."*
- *"The transaction workflow is a demonstration prototype designed for step-up verification."*

### ❌ Claims to Avoid
- ⛔ *"VoiceGuard verifies caller identity or authenticates the human speaker."*
- ⛔ *"Our model achieves 99.9% accuracy on voice clone detection."*
- ⛔ *"The system intercepts native phone calls at the OS kernel level."*
- ⛔ *"VoiceGuard directly blocks real UPI, NEFT, or bank accounts."*
- ⛔ *"Softmax scores are calibrated probabilities of deepfake authenticity."*
