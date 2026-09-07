# VoiceGuard — Hackathon Demonstration Guide (SIH 2026)

**Single Source of Truth for Live Demonstrations**  
*Covers environment setup, pre-flight readiness checks, complete demo flow, and judge Q&A.*

**Product Positioning**:  
> "VoiceGuard is a privacy-aware prototype for detecting suspicious synthetic-speech characteristics and requesting independent verification before a sensitive action. It processes audio ephemerally, checks audio quality before inference, aggregates evidence across multiple windows, and never treats silence or unavailable analysis as proof that a caller is genuine. The current model has significant evaluation limitations, so the prototype deliberately uses risk-based step-up verification rather than claiming identity authentication or real transaction blocking."

---

## 1. Complete Environment Setup

### Commands:
```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment:
# Windows (PowerShell):
.venv\Scripts\activate
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Verify configuration integrity
python -c "from voiceguard_config import get_config; print(get_config())"

# 5. Start the backend server
python backend.py

# 6. Verify readiness in a separate terminal
curl -k https://localhost:8443/ready

# 7. Run automated test suites
pytest tests/ -v
flutter test
dart analyze

# 8. Launch Flutter application in demo mode
flutter run -d windows --dart-define=DEMO_MODE=true
```

### Operational Details & Guidance:
- **Local Self-Signed Certificates**: If `cert.pem` and `key.pem` are not present, `generate_cert.py` is executed dynamically on startup to generate a self-signed certificate for `localhost:8443`. Local self-signed certificates are for development-only testing. Public deployment requires new trusted certificate/key material.
- **Model Download on First Start**: On initial startup, Hugging Face downloads the model weights for `garystafford/wav2vec2-deepfake-voice-detector` (~360 MB) into the local cache (`~/.cache/huggingface/hub`). Subsequent launches load instantly from disk.
- **Handling Readiness (HTTP 503)**: If `curl -k https://localhost:8443/ready` returns HTTP 503 or `{"status": "not_ready"}`, the model is still being materialized in memory. Allow 15–30 seconds for loading to complete.
- **Demo Mode vs. Secure Mode**:
  - In demo mode (`--dart-define=DEMO_MODE=true`), the Flutter client enables a local self-signed certificate exception strictly scoped to the VoiceGuard target host.
  - In secure mode (`DEMO_MODE=false` or omitted), standard strict TLS validation is enforced. Release builds strictly disallow demo mode.
  - The backend remains in secure mode. Flutter demo mode is enabled only to permit local self-signed certificate testing. This is not a public-deployment configuration.
- **`curl -k` Caution**: `curl -k` is used solely for local self-signed loopback testing. Public deployment must not use `curl -k`.

---

## 2. Pre-Flight Troubleshooting Checklist

| Symptom | Cause | Remediation |
| :--- | :--- | :--- |
| `GET /ready` returns `{"status":"not_ready"}` | Model weights are downloading or initializing into memory. | Wait 15–30 seconds for initial model load to finish. |
| Flutter shows "AI Engine Offline" | Backend is not started or port 8443 is blocked. | Verify backend is running on `https://127.0.0.1:8443`. |
| Audio capture permission error | Microphone access denied by OS. | Grant microphone permissions in Windows/device settings. |

---

## 3. Live Demonstration Walkthrough

### Scenario 1: File Screening & Quality Diagnostics
1. From the VoiceGuard home screen, tap **Analyze Audio File**.
2. Select `audio_samples/genuine_sample.wav`.
3. Tap **Analyze Voice Clip**:
   - Status: `LOW DETECTED SPOOF EVIDENCE` / `Continue with caution`
   - Spoof Score: $< 0.30$
   - Audio Quality Diagnostics: RMS energy, SNR estimate, clipping ratio, and voiced frame ratio.
4. Select `audio_samples/spoof_sample.wav`:
   - Status: `VERIFICATION REQUIRED`
   - Spoof Score: Elevated ($\ge 0.85$)
   - Reason Code: `elevated_spoof_score`

### Scenario 2: Live Call Simulation & Step-Up Hold
1. Tap **Live Call Protection**.
2. UX dialog explains the acoustic air-gap:
   *"Please enable speakerphone so VoiceGuard can monitor conversation audio. Audio is processed ephemerally without persistent storage."*
3. Tap **Continue** to start monitoring.
4. Normal Conversation:
   - Status: `ALLOW WITH CAUTION`.
5. Cloned Voice Demonstration:
   - Play a prepared synthetic or converted-speech sample through the supported microphone/speakerphone demonstration path.
   - *Note on Speakerphone Testing*: Standard mobile speakers filter out frequencies above 3.5 kHz, which can drop raw scores on `wav2vec2`. For physical loudspeaker testing, run the backend with `VOICEGUARD_ACTIVE_MODEL=wavlm_mlp` for acoustic robustness, or place the mic in close proximity (see `docs/LIVE_PATH_COMPARISON.md`).
   - Window 1: Spoof score spikes $\ge 0.85$ $\rightarrow$ Status: `VERIFICATION REQUIRED`.
   - Window 2: Persistent elevated evidence ($N=2$) $\rightarrow$ Status: **`ACTION HELD`**.
   - The simulated transaction card ("₹2,00,000 to ABC Suppliers") disables immediately with label: **`TRANSACTION HELD`**.
   - Expand the **`LIVE DIAGNOSTICS (DEV MODE)`** panel to demonstrate live stream format (16kHz mono PCM), chunk RMS, model loaded readiness, freshness, and the active `canShowLowRisk` gate to the judges.
   - Prominent disclaimer: **"Demo mode — no real financial transaction is executed."**
6. Step-Up Challenge Demonstration:
   - Tap **Verify Caller First**: VoiceGuard presents simulated step-up verification options, such as confirming through the official app, calling a saved number, or contacting a trusted person. No real financial or telephony service is invoked.
7. Cooldown Recovery:
   - Provide clean speech for 3 consecutive clean windows ($M=3$).
   - System transitions back to `ALLOW WITH CAUTION`.

### Scenario 3: Silence Resilience & Ephemeral Processing
1. During active monitoring, mute the microphone or remain silent:
   - State transitions to **`INSUFFICIENT_EVIDENCE`**.
   - Note: *Silence never returns bonafide, and never clears an active security hold.*
   - Transition flow:
     ```text
     ACTION_HELD
         → silence/backend failure
         → INSUFFICIENT_EVIDENCE
         → sensitive action remains held
     ```
2. Switch to the backend server terminal console:
   - Show the real-time structured privacy log:
     ```text
     PRIVACY | Temp audio purged — no persistent storage of raw audio
     ```
   - Emphasize ephemeral audio handling: temporary audio is processed for inference and cleaned up after processing. Structured cleanup events are logged without storing raw audio content.
   - Note qualification: Operating-system memory, swap files, crash dumps, client buffers, infrastructure logs, and backups are outside the guarantees of this prototype.

---

## 4. Five-Minute Judge Presentation Script

> **[Minute 1: Problem Statement & Positioning]**  
> *"Respected evaluators, generative speech models and voice cloning enable malicious actors to impersonate family members and corporate officers in phone calls to attempt fraudulent transfers.  
> VoiceGuard is a privacy-aware prototype for detecting suspicious synthetic-speech characteristics and requesting independent verification before a sensitive action. It processes audio ephemerally, checks audio quality before inference, aggregates evidence across multiple windows, and never treats silence or unavailable analysis as proof that a caller is genuine. The current model has significant evaluation limitations, so the prototype deliberately uses risk-based step-up verification rather than claiming identity authentication or real transaction blocking."*

> **[Minute 2: Pre-Inference Audio Quality Gating]**  
> *"Our system does not claim to verify caller identity; it evaluates acoustic generation characteristics using a wav2vec2 model.  
> Crucially, we enforce an **Audio Quality Engine** before inference: digital silence, clipping, or poor SNR produce `INSUFFICIENT_EVIDENCE`. Silence never registers as genuine speech and never clears a security hold."*

> **[Minute 3: Multi-Window Aggregation & Step-Up Verification]**  
> *"In our live call monitor, an isolated score spike generates `VERIFICATION_REQUIRED`. Exponential Moving Average smoothing reduces the impact of isolated score spikes; persistent high evidence across multiple windows is required before an action hold.  
> As seen on the screen, when persistent spoof evidence is detected, the sensitive transfer card disables. VoiceGuard presents simulated step-up verification options, such as confirming through the official app, calling a saved number, or contacting a trusted person. No real financial or telephony service is invoked."*

> **[Minute 4: Ephemeral Audio Handling]**  
> *"VoiceGuard is designed around data-minimization and ephemeral-processing principles. Temporary audio is processed for inference and cleaned up after processing. Structured cleanup events are logged without intentionally storing raw audio content. This prototype has not undergone a formal legal or regulatory compliance assessment."*

> **[Minute 5: Honest Technical Disclosure & Verification]**  
> *"In our evaluation on a 250-clip subset of ASVspoof 2019 LA (50 bona-fide and 200 spoof samples), all 50 bona-fide samples were classified above threshold 0.30, producing an observed FPR of 100% on this subset due to score overlap. Rather than hiding this limitation with fabricated metrics, our entire architecture—from temporal smoothing to step-up verification—is designed specifically around handling uncertainty safely.  
> We have 54 automated tests passing and static analysis reporting zero issues. Thank you, and we welcome your questions."*

---

## 5. Claims Cheat Sheet for Judges

### ✅ Safe Claims
- *"VoiceGuard evaluates acoustic indicators of synthetic speech and voice-cloning characteristics."*
- *"VoiceGuard processes audio ephemerally; temporary audio is purged after processing with structured cleanup logging."*
- *"Audio quality gates ensure silence returns INSUFFICIENT_EVIDENCE and cannot clear security holds."*
- *"Exponential Moving Average smoothing reduces the impact of isolated score spikes; persistent high evidence is required before an action hold."*
- *"The transaction workflow is a demonstration prototype designed for step-up verification."*
- *"Local prototype execution passed: backend readiness, API communication, Flutter integration, and mock transaction flow were verified."*

### ❌ Claims to Avoid
- ⛔ *"VoiceGuard verifies caller identity or authenticates the human speaker."*
- ⛔ *"VoiceGuard achieves 99.9% accuracy or is a production-ready detector."*
- ⛔ *"The system intercepts native cellular calls or taps baseband audio."*
- ⛔ *"VoiceGuard executes real payment blocking or integrates with UPI/banks."*
- ⛔ *"Guaranteed zero retention, zero-persistence privacy, or cryptographic audit logging."*
- ⛔ *"DPDP compliant or GDPR certified."*
- ⛔ *"Robust performance on Indian languages or cellular network codecs."*
- ⛔ *"Softmax outputs are calibrated Bayesian probabilities."*
