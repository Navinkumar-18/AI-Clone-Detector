# VoiceGuard — Known Limitations

## 1. Controlled-Demo Readiness Rating

```text
Controlled prototype demonstration readiness: 8/10
ML production readiness: 3/10
Real financial and native telephony deployment: not ready
```

The prototype is suitable for a controlled SIH demonstration after the claims and documentation are corrected. It is not ready for real financial, identity, or telephony deployment.

---

## 2. Machine Learning Limitations

1. **Score Overlap & False-Positive Rate**:
   - Evaluated a 250-clip subset of the ASVspoof 2019 LA evaluation set: 50 bona-fide and 200 spoof samples.
   - On this evaluated subset, using the production model and threshold 0.30, all 50 bona-fide samples were classified above the spoof threshold, producing an observed FPR of 100% on this subset. The committed threshold report records a 100% false-positive rate at the reviewed threshold. A complete threshold sweep is required before making any claim about all thresholds.
   - The score overlap is consistent with poor class separation under this evaluation, although the exact cause requires further investigation.
   - Raw score distribution on evaluated subset:
     | Class | N | Min | p25 | Median | p75 | Max | Mean |
     | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
     | **bonafide** | 50 | **0.8107** | 0.8699 | **0.8976** | 0.9877 | **0.9952** | **0.9080** |
     | **spoof** | 200 | **0.0316** | 0.3756 | **0.8620** | 0.8932 | **0.9953** | **0.7050** |

2. **Domain & Language Mismatch**:
   - The pretrained model was evaluated on English speech. It has not been fine-tuned or evaluated on:
     - Indian regional languages (Hindi, Tamil, Telugu, Kannada, Bengali, etc.)
     - Indian-accented English speech
     - Telephony-band audio (8 kHz AMR/GSM)
     - Real-world mobile microphone acoustic reverberation

3. **Uncalibrated Heuristic Outputs**:
   - The `prob_fake` softmax output is not a calibrated Bayesian probability. A score of 0.85 does not represent an 85% mathematical probability of deepfake origin.

4. **Attack Algorithm Coverage**:
   - Evaluated against ASVspoof 2019 LA systems (A07-A19). Modern zero-shot voice cloning systems (e.g. XTTS-v2, ElevenLabs, OpenVoice) require dedicated empirical evaluation.

5. **Candidate WavLM-base + MLP Limitations (Score Saturation & Near-Chance Discrimination)**:
   - Evaluated on the same 250-clip subset (`evaluation/manifest_250.json`).
   - While WavLM reduced the FPR at threshold 0.30 from 100% to 72% and reduced CPU inference latency by 55% (474 ms vs 1044 ms), its ROC-AUC is 0.4879 (near-chance discrimination).
   - Scores saturate heavily near the upper sigmoid boundary: over 50% of bona-fide samples and over 75% of spoof samples produce $\text{prob\_fake} = 1.0000$ (median score for both classes is $\approx 1.0000$).
   - The checkpoint metadata in `models/best_mlp_wavlm_base.pt` records an internal training EER of 53.5%, indicating that this research checkpoint was trained on an exploratory subset. It is preserved for experimentation and configuration-driven switching, but is not suitable for production default replacement without retraining.

---


## 3. System & Telephony Limitations

1. **Native Telephony & Call Interception**:
   - Native cellular call interception, real telephony integration, UPI integration, bank integration, and real transaction blocking were not tested and are outside the prototype scope.
   - The application requires speakerphone audio capture to monitor live conversations.

2. **No Speaker Verification**:
   - The system does not verify who is speaking or authenticate caller identity. It assesses synthetic speech characteristics, not biometric identity.

3. **Simulated Financial Protection**:
   - The transaction workflow and card are simulated demonstration artifacts. No real banking APIs (UPI, IMPS, NEFT) or card gateways are invoked.

4. **Single-Process Rate Limiter**:
   - The in-memory rate limiter is suitable for a single-process prototype only. It is not sufficient for distributed production deployment.

5. **Self-Signed TLS**:
   - Local self-signed certificates are for development-only testing. Public deployment requires new trusted certificate/key material.

6. **Acoustic Loudspeaker & Speakerphone Transducer Degradation**:
   - Smartphone loudspeaker playback introduces acoustic bandpass filtering (300 Hz – 3500 Hz), transducer soft-clipping, and ambient room noise.
   - Pretrained full-bandwidth acoustic models (such as `wav2vec2`) experience score attenuation when AI audio is played through low-cost micro-speakers (score drops from 0.484 to 0.047).
   - While client fail-closed guards prevent silence or unverified states from turning green, robust over-the-air detection requires backends trained with acoustic augmentation (e.g. `wavlm_mlp`). See `docs/LIVE_PATH_COMPARISON.md` and `docs/LIVE_CAPTURE_DEBUG.md`.

---

## 4. Ephemeral Storage Boundaries & Privacy Limitations

1. **Privacy Scope & Boundaries**:
   - VoiceGuard is designed for ephemeral audio processing. Temporary audio is processed for inference and cleaned up after processing. Structured cleanup events are logged without intentionally storing raw audio content.
   - **Qualification**: Operating-system memory, swap files, crash dumps, client buffers, infrastructure logs, and backups are outside the guarantees of this prototype.
   - Designed around data-minimization and ephemeral-processing principles. This prototype has not undergone a formal legal or regulatory compliance assessment.
