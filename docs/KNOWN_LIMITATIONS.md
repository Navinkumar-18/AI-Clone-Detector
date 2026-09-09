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

2. **Direct-Upload Accuracy vs. Acoustic Capture Domain Shift (Empirical Ground Truth)**:
   - **Direct Upload (Clean Digital Path)**: Tested on 2026-09-08 with a known AI-synthesized voice clip uploaded directly via Analyze Audio File / `POST /predict`. The production `wav2vec2` model accurately flags the synthetic speech with a raw spoof score of **98.9% (Critical / High Risk)**.
   - **Acoustic Re-Capture (Loudspeaker → Microphone)**: When the exact same AI clip is played through a physical loudspeaker and re-captured via microphone (tested across laptop speaker → earphone mic and smartphone loudspeaker → mic), the score drops from **98.9% to ~0% (Low Risk)**.
   - **Domain Shift Mechanism**: Smartphone and laptop micro-speakers act as strong acoustic bandpass filters (300 Hz – 3500 Hz), clip high-amplitude transients, and introduce room reverberation. The production `wav2vec2` model was trained on direct digital speech representations; the acoustic channel transfer function strips away the high-frequency spectral artifacts the model relies on to detect synthetic speech.
   - **Real Human Voice Control**: Direct-mic recordings of real human speech via clean upload evaluate to low risk under clean conditions. See `evaluation/evaluate_clean_control.py` and `evaluation/clean_control_results.json`.

3. **Prosody vs. Content Sensitivity Characterization (2x2 Matrix Protocol)**:
   - During live call monitoring, speaking a demanding phrase ("transfer two lakh rupees") with vocal urgency caused scores to jump from ~3% to 93–97% (Action Held) within two aggregation windows, whereas calm speech scored consistently lower.
   - To characterize whether this sensitivity is driven by vocal delivery (prosodic urgency, pitch elevation, strain), semantic text phrasing, or acoustic noise/RMS artifacts, an unbiased 2x2 test matrix is implemented in `evaluation/evaluate_prosody_bias.py`:
     1. Calm delivery + Neutral script (*"The weather is pleasant today and the sun is shining brightly."*)
     2. Calm delivery + Money-transfer script (*"Please transfer two lakh rupees to this account immediately."*)
     3. Urgent/demanding delivery + Neutral script
     4. Urgent/demanding delivery + Money-transfer script
   - **Consistency Protocol**: All clips must be recorded using the developer's phone voice recorder in the same room, at a constant ~15–20 cm mic distance, within the same session.
   - **Analysis Framework**: The script computes empirical deltas to determine whether score elevations correlate primarily with:
     - Delivery/prosody (vocal strain and urgency altering frame-level embeddings),
     - Semantic phrasing,
     - Interactive combination of both, or
     - Inconclusive noise/RMS variance (noting that the original 97% spike occurred during a session where the RMS gate and Silent-badge bug were active).

4. **Domain & Language Mismatch**:
   - The pretrained model was evaluated on English speech. It has not been fine-tuned or evaluated on:
     - Indian regional languages (Hindi, Tamil, Telugu, Kannada, Bengali, etc.)
     - Indian-accented English speech
     - Telephony-band audio (8 kHz AMR/GSM)
     - Real-world mobile microphone acoustic reverberation

5. **Uncalibrated Heuristic Outputs**:
   - The `prob_fake` softmax output is not a calibrated Bayesian probability. A score of 0.85 does not represent an 85% mathematical probability of deepfake origin.

6. **Attack Algorithm Coverage**:
   - Evaluated against ASVspoof 2019 LA systems (A07-A19). Modern zero-shot voice cloning systems (e.g. XTTS-v2, ElevenLabs, OpenVoice) require dedicated empirical evaluation.

7. **Candidate WavLM-base + MLP Limitations (Score Saturation & Near-Chance Discrimination)**:
   - Evaluated on the same 250-clip subset (`evaluation/manifest_250.json`).
   - While WavLM reduced the FPR at threshold 0.30 from 100% to 72% and reduced CPU inference latency by 55% (474 ms vs 1044 ms), its ROC-AUC is 0.4879 (near-chance discrimination).
   - Scores saturate heavily near the upper sigmoid boundary: over 50% of bona-fide samples and over 75% of spoof samples produce $\text{prob\_fake} = 1.0000$ (median score for both classes is $\approx 1.0000$).
   - The checkpoint metadata in `models/best_mlp_wavlm_base.pt` records an internal training EER of 53.5%, indicating that this research checkpoint was trained on an exploratory subset. It is preserved for experimentation and configuration-driven switching, but is not suitable for production default replacement without retraining.

---

## 3. System & Telephony Limitations

1. **UI Silent-Badge vs. Active Score Contradiction (Identified & Resolved)**:
   - **Symptom**: During live calls, the "Audio Input" indicator continuously displayed `SILENT (ENABLE SPEAKERPHONE)` while the risk score was actively oscillating between 0% and 97%.
   - **Root Cause**: The Silent badge was derived from an instantaneous visualizer sub-buffer variable `_audioLevel < 0.03` (calculated on 50–100ms raw PCM chunks, equivalent to RMS < 0.006). Meanwhile, the analysis engine evaluated the full 4-second rolling buffer against `_currentRms < 0.003`. Moderate-volume speech (RMS ~0.005) or micro-pauses between syllables falsely triggered the badge despite valid speech scoring.
   - **Resolution**:
     - Decoupled the badge from `_audioLevel` and synchronized it strictly with the RMS gate threshold: `_currentRms < 0.003 && _callState == _CallState.active`.
     - Applied exponential moving average (EMA) smoothing to `_audioLevel` to stabilize visualizer waveform bars.
     - Added regression test `test/silent_badge_consistency_test.dart` proving the badge and gate state can never visually disagree.

2. **Backend Disconnect & Fail-Closed Behavior (Root Cause & Hardening)**:
   - **Symptom**: Client encountered `BACKEND_UNAVAILABLE` mid-session (~20s into a live call). Fail-closed kicked in as intended, latching `ACTION_HELD`.
   - **Root-Cause Investigation**:
     - The server enforced an in-memory rate limit of 60 req/min across all paths. Routine traffic accounted for ~42 req/min (30 analysis req/min + 12 health check req/min). Rapid session restarts, pre-call health probes, or ngrok proxy traffic shared the exact same bucket and could trigger HTTP 429.
     - Server logs were directed to `sys.stderr` only and exceptions were converted to HTTP 422 without logging stack traces, preventing real-time diagnostics.
     - Any HTTP error (422, 429, 500, 503) or socket drop immediately transitioned the client to `backend_unavailable`.
   - **Hardening Implemented**:
     - Exempted `GET /health` and `GET /ready` from rate limiting in middleware.
     - Increased default rate limit to 120 req/min in `config/model_config.yaml`.
     - Added persistent file logging to `backend.log` so all requests, request IDs, and stack traces are permanently captured.
     - Added structured `LIVE_ERROR_EVENT` logging for 422, 429, and 500 events.
     - Guaranteed fail-closed behavior is fully preserved throughout outages.

3. **Native Telephony & Call Interception**:
   - Native cellular call interception, real telephony integration, UPI integration, bank integration, and real transaction blocking were not tested and are outside the prototype scope.
   - The application requires speakerphone audio capture to monitor live conversations.

4. **No Speaker Verification**:
   - The system does not verify who is speaking or authenticate caller identity. It assesses synthetic speech characteristics, not biometric identity.

5. **Simulated Financial Protection**:
   - The transaction workflow and card are simulated demonstration artifacts. No real banking APIs (UPI, IMPS, NEFT) or card gateways are invoked.

6. **Single-Process Rate Limiter**:
   - The in-memory rate limiter is suitable for a single-process prototype only. It is not sufficient for distributed production deployment.

7. **Self-Signed TLS**:
   - Local self-signed certificates are for development-only testing. Public deployment requires trusted certificate/key material.

---

## 4. Ephemeral Storage Boundaries & Privacy Limitations

1. **Privacy Scope & Boundaries**:
   - VoiceGuard is designed for ephemeral audio processing. Temporary audio is processed for inference and cleaned up after processing. Structured cleanup events are logged without intentionally storing raw audio content.
   - **Qualification**: Operating-system memory, swap files, crash dumps, client buffers, infrastructure logs, and backups are outside the guarantees of this prototype.
   - Designed around data-minimization and ephemeral-processing principles. This prototype has not undergone a formal legal or regulatory compliance assessment.

---

## 5. Explicit Out-of-Scope Roadmap (Post-Demo Milestones)

To protect system stability ahead of the Sept 10 SIH demo, the following items are intentionally **out of scope** for this pass and flagged for post-demo engineering:
1. **Model Retraining & Fine-Tuning**: Retraining backends on acoustic replay datasets (e.g. ASVspoof 2019 Physical Access / PA partition) or applying data augmentation (reverberation, codec compression).
2. **Threshold Shifts**: Altering production decision boundaries without an exhaustive multi-dataset sweep.
3. **Acoustic Preprocessing**: Introducing hardware-level echo cancellation or neural dereverberation filters into the live capture loop.
