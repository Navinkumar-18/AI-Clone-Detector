# VoiceGuard Live Path 3-Way Comparison: Digital vs Live vs Acoustic Speakerphone

## 1. Overview & Evaluation Goals

This audit compares raw detection responses across three execution paths on identical synthetic voice test audio (`test_clip.wav`):
1. **Path A: Direct File Screening (`POST /predict`)** — full audio file uploaded directly via multipart form data.
2. **Path B: Clean Live Chunks (`POST /live/analyze`)** — audio segmented into sequential 2-second windows matching client-side streaming.
3. **Path C: Physical Speakerphone Simulation (`POST /live/analyze`)** — audio subjected to acoustic degradation mimicking mobile speakerphone playback:
   - Bandpass filter (300 Hz - 3500 Hz phone loudspeaker cutoff)
   - Transducer saturation / soft-clipping
   - Ambient acoustic room noise (-32 dB SNR)

Evaluations were performed across both available backend models:
- Default `wav2vec2` (`garystafford/wav2vec2-deepfake-voice-detector`)
- Research `wavlm_mlp` (`microsoft/wavlm-base` + custom trained MLP head)

---

## 2. Comparative Results Summary

| Metric | Path A: `/predict` | Path B: `/live/analyze` (Clean Chunks) | Path C: `/live/analyze` (Speakerphone Capture) |
|---|---|---|---|
| **Audio Input** | 2.0s digital WAV | 2.0s segmented PCM chunk | 2.0s over-the-air degraded chunk |
| **wav2vec2 Spoof Score** | `0.4843` | `0.4844` | **`0.0468`** |
| **wav2vec2 Risk Level** | Medium Risk | Medium Risk | **Low Risk (Acoustic Drop)** |
| **wav2vec2 Decision** | `verification_required` | `verification_required` | `low_risk` |
| **wav2vec2 Action** | `verify` | `verify` | `allow_with_caution` |
| **wavlm_mlp Spoof Score** | `0.9409` | `0.9442` | **`0.9963`** |
| **wavlm_mlp Risk Level** | High Risk | High Risk | **High Risk** |
| **wavlm_mlp Decision** | `verification_required` | `verification_required` | `verification_required` |
| **wavlm_mlp Action** | `verify` | `verify` | `verify` |

---

## 3. Analysis: Why AI Audio Displayed as Low Risk Near Physical Speakers

### 3.1 Acoustic Transmission Path Sensitivity
The speakerphone path substantially changes the captured signal and may attenuate or distort features used by the model. In this test, the Wav2Vec2 score dropped from 0.4844 to 0.0468, indicating sensitivity to the acoustic transmission path. The precise frequency-level cause requires further analysis.

While the candidate `wavlm_mlp` model maintained an elevated spoof score under this simulated acoustic path (`0.9963`), this physical test does not prove that WavLM is production-ready. As documented in the model experiment audit (`docs/MODEL_EXPERIMENT_AUDIT.md`), `wavlm_mlp` remains experimental, exhibiting a 72% false-positive rate on evaluated bona-fide clips and near-chance discrimination (ROC-AUC: 0.4879).

Therefore:
- The fail-closed UI and live state machine are hardened and ready for demonstration.
- The underlying voice-deepfake ML models remain experimental prototypes.
- Green is strictly defined and displayed as "low detected spoof evidence under valid capture", never as "genuine," "safe," or "authenticated."

### 3.2 Client-Side Fail-Open Bug (Now Fixed)
Previously in `LiveCallScreen`:
```dart
// BUGGY LEGACY CODE:
if (_emaScore < mediumThreshold) {
  risk = 'low';
  action = 'allow_with_caution';
  decision = 'low_risk';
}
```
When `wav2vec2`'s score dropped below `0.30`, the client unconditionally entered `low_risk` and rendered Green, enabling the transaction card.

### 3.3 New Fail-Closed Protection (Verified)
Under the hardened pipeline:
1. Green requires `canShowLowRisk == true`.
2. `canShowLowRisk` requires:
   - `modelLoaded == true` (default `false` if missing or unverified)
   - `speechDetected == true`
   - `audioQuality.status == 'acceptable'`
   - `isFresh() == true` (< 15 seconds old)
   - Zero blocking reason codes
3. Even if acoustic attenuation drops the raw score, any intermittent silence, weak capture, or backend outage drops immediately to Grey (`INSUFFICIENT_EVIDENCE`), never defaulting to Green.
4. The candidate `wavlm_mlp` backend (`VOICEGUARD_ACTIVE_MODEL=wavlm_mlp`) demonstrates that different feature representations respond differently to loudspeaker playback, though its experimental status requires retraining before any production consideration.

---

## 4. Raw API Responses

### 4.1 Path A: `POST /predict` (wavlm_mlp)
```json
{
  "decision": "verification_required",
  "action": "verify",
  "label": "spoof",
  "spoof_score": 0.940928,
  "confidence": 0.940928,
  "risk_level": "high",
  "risk_percentage": 94.1,
  "speech_detected": true,
  "audio_quality": {
    "status": "acceptable",
    "duration_seconds": 2.0,
    "rms": 0.282842,
    "snr_db": 0.0,
    "clipping_ratio": 0.0,
    "voiced_ratio": 1.0
  },
  "reason_codes": [
    "elevated_spoof_score"
  ],
  "model_version": "research-mlp-v1",
  "model_backend": "wavlm_mlp",
  "score_type": "uncalibrated_sigmoid_score",
  "threshold_version": "wavlm-v1",
  "model_loaded": true,
  "request_id": "8eaaed6b-98d8-4728-9e1c-983acde06fe5"
}
```

### 4.2 Path B: `POST /live/analyze` Clean Chunks (wavlm_mlp)
```json
{
  "decision": "verification_required",
  "action": "verify",
  "label": "spoof",
  "spoof_score": 0.944219,
  "confidence": 0.944219,
  "risk_level": "high",
  "risk_percentage": 94.4,
  "speech_detected": true,
  "audio_quality": {
    "status": "acceptable",
    "duration_seconds": 2.0,
    "rms": 0.282815,
    "snr_db": 0.0,
    "clipping_ratio": 0.0,
    "voiced_ratio": 1.0
  },
  "reason_codes": [
    "elevated_spoof_score"
  ],
  "model_version": "research-mlp-v1",
  "model_backend": "wavlm_mlp",
  "score_type": "uncalibrated_sigmoid_score",
  "threshold_version": "wavlm-v1",
  "model_loaded": true,
  "request_id": "a4680e97-db1a-4d33-a9f0-873da7b91f04"
}
```

### 4.3 Path C: `POST /live/analyze` Speakerphone Capture (wavlm_mlp)
```json
{
  "decision": "verification_required",
  "action": "verify",
  "label": "spoof",
  "spoof_score": 0.996285,
  "confidence": 0.996285,
  "risk_level": "high",
  "risk_percentage": 99.6,
  "speech_detected": true,
  "audio_quality": {
    "status": "acceptable",
    "duration_seconds": 2.0,
    "rms": 0.262884,
    "snr_db": 0.0,
    "clipping_ratio": 0.0,
    "voiced_ratio": 1.0
  },
  "reason_codes": [
    "elevated_spoof_score"
  ],
  "model_version": "research-mlp-v1",
  "model_backend": "wavlm_mlp",
  "score_type": "uncalibrated_sigmoid_score",
  "threshold_version": "wavlm-v1",
  "model_loaded": true,
  "request_id": "918faa1a-baa9-4131-a47f-9220b7ceba49"
}
```
