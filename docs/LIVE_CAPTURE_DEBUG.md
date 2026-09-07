# VoiceGuard Live Acoustic Capture & Fail-Closed Architecture

## 1. Executive Summary

During live physical testing where AI-generated audio was played near a smartphone microphone, the VoiceGuard Flutter client was observed displaying a green / `LOW_RISK` signal. 

A thorough end-to-end investigation identified **two distinct root causes**:
1. **Client & UI Fail-Open Logic**:
   - `RiskIndicator` and `TransactionCard` wildcard handlers (`_ =>`) defaulted directly to green (`0xFF2E7D32`) and allowed transactions.
   - `LiveCallScreen` unconditionally set `risk = 'low'` and `decision = 'low_risk'` whenever the rolling EMA score fell below the medium threshold (`0.30`), without verifying whether valid speech was present, whether the backend model was loaded, or whether blocking reason codes existed.
   - Network errors, timeouts, or 503 errors silently kept the old UI state indefinitely without failing closed.
2. **Acoustic Transducer Domain Shift**:
   - The speakerphone path substantially changes the captured signal and may attenuate or distort features used by the model. In this test, the Wav2Vec2 score dropped from 0.4844 to 0.0468, indicating sensitivity to the acoustic transmission path. The precise frequency-level cause requires further analysis.
   - In contrast, the candidate `wavlm_mlp` model produced scores of `0.9409` on digital audio and `0.9963` under speakerphone simulation. However, as established in the model experiment audit, `wavlm_mlp` remains experimental with high false-positive rates (FPR: 72%, ROC-AUC: 0.4879) and is not production-ready.
   - Green is strictly defined and displayed as "low detected spoof evidence under valid capture", never as "genuine," "safe," or "authenticated."

---

## 2. End-to-End Pipeline Architecture

```text
Physical Audio (Loudspeaker / Speakerphone)
       │
       ▼
Microphone Capture (record package)
       │ (16 kHz, Mono, 16-bit PCM, 2 bytes/sample = 32,000 bytes/sec)
       ▼
Client Pre-Analysis Validation
  ├── Format Validation: Check sample rate, channels, bit-depth alignment
  ├── Digital Silence Gate: RMS < 0.003 → Immediately transition to INSUFFICIENT_EVIDENCE
       │
       ▼ (If RMS >= 0.003)
PCM-to-WAV Wrapping (44-byte canonical RIFF/WAVE header)
       │
       ▼
POST /live/analyze (Multipart form-data)
       │
       ├── Backend Audio Quality Gate (RMS, SNR, clipping, speech detection)
       ├── Active Model Backend Inference (WavLM-MLP or Wav2Vec2)
       └── Canonical DetectionResponse (threshold-v2 schema, model_loaded flag)
       │
       ▼
Client Risk Aggregation & Fail-Closed Enforcement
  ├── Check 1: modelLoaded == true (fail-closed: missing/null defaults to false)
  ├── Check 2: speechDetected == true
  ├── Check 3: audioQuality.status == 'acceptable'
  ├── Check 4: response is fresh (< 15 seconds)
  ├── Check 5: no blocking reason codes present
  └── Check 6: neither ACTION_HELD nor cooldown active
       │
       ▼
UI Rendering (RiskIndicator & TransactionCard)
  ├── Green (LOW_RISK) ONLY if ALL 6 checks pass
  ├── Grey (INSUFFICIENT_EVIDENCE) on silence, unverified audio, or offline engine
  ├── Orange (VERIFICATION_REQUIRED) on elevated anomaly score
  └── Red (ACTION_HELD) on persistent spoofing (latched until cooldown)
```

---

## 3. The 3 Implementation Cautions Enforced

### Caution 1: `modelLoaded` Production Parsing
- In `LiveAnalysisResult.fromJson` and `PredictResult.fromJson`, `modelLoaded` is parsed strictly as:
  ```dart
  modelLoaded: json['model_loaded'] == true
  ```
- Missing, null, or malformed values **strictly default to `false`**.
- A missing readiness field **never enables green**.

### Caution 2: Capture Format Validation
- The assumption that 32,000 bytes = 1 second is valid **only after confirming the microphone configuration**:
  - Sample Rate: `16,000 Hz`
  - Channels: `1` (Mono)
  - Bit Depth: `16-bit signed PCM`
  - Bytes per sample: `2`
- Buffer size slicing strictly computes sample alignment:
  ```dart
  final bytesPerSec = _captureSampleRate * _bytesPerSample;
  final alignedBufferLength = _pcmBuffer.length - (_pcmBuffer.length % _bytesPerSample);
  ```
- Buffer length is verified before slicing, and odd trailing bytes are discarded.

### Caution 3: Informational vs Blocking Reason Codes
- Informational reason codes (e.g., `chunk_analyzed_ok`, `clean_audio_profile`) **do not block green**.
- Green is blocked **only by blocking reason codes**:
  - `no_speech`
  - `audio_quality_poor`
  - `backend_unavailable`
  - `model_unavailable`
  - `analysis_stale`
  - `capture_unavailable`
  - `empty_or_invalid_capture`
  - `elevated_spoof_score`
  - `persistent_high_spoof_evidence`
  - `persistent_high_spoof_score`
  - `action_remains_held`
  - `recovering_from_action_held`

---

## 4. Live UI Dev Diagnostics Panel

In `LiveCallScreen`, a collapsible `LIVE DIAGNOSTICS (DEV MODE)` tile is integrated directly into the live call view:
- **Mic Stream Config**: `16000 Hz / 1ch / 16bit PCM`
- **Bytes / Sample**: `2 bytes`
- **Rolling Buffer**: Byte count and computed audio seconds
- **Chunk RMS (Speech Gate)**: Numerical RMS value and status (`SILENCE (<0.003)` vs `ACTIVE`)
- **Backend Connection**: `ONLINE (200 OK)` vs `OFFLINE / UNREACHABLE`
- **Model Loaded Flag**: `TRUE (READY)` vs `FALSE (FAIL-CLOSED)`
- **Freshness**: Elapsed seconds since last valid response
- **canShowLowRisk Gate**: Boolean flag governing the green UI state
- **Effective Decision & Action**: Real-time aggregated states
- **Consecutive High Windows & Hold Status**: Persistence and cooldown counters
- **Active Reason Codes**: Raw reason codes list

---

## 5. Security Demonstration of Success Conditions

| Scenario | Input Condition | Effective Decision | UI Color | Action / Transaction Button |
|---|---|---|---|---|
| **Initial State** | Zero audio frames processed | `insufficient_evidence` | Grey (`0xFF616161`) | Disabled (`unavailable`) |
| **Silence** | Muted mic / RMS < 0.003 | `insufficient_evidence` | Grey (`0xFF616161`) | Disabled (`unavailable`) |
| **Backend Stopped** | Backend 503 / network drop | `backend_unavailable` | Red / Grey | Disabled (`unavailable`) |
| **Invalid Capture** | Truncated / empty audio | `capture_unavailable` | Grey (`0xFF757575`) | Disabled (`unavailable`) |
| **High-Risk Persistent** | 2+ windows score >= 0.85 | `action_held` | Red (`0xFFC62828`) | Disabled (`hold`) |
| **Silence After Action Held**| Mic goes silent after hold | `action_held` | Red (`0xFFC62828`) | Action remains held (`hold`) |
| **Valid Fresh Low-Risk** | Fresh, speech, acceptable quality, low score | `low_risk` | Green (`0xFF2E7D32`) | Enabled (`allow_with_caution`) |
