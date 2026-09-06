# VoiceGuard — Data Flow and Privacy Architecture

**Design Principles**: Designed around data-minimization and ephemeral-processing principles. This prototype has not undergone a formal legal or regulatory compliance assessment.  
**Processing Model**: Ephemeral audio handling with structured cleanup logging  

---

## 1. Ephemeral Audio Processing

VoiceGuard is designed for ephemeral audio processing. Temporary audio is processed for inference and cleaned up after processing. Structured cleanup events are logged without intentionally storing raw audio content.

> [!WARNING]
> **Privacy Scope & Limitations**:  
> Operating-system memory, swap files, crash dumps, client buffers, infrastructure logs, and backups are outside the guarantees of this prototype.

---

## 2. Physical Audio Capture Architecture

```
                      [ Incoming Phone Call ]
                                 │
                                 ▼
                     [ Mobile Device Speaker ]
                                 │
                 (Acoustic Air-Gap / Audio Wave)
                                 │
                                 ▼
                      [ Device Microphone ]
                                 │
                   (Audible to VoiceGuard App)
                                 │
                                 ▼
                 [ VoiceGuard Flutter PCM Stream ]
```

### Why Speakerphone Mode is Required:
Modern mobile operating systems strictly prohibit third-party applications from intercepting, recording, or tapping into the hardware baseband voice-call audio stream without system-level firmware privileges or root access. 

VoiceGuard captures audio through the device microphone while the user enables speakerphone. An in-app UX dialog guides the user to enable speakerphone before monitoring starts. Native cellular call interception and real telephony integration were not tested and are outside the prototype scope.

---

## 3. End-to-End Data Flow

```
1. Capture & Chunking (Client):
   PCM 16-bit Mono @ 16kHz.
   Rolling buffer captures 4-second chunk every 2 seconds (overlapping).

2. Transport (Client → Server):
   WAV-wrapped binary sent over HTTPS.
   Request tagged with client-generated or server-injected UUID (X-Request-ID).

3. Ingestion & Streaming Validation (Server):
   Server reads multipart upload via chunked stream (64 KB buffers).
   Accumulated bytes checked against maximum limit (10 MB). Oversized streams rejected (HTTP 413).

4. Transient Storage & Decoding (Server):
   Bytes written to ephemeral unique temp file: /tmp/vg_<UUID>.wav.
   librosa decodes audio to float32 mono @ 16kHz.

5. Pre-Inference Quality Gating (Server):
   AudioQualityEngine inspects RMS energy, clipping ratio, voiced frame ratio, and SNR.
   If silent / invalid: Immediately unlinks file and returns INSUFFICIENT_EVIDENCE.

6. Non-Blocking ML Inference (Server):
   Synchronous wav2vec2 model evaluated in a bounded ThreadPoolExecutor.
   Softmax output computes: prob_real and prob_fake (spoof_score).

7. Immediate Purge & Audit Logging (Server):
   In the request finally block, os.remove(tmp_path) is executed unconditionally.
   Audit log line emitted:
   "PRIVACY | Temp audio purged — no persistent storage of raw audio"

8. Response Dispatch (Server → Client):
   Only scalar metadata (spoof_score, risk_level, reason_codes, audio_quality) returned.
   Zero audio bytes returned or persisted.

9. Client-Side Aggregation (Client):
   RiskAggregator updates EMA and consecutive-threat counters.
   Updates visual risk gauge and controls simulated transaction hold state.
```

---

## 4. Privacy Principles

### Principle 1: No Biometric Enrollment or Voiceprint Storage
- VoiceGuard does not map, extract, or store voice biometric templates (such as x-vectors, d-vectors, or speaker embeddings).
- The system is an acoustic deepfake detector, not a speaker verification or voice identification engine. It cannot identify who is speaking, only whether the speech exhibits synthetic generation characteristics.

### Principle 2: Ephemeral Lifecycle
- Temporary audio is processed for inference and cleaned up after processing. Structured cleanup events are logged without storing raw audio content.
- Lifetime of audio on disk is bounded by request execution (typically 200 ms – 600 ms).
- No database tables, cloud buckets, or persistent storage volumes record audio payloads.

### Principle 3: No Transcription or Linguistic Analysis
- VoiceGuard analyzes raw spectral and acoustic wave properties.
- It does not perform automatic speech recognition (ASR), does not transcribe speech to text, and does not analyze semantic conversation context.

---

## 5. Privacy Audit Verification

Every request generates a structured server log confirming the purge:

```text
2026-09-06T23:15:05 | INFO | PREDICT | request_id=8f3b23c9-943f-4e09-b697-3f338d1502be | label=bonafide | spoof_score=0.0412 | risk=low
2026-09-06T23:15:05 | INFO | PRIVACY | Temp audio purged — no persistent storage of raw audio
```
