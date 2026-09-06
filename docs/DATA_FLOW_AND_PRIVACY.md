# VoiceGuard — Data Flow and Privacy Architecture

**Compliance Alignment**: Digital Personal Data Protection Act (DPDP 2023, India) & GDPR (General Data Protection Regulation)  
**Core Guarantee**: **Zero-Persistence Audio Handling**  

---

## 1. Physical Audio Capture Architecture

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
Modern mobile operating systems (Android 10+ and iOS) strictly prohibit third-party applications from intercepting, recording, or tapping into the hardware baseband voice-call audio stream without system-level firmware privileges or root access. 

To provide a secure, legal, and privacy-compliant demonstration without compromising OS sandboxing, VoiceGuard captures audio through the device microphone while the user enables speakerphone. An in-app UX nudge guides the user to enable speakerphone before monitoring starts.

---

## 2. End-to-End Data Flow

```
1. Capture & Chunking (Client):
   PCM 16-bit Mono @ 16kHz
   Rolling buffer captures 4-second chunk every 2 seconds (overlapping).

2. Transport (Client → Server):
   WAV-wrapped binary sent over HTTPS (TLS 1.3).
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

## 3. Privacy Guarantees & Legal Compliance

### Principle 1: No Biometric Enrollment or Template Storage
- VoiceGuard does **not** map, extract, or store voice biometric templates (such as x-vectors, d-vectors, or speaker embeddings).
- The system is an acoustic deepfake detector, not a speaker verification or voice identification engine. It cannot identify who is speaking, only whether the speech exhibits synthetic generation artifacts.

### Principle 2: Ephemeral Lifecycle & Zero Data at Rest
- Raw audio, PCM buffers, and decoded waveforms exist solely in transient volatility (RAM and short-lived temporary files).
- Lifetime of audio on disk: strictly bounded by request duration (typically 200 ms – 600 ms).
- No database tables, cloud buckets, or log aggregators ever record audio payloads.

### Principle 3: No Transcription or Linguistic Analysis
- VoiceGuard analyzes raw spectral and acoustic wave properties.
- It does **not** perform automatic speech recognition (ASR), does **not** transcribe speech to text, and does **not** analyze the semantic content or context of private conversations.

---

## 4. Privacy Audit Verification

Every request generates a traceable, sanitized server log confirming the purge:

```text
2026-09-06T23:15:05 | INFO | PREDICT | request_id=8f3b23c9-943f-4e09-b697-3f338d1502be | label=bonafide | spoof_score=0.0412 | risk=low
2026-09-06T23:15:05 | INFO | PRIVACY | Temp audio purged — no persistent storage of raw audio
```

To verify zero retention on disk:
```bash
# Verify no residual WAV files remain in system temp directory:
ls /tmp/vg_* 2>/dev/null || dir %TEMP%\vg_*
# Result: File Not Found (Zero persistence confirmed)
```
