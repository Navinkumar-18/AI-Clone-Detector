# VoiceGuard — Threat Model & Security Architecture

**Framework**: STRIDE-Based Threat Modeling  
**Target Architecture**: Real-Time Acoustic Deepfake Gating & Risk Aggregation  

---

## 1. System Overview

VoiceGuard is a client-server security prototype designed to evaluate incoming speech streams for synthetic acoustic artifacts and protect simulated high-value operations through temporal risk aggregation and step-up verification.

---

## 2. Threat Actors & Capabilities

| Threat Actor | Motivation | Capabilities |
| :--- | :--- | :--- |
| **Opportunistic Voice Impersonator** | Financial gain via social engineering | Access to public consumer TTS/VC tools (ElevenLabs, PlayHT). Clones victim voice from social media clips. |
| **Targeted Executive Impersonator** | High-value CEO fraud / wire transfer | High-quality bespoke diffusion/neural vocoder models, access to target caller ID spoofing. |
| **Network Eavesdropper (Man-in-the-Middle)** | Intercept private conversation / replay auth | Network position on local Wi-Fi / cellular data path. Attempts to inspect audio or spoof server responses. |
| **Malicious Client / DoS Attacker** | Disrupt detection availability | Flooding API with massive file payloads, corrupted audio streams, or concurrent requests. |

---

## 3. Trust Boundaries & Assumptions

```
 [ Untrusted Environment ]           [ Client Trust Boundary ]          [ Server Trust Boundary ]
 ┌──────────────────────┐            ┌───────────────────────────┐      ┌───────────────────────────┐
 │ Remote Caller Stream │ ──(Audio)─►│ Smartphone Microphone /   │      │ VoiceGuard FastAPI Server │
 │ (Human / Cloned AI)  │            │ Speakerphone Capture      │      │                           │
 └──────────────────────┘            │                           │      │ - Audio Quality Filter    │
                                     │ - Sliding Risk Aggregator │      │ - wav2vec2 Deepfake Model │
                                     │ - Transaction Interceptor │      │ - Ephemeral Temp Storage  │
                                     │ - Scoped TLS Client       │      └───────────────────────────┘
                                     └─────────────┬─────────────┘                    ▲
                                                   │                                  │
                                                   └───[ HTTPS / TLS 1.3 ]────────────┘
                                                       (Encrypted Transient Chunks)
```

### Core Security Assumptions:
1. **Zero-Trust Network**: All communications between client and server are assumed to traverse potentially hostile networks. Transport Layer Security (TLS) is mandatory.
2. **Untrusted Input Stream**: Audio input is untrusted and may contain silence, clipped noise, corrupted file headers, or adversarial payloads.
3. **Ephemeral Audio Handling**: The server processes temporary audio for inference and cleans up files immediately without writing caller voiceprints, raw PCM arrays, or audio files to persistent databases or long-term storage.

---

## 4. Attack Vectors & Implemented Mitigations

### Attack 1: Real-Time Generative Voice-Cloning Impersonation
- **Description**: Adversary trains or fine-tunes a voice-cloning model (e.g., ElevenLabs, XTTS, RVC) on 3 seconds of target voice to authorize a sensitive transfer.
- **Mitigation**: Pretrained wav2vec2 feature extractor detects acoustic generation characteristics (spectral discontinuities, phase inconsistencies, vocoder signatures). Softmax spoof scores elevate risk.
- **Defense-in-Depth**: Client-side **Risk Aggregator** requires 2 consecutive elevated windows before triggering `ACTION_HELD`, holding the transaction and presenting simulated step-up verification options.

### Attack 2: Silence / Muted Microphone Evasion
- **Description**: Adversary stays silent or mutes microphone during sensitive transaction prompts to avoid detection, attempting to bypass security with false bonafide scores.
- **Mitigation**: Pre-inference **Audio Quality Engine** (`audio_quality.py`) measures RMS energy. Signals below silence threshold return `insufficient_evidence` with `speech_detected: false`.
- **Security Invariant**: Silence **never** returns `bonafide` and **never** clears an active `ACTION_HELD` state.
  ```text
  ACTION_HELD
      → silence/backend failure
      → INSUFFICIENT_EVIDENCE
      → sensitive action remains held
  ```

### Attack 3: Single-Spike Audio Glitch Manipulation
- **Description**: Transient network packets, cellular dropout, or coughing causes an isolated artifact, which in a naive system would immediately trigger holds.
- **Mitigation**: Exponential Moving Average ($\alpha = 0.3$) smoothing reduces the impact of isolated score spikes; persistent high evidence is required before an action hold. A single spike generates `VERIFICATION_REQUIRED`, while only persistent evidence locks `ACTION_HELD`. Recovery requires 3 consecutive clean windows (hysteresis).

### Attack 4: Denial-of-Service via Oversized Payloads (Zip Bombs / Massive Audio)
- **Description**: Adversary streams multi-gigabyte files to crash server memory.
- **Mitigation**: `_read_upload_limited()` streams uploads in 64 KB chunks and enforces a strict 10 MB maximum limit, rejecting with `HTTP 413 Payload Too Large` before loading into RAM. Decoded audio duration is capped at 60 seconds.

### Attack 5: Concurrency Overload & Event Loop Starvation
- **Description**: Flooding concurrent inference requests blocks async I/O handlers.
- **Mitigation**: PyTorch inference is decoupled from the asyncio event loop into a bounded `ThreadPoolExecutor` governed by an `asyncio.Semaphore`. Excess traffic is throttled via per-client IP rate limiting (60 req/min, `HTTP 429`). The in-memory rate limiter is suitable for a single-process prototype only and is not sufficient for distributed production deployment.

### Attack 6: Man-in-the-Middle (MITM) & Compromised Certificate Re-use
- **Description**: Using tracked repository certificates to intercept live audio traffic.
- **Mitigation**: Removed `cert.pem` and `key.pem` from Git tracking on the hardening branch and added certificate/key patterns to `.gitignore`. Because these files were previously committed, the old private key must be treated as compromised. Historical removal was not automatically performed. Public deployment requires new trusted certificate/key material. Insecure TLS bypass is disabled by default (`kDemoMode = false`) and forbidden in production release builds. Demo TLS exceptions in `IOClient` are strictly scoped only to the configured VoiceGuard host.

---

## 5. Residual Risks & Out-of-Scope Threats

1. **Human Social Engineering Without Voice Cloning**: An authentic human fraudster tricking a victim is outside the scope of deepfake acoustic detection.
2. **Severe Telephony Compression (2G GSM / AMR 4.75kbps)**: Low-bitrate telephony can strip high-frequency vocoder cues, increasing false positive rates.
3. **Compromised Operating System / Rooted Device**: If the host device OS is compromised at the kernel level, local microphone audio or display memory could be intercepted.
4. **Ephemeral Storage Scope**: Operating-system memory, swap files, crash dumps, client buffers, infrastructure logs, and backups are outside the guarantees of this prototype.
