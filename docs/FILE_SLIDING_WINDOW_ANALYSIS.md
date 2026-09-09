# VoiceGuard File Sliding-Window Analysis Contract & Implementation

This document describes the design, API contract, fail-closed safety invariants, and UI telemetry presentation for VoiceGuard's uploaded-audio sliding-window analysis feature (`POST /file/analyze_windows`).

---

## 1. Overview & Goal

When a user uploads an audio recording or screens a voice file, VoiceGuard partitions the file into overlapping 4.0-second analysis windows advancing by a 2.0-second stride (50% overlap). Every window displays full acoustic and inference telemetry matching the VoiceGuard Live Call Analysis experience.

---

## 2. Windowing Architecture: 4-Second Window & 2-Second Stride

Audio files are sampled at **16,000 Hz**, mono, 16-bit PCM:
- **Full window length ($W$):** `4,000 ms` = `64,000 samples` (128,000 bytes)
- **Stride ($S$):** `2,000 ms` = `32,000 samples` (64,000 bytes)
- **Overlap:** `2,000 ms` (50% overlap across consecutive windows)

### Window Generation Algorithm
For an audio file of duration $D$:
```text
start = 0.0
while start + 4.0 <= D:
    process [start, start + 4.0]
    start += 2.0
```

For example, a **10.0-second** audio file generates exactly 4 complete windows:
- **Window 0:** `00:00–00:04` (samples 0 to 64,000)
- **Window 1:** `00:02–00:06` (samples 32,000 to 96,000)
- **Window 2:** `00:04–00:08` (samples 64,000 to 128,000)
- **Window 3:** `00:06–00:10` (samples 96,000 to 160,000)

Every complete window satisfies:
- `duration_seconds == 4.0`
- `window_complete == true`

---

## 3. Short-File & Final-Partial-Window Policies

### Short-File Policy ($D < 4.0\text{ seconds}$)
If the total file duration is less than the required 4.0-second model window:
- The system **never** pads with silence or pretends the audio was 4 seconds long.
- Model inference is **bypassed**.
- Exactly one window descriptor is returned:
  - `window_complete = false`
  - `decision = "insufficient_evidence"`
  - `action = "unavailable"`
  - `quality_status = "short_audio"`
  - `reason_codes = ["short_audio"]`
  - `spoof_score = null`, `confidence = null`, `rms = null`

### Final Partial Segment Policy ($D \ge 4.0\text{ seconds}$)
If the file duration $D$ does not align with a 2-second stride multiple and audio remains after the last complete window:
- For example, a **7.0-second** file yields:
  - Window 0: `00:00–00:04` (`window_complete = true`, inference executed)
  - Window 1: `00:02–00:06` (`window_complete = true`, inference executed)
  - Partial segment: `00:04–00:07` (`duration_seconds = 3.0`, `window_complete = false`)
- The partial segment is marked:
  - `window_complete = false`
  - `decision = "insufficient_evidence"`
  - `action = "verify"`
  - `quality_status = "partial_window"`
  - `reason_codes = ["partial_window"]`
- **Zero incomplete or partial segments ever reach the model inference engine.**

---

## 4. Per-Window & File-Level Response Schemas

### Window Item Schema (`SlidingWindowItemResponse`)
```python
class SlidingWindowItemResponse(BaseModel):
    window_index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    window_complete: bool

    decision: str
    action: str
    label: str
    spoof_score: Optional[float]
    confidence: Optional[float]
    score_type: str

    risk_level: str
    speech_detected: bool
    quality_status: str

    rms: Optional[float]
    snr_db: Optional[float]
    voiced_ratio: Optional[float]
    reason_codes: list[str]

    model_backend: str
    model_version: str
    threshold_version: str
    model_loaded: bool
```

### File-Level Aggregation Schema (`FileSlidingAnalysisResponse`)
```python
class FileSlidingAnalysisResponse(BaseModel):
    total_duration_seconds: float
    window_length_seconds: float
    stride_seconds: float
    window_count: int
    complete_window_count: int

    high_risk_windows: int
    maximum_spoof_score: Optional[float]
    average_spoof_score: Optional[float]
    ema_score: Optional[float]

    decision: str
    action: str
    persistence_triggered: bool
    consecutive_high_count: int
    reason_codes: list[str]

    model_backend: str
    model_version: str
    score_type: str
    threshold_version: str
    model_loaded: bool

    windows: list[SlidingWindowItemResponse]
```

See [evaluation/file_sliding_analysis_example.json](../evaluation/file_sliding_analysis_example.json) for a complete example payload.

---

## 5. Risk Semantics & Aggregation Rules

File-level decision logic matches the Live Call Analysis rules:
1. **Exponential Moving Average (EMA):**
   $$\text{EMA}_t = \alpha \cdot s_t + (1 - \alpha) \cdot \text{EMA}_{t-1}$$
   where $\alpha = 0.3$ (from `cfg.risk_aggregation.ema_alpha`).
2. **Persistence Rule:**
   If consecutive high-risk windows reach $N = 2$ (`cfg.risk_aggregation.persistent_high_windows`), `persistence_triggered = true`.
3. **Action Held Rule:**
   When `persistence_triggered == true`, file-level `decision = "action_held"` and `action = "hold"`.
4. **Elevated Anomaly Rule:**
   If `high_risk_windows > 0` or $\text{EMA} \ge 0.85$, `decision = "verification_required"` and `action = "verify"`.

---

## 6. Differences Between VoiceGuard Endpoints

| Feature | `POST /predict` | `POST /live/analyze` | `POST /file/analyze_windows` |
| :--- | :--- | :--- | :--- |
| **Primary Use Case** | Single uploaded audio file screening | Real-time chunk from microphone stream | Detailed file screening with rolling windows |
| **Input Format** | Audio file (WAV, MP3, M4A, FLAC, OGG, AAC) | Single WAV chunk (exactly 4.0s) | Audio file (WAV, MP3, M4A, FLAC, OGG, AAC) |
| **Windowing** | Evaluates whole file up to max duration | Single 4-second chunk from circular buffer | Partitioned into 4.0s windows with 2.0s stride |
| **Response Schema** | `DetectionResponse` (single file summary) | `DetectionResponse` (clip-level evidence) | `FileSlidingAnalysisResponse` (windows + summary) |
| **Aggregation** | Single model inference | Client-side `LiveRiskAggregator` | Server-side sliding-window aggregator |
| **Preservation** | **Preserved untouched** as fallback & regression path | **Preserved untouched** for live calls | **New dedicated endpoint** |

---

## 7. Fail-Closed UI Rules

Under VoiceGuard security requirements:
- A window can **only** be displayed in green (`low_risk`) if:
  $$\begin{aligned}
  &\text{decision} == \text{"low\_risk"} \\
  &\land \text{window\_complete} == \text{true} \\
  &\land \text{speech\_detected} == \text{true} \\
  &\land \text{quality\_status} == \text{"acceptable"} \\
  &\land \text{model\_loaded} == \text{true} \\
  &\land \text{no blocking reason codes}
  \end{aligned}$$
- The overall file result is never green if:
  - Any required result is missing.
  - The model backend is unavailable (`model_loaded == false`).
  - The file cannot be decoded.
  - All windows are incomplete ($D < 4.0\text{s}$).
  - Audio quality is silent or insufficient.
  - The response is malformed or backend fails.
- In these cases, the UI displays `INSUFFICIENT EVIDENCE`, `ANALYSIS UNAVAILABLE`, or `MODEL UNAVAILABLE`.

### Client Fallback Behavior
If `/file/analyze_windows` returns 404 or 405 (e.g. on an un-upgraded deployment), the mobile client falls back to `/predict` and displays:
```text
Sliding-window details unavailable; showing single-file analysis.
```
No fake or synthesized window details are ever fabricated.

---

## 8. Limitations & Disclaimers

1. **Acoustic & Environmental Constraints:**
   High background noise, extreme reverberation, or speakerphone coupling may degrade audio quality metrics (low SNR, clipping), prompting `INSUFFICIENT EVIDENCE` rather than definitive classification.
2. **Model Domain & Generalization:**
   Synthetic voice detection algorithms are calibrated on benchmark deepfake distributions. Novel vocoders, unseen codecs, or adversarial perturbations may exhibit reduced confidence.
3. **No Caller Identity Authentication:**
   VoiceGuard detects synthetic voice characteristics and acoustic manipulation. It is **not** a speaker biometric verification system and does not claim to authenticate who the caller is.
4. **Simulated Prevention / Banking Transactions:**
   The `TransactionCard` and step-up verification actions represent demo simulations of financial risk intervention workflows. No real bank APIs or monetary transfers are executed.
