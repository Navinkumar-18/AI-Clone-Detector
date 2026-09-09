# VoiceGuard Live-Window Contract

## Summary

VoiceGuard's current live-analysis contract requires a **complete 4-second mono, 16 kHz audio window** before any inference is performed. This contract prevents unstable inference on partial startup or timer-flush chunks. The exact training-duration distribution of the model is documented separately (see [MODEL_EVALUATION.md](MODEL_EVALUATION.md)).

---

## Contract Parameters

These values are the single source of truth and live in `config/model_config.yaml → live_analysis`:

| Parameter | Value | Description |
|---|---|---|
| `full_window_ms` | 4000 ms | Minimum audio length before inference is attempted |
| `stride_ms` | 2000 ms | How much to advance the rolling buffer after each window |
| `sample_rate` | 16 000 Hz | Must match `audio.sample_rate` |
| `channels` | 1 | Mono only |
| `bits_per_sample` | 16 | Signed PCM |

---

## Partial-Window Rejection Policy

Any audio chunk shorter than `full_window_ms` sent to `POST /live/analyze` **must not**:

- Reach model inference.
- Update the EMA risk score.
- Increment the consecutive-high-window counter.
- Count as a clean recovery window.
- Produce `LOW_RISK`.
- Clear `ACTION_HELD`.
- Reset `VERIFICATION_REQUIRED`.

The backend returns:

```json
{
  "decision": "insufficient_evidence",
  "action": "verify",
  "label": "unknown",
  "speech_detected": false,
  "reason_codes": ["partial_window"]
}
```

This guard is **not applied to `POST /predict`**. The upload path accepts any duration that passes the normal quality checks (minimum 0.5 s configured in `audio.minimum_duration_seconds`).

---

## Rolling-Buffer Algorithm (Flutter)

```
while pcm_buffer_duration_ms >= full_window_ms:
    window = pcm_buffer[0 .. full_window_bytes]
    remove stride_bytes from front of pcm_buffer   # preserve overlap tail
    send window to /live/analyze

if pcm_buffer_duration_ms < full_window_ms:
    display: ANALYZING / INSUFFICIENT_EVIDENCE
    do NOT send partial buffer to backend
```

The buffer is trimmed to a maximum of `3 × full_window_ms` bytes to prevent unbounded growth.

When audio capture stops, any remaining partial buffer is **not classified**.

---

## First-Window Behavior

After starting a live call, the first `full_window_ms` (4 seconds) of audio is accumulated before any inference runs. During this accumulation period the UI must display:

```
ANALYZING  or  INSUFFICIENT_EVIDENCE
```

It must **never** show green / LOW_RISK during the initial accumulation period.

---

## Preprocessing Chain

Both `POST /predict` and `POST /live/analyze` use the same decoding and preprocessing path:

1. **Save to temp file** — raw bytes written to disk.
2. **Decode with librosa** — `librosa.load(path, sr=16000, mono=True)` → float32 array in `[-1.0, 1.0]`.
3. **PyAV fallback** — if librosa fails (e.g., WEBM, OPUS).
4. **Quality check** — RMS, clipping, voiced ratio, SNR.
5. **Inference** — passed directly to the model backend.

The live path adds one additional gate before step 4:
- Duration check: `len(audio) / sr * 1000 >= full_window_ms`.

---

## Configuration Locations

| File | Key |
|---|---|
| `config/model_config.yaml` | `live_analysis.*` (authoritative) |
| `voiceguard_config.py` | `LiveAnalysisConfig` dataclass |
| `lib/config.dart` | `kLiveFullWindowMs`, `kLiveStrideMs`, `kLiveSampleRate` |
| `tests/test_live_window_contract.py` | All contract enforcement tests |
| `evaluation/compare_live_paths.py` | `chunk_size = sr * 4`, `step_size = sr * 2` |

---

## Known Limitations

- The exact training-time window distribution of `garystafford/wav2vec2-deepfake-voice-detector` is not publicly documented. The 4-second contract is adopted because: (a) wav2vec2 accepts variable-length inputs, (b) complete windows produce more consistent results than short fragments, and (c) the upload path (which produces validated results) typically provides multi-second audio.
- Live microphone capture quality is device- and OS-dependent. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).
- Speakerphone acoustic replay changes the spectral characteristics of the captured signal.
