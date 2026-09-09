# VoiceGuard Live-Window vs Upload Comparison

## Purpose

This document records the methodology and results of comparing:

- **Path A** — `POST /predict` with the complete audio file (upload path).
- **Path B** — `POST /live/analyze` with deterministic 4-second windows (clean live path).
- **Path C** — `POST /live/analyze` with simulated speakerphone acoustic degradation.

The comparison verifies that the **same complete audio waveform** produces equivalent model inputs and equivalent scores across both paths.

---

## Parity Test Methodology

```
Same WAV file
    → Path A: POST /predict (full file)
    → Path B: POST /live/analyze (4-second windows, stride 2s)
    → compare spoof_score, label, risk_level per window
```

Both paths call `_load_audio_file(path, sr=16000)` → `librosa.load(..., mono=True)` → the same float32 array → the same model backend.

Parity is expected to hold within floating-point tolerance (±0.001) for the same audio segment.

---

## Fields Recorded per Path

| Field | Description |
|---|---|
| `path` | A / B / C |
| `window_duration_ms` | Actual decoded audio duration |
| `sample_rate` | Confirmed from WAV header |
| `channels` | 1 = mono |
| `bits_per_sample` | 16 |
| `rms` | Root-mean-square energy |
| `speech_detected` | Boolean |
| `audio_quality` | acceptable / poor / silent / invalid |
| `spoof_score` | Raw prob_fake [0–1] |
| `score_type` | uncalibrated_softmax_score |
| `decision` | low_risk / verification_required / insufficient_evidence |
| `action` | allow_with_caution / verify / hold |
| `reason_codes` | List of codes |
| `model_backend` | wav2vec2 / wavlm_mlp |
| `latency_ms` | Wall-clock request latency |

---

## Running the Comparison

```bash
python evaluation/compare_live_paths.py test_clip.wav
# Results saved to: evaluation/live_vs_upload_comparison.json
```

---

## Interpretation

| Result | Meaning |
|---|---|
| `spoof_score` within ±0.05 across all 4-second windows | Parity confirmed |
| Score differs by > 0.05 | Document the exact cause (segment offset, length mismatch, resampling) |
| Path C scores diverge | Expected — speakerphone degrades the acoustic signal (bandpass + transducer saturation + room noise) |

---

## Known Discrepancy Sources

1. **Segment offset**: Path B windows are at 0s–4s, 2s–6s, 4s–8s. Path A uses the full file. Scores may differ slightly per segment position.
2. **Speakerphone degradation (Path C)**: Bandpass (300–3500 Hz), tanh soft-clipping, and ambient noise simulate acoustic transducer behavior and are expected to change the score.
3. **Buffering artifacts**: Any live chunk shorter than 4000 ms is rejected before reaching the model and does not appear in the score comparison.

---

## Separation of Result Categories

The upload path produces the expected result on the tested AI and human audio files. The live path is being hardened to ensure that only complete, consistently formatted windows reach inference. Live microphone and speakerphone behavior remains device- and acoustic-path dependent.

| Category | Status |
|---|---|
| Model result on complete uploaded files | **Validated** — AI-gen and human audio produce expected results |
| Complete live-window output (Path B) | **Parity expected** — same preprocessing and model path |
| Live microphone capture (physical device) | **Device-dependent** — not fully validated across devices |
| Speakerphone replay (Path C) | **Expected degradation** — acoustic path changes the signal |
| UI and aggregator safety behavior | **Enforced** — partial windows, silence, and failures fail closed |

---

## Disclaimer

Do not claim:

- Universal AI-voice detection.
- Caller authentication.
- Production-grade accuracy.
- Cross-device live-capture consistency.

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for the full limitation list.
