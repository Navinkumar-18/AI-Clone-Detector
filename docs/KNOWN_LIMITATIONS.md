# VoiceGuard — Known Limitations

## Model Limitations

1. **Score overlap on ASVspoof 2019 LA**: The committed threshold report records a 100% false-positive rate at the reviewed threshold. The available evaluation artifact shows substantial score overlap between bona-fide and spoof samples.

2. **Domain mismatch**: The model was trained on English speech from the ASVspoof 2019 dataset. It has NOT been evaluated on:
   - Indian-language speech (Hindi, Tamil, Telugu, etc.)
   - Telephony-quality audio (8kHz, codec artifacts)
   - Mobile microphone recordings in noisy environments
   - Indian accent English speech

3. **Uncalibrated outputs**: The `prob_fake` softmax score is NOT a calibrated probability. A score of 0.90 does NOT mean "90% chance of being fake."

4. **Attack coverage**: Only tested against ASVspoof 2019 LA attack algorithms (TTS/VC systems A07-A19). Modern voice cloning systems (e.g., XTTS, Bark, RVC) were not in the evaluation.

## System Limitations

1. **No call interception**: The app captures microphone audio via the `record` package. It does NOT intercept actual cellular or VoIP call audio. Speakerphone mode is required.

2. **No speaker verification**: The system does NOT verify who is speaking. It detects synthetic speech characteristics, not speaker identity.

3. **No real transaction blocking**: Transaction features are simulated for demonstration. No bank API, UPI, or payment system is integrated.

4. **Single-process rate limiter**: The in-memory rate limiter is suitable only for a single-process demo deployment.

5. **Self-signed TLS**: The demo uses self-signed certificates. These are NOT suitable for production deployment.

6. **Event-loop isolation**: Inference runs in a thread pool executor. Under heavy load, the bounded concurrency limit may cause requests to queue.

## Privacy Limitations

1. **Audio transmission**: Audio is transmitted to the backend over HTTPS for inference. In a production system, on-device inference would be preferred.

2. **Temporary files**: Audio is stored in temporary files during inference and deleted immediately after. The deletion is best-effort — a process crash could leave orphaned temp files.

## UI/UX Limitations

1. **No offline mode**: The app requires a backend connection. If the backend is unreachable, analysis cannot proceed.

2. **Latency**: Each analysis window takes 1-3 seconds for inference plus network round-trip. Real-time detection has a 4-6 second delay.

3. **No historical analysis**: The app does not persist call history or analysis results between sessions.
