/// config.dart — backend URL configuration and live analysis constants.
///
/// Change [kBackendBaseUrl] here when swapping between the stub and the real backend.
/// Do NOT change this value anywhere else in the codebase.
///
/// Common values:
///   Android emulator → host machine:  'http://10.0.2.2:8000'
///   Web / desktop (localhost):         'http://localhost:8000'
///   Physical Android device (LAN):     'http://192.168.x.x:8000'  ← replace with your machine's LAN IP
const String kBackendBaseUrl = 'http://10.0.2.2:8000';

/// Request timeout for the /predict call.
/// Large audio files on a slow demo-day Wi-Fi may need more time.
const Duration kRequestTimeout = Duration(seconds: 30);

// ---------------------------------------------------------------------------
// Live Call Analysis constants
// ---------------------------------------------------------------------------

/// Duration of each analysis window sent to the backend (seconds).
/// The rolling buffer captures this much audio before sending.
const int kLiveChunkDurationSec = 4;

/// Interval between consecutive analysis sends (seconds).
/// Together with kLiveChunkDurationSec, this creates overlapping windows:
///   0s──────4s
///         2s──────6s
///               4s──────8s
const int kLiveAnalysisIntervalSec = 2;

/// Sample rate for live audio capture — must match backend's SAMPLE_RATE.
const int kLiveSampleRate = 16000;

/// How often to check backend connectivity during a live call (seconds).
const int kHealthCheckIntervalSec = 5;

/// Request timeout for live chunk analysis — shorter than full-file predict
/// because chunks are small (4 seconds of audio).
const Duration kLiveRequestTimeout = Duration(seconds: 15);
