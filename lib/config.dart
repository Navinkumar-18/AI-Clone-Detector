/// Backend base URL configuration.
///
/// Choose the appropriate URL for your environment:
///
/// 1. Physical Android Phone (Wi-Fi / LAN):
///    Must point to your Windows PC's IPv4 address on the same local network.
///    Find your PC's IP via `ipconfig` in PowerShell (e.g. Wireless LAN adapter Wi-Fi).
///
/// 2. Android Emulator:
///    Uses 'http://10.0.2.2:8000' (special emulator alias routing to host 127.0.0.1).
///
/// 3. Web / Desktop:
///    Uses 'http://localhost:8000'.
const String kEmulatorBackendUrl = 'http://10.0.2.2:8000';

/// Physical Android device LAN URL.
/// Update this if your PC's LAN IP changes (e.g. on a different Wi-Fi router).
const String kPhysicalDeviceBackendUrl = 'http://192.168.137.45:8000';

/// Active backend URL used across all API calls.
/// Defaults to [kPhysicalDeviceBackendUrl] for physical device testing.
/// Can also be overridden at build/run time via:
///   flutter run --dart-define=BACKEND_URL=http://10.0.2.2:8000
const String kBackendBaseUrl = String.fromEnvironment(
  'BACKEND_URL',
  defaultValue: kPhysicalDeviceBackendUrl,
);

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
