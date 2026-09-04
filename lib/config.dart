/// config.dart — backend URL configuration.
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
