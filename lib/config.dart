/// Backend configuration for VoiceGuard.
///
/// Contains:
///   - [BackendConfig]: singleton managing the server URL (persisted via SharedPreferences).
///   - Compile-time fallback URLs and live-call-analysis constants.
///   - Demo mode flag controlling TLS bypass behavior.
library;

import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

// ---------------------------------------------------------------------------
// Default backend URLs (compile-time constants, internal fallbacks only)
// ---------------------------------------------------------------------------

/// Default URL for the Android Emulator (10.0.2.2 aliases host loopback).
const String kEmulatorBackendUrl = 'https://10.0.2.2:8443';

/// Default URL for a physical Android device on the same LAN.
/// Update this if your PC's LAN IP changes.
const String kPhysicalDeviceBackendUrl = 'https://192.168.137.45:8443';

/// Default backend URL used as the initial fallback.
/// Dynamically resolves to:
/// - Desktop (Windows/macOS/Linux): https://127.0.0.1:8443
/// - Android / Mobile: https://10.0.2.2:8443 (emulator loopback)
/// Can be overridden at build/run time via:
///   flutter run --dart-define=BACKEND_URL=https://...
String get kDefaultBackendUrl {
  const env = String.fromEnvironment('BACKEND_URL');
  if (env.isNotEmpty) return env;
  try {
    if (Platform.isWindows || Platform.isMacOS || Platform.isLinux) {
      return 'https://127.0.0.1:8443';
    }
  } catch (_) {
    // Web or non-IO platform fallback
  }
  return kEmulatorBackendUrl;
}

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
/// This value is obtained from config/model_config.yaml via the backend.
const int kLiveSampleRate = 16000;

/// How often to check backend connectivity during a live call (seconds).
const int kHealthCheckIntervalSec = 5;

/// Request timeout for live chunk analysis — shorter than full-file predict
/// because chunks are small (4 seconds of audio).
const Duration kLiveRequestTimeout = Duration(seconds: 15);

// ---------------------------------------------------------------------------
// Demo mode — controls TLS bypass and UI indicators
// ---------------------------------------------------------------------------

/// Demo mode flag. Defaults to FALSE (secure by default).
/// Insecure TLS bypass is DISABLED by default.
///
/// To enable for local testing with self-signed certificates:
///   flutter run --dart-define=DEMO_MODE=true
///
/// This must NEVER be enabled in release/production builds.
const bool kDemoMode = bool.fromEnvironment('DEMO_MODE', defaultValue: false);

// ---------------------------------------------------------------------------
// BackendConfig — runtime-configurable server URL
// ---------------------------------------------------------------------------

/// SharedPreferences key for the persisted backend URL.
const String _kBackendUrlKey = 'voiceguard_backend_url';

/// Singleton managing the active backend URL.
///
/// [init] must be called once at startup (before runApp).  After that,
/// [baseUrl] returns the current URL and [setBaseUrl] validates + saves
/// a new URL (checking /health before committing).
class BackendConfig {
  BackendConfig._();

  static String _baseUrl = kDefaultBackendUrl;

  /// Current active backend base URL (no trailing slash).
  static String get baseUrl => _baseUrl;

  /// Load any previously-saved URL from SharedPreferences.
  /// Must be called once before runApp().
  static Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString(_kBackendUrlKey);
    if (saved != null && saved.isNotEmpty) {
      _baseUrl = saved;
    }
  }

  /// Validate [url] by hitting its /health endpoint, then persist if healthy.
  ///
  /// Returns `true` on success, `false` if the URL is unreachable or invalid.
  static Future<bool> setBaseUrl(String url) async {
    // Normalize: strip trailing slash
    final normalized = url.endsWith('/') ? url.substring(0, url.length - 1) : url;

    // Validate by calling /health
    try {
      final client = HttpClient();
      // Scoped only to target host and only when demo mode is active
      if (kDemoMode && !kReleaseMode) {
        final targetHost = Uri.parse(normalized).host;
        client.badCertificateCallback = (cert, host, port) => host == targetHost;
      }
      final request = await client.getUrl(Uri.parse('$normalized/health'));
      final response = await request.close().timeout(const Duration(seconds: 5));
      final body = await response.transform(utf8.decoder).join();
      client.close();

      if (response.statusCode == 200) {
        final json = jsonDecode(body) as Map<String, dynamic>;
        if (json['model_loaded'] == true) {
          _baseUrl = normalized;
          final prefs = await SharedPreferences.getInstance();
          await prefs.setString(_kBackendUrlKey, normalized);
          return true;
        }
      }
      return false;
    } catch (_) {
      return false;
    }
  }

  /// Reset to the compile-time default and clear persisted value.
  static Future<void> resetToDefault() async {
    _baseUrl = kDefaultBackendUrl;
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_kBackendUrlKey);
  }
}
