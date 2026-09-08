/// All API logic for VoiceGuard.
///
/// To swap the backend: use the settings screen or edit [kDefaultBackendUrl] in config.dart.
/// To change request/response shape: edit models and methods here only.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:http/io_client.dart' as http_io;
import 'config.dart';
import 'models/live_analysis_result.dart';

// ---------------------------------------------------------------------------
// Result model — /predict contract (updated for canonical schema)
// ---------------------------------------------------------------------------

/// Typed result returned by /predict.
/// Uses the canonical DetectionResponse schema from the backend.
class PredictResult {
  final String decision;      // "low_risk" | "verification_required" | "action_held" | "insufficient_evidence"
  final String action;        // "allow_with_caution" | "verify" | "hold" | "unavailable"
  final String label;         // "bonafide" | "spoof" | "unknown"
  final double confidence;    // 0.0 – 1.0
  final double spoofScore;    // 0.0 – 1.0
  final String riskLevel;     // "low" | "medium" | "high" | "unknown"
  final double riskPercentage;
  final bool speechDetected;
  final List<String> reasonCodes;
  final String modelVersion;
  final String modelBackend;
  final String scoreType;
  final String thresholdVersion;
  final bool modelLoaded;     // fail-closed: default is false
  final DateTime receivedAt;

  PredictResult({
    required this.decision,
    required this.action,
    required this.label,
    required this.confidence,
    required this.spoofScore,
    required this.riskLevel,
    this.riskPercentage = 0.0,
    this.speechDetected = true,
    this.reasonCodes = const [],
    this.modelVersion = '',
    this.modelBackend = '',
    this.scoreType = '',
    this.thresholdVersion = '',
    this.modelLoaded = false,
    DateTime? receivedAt,
  }) : receivedAt = receivedAt ?? DateTime.now();

  factory PredictResult.fromJson(Map<String, dynamic> json) {
    List<String> reasons = [];
    if (json['reason_codes'] is List) {
      reasons = (json['reason_codes'] as List).map((e) => e.toString()).toList();
    }

    return PredictResult(
      decision: json['decision'] as String? ?? _legacyDecision(json),
      action: json['action'] as String? ?? 'verify',
      label: json['label'] as String? ?? 'unknown',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      spoofScore: (json['spoof_score'] as num?)?.toDouble() ?? 0.0,
      riskLevel: json['risk_level'] as String? ?? 'unknown',
      riskPercentage: (json['risk_percentage'] as num?)?.toDouble() ?? 0.0,
      speechDetected: json['speech_detected'] as bool? ?? false,
      reasonCodes: reasons,
      modelVersion: json['model_version'] as String? ?? '',
      modelBackend: json['model_backend'] as String? ?? '',
      scoreType: json['score_type'] as String? ?? '',
      thresholdVersion: json['threshold_version'] as String? ?? '',
      modelLoaded: json['model_loaded'] as bool? ?? false,
      receivedAt: DateTime.now(),
    );
  }

  static String _legacyDecision(Map<String, dynamic> json) {
    final risk = json['risk_level'] as String? ?? 'unknown';
    return switch (risk) {
      'low' => 'low_risk',
      'medium' => 'verification_required',
      'high' => 'verification_required',
      _ => 'insufficient_evidence',
    };
  }

  bool isFresh([int maxAgeSeconds = 30]) =>
      DateTime.now().difference(receivedAt).inSeconds < maxAgeSeconds;

  bool get canShowLowRisk =>
      decision == 'low_risk' &&
      speechDetected == true &&
      modelLoaded == true &&
      isFresh() &&
      !reasonCodes.any(LiveAnalysisResult.blockingReasonCodes.contains);
}

// ---------------------------------------------------------------------------
// Readiness info from GET /ready
// ---------------------------------------------------------------------------

class BackendReadiness {
  final String status;       // "ready" | "not_ready"
  final bool modelLoaded;
  final String modelBackend;
  final String scoreType;
  final String modelVersion;
  final String thresholdVersion;
  final String device;
  final bool demoMode;

  const BackendReadiness({
    required this.status,
    required this.modelLoaded,
    this.modelBackend = '',
    this.scoreType = '',
    this.modelVersion = '',
    this.thresholdVersion = '',
    this.device = '',
    this.demoMode = false,
  });

  bool get isReady => status == 'ready' && modelLoaded;

  factory BackendReadiness.fromJson(Map<String, dynamic> json) {
    return BackendReadiness(
      status: json['status'] as String? ?? 'not_ready',
      modelLoaded: json['model_loaded'] as bool? ?? false,
      modelBackend: json['model_backend'] as String? ?? '',
      scoreType: json['score_type'] as String? ?? '',
      modelVersion: json['model_version'] as String? ?? '',
      thresholdVersion: json['threshold_version'] as String? ?? '',
      device: json['device'] as String? ?? '',
      demoMode: json['demo_mode'] as bool? ?? false,
    );
  }
}

// ---------------------------------------------------------------------------
// Typed exceptions
// ---------------------------------------------------------------------------

/// Backend returned a non-200 status code.
class BackendErrorException implements Exception {
  final int statusCode;
  final String detail;
  const BackendErrorException(this.statusCode, [this.detail = '']);
}

/// Network-level failure: connection refused, timeout, no route to host, etc.
class BackendUnreachableException implements Exception {
  final String detail;
  const BackendUnreachableException(this.detail);
}

/// Server rejected the file (HTTP 415 / 422) — unsupported or corrupted audio.
class UnsupportedFileException implements Exception {
  final String? message;
  const UnsupportedFileException([this.message]);
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

class VoiceGuardApiClient {
  VoiceGuardApiClient._();

  static http.Client? _customClient;

  /// Scoped HTTP client.
  /// When demo mode is explicitly active (kDemoMode && !kReleaseMode),
  /// self-signed certificates are accepted strictly for the VoiceGuard backend host.
  /// Secure mode (default) keeps standard certificate verification enabled.
  /// No global certificate override affects unrelated HTTP traffic.
  static http.Client get _client {
    if (_customClient != null) return _customClient!;
    if (!kReleaseMode || kDemoMode) {
      final ioHttpClient = HttpClient()
        ..badCertificateCallback = (cert, host, port) {
          try {
            final backendUri = Uri.parse(BackendConfig.baseUrl);
            final targetHost = backendUri.host;
            final isLoopback = (host == '127.0.0.1' || host == 'localhost' || host == '10.0.2.2');
            final isTargetLoopback = (targetHost == '127.0.0.1' || targetHost == 'localhost' || targetHost == '10.0.2.2');
            return host == targetHost || (isLoopback && isTargetLoopback);
          } catch (_) {
            return false;
          }
        };
      _customClient = http_io.IOClient(ioHttpClient);
    } else {
      _customClient = http.Client();
    }
    return _customClient!;
  }

  /// Reset client cache (e.g., when base URL changes).
  static void resetClient() {
    _customClient?.close();
    _customClient = null;
  }

  /// Send [filePath] to POST /predict and return a typed [PredictResult].
  ///
  /// Throws [BackendUnreachableException], [UnsupportedFileException], or
  /// [BackendErrorException] on failure — never a raw socket exception.
  static Future<PredictResult> predict(String filePath, {String? originalFilename}) async {
    final uri = Uri.parse('${BackendConfig.baseUrl}/predict');
    final request = http.MultipartRequest('POST', uri);

    final filename = (originalFilename != null && originalFilename.isNotEmpty)
        ? originalFilename
        : filePath.split(Platform.pathSeparator).last;

    try {
      request.files.add(await http.MultipartFile.fromPath(
        'file',
        filePath,
        filename: filename,
      ));
    } on FileSystemException {
      throw const UnsupportedFileException();
    }

    http.StreamedResponse streamedResponse;
    try {
      streamedResponse = await _client.send(request).timeout(kRequestTimeout);
    } on SocketException catch (e) {
      throw BackendUnreachableException(e.message);
    } on TimeoutException {
      throw const BackendUnreachableException('Request timed out');
    } catch (e) {
      throw BackendUnreachableException(e.toString());
    }

    http.Response response;
    try {
      response = await http.Response.fromStream(streamedResponse);
    } catch (e) {
      throw BackendUnreachableException('Failed to read server response: $e');
    }

    if (response.statusCode == 413) {
      throw const UnsupportedFileException('Audio file exceeds the maximum allowed size (10 MB).');
    }
    if (response.statusCode == 415 || response.statusCode == 422) {
      String detail = '';
      try {
        final errJson = jsonDecode(response.body) as Map<String, dynamic>;
        detail = errJson['detail']?.toString() ?? '';
      } catch (_) {}
      throw UnsupportedFileException(detail.isNotEmpty ? detail : null);
    }
    if (response.statusCode == 429) {
      throw const BackendErrorException(429, 'Rate limit exceeded. Please wait a moment.');
    }
    if (response.statusCode != 200) {
      String detail = '';
      try {
        final errJson = jsonDecode(response.body) as Map<String, dynamic>;
        detail = errJson['detail']?.toString() ?? '';
      } catch (_) {}
      throw BackendErrorException(response.statusCode, detail);
    }

    try {
      final json = jsonDecode(response.body) as Map<String, dynamic>;
      return PredictResult.fromJson(json);
    } catch (_) {
      throw const BackendErrorException(200, 'Invalid response format from server.');
    }
  }

  // -------------------------------------------------------------------------
  // Live Call Analysis
  // -------------------------------------------------------------------------

  /// Send a raw WAV audio chunk to POST /live/analyze for real-time detection.
  ///
  /// Returns clip-level evidence. Call-level decisions are made by the
  /// client-side risk aggregator.
  static Future<LiveAnalysisResult> analyzeLiveChunk(Uint8List wavBytes) async {
    final uri = Uri.parse('${BackendConfig.baseUrl}/live/analyze');
    final request = http.MultipartRequest('POST', uri);

    request.files.add(http.MultipartFile.fromBytes(
      'file',
      wavBytes,
      filename: 'live_chunk.wav',
    ));

    http.StreamedResponse streamedResponse;
    try {
      streamedResponse = await _client.send(request).timeout(kLiveRequestTimeout);
    } on SocketException catch (e) {
      throw BackendUnreachableException(e.message);
    } on TimeoutException {
      throw const BackendUnreachableException('Live analysis timed out');
    } catch (e) {
      throw BackendUnreachableException(e.toString());
    }

    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode != 200) {
      throw BackendErrorException(response.statusCode);
    }

    try {
      final json = jsonDecode(response.body) as Map<String, dynamic>;
      return LiveAnalysisResult.fromJson(json);
    } catch (_) {
      throw const BackendErrorException(200);
    }
  }

  // -------------------------------------------------------------------------
  // Health & Readiness
  // -------------------------------------------------------------------------

  /// Check backend liveness via GET /health.
  /// Returns `true` if the backend process is alive and model is loaded.
  static Future<bool> checkHealth() async {
    try {
      final uri = Uri.parse('${BackendConfig.baseUrl}/health');
      final response = await _client.get(uri).timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        final json = jsonDecode(response.body) as Map<String, dynamic>;
        return json['model_loaded'] == true;
      }
      debugPrint('[VoiceGuard] Health check returned status ${response.statusCode}');
      return false;
    } catch (e) {
      debugPrint('[VoiceGuard] Health check error: $e');
      return false;
    }
  }

  /// Check backend readiness via GET /ready.
  /// Returns a [BackendReadiness] with model version, threshold version, etc.
  /// The Flutter app must not display "Protection Active" if the backend
  /// is alive but not ready.
  static Future<BackendReadiness?> checkReadiness() async {
    try {
      final uri = Uri.parse('${BackendConfig.baseUrl}/ready');
      final response = await _client.get(uri).timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        final json = jsonDecode(response.body) as Map<String, dynamic>;
        return BackendReadiness.fromJson(json);
      }
      return null;
    } catch (e) {
      debugPrint('[VoiceGuard] Readiness check error: $e');
      return null;
    }
  }
}
