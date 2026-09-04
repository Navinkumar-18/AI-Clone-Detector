/// All /predict HTTP logic for VoiceGuard.
///
/// To swap the backend: edit [kBackendBaseUrl] in config.dart only.
/// To change request/response shape: edit [PredictResult] and [VoiceGuardApiClient.predict] here only.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'config.dart';

// ---------------------------------------------------------------------------
// Result model
// ---------------------------------------------------------------------------

/// Typed result returned by /predict.
/// Mirrors the agreed API contract exactly — do not change without coordinating
/// with the backend team.
class PredictResult {
  final String label;       // "bonafide" | "spoof"
  final double confidence;  // 0.0 – 1.0
  final String riskLevel;   // "low" | "medium" | "high"

  const PredictResult({
    required this.label,
    required this.confidence,
    required this.riskLevel,
  });

  factory PredictResult.fromJson(Map<String, dynamic> json) {
    return PredictResult(
      label: json['label'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      riskLevel: json['risk_level'] as String,
    );
  }
}

// ---------------------------------------------------------------------------
// Typed exceptions — lets the UI show specific messages without leaking raw
// exception text to the user.
// ---------------------------------------------------------------------------

/// Backend returned a non-200 status code.
class BackendErrorException implements Exception {
  final int statusCode;
  const BackendErrorException(this.statusCode);
}

/// Network-level failure: connection refused, timeout, no route to host, etc.
class BackendUnreachableException implements Exception {
  final String detail;
  const BackendUnreachableException(this.detail);
}

/// Server rejected the file (HTTP 415 / 422) — unsupported or corrupted audio.
class UnsupportedFileException implements Exception {
  const UnsupportedFileException();
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

class VoiceGuardApiClient {
  VoiceGuardApiClient._();

  /// Send [filePath] to POST /predict and return a typed [PredictResult].
  ///
  /// Throws [BackendUnreachableException], [UnsupportedFileException], or
  /// [BackendErrorException] on failure — never a raw socket exception.
  static Future<PredictResult> predict(String filePath) async {
    final uri = Uri.parse('$kBackendBaseUrl/predict');
    final request = http.MultipartRequest('POST', uri);

    try {
      request.files.add(await http.MultipartFile.fromPath('file', filePath));
    } on FileSystemException {
      // File vanished between selection and upload (e.g. temp file cleaned up).
      throw const UnsupportedFileException();
    }

    http.StreamedResponse streamedResponse;
    try {
      streamedResponse = await request.send().timeout(kRequestTimeout);
    } on SocketException catch (e) {
      throw BackendUnreachableException(e.message);
    } on TimeoutException {
      throw const BackendUnreachableException('Request timed out');
    } catch (e) {
      // Catch-all for other network-layer issues (e.g. bad TLS, DNS failure).
      throw BackendUnreachableException(e.toString());
    }

    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 415 || response.statusCode == 422) {
      throw const UnsupportedFileException();
    }

    if (response.statusCode != 200) {
      throw BackendErrorException(response.statusCode);
    }

    try {
      final json = jsonDecode(response.body) as Map<String, dynamic>;
      return PredictResult.fromJson(json);
    } catch (_) {
      // Malformed JSON from backend — treat as server error.
      throw const BackendErrorException(200);
    }
  }
}
