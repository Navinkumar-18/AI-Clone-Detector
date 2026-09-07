/// Typed model for the /live/analyze and /predict API response.
///
/// Updated to match the canonical backend DetectionResponse schema.
/// The backend returns clip-level evidence using a unified response model.
///
/// Backend schema version: threshold-v2
library;

class AudioQualityResult {
  final String status; // "acceptable" | "poor" | "silent" | "invalid" | "unknown"
  final double durationSeconds;
  final double rms;
  final double? snrDb;
  final double clippingRatio;
  final double voicedRatio;

  const AudioQualityResult({
    required this.status,
    this.durationSeconds = 0.0,
    this.rms = 0.0,
    this.snrDb,
    this.clippingRatio = 0.0,
    this.voicedRatio = 0.0,
  });

  factory AudioQualityResult.fromJson(Map<String, dynamic> json) {
    return AudioQualityResult(
      status: json['status'] as String? ?? 'unknown',
      durationSeconds: (json['duration_seconds'] as num?)?.toDouble() ?? 0.0,
      rms: (json['rms'] as num?)?.toDouble() ?? 0.0,
      snrDb: (json['snr_db'] as num?)?.toDouble(),
      clippingRatio: (json['clipping_ratio'] as num?)?.toDouble() ?? 0.0,
      voicedRatio: (json['voiced_ratio'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class EvidenceResult {
  final int windowCount;
  final int highRiskWindowCount;
  final int analysisAgeMs;

  const EvidenceResult({
    this.windowCount = 0,
    this.highRiskWindowCount = 0,
    this.analysisAgeMs = 0,
  });

  factory EvidenceResult.fromJson(Map<String, dynamic> json) {
    return EvidenceResult(
      windowCount: json['window_count'] as int? ?? 0,
      highRiskWindowCount: json['high_risk_window_count'] as int? ?? 0,
      analysisAgeMs: json['analysis_age_ms'] as int? ?? 0,
    );
  }
}

class LiveAnalysisResult {
  // --- New canonical fields ---
  final String decision;       // "low_risk" | "verification_required" | "action_held" | "insufficient_evidence"
  final String action;         // "allow_with_caution" | "verify" | "hold" | "unavailable"
  final String label;          // "bonafide" | "spoof" | "unknown"
  final double spoofScore;     // Raw prob_fake 0.0–1.0
  final double confidence;     // 0.0–1.0
  final String riskLevel;      // "low" | "medium" | "high" | "unknown"
  final double riskPercentage; // 0.0–100.0
  final bool speechDetected;   // false if chunk was silence / muted mic
  final AudioQualityResult audioQuality;
  final EvidenceResult evidence;
  final List<String> reasonCodes;
  final String modelVersion;
  final String modelBackend;
  final String scoreType;
  final String thresholdVersion;
  final bool modelLoaded;      // fail-closed: default is false
  final String requestId;
  final DateTime receivedAt;

  const LiveAnalysisResult({
    required this.decision,
    required this.action,
    required this.label,
    required this.spoofScore,
    required this.confidence,
    required this.riskLevel,
    this.riskPercentage = 0.0,
    this.speechDetected = true,
    this.audioQuality = const AudioQualityResult(status: 'unknown'),
    this.evidence = const EvidenceResult(),
    this.reasonCodes = const [],
    this.modelVersion = '',
    this.modelBackend = '',
    this.scoreType = '',
    this.thresholdVersion = '',
    this.modelLoaded = false,
    this.requestId = '',
    DateTime? receivedAt,
  }) : receivedAt = receivedAt ?? const _ConstDateTime();

  /// Parse from the canonical backend DetectionResponse JSON.
  ///
  /// Backward-compatible: falls back to legacy field names if new fields
  /// are absent, so older backend versions still work during transitions.
  factory LiveAnalysisResult.fromJson(Map<String, dynamic> json) {
    // Audio quality
    AudioQualityResult aq;
    if (json['audio_quality'] is Map<String, dynamic>) {
      aq = AudioQualityResult.fromJson(json['audio_quality'] as Map<String, dynamic>);
    } else {
      aq = const AudioQualityResult(status: 'unknown');
    }

    // Evidence
    EvidenceResult ev;
    if (json['evidence'] is Map<String, dynamic>) {
      ev = EvidenceResult.fromJson(json['evidence'] as Map<String, dynamic>);
    } else {
      ev = const EvidenceResult();
    }

    // Reason codes
    List<String> reasons;
    if (json['reason_codes'] is List) {
      reasons = (json['reason_codes'] as List).map((e) => e.toString()).toList();
    } else {
      reasons = const [];
    }

    return LiveAnalysisResult(
      decision: json['decision'] as String? ?? _legacyDecision(json),
      action: json['action'] as String? ?? 'verify',
      label: json['label'] as String? ?? 'unknown',
      spoofScore: (json['spoof_score'] as num?)?.toDouble()
          ?? (json['detection_score'] as num?)?.toDouble()
          ?? (json['prob_fake'] as num?)?.toDouble()
          ?? 0.0,
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      riskLevel: json['risk_level'] as String? ?? 'unknown',
      riskPercentage: (json['risk_percentage'] as num?)?.toDouble() ?? 0.0,
      speechDetected: json['speech_detected'] as bool? ?? false,
      audioQuality: aq,
      evidence: ev,
      reasonCodes: reasons,
      modelVersion: json['model_version'] as String? ?? '',
      modelBackend: json['model_backend'] as String? ?? '',
      scoreType: json['score_type'] as String? ?? '',
      thresholdVersion: json['threshold_version'] as String? ?? '',
      // CAUTION 1: Never default model_loaded to true. If missing or malformed -> false.
      modelLoaded: json['model_loaded'] as bool? ?? false,
      requestId: json['request_id'] as String? ?? '',
      receivedAt: DateTime.now(),
    );
  }

  /// Legacy fallback: derive decision from risk_level for older backends.
  static String _legacyDecision(Map<String, dynamic> json) {
    final risk = json['risk_level'] as String? ?? 'unknown';
    return switch (risk) {
      'low' => 'low_risk',
      'medium' => 'verification_required',
      'high' => 'verification_required',
      _ => 'insufficient_evidence',
    };
  }

  /// Whether this result indicates insufficient evidence.
  bool get isInsufficientEvidence => decision == 'insufficient_evidence';

  /// Blocking reason codes that immediately disqualify the result from green/low-risk.
  static const Set<String> blockingReasonCodes = {
    'no_speech',
    'audio_quality_poor',
    'backend_unavailable',
    'model_unavailable',
    'analysis_stale',
    'capture_unavailable',
    'empty_or_invalid_capture',
    'elevated_spoof_score',
    'persistent_high_spoof_evidence',
    'persistent_high_spoof_score',
    'action_remains_held',
    'recovering_from_action_held',
  };

  /// Freshness check: returns true if result was received within [maxAgeSeconds].
  bool isFresh([int maxAgeSeconds = 15]) =>
      DateTime.now().difference(receivedAt).inSeconds < maxAgeSeconds;

  /// FAIL-CLOSED GREEN-STATE RULE:
  /// Green/LOW_RISK may be displayed ONLY when EVERY condition below is true:
  ///   1. decision == 'low_risk'
  ///   2. speechDetected == true
  ///   3. audioQuality.status == 'acceptable'
  ///   4. modelLoaded == true (fail-closed: never defaults to true)
  ///   5. response is fresh (< 15 seconds)
  ///   6. no blocking reason codes present
  bool get canShowLowRisk =>
      decision == 'low_risk' &&
      speechDetected == true &&
      audioQuality.status == 'acceptable' &&
      modelLoaded == true &&
      isFresh() &&
      !reasonCodes.any(blockingReasonCodes.contains);
}

/// Compile-time constant fallback for receivedAt.
class _ConstDateTime implements DateTime {
  const _ConstDateTime();

  @override
  bool isAfter(DateTime other) => false;
  @override
  bool isBefore(DateTime other) => false;
  @override
  bool isAtSameMomentAs(DateTime other) => false;
  @override
  int compareTo(DateTime other) => 0;
  @override
  DateTime add(Duration duration) => this;
  @override
  DateTime subtract(Duration duration) => this;
  @override
  Duration difference(DateTime other) => const Duration(seconds: 99999);
  @override
  int get millisecondsSinceEpoch => 0;
  @override
  int get microsecondsSinceEpoch => 0;
  @override
  String get timeZoneName => 'UTC';
  @override
  Duration get timeZoneOffset => Duration.zero;
  @override
  int get year => 1970;
  @override
  int get month => 1;
  @override
  int get day => 1;
  @override
  int get hour => 0;
  @override
  int get minute => 0;
  @override
  int get second => 0;
  @override
  int get millisecond => 0;
  @override
  int get microsecond => 0;
  @override
  int get weekday => 4;
  @override
  bool get isUtc => true;
  @override
  DateTime toLocal() => this;
  @override
  DateTime toUtc() => this;
  @override
  String toIso8601String() => '1970-01-01T00:00:00.000Z';
}
