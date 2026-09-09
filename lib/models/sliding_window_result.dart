/// sliding_window_result.dart — 4-second sliding window telemetry and results.
///
/// Implements the 4.0s window / 2.0s stride contract:
///   Window 0: 00:00–00:04
///   Window 1: 00:02–00:06
///   Window 2: 00:04–00:08
///   ...
library;

class SlidingWindowItem {
  final int windowIndex;
  final double startSeconds;
  final double endSeconds;
  final double durationSeconds;
  final bool windowComplete;

  final String decision;
  final String action;
  final String label;
  final double? spoofScore;
  final double? confidence;
  final String scoreType;
  final String riskLevel;

  final bool speechDetected;
  final String qualityStatus;
  final double? rms;
  final double? snrDb;
  final double? voicedRatio;
  final List<String> reasonCodes;

  final String modelBackend;
  final String modelVersion;
  final String thresholdVersion;
  final bool modelLoaded;

  const SlidingWindowItem({
    required this.windowIndex,
    required this.startSeconds,
    required this.endSeconds,
    required this.durationSeconds,
    required this.windowComplete,
    required this.decision,
    required this.action,
    required this.label,
    this.spoofScore,
    this.confidence,
    this.scoreType = '',
    required this.riskLevel,
    this.speechDetected = true,
    this.qualityStatus = 'acceptable',
    this.rms,
    this.snrDb,
    this.voicedRatio,
    this.reasonCodes = const [],
    this.modelBackend = '',
    this.modelVersion = '',
    this.thresholdVersion = '',
    this.modelLoaded = false,
  });

  int get index => windowIndex;

  String get timeRangeFormatted {
    final sMin = (startSeconds ~/ 60).toString().padLeft(2, '0');
    final sSec = (startSeconds.toInt() % 60).toString().padLeft(2, '0');
    final eMin = (endSeconds ~/ 60).toString().padLeft(2, '0');
    final eSec = (endSeconds.toInt() % 60).toString().padLeft(2, '0');
    return '$sMin:$sSec–$eMin:$eSec';
  }

  /// Fail-closed window evaluation rule:
  /// decision == low_risk
  /// AND window_complete == true
  /// AND speech_detected == true
  /// AND quality_status == acceptable
  /// AND model_loaded == true
  /// AND no blocking reason codes
  bool get canShowLowRisk =>
      decision == 'low_risk' &&
      windowComplete == true &&
      speechDetected == true &&
      qualityStatus == 'acceptable' &&
      modelLoaded == true &&
      !reasonCodes.any((r) =>
          r.contains('blocking') ||
          r.contains('elevated') ||
          r.contains('fail') ||
          r.contains('error') ||
          r.contains('poor') ||
          r.contains('silent') ||
          r.contains('short') ||
          r.contains('partial'));

  factory SlidingWindowItem.fromJson(Map<String, dynamic> json) {
    List<String> reasons = [];
    if (json['reason_codes'] is List) {
      reasons = (json['reason_codes'] as List).map((e) => e.toString()).toList();
    }
    return SlidingWindowItem(
      windowIndex: json['window_index'] as int? ?? json['index'] as int? ?? 0,
      startSeconds: (json['start_seconds'] as num?)?.toDouble() ?? 0.0,
      endSeconds: (json['end_seconds'] as num?)?.toDouble() ?? 4.0,
      durationSeconds: (json['duration_seconds'] as num?)?.toDouble() ?? 4.0,
      windowComplete: json['window_complete'] as bool? ?? true,
      decision: json['decision'] as String? ?? 'insufficient_evidence',
      action: json['action'] as String? ?? 'unavailable',
      label: json['label'] as String? ?? 'unknown',
      spoofScore: (json['spoof_score'] as num?)?.toDouble(),
      confidence: (json['confidence'] as num?)?.toDouble(),
      scoreType: json['score_type'] as String? ?? '',
      riskLevel: json['risk_level'] as String? ?? 'unknown',
      speechDetected: json['speech_detected'] as bool? ?? false,
      qualityStatus: json['quality_status'] as String? ?? 'unknown',
      rms: (json['rms'] as num?)?.toDouble(),
      snrDb: (json['snr_db'] as num?)?.toDouble(),
      voicedRatio: (json['voiced_ratio'] as num?)?.toDouble(),
      reasonCodes: reasons,
      modelBackend: json['model_backend'] as String? ?? '',
      modelVersion: json['model_version'] as String? ?? '',
      thresholdVersion: json['threshold_version'] as String? ?? '',
      modelLoaded: json['model_loaded'] as bool? ?? false,
    );
  }
}

class FileSlidingAnalysisResult {
  final double totalDurationSeconds;
  final double windowLengthSeconds;
  final double strideSeconds;
  final int windowCount;
  final int completeWindowCount;

  final int highRiskWindows;
  final double? maximumSpoofScore;
  final double? averageSpoofScore;
  final double? emaScore;

  final String decision;
  final String action;
  final bool persistenceTriggered;
  final int consecutiveHighCount;
  final List<String> reasonCodes;

  final String modelBackend;
  final String modelVersion;
  final String scoreType;
  final String thresholdVersion;
  final bool modelLoaded;

  final List<SlidingWindowItem> windows;
  final bool isFallbackPredict;

  const FileSlidingAnalysisResult({
    required this.totalDurationSeconds,
    this.windowLengthSeconds = 4.0,
    this.strideSeconds = 2.0,
    required this.windowCount,
    required this.completeWindowCount,
    required this.highRiskWindows,
    this.maximumSpoofScore,
    this.averageSpoofScore,
    this.emaScore,
    required this.decision,
    required this.action,
    this.persistenceTriggered = false,
    this.consecutiveHighCount = 0,
    this.reasonCodes = const [],
    this.modelBackend = '',
    this.modelVersion = '',
    this.scoreType = '',
    this.thresholdVersion = '',
    this.modelLoaded = false,
    required this.windows,
    this.isFallbackPredict = false,
  });

  String get overallRiskLevel {
    if (highRiskWindows > 0 || persistenceTriggered) return 'high';
    if (windows.any((w) => w.riskLevel == 'medium' && w.windowComplete)) return 'medium';
    if (canShowLowRisk) return 'low';
    return 'unknown';
  }

  String get maxRisk => overallRiskLevel;

  double get overallSpoofScore => maximumSpoofScore ?? averageSpoofScore ?? 0.0;

  /// Fail-closed file evaluation rule
  bool get canShowLowRisk =>
      decision == 'low_risk' &&
      modelLoaded == true &&
      completeWindowCount > 0 &&
      !isFallbackPredict &&
      !reasonCodes.any((r) =>
          r.contains('insufficient') ||
          r.contains('poor') ||
          r.contains('model_unavailable') ||
          r.contains('short_audio') ||
          r.contains('no_valid_windows') ||
          r.contains('unavailable'));

  factory FileSlidingAnalysisResult.fromJson(Map<String, dynamic> json) {
    List<String> reasons = [];
    if (json['reason_codes'] is List) {
      reasons = (json['reason_codes'] as List).map((e) => e.toString()).toList();
    }
    final rawWindows = json['windows'] as List? ?? [];
    final windows = rawWindows
        .map((w) => SlidingWindowItem.fromJson(w as Map<String, dynamic>))
        .toList();

    return FileSlidingAnalysisResult(
      totalDurationSeconds: (json['total_duration_seconds'] as num?)?.toDouble() ?? 0.0,
      windowLengthSeconds: (json['window_length_seconds'] as num?)?.toDouble() ?? 4.0,
      strideSeconds: (json['stride_seconds'] as num?)?.toDouble() ?? 2.0,
      windowCount: json['window_count'] as int? ?? windows.length,
      completeWindowCount: json['complete_window_count'] as int? ??
          windows.where((w) => w.windowComplete).length,
      highRiskWindows: json['high_risk_windows'] as int? ?? 0,
      maximumSpoofScore: (json['maximum_spoof_score'] as num?)?.toDouble(),
      averageSpoofScore: (json['average_spoof_score'] as num?)?.toDouble(),
      emaScore: (json['ema_score'] as num?)?.toDouble(),
      decision: json['decision'] as String? ?? 'insufficient_evidence',
      action: json['action'] as String? ?? 'unavailable',
      persistenceTriggered: json['persistence_triggered'] as bool? ?? false,
      consecutiveHighCount: json['consecutive_high_count'] as int? ?? 0,
      reasonCodes: reasons,
      modelBackend: json['model_backend'] as String? ?? '',
      modelVersion: json['model_version'] as String? ?? '',
      scoreType: json['score_type'] as String? ?? '',
      thresholdVersion: json['threshold_version'] as String? ?? '',
      modelLoaded: json['model_loaded'] as bool? ?? false,
      windows: windows,
      isFallbackPredict: false,
    );
  }
}
