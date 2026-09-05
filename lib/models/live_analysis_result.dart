/// Typed model for the /live/analyze API response.
///
/// Used by the live call analysis mode. The backend returns detection_score
/// (prob_fake from Wav2Vec2 softmax) and a security action derived from
/// the same risk thresholds as /predict.
library;

class LiveAnalysisResult {
  final String label;          // "bonafide" | "spoof"
  final double confidence;     // 0.0–1.0
  final String riskLevel;      // "low" | "medium" | "high"
  final double detectionScore; // prob_fake 0.0–1.0 (the spoof detection score)
  final double probReal;       // prob_real 0.0–1.0
  final double probFake;       // prob_fake 0.0–1.0
  final String action;         // "allow" | "verify" | "block"

  const LiveAnalysisResult({
    required this.label,
    required this.confidence,
    required this.riskLevel,
    required this.detectionScore,
    required this.probReal,
    required this.probFake,
    required this.action,
  });

  factory LiveAnalysisResult.fromJson(Map<String, dynamic> json) {
    return LiveAnalysisResult(
      label: json['label'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      riskLevel: json['risk_level'] as String,
      detectionScore: (json['detection_score'] as num).toDouble(),
      probReal: (json['prob_real'] as num).toDouble(),
      probFake: (json['prob_fake'] as num).toDouble(),
      action: json['action'] as String,
    );
  }
}
