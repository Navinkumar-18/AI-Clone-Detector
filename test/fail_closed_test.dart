import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sih2026/models/live_analysis_result.dart';
import 'package:sih2026/widgets/risk_indicator.dart';

void main() {
  group('LiveAnalysisResult Fail-Closed Tests', () {
    test('Defaults modelLoaded to false in fromJson when field is missing (Caution 1)', () {
      final json = {
        'risk_level': 'low',
        'spoof_score': 0.05,
        'action': 'allow_with_caution',
        'decision': 'low_risk',
        'speech_detected': true,
        'label': 'bonafide',
        'confidence': 0.95,
        'reason_codes': <String>[],
        'audio_quality': {'status': 'acceptable'},
      };
      final result = LiveAnalysisResult.fromJson(json);
      expect(result.modelLoaded, isFalse,
          reason: 'Missing model_loaded field must default to false');
      expect(result.canShowLowRisk, isFalse,
          reason: 'canShowLowRisk must be blocked when modelLoaded is false');
    });

    test('Defaults modelLoaded to false in fromJson when field is null or non-bool', () {
      final json = {
        'risk_level': 'low',
        'spoof_score': 0.05,
        'action': 'allow_with_caution',
        'decision': 'low_risk',
        'speech_detected': true,
        'label': 'bonafide',
        'confidence': 0.95,
        'reason_codes': <String>[],
        'model_loaded': null,
        'audio_quality': {'status': 'acceptable'},
      };
      final result = LiveAnalysisResult.fromJson(json);
      expect(result.modelLoaded, isFalse);
      expect(result.canShowLowRisk, isFalse);
    });

    test('canShowLowRisk is blocked by speechDetected = false', () {
      final result = LiveAnalysisResult(
        riskLevel: 'low',
        spoofScore: 0.05,
        confidence: 0.95,
        label: 'bonafide',
        action: 'allow_with_caution',
        decision: 'low_risk',
        speechDetected: false,
        audioQuality: const AudioQualityResult(status: 'acceptable'),
        reasonCodes: const ['no_speech'],
        modelLoaded: true,
        receivedAt: DateTime.now(),
      );
      expect(result.canShowLowRisk, isFalse);
    });

    test('canShowLowRisk is blocked by blocking reason codes', () {
      final blockingCodes = [
        'no_speech',
        'audio_quality_poor',
        'backend_unavailable',
        'model_unavailable',
        'analysis_stale',
        'capture_unavailable',
        'empty_or_invalid_capture',
        'elevated_spoof_score',
        'persistent_high_spoof_evidence',
        'action_remains_held',
      ];

      for (final code in blockingCodes) {
        final result = LiveAnalysisResult(
          riskLevel: 'low',
          spoofScore: 0.05,
          confidence: 0.95,
          label: 'bonafide',
          action: 'allow_with_caution',
          decision: 'low_risk',
          speechDetected: true,
          audioQuality: const AudioQualityResult(status: 'acceptable'),
          reasonCodes: [code],
          modelLoaded: true,
          receivedAt: DateTime.now(),
        );
        expect(result.canShowLowRisk, isFalse,
            reason: 'Blocking code $code must prevent canShowLowRisk');
      }
    });

    test('canShowLowRisk permits non-blocking informational reason codes (Caution 3)', () {
      final result = LiveAnalysisResult(
        riskLevel: 'low',
        spoofScore: 0.12,
        confidence: 0.88,
        label: 'bonafide',
        action: 'allow_with_caution',
        decision: 'low_risk',
        speechDetected: true,
        audioQuality: const AudioQualityResult(status: 'acceptable'),
        reasonCodes: const ['chunk_analyzed_ok', 'clean_audio_profile'],
        modelLoaded: true,
        receivedAt: DateTime.now(),
      );
      expect(result.canShowLowRisk, isTrue,
          reason: 'Informational codes should not prevent green');
    });

    test('canShowLowRisk is blocked when result is stale (> 15 seconds old)', () {
      final result = LiveAnalysisResult(
        riskLevel: 'low',
        spoofScore: 0.05,
        confidence: 0.95,
        label: 'bonafide',
        action: 'allow_with_caution',
        decision: 'low_risk',
        speechDetected: true,
        audioQuality: const AudioQualityResult(status: 'acceptable'),
        reasonCodes: const [],
        modelLoaded: true,
        receivedAt: DateTime.now().subtract(const Duration(seconds: 30)),
      );
      expect(result.isFresh(), isFalse);
      expect(result.canShowLowRisk, isFalse);
    });

    test('canShowLowRisk is blocked when spoofScore >= 0.30', () {
      final result = LiveAnalysisResult(
        riskLevel: 'medium',
        spoofScore: 0.45,
        confidence: 0.60,
        label: 'spoof',
        action: 'verify',
        decision: 'verification_required',
        speechDetected: true,
        audioQuality: const AudioQualityResult(status: 'acceptable'),
        reasonCodes: const [],
        modelLoaded: true,
        receivedAt: DateTime.now(),
      );
      expect(result.canShowLowRisk, isFalse);
    });
  });

  group('RiskIndicator Fail-Closed UI Tests', () {
    test('effectiveDecision converts unverified low_risk to insufficient_evidence', () {
      const indicator = RiskIndicator(
        riskLevel: 'low',
        action: 'allow_with_caution',
        decision: 'low_risk',
        canShowLowRisk: false, // Unverified / missing model / silent
      );
      expect(indicator.effectiveDecision, equals('insufficient_evidence'));
    });

    test('effectiveDecision preserves low_risk only when canShowLowRisk is true', () {
      const indicator = RiskIndicator(
        riskLevel: 'low',
        action: 'allow_with_caution',
        decision: 'low_risk',
        canShowLowRisk: true,
      );
      expect(indicator.effectiveDecision, equals('low_risk'));
    });

    test('Risk colors: low_risk without verification fails closed to grey, green only with flag', () {
      // Without canShowLowRisk: false (default), low risk must be grey!
      expect(RiskIndicator.riskColor('low'), equals(const Color(0xFF616161)));
      // With canShowLowRisk: true, low risk is green
      expect(RiskIndicator.riskColor('low', canShowLowRisk: true), equals(const Color(0xFF2E7D32)));

      expect(RiskIndicator.riskColor('medium'), equals(const Color(0xFFE65100)));
      expect(RiskIndicator.riskColor('high'), equals(const Color(0xFFC62828)));
      // Wildcard must be Grey (0xFF616161), never Green!
      expect(RiskIndicator.riskColor('unknown'), equals(const Color(0xFF616161)));
      expect(RiskIndicator.riskColor('unrecognized_state'), equals(const Color(0xFF616161)));
    });
  });
}
