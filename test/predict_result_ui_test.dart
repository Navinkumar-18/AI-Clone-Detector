// test/predict_result_ui_test.dart
// Unit and Widget tests for uploaded-audio analysis via POST /predict.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sih2026/api_client.dart';
import 'package:sih2026/main.dart';
import 'package:sih2026/widgets/risk_indicator.dart';
import 'package:sih2026/widgets/transaction_card.dart';

void main() {
  group('Uploaded Audio Single-File Analysis Tests (POST /predict)', () {
    test('PredictResult parses all required detection and acoustic telemetry fields', () {
      final json = {
        'decision': 'low_risk',
        'action': 'allow_with_caution',
        'label': 'bonafide',
        'spoof_score': 0.042,
        'confidence': 0.958,
        'risk_level': 'low',
        'risk_percentage': 4.2,
        'speech_detected': true,
        'audio_quality': {
          'status': 'acceptable',
          'duration_seconds': 12.5,
          'rms': 0.0384,
          'snr_db': 21.4,
          'clipping_ratio': 0.0,
          'voiced_ratio': 0.76,
        },
        'reason_codes': <String>[],
        'model_version': 'voiceguard-v1',
        'model_backend': 'wav2vec2',
        'score_type': 'uncalibrated_softmax_score',
        'threshold_version': 'threshold-v2',
        'model_loaded': true,
      };

      final result = PredictResult.fromJson(json);

      expect(result.decision, equals('low_risk'));
      expect(result.action, equals('allow_with_caution'));
      expect(result.label, equals('bonafide'));
      expect(result.spoofScore, closeTo(0.042, 0.001));
      expect(result.confidence, closeTo(0.958, 0.001));
      expect(result.riskLevel, equals('low'));
      expect(result.speechDetected, isTrue);
      expect(result.modelLoaded, isTrue);
      expect(result.audioQualityStatus, equals('acceptable'));
      expect(result.durationSeconds, closeTo(12.5, 0.01));
      expect(result.rms, closeTo(0.0384, 0.0001));
      expect(result.snrDb, closeTo(21.4, 0.1));
      expect(result.voicedRatio, closeTo(0.76, 0.01));
      expect(result.modelBackend, equals('wav2vec2'));
      expect(result.scoreType, equals('uncalibrated_softmax_score'));
      expect(result.modelVersion, equals('voiceguard-v1'));
      expect(result.thresholdVersion, equals('threshold-v2'));
      expect(result.canShowLowRisk, isTrue);
    });

    test('PredictResult fail-closed: modelLoaded=false blocks canShowLowRisk', () {
      final json = {
        'decision': 'low_risk',
        'action': 'allow_with_caution',
        'label': 'bonafide',
        'spoof_score': 0.042,
        'confidence': 0.958,
        'risk_level': 'low',
        'speech_detected': true,
        'audio_quality': {'status': 'acceptable'},
        'model_loaded': false, // UNLOADED
      };

      final result = PredictResult.fromJson(json);
      expect(result.canShowLowRisk, isFalse);
    });

    test('PredictResult fail-closed: poor audio quality blocks canShowLowRisk', () {
      final json = {
        'decision': 'low_risk',
        'action': 'allow_with_caution',
        'label': 'bonafide',
        'spoof_score': 0.042,
        'confidence': 0.958,
        'risk_level': 'low',
        'speech_detected': true,
        'audio_quality': {'status': 'poor'}, // POOR
        'model_loaded': true,
      };

      final result = PredictResult.fromJson(json);
      expect(result.canShowLowRisk, isFalse);
    });

    test('PredictResult fail-closed: spoof label blocks canShowLowRisk', () {
      final json = {
        'decision': 'verification_required',
        'action': 'verify',
        'label': 'spoof', // SPOOF
        'spoof_score': 0.85,
        'confidence': 0.85,
        'risk_level': 'high',
        'speech_detected': true,
        'audio_quality': {'status': 'acceptable'},
        'model_loaded': true,
      };

      final result = PredictResult.fromJson(json);
      expect(result.canShowLowRisk, isFalse);
    });

    testWidgets('HomeScreen renders complete-file upload result view with all required fields', (tester) async {
      tester.view.physicalSize = const Size(1080, 2400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(() => tester.view.resetPhysicalSize());

      final predictResult = PredictResult(
        decision: 'low_risk',
        action: 'allow_with_caution',
        label: 'bonafide',
        spoofScore: 0.035,
        confidence: 0.965,
        riskLevel: 'low',
        riskPercentage: 3.5,
        speechDetected: true,
        audioQualityStatus: 'acceptable',
        durationSeconds: 8.4,
        rms: 0.0412,
        snrDb: 22.8,
        voicedRatio: 0.81,
        modelBackend: 'wav2vec2',
        modelVersion: 'voiceguard-v1',
        scoreType: 'uncalibrated_softmax_score',
        thresholdVersion: 'threshold-v2',
        modelLoaded: true,
        reasonCodes: const [],
      );

      await tester.pumpWidget(MaterialApp(
        home: HomeScreen(initialPredictResult: predictResult),
      ));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // 1. Status bar
      expect(find.text('VOICE INTEGRITY VERIFIED'), findsOneWidget);

      // 2. RiskIndicator
      expect(find.byType(RiskIndicator), findsOneWidget);

      // 3. Core summary card
      expect(find.text('VOICE INTEGRITY ASSESSMENT'), findsOneWidget);
      expect(find.text('BONAFIDE'), findsOneWidget);
      expect(find.text('3.5%'), findsOneWidget); // spoof score pct
      expect(find.text('96.5%'), findsOneWidget); // confidence pct
      expect(find.text('ALLOW WITH CAUTION'), findsOneWidget);

      // 4. Acoustic & Quality telemetry
      expect(find.text('ACOUSTIC & SIGNAL TELEMETRY'), findsOneWidget);
      expect(find.text('ACCEPTABLE'), findsOneWidget);
      expect(find.text('8.4s'), findsOneWidget); // duration
      expect(find.text('22.8 dB'), findsOneWidget); // SNR
      expect(find.text('81.0%'), findsOneWidget); // voiced ratio

      // 5. Model metadata
      expect(find.text('DETECTION PIPELINE & AUDIT METADATA'), findsOneWidget);
      expect(find.text('wav2vec2'), findsAtLeastNWidgets(1));
      expect(find.text('uncalibrated_softmax_score'), findsOneWidget);
      expect(find.text('voiceguard-v1'), findsOneWidget);
      expect(find.text('threshold-v2'), findsOneWidget);
      expect(find.text('Model Loaded & Active'), findsOneWidget);

      // 6. Security reason codes
      expect(find.text('SECURITY REASON CODES'), findsOneWidget);

      // 7. Step-up recommendation
      expect(find.text('STEP-UP VERIFICATION RECOMMENDATION'), findsOneWidget);

      // 8. TransactionCard
      expect(find.byType(TransactionCard), findsOneWidget);

      // 9. Disclaimer
      expect(find.textContaining('VoiceGuard provides machine-learning probabilistic evidence'), findsOneWidget);

      // 10. NO sliding window cards in upload view
      expect(find.text('4-SECOND SLIDING WINDOWS'), findsNothing);
    });

    testWidgets('HomeScreen renders high-risk spoof complete-file result with fail-closed state', (tester) async {
      tester.view.physicalSize = const Size(1080, 2400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(() => tester.view.resetPhysicalSize());

      final spoofResult = PredictResult(
        decision: 'verification_required',
        action: 'verify',
        label: 'spoof',
        spoofScore: 0.912,
        confidence: 0.912,
        riskLevel: 'high',
        riskPercentage: 91.2,
        speechDetected: true,
        audioQualityStatus: 'acceptable',
        durationSeconds: 15.0,
        rms: 0.0521,
        snrDb: 18.5,
        voicedRatio: 0.88,
        modelBackend: 'wav2vec2',
        modelVersion: 'voiceguard-v1',
        scoreType: 'uncalibrated_softmax_score',
        thresholdVersion: 'threshold-v2',
        modelLoaded: true,
        reasonCodes: const ['elevated_spoof_score'],
      );

      await tester.pumpWidget(MaterialApp(
        home: HomeScreen(initialPredictResult: spoofResult),
      ));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      expect(find.text('HIGH RISK DEEPFAKE DETECTED'), findsOneWidget);
      expect(find.text('SPOOF'), findsOneWidget);
      expect(find.text('91.2%'), findsAtLeastNWidgets(1));
      expect(find.text('VERIFY'), findsOneWidget);
      expect(find.text('elevated_spoof_score'), findsOneWidget);
      expect(find.textContaining('Elevated risk detected'), findsNothing);
      expect(find.textContaining('High deepfake probability detected'), findsOneWidget);
      expect(find.text('4-SECOND SLIDING WINDOWS'), findsNothing);
    });

    testWidgets('HomeScreen and TransactionCard dialog do not overflow on narrow mobile screen with long filename', (tester) async {
      // 360 x 740 is a standard compact mobile phone width
      tester.view.physicalSize = const Size(360, 740);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(() => tester.view.resetPhysicalSize());

      final bonafideResult = PredictResult(
        decision: 'low_risk',
        action: 'allow_with_caution',
        label: 'bonafide',
        spoofScore: 0.05,
        confidence: 0.95,
        riskLevel: 'low',
        speechDetected: true,
        modelLoaded: true,
        audioQualityStatus: 'acceptable',
        durationSeconds: 55.7,
        rms: 0.1216,
        snrDb: 35.7,
        voicedRatio: 0.82,
        modelBackend: 'wav2vec2-base',
        scoreType: 'uncalibrated_softmax_score',
        thresholdVersion: 'threshold-v2',
        reasonCodes: const [],
      );

      await tester.pumpWidget(MaterialApp(
        home: HomeScreen(
          initialPredictResult: bonafideResult,
          initialFilename: 'AUD-20260908-WA0004.mp3',
        ),
      ));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // Check status bar with long filename rendered without overflow
      expect(find.text('VOICE INTEGRITY VERIFIED'), findsOneWidget);
      expect(find.text('AUD-20260908-WA0004.mp3'), findsOneWidget);

      // Scroll to transaction card and tap "CONTINUE WITH CAUTION"
      final continueBtn = find.text('CONTINUE WITH CAUTION');
      await tester.scrollUntilVisible(continueBtn, 200);
      await tester.pump(const Duration(milliseconds: 100));

      await tester.tap(continueBtn);
      await tester.pump(const Duration(milliseconds: 100));

      // Verify dialog appears and title fits without overflow
      expect(find.text('Transfer Verification Passed'), findsOneWidget);
      expect(find.text('OK'), findsOneWidget);

      // Dismiss dialog
      await tester.tap(find.text('OK'));
      await tester.pump(const Duration(milliseconds: 100));
    });
  });
}

