// file_sliding_window_test.dart — Unit and Widget tests for uploaded audio sliding window analysis.
//
// Tests required:
// 1. test_window_list_renders_all_windows
// 2. test_window_expansion_shows_acoustic_details
// 3. test_short_file_is_not_marked_complete
// 4. test_missing_window_data_fails_closed
// 5. test_model_unavailable_is_not_green
// 6. test_file_summary_matches_window_results
// 7. test_transaction_card_respects_overall_decision

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sih2026/main.dart';
import 'package:sih2026/models/sliding_window_result.dart';
import 'package:sih2026/widgets/risk_indicator.dart';
import 'package:sih2026/widgets/transaction_card.dart';

void main() {
  group('Uploaded Audio Sliding Window Analysis Tests', () {
    // 1. test_window_list_renders_all_windows
    testWidgets('test_window_list_renders_all_windows', (WidgetTester tester) async {
      tester.view.physicalSize = const Size(1080, 2400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(() => tester.view.resetPhysicalSize());

      final windows = [
        const SlidingWindowItem(
          windowIndex: 0,
          startSeconds: 0.0,
          endSeconds: 4.0,
          durationSeconds: 4.0,
          windowComplete: true,
          decision: 'low_risk',
          action: 'allow_with_caution',
          label: 'bonafide',
          spoofScore: 0.05,
          confidence: 0.95,
          scoreType: 'prob_fake',
          riskLevel: 'low',
          speechDetected: true,
          qualityStatus: 'acceptable',
          rms: 0.035,
          snrDb: 19.5,
          voicedRatio: 0.85,
          modelBackend: 'wav2vec2',
          modelVersion: 'model-v2',
          thresholdVersion: 'threshold-v2',
          modelLoaded: true,
        ),
        const SlidingWindowItem(
          windowIndex: 1,
          startSeconds: 2.0,
          endSeconds: 6.0,
          durationSeconds: 4.0,
          windowComplete: true,
          decision: 'low_risk',
          action: 'allow_with_caution',
          label: 'bonafide',
          spoofScore: 0.08,
          confidence: 0.92,
          scoreType: 'prob_fake',
          riskLevel: 'low',
          speechDetected: true,
          qualityStatus: 'acceptable',
          rms: 0.038,
          snrDb: 18.2,
          voicedRatio: 0.82,
          modelBackend: 'wav2vec2',
          modelVersion: 'model-v2',
          thresholdVersion: 'threshold-v2',
          modelLoaded: true,
        ),
        const SlidingWindowItem(
          windowIndex: 2,
          startSeconds: 4.0,
          endSeconds: 8.0,
          durationSeconds: 4.0,
          windowComplete: true,
          decision: 'low_risk',
          action: 'allow_with_caution',
          label: 'bonafide',
          spoofScore: 0.06,
          confidence: 0.94,
          scoreType: 'prob_fake',
          riskLevel: 'low',
          speechDetected: true,
          qualityStatus: 'acceptable',
          rms: 0.034,
          snrDb: 20.1,
          voicedRatio: 0.88,
          modelBackend: 'wav2vec2',
          modelVersion: 'model-v2',
          thresholdVersion: 'threshold-v2',
          modelLoaded: true,
        ),
      ];

      final result = FileSlidingAnalysisResult(
        totalDurationSeconds: 8.0,
        windowLengthSeconds: 4.0,
        strideSeconds: 2.0,
        windowCount: 3,
        completeWindowCount: 3,
        highRiskWindows: 0,
        maximumSpoofScore: 0.08,
        averageSpoofScore: 0.063,
        emaScore: 0.065,
        decision: 'low_risk',
        action: 'allow_with_caution',
        persistenceTriggered: false,
        consecutiveHighCount: 0,
        modelBackend: 'wav2vec2',
        modelVersion: 'model-v2',
        scoreType: 'prob_fake',
        thresholdVersion: 'threshold-v2',
        modelLoaded: true,
        windows: windows,
      );

      await tester.pumpWidget(MaterialApp(home: HomeScreen(initialResult: result)));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      expect(find.text('4-SECOND SLIDING WINDOWS'), findsOneWidget);
      expect(find.text('3 Windows (2s Stride)'), findsOneWidget);
      expect(find.text('00:00–00:04'), findsOneWidget);
      expect(find.text('00:02–00:06'), findsOneWidget);
      expect(find.text('00:04–00:08'), findsOneWidget);
    });

    // 2. test_window_expansion_shows_acoustic_details
    testWidgets('test_window_expansion_shows_acoustic_details', (WidgetTester tester) async {
      tester.view.physicalSize = const Size(1080, 2400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(() => tester.view.resetPhysicalSize());

      final window = const SlidingWindowItem(
        windowIndex: 0,
        startSeconds: 0.0,
        endSeconds: 4.0,
        durationSeconds: 4.0,
        windowComplete: true,
        decision: 'low_risk',
        action: 'allow_with_caution',
        label: 'bonafide',
        spoofScore: 0.075,
        confidence: 0.925,
        scoreType: 'uncalibrated_softmax_score',
        riskLevel: 'low',
        speechDetected: true,
        qualityStatus: 'acceptable',
        rms: 0.0412,
        snrDb: 21.4,
        voicedRatio: 0.87,
        reasonCodes: ['clean_speech'],
        modelBackend: 'wav2vec2-base',
        modelVersion: 'v2.1',
        thresholdVersion: 'th-v2.0',
        modelLoaded: true,
      );

      final result = FileSlidingAnalysisResult(
        totalDurationSeconds: 4.0,
        windowLengthSeconds: 4.0,
        strideSeconds: 2.0,
        windowCount: 1,
        completeWindowCount: 1,
        highRiskWindows: 0,
        maximumSpoofScore: 0.075,
        averageSpoofScore: 0.075,
        emaScore: 0.075,
        decision: 'low_risk',
        action: 'allow_with_caution',
        persistenceTriggered: false,
        consecutiveHighCount: 0,
        modelBackend: 'wav2vec2-base',
        modelVersion: 'v2.1',
        scoreType: 'uncalibrated_softmax_score',
        thresholdVersion: 'th-v2.0',
        modelLoaded: true,
        windows: [window],
      );

      await tester.pumpWidget(MaterialApp(home: HomeScreen(initialResult: result)));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // Tap to expand window card
      final windowTile = find.text('00:00–00:04');
      expect(windowTile, findsOneWidget);
      await tester.tap(windowTile);
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // Verify expanded telemetry items are displayed
      expect(find.text('Energy RMS'), findsOneWidget);
      expect(find.text('0.0412'), findsOneWidget);
      expect(find.text('Signal-to-Noise'), findsOneWidget);
      expect(find.text('21.4 dB'), findsOneWidget);
      expect(find.text('Voiced Speech Ratio'), findsOneWidget);
      expect(find.text('87.0%'), findsOneWidget);
      expect(find.text('Model Backend'), findsOneWidget);
      expect(find.text('wav2vec2-base'), findsOneWidget);
      expect(find.text('Score Type'), findsOneWidget);
      expect(find.text('uncalibrated_softmax_score'), findsOneWidget);
      expect(find.text('Window Completeness'), findsOneWidget);
      expect(find.text('COMPLETE (4.0s)'), findsOneWidget);
    });

    // 3. test_short_file_is_not_marked_complete
    test('test_short_file_is_not_marked_complete', () {
      final shortWindow = const SlidingWindowItem(
        windowIndex: 0,
        startSeconds: 0.0,
        endSeconds: 2.5,
        durationSeconds: 2.5,
        windowComplete: false,
        decision: 'insufficient_evidence',
        action: 'unavailable',
        label: 'unknown',
        spoofScore: null,
        confidence: null,
        scoreType: 'prob_fake',
        riskLevel: 'unknown',
        speechDetected: false,
        qualityStatus: 'short_audio',
        reasonCodes: ['short_audio'],
        modelLoaded: true,
      );

      final result = FileSlidingAnalysisResult(
        totalDurationSeconds: 2.5,
        windowLengthSeconds: 4.0,
        strideSeconds: 2.0,
        windowCount: 1,
        completeWindowCount: 0,
        highRiskWindows: 0,
        decision: 'insufficient_evidence',
        action: 'unavailable',
        reasonCodes: const ['short_audio'],
        modelLoaded: true,
        windows: [shortWindow],
      );

      expect(shortWindow.windowComplete, isFalse);
      expect(result.completeWindowCount, equals(0));
      expect(result.canShowLowRisk, isFalse);
      expect(shortWindow.canShowLowRisk, isFalse);
      expect(result.decision, equals('insufficient_evidence'));
    });

    // 4. test_missing_window_data_fails_closed
    test('test_missing_window_data_fails_closed', () {
      // Empty windows list
      final emptyResult = const FileSlidingAnalysisResult(
        totalDurationSeconds: 0.0,
        windowCount: 0,
        completeWindowCount: 0,
        highRiskWindows: 0,
        decision: 'low_risk', // Even if decision claims low_risk
        action: 'allow_with_caution',
        modelLoaded: true,
        windows: [],
      );
      expect(emptyResult.canShowLowRisk, isFalse);

      // Incomplete window with null spoof score and missing speech
      final incompleteWindow = const SlidingWindowItem(
        windowIndex: 0,
        startSeconds: 0.0,
        endSeconds: 3.0,
        durationSeconds: 3.0,
        windowComplete: false,
        decision: 'low_risk',
        action: 'allow_with_caution',
        label: 'unknown',
        spoofScore: null,
        riskLevel: 'low',
        speechDetected: false,
        qualityStatus: 'silent',
        reasonCodes: ['no_speech'],
        modelLoaded: true,
      );
      expect(incompleteWindow.canShowLowRisk, isFalse);
    });

    // 5. test_model_unavailable_is_not_green
    testWidgets('test_model_unavailable_is_not_green', (WidgetTester tester) async {
      final unavailWindow = const SlidingWindowItem(
        windowIndex: 0,
        startSeconds: 0.0,
        endSeconds: 4.0,
        durationSeconds: 4.0,
        windowComplete: true,
        decision: 'insufficient_evidence',
        action: 'unavailable',
        label: 'unknown',
        riskLevel: 'unknown',
        reasonCodes: ['model_unavailable'],
        modelLoaded: false, // Model is NOT loaded
      );

      final unavailResult = FileSlidingAnalysisResult(
        totalDurationSeconds: 4.0,
        windowCount: 1,
        completeWindowCount: 1,
        highRiskWindows: 0,
        decision: 'insufficient_evidence',
        action: 'unavailable',
        reasonCodes: const ['model_unavailable'],
        modelLoaded: false,
        windows: [unavailWindow],
      );

      expect(unavailWindow.canShowLowRisk, isFalse);
      expect(unavailResult.canShowLowRisk, isFalse);

      await tester.pumpWidget(MaterialApp(
        home: Scaffold(
          body: RiskIndicator(
            riskLevel: unavailResult.maxRisk,
            action: unavailResult.action,
            decision: unavailResult.decision,
            canShowLowRisk: unavailResult.canShowLowRisk,
          ),
        ),
      ));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // When canShowLowRisk is false, RiskIndicator must NOT show "Protection Active" / green
      expect(find.text('PROTECTION ACTIVE'), findsNothing);
      expect(find.text('INSUFFICIENT EVIDENCE'), findsOneWidget);
    });

    // 6. test_file_summary_matches_window_results
    testWidgets('test_file_summary_matches_window_results', (WidgetTester tester) async {
      tester.view.physicalSize = const Size(1080, 2400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(() => tester.view.resetPhysicalSize());

      final windows = [
        const SlidingWindowItem(
          windowIndex: 0,
          startSeconds: 0.0,
          endSeconds: 4.0,
          durationSeconds: 4.0,
          windowComplete: true,
          decision: 'verification_required',
          action: 'verify',
          label: 'spoof',
          spoofScore: 0.88,
          confidence: 0.88,
          riskLevel: 'high',
          speechDetected: true,
          qualityStatus: 'acceptable',
          modelLoaded: true,
        ),
        const SlidingWindowItem(
          windowIndex: 1,
          startSeconds: 2.0,
          endSeconds: 6.0,
          durationSeconds: 4.0,
          windowComplete: true,
          decision: 'verification_required',
          action: 'verify',
          label: 'spoof',
          spoofScore: 0.92,
          confidence: 0.92,
          riskLevel: 'high',
          speechDetected: true,
          qualityStatus: 'acceptable',
          modelLoaded: true,
        ),
      ];

      final result = FileSlidingAnalysisResult(
        totalDurationSeconds: 6.0,
        windowLengthSeconds: 4.0,
        strideSeconds: 2.0,
        windowCount: 2,
        completeWindowCount: 2,
        highRiskWindows: 2,
        maximumSpoofScore: 0.92,
        averageSpoofScore: 0.90,
        emaScore: 0.908,
        decision: 'action_held',
        action: 'hold',
        persistenceTriggered: true,
        consecutiveHighCount: 2,
        reasonCodes: const ['persistent_high_risk'],
        modelLoaded: true,
        windows: windows,
      );

      // Verify computed properties match the windows
      expect(result.highRiskWindows, equals(2));
      expect(result.completeWindowCount, equals(2));
      expect(result.maximumSpoofScore, equals(0.92));
      expect(result.persistenceTriggered, isTrue);
      expect(result.maxRisk, equals('high'));

      await tester.pumpWidget(MaterialApp(home: HomeScreen(initialResult: result)));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // Verify summary items in HomeScreen
      expect(find.text('2 / 2 complete'), findsOneWidget);
      expect(find.text('4.0s / 2.0s'), findsOneWidget);
      expect(find.text('YES (ACTION HELD)'), findsOneWidget);
    });

    // 7. test_transaction_card_respects_overall_decision
    testWidgets('test_transaction_card_respects_overall_decision', (WidgetTester tester) async {
      await tester.pumpWidget(const MaterialApp(
        home: Scaffold(
          body: TransactionCard(
            riskLevel: 'high',
            decision: 'action_held',
            reasonCodes: ['persistent_high_risk'],
            evidenceWindows: 3,
            canShowLowRisk: false,
          ),
        ),
      ));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // In ACTION_HELD state, the transaction is held and button is disabled
      expect(find.text('TRANSACTION HELD'), findsOneWidget);
      expect(find.text('CONTINUE WITH CAUTION'), findsNothing);
    });
  });
}
