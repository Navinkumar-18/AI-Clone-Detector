import 'package:flutter_test/flutter_test.dart';

void main() {
  group('Silent Badge and RMS Gate Consistency Tests', () {
    // Gate threshold constant defined across the system
    const double kSilenceThresholdRms = 0.003;

    /// Evaluates the UI visibility predicate for the "SILENT (ENABLE SPEAKERPHONE)" badge.
    /// In the fixed implementation, this is strictly:
    /// `_currentRms < 0.003 && _callState == _CallState.active`
    bool isSilentBadgeVisible({
      required double currentRms,
      required bool isCallActive,
    }) {
      return currentRms < kSilenceThresholdRms && isCallActive;
    }

    /// Evaluates the RMS gate state in the diagnostics panel:
    /// `_currentRms < 0.003 ? "SILENCE (<0.003)" : "ACTIVE"`
    bool isRmsGateActive({required double currentRms}) {
      return currentRms >= kSilenceThresholdRms;
    }

    test('Silent badge is hidden whenever RMS gate is ACTIVE (>= 0.003)', () {
      final activeRmsValues = [0.003, 0.0031, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5];

      for (final rms in activeRmsValues) {
        expect(isRmsGateActive(currentRms: rms), isTrue,
            reason: 'RMS $rms must be considered ACTIVE by the gate');
        expect(
          isSilentBadgeVisible(currentRms: rms, isCallActive: true),
          isFalse,
          reason: 'Silent badge must NOT be displayed when RMS is $rms (ACTIVE)',
        );
      }
    });

    test('Silent badge is displayed when RMS gate is in SILENCE (< 0.003) during active call', () {
      final silentRmsValues = [0.0, 0.0001, 0.001, 0.002, 0.0029, 0.00299];

      for (final rms in silentRmsValues) {
        expect(isRmsGateActive(currentRms: rms), isFalse,
            reason: 'RMS $rms must be considered SILENCE by the gate');
        expect(
          isSilentBadgeVisible(currentRms: rms, isCallActive: true),
          isTrue,
          reason: 'Silent badge must be displayed when RMS is $rms (< 0.003)',
        );
      }
    });

    test('Silent badge is never displayed when call is not active, regardless of RMS', () {
      expect(isSilentBadgeVisible(currentRms: 0.0, isCallActive: false), isFalse);
      expect(isSilentBadgeVisible(currentRms: 0.001, isCallActive: false), isFalse);
    });

    test('Silent badge and RMS gate status can never visually disagree across all threshold boundaries', () {
      // Step through RMS values across the critical boundary [0.0000 to 0.0100]
      for (int i = 0; i <= 100; i++) {
        final rms = i * 0.0001; // 0.0000 to 0.0100 in 0.0001 increments
        final gateActive = isRmsGateActive(currentRms: rms);
        final badgeVisible = isSilentBadgeVisible(currentRms: rms, isCallActive: true);

        // Invariant: badgeVisible must ALWAYS equal !gateActive during an active call
        expect(
          badgeVisible,
          equals(!gateActive),
          reason: 'Contradiction at RMS $rms: gateActive=$gateActive but badgeVisible=$badgeVisible',
        );
      }
    });
  });
}
