/// Reusable risk level indicator widget.
///
/// Shows a colored badge with risk level text + security action text.
/// Used by both the live call screen and the result view.
///
/// Decision states:
///   LOW_RISK             → Green  → Low detected spoof evidence
///   VERIFICATION_REQUIRED → Orange → Additional verification required
///   ACTION_HELD          → Red    → Action held pending verification
///   INSUFFICIENT_EVIDENCE → Grey   → Insufficient evidence for assessment
library;

import 'package:flutter/material.dart';

class RiskIndicator extends StatelessWidget {
  final String riskLevel; // "low" | "medium" | "high" | "unknown"
  final String action;    // "allow_with_caution" | "verify" | "hold" | "unavailable"
  final String decision;  // "low_risk" | "verification_required" | "action_held" | "insufficient_evidence" | ...
  final bool canShowLowRisk;

  const RiskIndicator({
    super.key,
    required this.riskLevel,
    required this.action,
    this.decision = '',
    this.canShowLowRisk = false,
  });

  String get effectiveDecision {
    final raw = decision.isNotEmpty ? decision : _decisionFromRisk(riskLevel);
    // FAIL-CLOSED: low_risk can ONLY be effective when canShowLowRisk is true.
    if (raw == 'low_risk' && !canShowLowRisk) {
      return 'insufficient_evidence';
    }
    return raw;
  }

  @override
  Widget build(BuildContext context) {
    final effDecision = effectiveDecision;
    final color = _decisionColor(effDecision);
    final icon = _decisionIcon(effDecision);
    final title = _decisionTitle(effDecision);
    final subtitle = _decisionSubtitle(effDecision);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        border: Border.all(color: color.withValues(alpha: 0.4), width: 2),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          Icon(icon, size: 36, color: color),
          const SizedBox(height: 8),
          Text(
            title,
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w900,
              color: color,
              letterSpacing: 1,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            subtitle,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 13,
              color: color.withValues(alpha: 0.8),
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }

  static Color riskColor(String level, {bool canShowLowRisk = false}) =>
      _riskColor(level, canShowLowRisk: canShowLowRisk);

  static Color _riskColor(String level, {bool canShowLowRisk = false}) {
    return switch (level) {
      'high' => const Color(0xFFC62828),
      'medium' => const Color(0xFFE65100),
      'low' => canShowLowRisk ? const Color(0xFF2E7D32) : const Color(0xFF616161),
      _ => const Color(0xFF616161), // Fail-closed: grey for unknown/unverified
    };
  }

  static Color _decisionColor(String decision) {
    return switch (decision) {
      'low_risk' => const Color(0xFF2E7D32),             // Green ONLY for confirmed low_risk
      'action_held' => const Color(0xFFC62828),          // Red
      'verification_required' => const Color(0xFFE65100),// Orange
      'analysis_stale' => const Color(0xFFE65100),       // Orange warning
      'backend_unavailable' => const Color(0xFFC62828),  // Red
      'audio_quality_poor' => const Color(0xFF757575),   // Grey
      'capture_unavailable' => const Color(0xFF757575),  // Grey
      'empty_or_invalid_capture' => const Color(0xFF757575),
      'model_unavailable' => const Color(0xFF757575),    // Grey
      'permission_denied' => const Color(0xFFE65100),    // Orange
      'capture_stopped' => const Color(0xFF616161),      // Grey
      _ => const Color(0xFF616161),                      // Fail-closed: grey for any fallback
    };
  }

  static IconData _decisionIcon(String decision) {
    return switch (decision) {
      'low_risk' => Icons.shield_rounded,
      'action_held' => Icons.gpp_bad_rounded,
      'verification_required' => Icons.gpp_maybe_rounded,
      'analysis_stale' => Icons.history_rounded,
      'backend_unavailable' => Icons.cloud_off_rounded,
      'model_unavailable' => Icons.miscellaneous_services_rounded,
      'audio_quality_poor' => Icons.volume_down_rounded,
      'capture_unavailable' => Icons.mic_off_rounded,
      'empty_or_invalid_capture' => Icons.mic_off_rounded,
      'permission_denied' => Icons.security_rounded,
      'capture_stopped' => Icons.stop_circle_outlined,
      _ => Icons.help_outline_rounded,
    };
  }

  static String _decisionTitle(String decision) {
    return switch (decision) {
      'low_risk' => 'LOW DETECTED SPOOF EVIDENCE',
      'action_held' => 'ACTION HELD',
      'verification_required' => 'VERIFICATION REQUIRED',
      'audio_quality_poor' => 'AUDIO QUALITY TOO POOR',
      'capture_unavailable' => 'CAPTURE UNAVAILABLE',
      'empty_or_invalid_capture' => 'CAPTURE UNAVAILABLE',
      'backend_unavailable' => 'BACKEND UNAVAILABLE',
      'model_unavailable' => 'MODEL UNAVAILABLE',
      'analysis_stale' => 'ANALYSIS STALE',
      'permission_denied' => 'MICROPHONE PERMISSION REQUIRED',
      'capture_stopped' => 'CAPTURE STOPPED',
      'analyzing' => 'ANALYZING ACOUSTICS...',
      'no_evidence' => 'WAITING FOR SPEECH',
      _ => 'INSUFFICIENT EVIDENCE',
    };
  }

  static String _decisionSubtitle(String decision) {
    return switch (decision) {
      'low_risk' =>
        'Low detected spoof evidence under fresh acoustic analysis.\nContinue with caution.',
      'action_held' =>
        'Persistent elevated spoof evidence detected.\nSensitive actions are held pending verification.',
      'verification_required' =>
        'Elevated spoof characteristics detected.\nVerify caller through an independent channel.',
      'audio_quality_poor' =>
        'Microphone signal is noisy, clipped, or silent.\nMove closer to the speaker or increase volume.',
      'capture_unavailable' =>
        'Audio capture is empty or corrupted.\nEnsure microphone is unmuted and functional.',
      'empty_or_invalid_capture' =>
        'No valid PCM audio was captured from the microphone.',
      'backend_unavailable' =>
        'Cannot reach detection server.\nSensitive actions cannot be permitted.',
      'model_unavailable' =>
        'AI detection model is not loaded or failed initialization.',
      'analysis_stale' =>
        'Audio analysis response timed out.\nRe-verifying current acoustic stream.',
      'permission_denied' =>
        'Microphone permission is required to analyze call audio.',
      'capture_stopped' =>
        'Audio capture has stopped. Start call to resume analysis.',
      'analyzing' =>
        'Capturing audio chunks. Waiting for speech analysis...',
      'no_evidence' =>
        'Place phone near speaker. VoiceGuard will analyze voice once speech begins.',
      _ =>
        'Audio quality or signal insufficient for assessment.\nContinue with caution and verify independently.',
    };
  }

  static String _decisionFromRisk(String riskLevel) {
    return switch (riskLevel) {
      'high' => 'action_held',
      'medium' => 'verification_required',
      'low' => 'low_risk',
      _ => 'insufficient_evidence',
    };
  }
}
