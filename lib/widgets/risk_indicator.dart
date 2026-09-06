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
  final String decision;  // "low_risk" | "verification_required" | "action_held" | "insufficient_evidence"

  const RiskIndicator({
    super.key,
    required this.riskLevel,
    required this.action,
    this.decision = '',
  });

  @override
  Widget build(BuildContext context) {
    final effectiveDecision = decision.isNotEmpty ? decision : _decisionFromRisk(riskLevel);
    final color = _decisionColor(effectiveDecision);
    final icon = _decisionIcon(effectiveDecision);
    final title = _decisionTitle(effectiveDecision);
    final subtitle = _decisionSubtitle(effectiveDecision);

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

  static Color riskColor(String level) => _riskColor(level);

  static Color _riskColor(String level) {
    return switch (level) {
      'high' => const Color(0xFFC62828),
      'medium' => const Color(0xFFE65100),
      'unknown' => const Color(0xFF616161),
      _ => const Color(0xFF2E7D32),
    };
  }

  static Color _decisionColor(String decision) {
    return switch (decision) {
      'action_held' => const Color(0xFFC62828),
      'verification_required' => const Color(0xFFE65100),
      'insufficient_evidence' => const Color(0xFF616161),
      _ => const Color(0xFF2E7D32),
    };
  }

  static IconData _decisionIcon(String decision) {
    return switch (decision) {
      'action_held' => Icons.gpp_bad_rounded,
      'verification_required' => Icons.gpp_maybe_rounded,
      'insufficient_evidence' => Icons.help_outline_rounded,
      _ => Icons.shield_rounded,
    };
  }

  static String _decisionTitle(String decision) {
    return switch (decision) {
      'action_held' => 'ACTION HELD',
      'verification_required' => 'VERIFICATION REQUIRED',
      'insufficient_evidence' => 'INSUFFICIENT EVIDENCE',
      _ => 'LOW DETECTED SPOOF EVIDENCE',
    };
  }

  static String _decisionSubtitle(String decision) {
    return switch (decision) {
      'action_held' =>
        'Persistent elevated spoof evidence detected.\nSensitive actions are held pending verification.',
      'verification_required' =>
        'Elevated spoof characteristics detected.\nVerify the caller independently before proceeding.',
      'insufficient_evidence' =>
        'Audio quality or signal insufficient for assessment.\nContinue with caution and verify independently.',
      _ => 'Low detected spoof evidence.\nContinue with caution.',
    };
  }

  static String _decisionFromRisk(String riskLevel) {
    return switch (riskLevel) {
      'high' => 'action_held',
      'medium' => 'verification_required',
      'unknown' => 'insufficient_evidence',
      _ => 'low_risk',
    };
  }
}
