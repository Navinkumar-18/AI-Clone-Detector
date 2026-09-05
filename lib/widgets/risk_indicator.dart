/// Reusable risk level indicator widget.
///
/// Shows a colored badge with risk level text + security action text.
/// Used by both the live call screen and potentially the result view.
library;

import 'package:flutter/material.dart';

class RiskIndicator extends StatelessWidget {
  final String riskLevel; // "low" | "medium" | "high"
  final String action;    // "allow" | "verify" | "block"

  const RiskIndicator({
    super.key,
    required this.riskLevel,
    required this.action,
  });

  @override
  Widget build(BuildContext context) {
    final color = _riskColor(riskLevel);
    final icon = _riskIcon(riskLevel);
    final title = _riskTitle(riskLevel);
    final subtitle = _riskSubtitle(riskLevel);

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
      _ => const Color(0xFF2E7D32),
    };
  }

  static IconData _riskIcon(String level) {
    return switch (level) {
      'high' => Icons.gpp_bad_rounded,
      'medium' => Icons.gpp_maybe_rounded,
      _ => Icons.verified_user_rounded,
    };
  }

  static String _riskTitle(String level) {
    return switch (level) {
      'high' => '🔴 HIGH RISK',
      'medium' => '🟠 ADDITIONAL VERIFICATION REQUIRED',
      _ => '🟢 VOICE VERIFIED',
    };
  }

  static String _riskSubtitle(String level) {
    return switch (level) {
      'high' =>
        'Possible voice clone detected.\nSensitive action should be blocked.',
      'medium' =>
        'Suspicious voice characteristics detected.\nVerify the caller independently before proceeding.',
      _ => 'Low impersonation risk.\nTransaction permitted.',
    };
  }
}
