/// Mock sensitive transaction card.
///
/// Demonstrates the "prevention" part of the SIH problem statement:
///   Detection → Risk Assessment → Prevention
///
/// The transaction button state is controlled by the current call-level decision:
///   LOW_RISK              → CONTINUE WITH CAUTION (enabled, demo only)
///   VERIFICATION_REQUIRED → VERIFY CALLER (warning dialog with options)
///   ACTION_HELD           → TRANSACTION HELD (disabled)
///   INSUFFICIENT_EVIDENCE → CANNOT ASSESS (disabled)
///
/// This is a DEMO transaction only. No real bank API, no real money.
/// Clearly labeled: "Demo mode — no real financial transaction is executed."
library;

import 'package:flutter/material.dart';

class TransactionCard extends StatelessWidget {
  final String riskLevel; // "low" | "medium" | "high" | "unknown"
  final String decision;  // "low_risk" | "verification_required" | "action_held" | "insufficient_evidence"
  final List<String> reasonCodes;
  final int evidenceWindows;
  final bool canShowLowRisk;

  const TransactionCard({
    super.key,
    required this.riskLevel,
    this.decision = '',
    this.reasonCodes = const [],
    this.evidenceWindows = 0,
    this.canShowLowRisk = false,
  });

  String get _effectiveDecision {
    final raw = decision.isNotEmpty ? decision : _decisionFromRisk(riskLevel);
    if (raw == 'low_risk' && !canShowLowRisk) {
      return 'insufficient_evidence';
    }
    return raw;
  }

  @override
  Widget build(BuildContext context) {
    final effectiveDecision = _effectiveDecision;
    final isAllowed = effectiveDecision == 'low_risk' && canShowLowRisk;
    final isHeld = effectiveDecision == 'action_held';
    final needsVerify = effectiveDecision == 'verification_required';
    final isInsufficient = effectiveDecision == 'insufficient_evidence' ||
        (!isAllowed && !isHeld && !needsVerify);
    final isDisabled = !isAllowed && !needsVerify;
    final color = isAllowed
        ? const Color(0xFF2E7D32)
        : isHeld
            ? const Color(0xFFC62828)
            : needsVerify
                ? const Color(0xFFE65100)
                : const Color(0xFF616161);

    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isDisabled
              ? Colors.white.withValues(alpha: 0.1)
              : isHeld
                  ? const Color(0xFFC62828).withValues(alpha: 0.4)
                  : Colors.white.withValues(alpha: 0.15),
          width: 1.5,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.2),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header with demo label
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF1F2937),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(15)),
            ),
            child: const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.account_balance, size: 18, color: Color(0xFF58A6FF)),
                    SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'SENSITIVE TRANSACTION',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color: Colors.white,
                          letterSpacing: 1,
                        ),
                      ),
                    ),
                  ],
                ),
                SizedBox(height: 4),
                Text(
                  'Protected by real-time voice verification',
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    color: Colors.white54,
                  ),
                ),
              ],
            ),
          ),

          // Transaction details
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _detailRow('Receiver', 'ABC Suppliers'),
                const SizedBox(height: 8),
                _detailRow('Amount', '\u20B92,00,000'),
                const SizedBox(height: 8),
                _detailRow('Purpose', 'Invoice Payment'),
                const SizedBox(height: 8),
                _detailRow('Evidence windows', '$evidenceWindows'),
                const SizedBox(height: 16),

                // Risk status banner
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.08),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: color.withValues(alpha: 0.3)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(
                            isHeld
                                ? Icons.block
                                : isInsufficient
                                    ? Icons.help_outline
                                    : needsVerify
                                        ? Icons.warning_amber_rounded
                                        : Icons.shield_outlined,
                            size: 18,
                            color: color,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              _statusText(effectiveDecision),
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w700,
                                color: color,
                              ),
                            ),
                          ),
                        ],
                      ),
                      if (reasonCodes.isNotEmpty) ...[
                        const SizedBox(height: 6),
                        Text(
                          'Reason: ${reasonCodes.join(", ")}',
                          style: TextStyle(
                            fontSize: 10,
                            color: color.withValues(alpha: 0.7),
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
                const SizedBox(height: 12),

                // Action button
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: isDisabled
                        ? null
                        : needsVerify
                            ? () => _showVerifyDialog(context)
                            : () => _showSuccessDialog(context),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: isDisabled
                          ? Colors.white10
                          : needsVerify
                              ? const Color(0xFFE65100)
                              : const Color(0xFF2E7D32),
                      foregroundColor: Colors.white,
                      disabledBackgroundColor: Colors.white10,
                      disabledForegroundColor: Colors.white38,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    child: Text(
                      _buttonText(effectiveDecision),
                      style: const TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 14,
                        letterSpacing: 0.5,
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 8),
                Center(
                  child: Text(
                    'High-value transfer requires verified acoustic integrity',
                    style: TextStyle(
                      fontSize: 10,
                      color: Colors.white.withValues(alpha: 0.4),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _detailRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 13,
            color: Colors.white.withValues(alpha: 0.5),
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            value,
            textAlign: TextAlign.end,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
        ),
      ],
    );
  }

  String _statusText(String dec) {
    return switch (dec) {
      'action_held' => 'ACTION HELD — Persistent elevated spoof evidence',
      'verification_required' => 'VERIFICATION REQUIRED — Independent verification needed',
      'low_risk' => 'LOW SPOOF EVIDENCE — Continue with caution',
      _ => 'INSUFFICIENT EVIDENCE — Cannot assess voice risk',
    };
  }

  String _buttonText(String dec) {
    return switch (dec) {
      'action_held' => 'TRANSACTION HELD',
      'verification_required' => 'VERIFY CALLER FIRST',
      'low_risk' => 'CONTINUE WITH CAUTION',
      _ => 'CANNOT ASSESS',
    };
  }

  void _showVerifyDialog(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: Color(0xFFE65100), size: 24),
            SizedBox(width: 8),
            Expanded(
              child: Text(
                'Independent Verification Required',
                style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold),
              ),
            ),
          ],
        ),
        content: const Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Elevated spoof characteristics detected. '
              'Verify the caller through an independent channel before '
              'authorizing this transaction.\n',
            ),
            Text(
              'Verification options:',
              style: TextStyle(fontWeight: FontWeight.w700),
            ),
            SizedBox(height: 8),
            _VerificationOption(
              icon: Icons.phone_android,
              text: 'Verify in the official banking app',
            ),
            _VerificationOption(
              icon: Icons.contact_phone,
              text: 'Call the saved contact number',
            ),
            _VerificationOption(
              icon: Icons.people,
              text: 'Contact a trusted person',
            ),
            _VerificationOption(
              icon: Icons.key,
              text: 'Use a pre-agreed verification phrase',
            ),
            _VerificationOption(
              icon: Icons.cancel_outlined,
              text: 'Cancel or delay the action',
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('UNDERSTOOD'),
          ),
        ],
      ),
    );
  }

  void _showSuccessDialog(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.check_circle_outline, color: Color(0xFF2E7D32), size: 24),
            SizedBox(width: 8),
            Expanded(
              child: Text(
                'Transfer Verification Passed',
                style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold),
              ),
            ),
          ],
        ),
        content: const Text(
          'Acoustic integrity verified.\n\n'
          'Transfer of \u20B92,00,000 to ABC Suppliers is authorized.\n'
          'Voice authenticity metrics meet active security policy criteria.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
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

/// Simple verification option row.
class _VerificationOption extends StatelessWidget {
  final IconData icon;
  final String text;

  const _VerificationOption({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        children: [
          Icon(icon, size: 16, color: Colors.black54),
          const SizedBox(width: 8),
          Flexible(
            child: Text(text, style: const TextStyle(fontSize: 13)),
          ),
        ],
      ),
    );
  }
}
