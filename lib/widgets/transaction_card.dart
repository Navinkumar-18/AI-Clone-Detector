/// Mock sensitive transaction card.
///
/// Demonstrates the "prevention" part of the SIH problem statement:
///   Detection → Risk Assessment → Prevention
///
/// The transaction button state is controlled by the current call-level risk:
///   LOW    → 🟢 TRANSACTION PERMITTED  → CONFIRM TRANSFER (enabled)
///   MEDIUM → 🟠 VERIFICATION REQUIRED  → VERIFY CALLER    (warning)
///   HIGH   → 🛑 TRANSACTION BLOCKED    → disabled/greyed
///
/// This is a DEMO transaction only. No real bank API, no real money.
library;

import 'package:flutter/material.dart';
import 'risk_indicator.dart';

class TransactionCard extends StatelessWidget {
  final String riskLevel; // "low" | "medium" | "high"

  const TransactionCard({super.key, required this.riskLevel});

  @override
  Widget build(BuildContext context) {
    final color = RiskIndicator.riskColor(riskLevel);
    final isBlocked = riskLevel == 'high';
    final needsVerify = riskLevel == 'medium';

    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isBlocked
              ? const Color(0xFFC62828).withValues(alpha: 0.3)
              : Colors.grey.withValues(alpha: 0.2),
          width: 1.5,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF1A237E).withValues(alpha: 0.05),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(15)),
            ),
            child: const Row(
              children: [
                Icon(Icons.account_balance, size: 18, color: Color(0xFF1A237E)),
                SizedBox(width: 8),
                Text(
                  'SENSITIVE TRANSACTION',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: Color(0xFF1A237E),
                    letterSpacing: 1,
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
                _detailRow('Amount', '₹2,00,000'),
                const SizedBox(height: 8),
                _detailRow('Purpose', 'Invoice Payment'),
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
                  child: Row(
                    children: [
                      Icon(
                        isBlocked
                            ? Icons.block
                            : needsVerify
                                ? Icons.warning_amber_rounded
                                : Icons.check_circle_outline,
                        size: 18,
                        color: color,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _statusText(riskLevel),
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            color: color,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),

                // Action button
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: isBlocked
                        ? null
                        : needsVerify
                            ? () => _showVerifyDialog(context)
                            : () => _showSuccessDialog(context),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: isBlocked
                          ? Colors.grey.shade300
                          : needsVerify
                              ? const Color(0xFFE65100)
                              : const Color(0xFF2E7D32),
                      foregroundColor: Colors.white,
                      disabledBackgroundColor: Colors.grey.shade300,
                      disabledForegroundColor: Colors.grey.shade500,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    child: Text(
                      _buttonText(riskLevel),
                      style: const TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 14,
                        letterSpacing: 0.5,
                      ),
                    ),
                  ),
                ),

                // Demo disclaimer
                const SizedBox(height: 8),
                const Center(
                  child: Text(
                    'DEMO ONLY — No real transaction',
                    style: TextStyle(
                      fontSize: 10,
                      color: Colors.grey,
                      fontStyle: FontStyle.italic,
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
          style: const TextStyle(
            fontSize: 13,
            color: Colors.black54,
          ),
        ),
        Text(
          value,
          style: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: Colors.black87,
          ),
        ),
      ],
    );
  }

  String _statusText(String level) {
    return switch (level) {
      'high' => 'VOICE RISK: HIGH — TRANSACTION BLOCKED',
      'medium' => 'VOICE RISK: MEDIUM — ADDITIONAL VERIFICATION REQUIRED',
      _ => 'VOICE RISK: LOW — TRANSACTION PERMITTED',
    };
  }

  String _buttonText(String level) {
    return switch (level) {
      'high' => '🛑 TRANSACTION BLOCKED',
      'medium' => '⚠ VERIFY CALLER',
      _ => '✅ CONFIRM TRANSFER',
    };
  }

  void _showVerifyDialog(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Verification Required'),
        content: const Text(
          'Suspicious voice characteristics detected.\n\n'
          'Please verify the caller through an independent channel '
          '(e.g. callback, video call) before authorizing this transaction.',
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
            Icon(Icons.check_circle, color: Color(0xFF2E7D32)),
            SizedBox(width: 8),
            Text('Transfer Initiated'),
          ],
        ),
        content: const Text(
          'Demo: ₹2,00,000 transfer to ABC Suppliers would proceed.\n\n'
          'This is a demonstration only — no real money was transferred.',
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
}
