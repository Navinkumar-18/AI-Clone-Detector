// Basic VoiceGuard widget smoke test.
//
// Verifies the app builds and the home screen renders
// with the expected two-mode UI.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:sih2026/main.dart';

void main() {
  testWidgets('Home screen shows VoiceGuard title and both action buttons',
      (WidgetTester tester) async {
    await tester.pumpWidget(const VoiceGuardApp());

    // App title
    expect(find.text('VoiceGuard'), findsOneWidget);

    // App subtitle
    expect(find.text('AI Voice Security System'), findsOneWidget);

    // Primary action — Live Call
    expect(find.text('LIVE CALL ANALYSIS'), findsOneWidget);

    // Secondary action — Audio File
    expect(find.text('ANALYZE AUDIO FILE'), findsOneWidget);

    // Record option
    expect(find.text('Record & Analyze'), findsOneWidget);
  });
}
