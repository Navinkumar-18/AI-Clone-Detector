// Basic VoiceGuard widget smoke test.
//
// Verifies the app builds and the home screen renders
// with the expected two-mode UI.

import 'package:flutter_test/flutter_test.dart';

import 'package:sih2026/main.dart';

void main() {
  testWidgets('App launches into HomeScreen with 3 options and navigates to Live Call',
      (WidgetTester tester) async {
    await tester.pumpWidget(const VoiceGuardApp());
    await tester.pump();

    // App title and subtitle
    expect(find.text('VoiceGuard'), findsOneWidget);
    expect(find.text('VOICEGUARD'), findsOneWidget);
    expect(find.text('AI Voice Security System'), findsOneWidget);

    // Option 1 — Live Call Analysis
    expect(find.text('LIVE CALL ANALYSIS'), findsOneWidget);

    // Option 2 — Analyze Audio File
    expect(find.text('ANALYZE AUDIO FILE'), findsOneWidget);

    // Option 3 — Record & Analyze
    expect(find.text('Record & Analyze'), findsOneWidget);

    // Settings action in AppBar
    expect(find.byTooltip('Server Settings'), findsOneWidget);

    // Tap LIVE CALL ANALYSIS to navigate to LiveCallScreen
    await tester.tap(find.text('LIVE CALL ANALYSIS'));
    await tester.pumpAndSettle();

    // Verify LiveCallScreen is presented
    expect(find.text('VoiceGuard Live'), findsOneWidget);
    expect(find.text('START LIVE CALL'), findsOneWidget);
  });
}
