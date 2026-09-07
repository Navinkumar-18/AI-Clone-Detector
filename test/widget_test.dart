// Basic VoiceGuard widget smoke test.
//
// Verifies the app builds and the home screen renders
// with the expected two-mode UI.

import 'package:flutter_test/flutter_test.dart';

import 'package:sih2026/main.dart';

void main() {
  testWidgets('App launches directly into Live Call mode with Start Live Call button',
      (WidgetTester tester) async {
    await tester.pumpWidget(const VoiceGuardApp());
    await tester.pump();

    // Live App title & Start Call button
    expect(find.text('VoiceGuard Live'), findsOneWidget);
    expect(find.text('START LIVE CALL'), findsOneWidget);

    // File screening and settings actions in AppBar
    expect(find.byTooltip('Screen Audio File'), findsOneWidget);
    expect(find.byTooltip('Server Settings'), findsOneWidget);
  });
}
