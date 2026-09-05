// main.dart — VoiceGuard app entry point and HomeScreen widget.
//
// Two analysis modes:
//   1. Live Call Analysis  → LiveCallScreen (continuous microphone analysis)
//   2. Analyze Audio File  → existing upload/record → single prediction
//
// API logic lives in api_client.dart.
// Backend URL and timeout constants live in config.dart.

import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import 'api_client.dart';
import 'screens/live_call_screen.dart';

// ---------------------------------------------------------------------------
// App root
// ---------------------------------------------------------------------------

void main() => runApp(const VoiceGuardApp());

class VoiceGuardApp extends StatelessWidget {
  const VoiceGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'VoiceGuard',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1565C0),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

// ---------------------------------------------------------------------------
// State enum
// ---------------------------------------------------------------------------

enum _ScreenState { idle, recording, loading, result, error }

// ---------------------------------------------------------------------------
// HomeScreen
// ---------------------------------------------------------------------------

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  // -- state --
  _ScreenState _screen = _ScreenState.idle;
  PredictResult? _result;
  String _errorMessage = '';

  // -- recording --
  final AudioRecorder _recorder = AudioRecorder();
  String? _recordingPath;
  late AnimationController _pulseCtrl;
  late Animation<double> _pulseAnim;

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    )..repeat(reverse: true);
    _pulseAnim = Tween<double>(begin: 0.5, end: 1.0).animate(_pulseCtrl);
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    _recorder.dispose();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Upload flow
  // ---------------------------------------------------------------------------

  Future<void> _pickAndAnalyze() async {
    final picked = await FilePicker.platform.pickFiles(type: FileType.audio);
    if (picked == null) return; // user cancelled
    await _runPrediction(picked.files.single.path!);
  }

  // ---------------------------------------------------------------------------
  // Recording flow
  // ---------------------------------------------------------------------------

  Future<void> _startRecording() async {
    // 1. Check / request microphone permission.
    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) {
      _showPermissionDeniedDialog();
      return;
    }

    // 2. Build a temp file path.
    final tmpDir = await getTemporaryDirectory();
    _recordingPath =
        '${tmpDir.path}/voiceguard_rec_${DateTime.now().millisecondsSinceEpoch}.wav';

    // 3. Start recording.
    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.wav, bitRate: 128000),
      path: _recordingPath!,
    );
    setState(() => _screen = _ScreenState.recording);
  }

  Future<void> _stopRecordingAndAnalyze() async {
    final path = await _recorder.stop();
    if (path == null || path.isEmpty) {
      setState(() {
        _screen = _ScreenState.error;
        _errorMessage = 'Recording failed — no audio was captured.';
      });
      return;
    }
    await _runPrediction(path);
  }

  // ---------------------------------------------------------------------------
  // Shared prediction call
  // ---------------------------------------------------------------------------

  Future<void> _runPrediction(String filePath) async {
    setState(() {
      _screen = _ScreenState.loading;
      _result = null;
      _errorMessage = '';
    });

    try {
      final result = await VoiceGuardApiClient.predict(filePath);
      setState(() {
        _screen = _ScreenState.result;
        _result = result;
      });
    } on BackendUnreachableException {
      setState(() {
        _screen = _ScreenState.error;
        _errorMessage =
            'Cannot reach the server.\n\nMake sure the backend is running and the URL in config.dart is correct.';
      });
    } on UnsupportedFileException {
      setState(() {
        _screen = _ScreenState.error;
        _errorMessage =
            'This file could not be analysed.\n\nPlease use a supported audio format (WAV, MP3, M4A, FLAC).';
      });
    } on BackendErrorException catch (e) {
      setState(() {
        _screen = _ScreenState.error;
        _errorMessage = 'The server returned an error (HTTP ${e.statusCode}).\n\nTry again or check backend logs.';
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Reset
  // ---------------------------------------------------------------------------

  void _reset() {
    setState(() {
      _screen = _ScreenState.idle;
      _result = null;
      _errorMessage = '';
      _recordingPath = null;
    });
  }

  // ---------------------------------------------------------------------------
  // Permission denied dialog
  // ---------------------------------------------------------------------------

  void _showPermissionDeniedDialog() {
    showDialog<void>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Microphone Access Required'),
        content: const Text(
          'VoiceGuard needs microphone access to record audio.\n\n'
          'Please grant the permission in your device settings and try again.',
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

  // ---------------------------------------------------------------------------
  // Navigate to Live Call screen
  // ---------------------------------------------------------------------------

  void _openLiveCallScreen() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const LiveCallScreen()),
    );
  }

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1565C0),
        foregroundColor: Colors.white,
        elevation: 2,
        title: const Row(
          children: [
            Icon(Icons.shield_outlined, size: 22),
            SizedBox(width: 8),
            Text(
              'VoiceGuard',
              style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 0.5),
            ),
          ],
        ),
        actions: [
          if (_screen == _ScreenState.result || _screen == _ScreenState.error)
            TextButton.icon(
              onPressed: _reset,
              icon: const Icon(Icons.refresh, color: Colors.white70),
              label: const Text(
                'Reset',
                style: TextStyle(color: Colors.white70),
              ),
            ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
          child: _buildBody(),
        ),
      ),
    );
  }

  Widget _buildBody() {
    return switch (_screen) {
      _ScreenState.idle => _buildIdleView(),
      _ScreenState.recording => _buildRecordingView(),
      _ScreenState.loading => _buildLoadingView(),
      _ScreenState.result => _buildResultView(),
      _ScreenState.error => _buildErrorView(),
    };
  }

  // ---------------------------------------------------------------------------
  // Idle view — two-button home screen
  // ---------------------------------------------------------------------------

  Widget _buildIdleView() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Icon(Icons.shield, size: 64, color: Color(0xFF1565C0)),
        const SizedBox(height: 16),
        const Text(
          'VOICEGUARD',
          style: TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.w900,
            color: Color(0xFF1A237E),
            letterSpacing: 3,
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 6),
        const Text(
          'AI Voice Security System',
          style: TextStyle(
            fontSize: 14,
            color: Colors.black45,
            letterSpacing: 1,
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 40),

        // Primary action — Live Call Analysis
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF1565C0),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 20),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
              ),
              elevation: 3,
              textStyle: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            onPressed: _openLiveCallScreen,
            icon: const Icon(Icons.mic, size: 24),
            label: const Text('LIVE CALL ANALYSIS'),
          ),
        ),
        const SizedBox(height: 16),

        // Secondary action — Analyze Audio File
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
            style: OutlinedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 20),
              side: const BorderSide(color: Color(0xFF1565C0), width: 1.5),
              foregroundColor: const Color(0xFF1565C0),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
              ),
              textStyle: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            onPressed: _pickAndAnalyze,
            icon: const Icon(Icons.folder_open, size: 24),
            label: const Text('ANALYZE AUDIO FILE'),
          ),
        ),
        const SizedBox(height: 12),

        // Record option
        SizedBox(
          width: double.infinity,
          child: TextButton.icon(
            style: TextButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
              foregroundColor: const Color(0xFFC62828),
              textStyle: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
              ),
            ),
            onPressed: _startRecording,
            icon: const Icon(Icons.fiber_manual_record, size: 18),
            label: const Text('Record & Analyze'),
          ),
        ),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Recording view
  // ---------------------------------------------------------------------------

  Widget _buildRecordingView() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        FadeTransition(
          opacity: _pulseAnim,
          child: Container(
            width: 80,
            height: 80,
            decoration: const BoxDecoration(
              color: Color(0xFFC62828),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.mic, color: Colors.white, size: 40),
          ),
        ),
        const SizedBox(height: 24),
        const Text(
          'Recording…',
          style: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.bold,
            color: Color(0xFFC62828),
          ),
        ),
        const SizedBox(height: 8),
        const Text(
          'Speak now. Tap Stop when you are done.',
          style: TextStyle(color: Colors.black54),
        ),
        const SizedBox(height: 40),
        ElevatedButton.icon(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFFC62828),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 36, vertical: 16),
            textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
          onPressed: _stopRecordingAndAnalyze,
          icon: const Icon(Icons.stop),
          label: const Text('Stop & Analyse'),
        ),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Loading view
  // ---------------------------------------------------------------------------

  Widget _buildLoadingView() {
    return const Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        CircularProgressIndicator(
          color: Color(0xFF1565C0),
          strokeWidth: 3,
        ),
        SizedBox(height: 24),
        Text(
          'Analysing…',
          style: TextStyle(
            fontSize: 18,
            color: Color(0xFF1565C0),
            fontWeight: FontWeight.w600,
          ),
        ),
        SizedBox(height: 8),
        Text(
          'Sending audio to the detection model.',
          style: TextStyle(color: Colors.black45),
        ),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Result view
  // ---------------------------------------------------------------------------

  Widget _buildResultView() {
    final r = _result!;
    final color = _riskColor(r.riskLevel);
    final isSpoof = r.label == 'spoof';
    final pct = (r.confidence * 100).toStringAsFixed(1);

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // ── Main verdict card ──────────────────────────────────────────────
        Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 32, horizontal: 24),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.08),
            border: Border.all(color: color.withValues(alpha: 0.4), width: 2),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Column(
            children: [
              // Verdict icon
              Icon(
                isSpoof ? Icons.warning_rounded : Icons.verified_user,
                size: 64,
                color: color,
              ),
              const SizedBox(height: 12),

              // BONAFIDE / SPOOF label — large, readable from across a table
              Text(
                r.label.toUpperCase(),
                style: TextStyle(
                  fontSize: 40,
                  fontWeight: FontWeight.w900,
                  color: color,
                  letterSpacing: 2,
                ),
              ),

              const SizedBox(height: 12),

              // Risk level badge
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 6),
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  'RISK: ${r.riskLevel.toUpperCase()}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                    letterSpacing: 1.2,
                  ),
                ),
              ),

              const SizedBox(height: 24),
              const Divider(),
              const SizedBox(height: 16),

              // Confidence section
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Confidence',
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 15,
                      color: Colors.black87,
                    ),
                  ),
                  Text(
                    '$pct%',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 20,
                      color: color,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: LinearProgressIndicator(
                  value: r.confidence,
                  minHeight: 14,
                  backgroundColor: color.withValues(alpha: 0.15),
                  valueColor: AlwaysStoppedAnimation<Color>(color),
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 32),

        // ── Reset button ───────────────────────────────────────────────────
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
            style: OutlinedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
              side: const BorderSide(color: Color(0xFF1565C0), width: 1.5),
              foregroundColor: const Color(0xFF1565C0),
              textStyle: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
            onPressed: _reset,
            icon: const Icon(Icons.refresh),
            label: const Text('Check Another Clip'),
          ),
        ),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Error view
  // ---------------------------------------------------------------------------

  Widget _buildErrorView() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Icon(Icons.error_outline, size: 64, color: Color(0xFFC62828)),
        const SizedBox(height: 20),
        const Text(
          'Something went wrong',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: Color(0xFFC62828),
          ),
        ),
        const SizedBox(height: 12),
        Text(
          _errorMessage,
          textAlign: TextAlign.center,
          style: const TextStyle(color: Colors.black54, height: 1.5, fontSize: 15),
        ),
        const SizedBox(height: 36),
        ElevatedButton.icon(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF1565C0),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 36, vertical: 14),
          ),
          onPressed: _reset,
          icon: const Icon(Icons.arrow_back),
          label: const Text('Try Again'),
        ),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /// Returns a readable color for each risk level.
  /// DESIGN DECISION — verify these values are legible on your demo screen.
  Color _riskColor(String level) {
    return switch (level) {
      'high' => const Color(0xFFC62828),   // dark red
      'medium' => const Color(0xFFE65100), // deep orange
      _ => const Color(0xFF2E7D32),        // dark green (low / unknown)
    };
  }
}