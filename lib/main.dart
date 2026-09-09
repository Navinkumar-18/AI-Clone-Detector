// main.dart — VoiceGuard app entry point and HomeScreen widget.
//
// Two analysis modes:
//   1. Live Call Analysis  → LiveCallScreen (continuous microphone analysis)
//   2. Analyze Audio File  → existing upload/record → single prediction
//
// API logic lives in api_client.dart.
// Backend URL and timeout constants live in config.dart.
//
// PRIVACY: Only PredictResult metadata (label, confidence, riskLevel) is
// retained in widget state. Raw audio is never stored beyond the HTTP request
// lifetime.

import 'dart:async';
import 'dart:io';
import 'dart:math';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import 'api_client.dart';
import 'config.dart';
import 'models/sliding_window_result.dart';
import 'screens/live_call_screen.dart';
import 'widgets/risk_indicator.dart';
import 'widgets/transaction_card.dart';

// ---------------------------------------------------------------------------
// App root
// ---------------------------------------------------------------------------

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Load any previously-saved backend URL from SharedPreferences
  await BackendConfig.init();

  runApp(const VoiceGuardApp());
}

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
  final FileSlidingAnalysisResult? initialResult;
  final PredictResult? initialPredictResult;
  final String? initialFilename;
  const HomeScreen({
    super.key,
    this.initialResult,
    this.initialPredictResult,
    this.initialFilename,
  });
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  // -- state --
  _ScreenState _screen = _ScreenState.idle;
  PredictResult? _predictResult;
  FileSlidingAnalysisResult? _slidingResult;
  String _errorMessage = '';
  String _activeFilename = '';
  int? _expandedWindowIndex;

  // -- recording & animation --
  final AudioRecorder _recorder = AudioRecorder();
  String? _recordingPath;
  late AnimationController _pulseCtrl;
  late Animation<double> _pulseAnim;
  Timer? _recordingTimer;
  Duration _recordingDuration = Duration.zero;
  double _audioLevel = 0.0;

  // -- backend & health --
  bool _backendConnected = false;
  Timer? _healthTimer;

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  @override
  void initState() {
    super.initState();
    _activeFilename = widget.initialFilename ?? '';
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    )..repeat(reverse: true);
    _pulseAnim = Tween<double>(begin: 0.4, end: 1.0).animate(_pulseCtrl);
    if (widget.initialPredictResult != null) {
      _screen = _ScreenState.result;
      _predictResult = widget.initialPredictResult;
    } else if (widget.initialResult != null) {
      _screen = _ScreenState.result;
      _slidingResult = widget.initialResult;
    } else {
      _checkHealth();
      _startHealthChecks();
    }
  }

  @override
  void dispose() {
    _healthTimer?.cancel();
    _recordingTimer?.cancel();
    _pulseCtrl.dispose();
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _checkHealth() async {
    final ok = await VoiceGuardApiClient.checkHealth();
    if (mounted) setState(() => _backendConnected = ok);
  }

  void _startHealthChecks() {
    _healthTimer?.cancel();
    _healthTimer = Timer.periodic(
      Duration(seconds: kHealthCheckIntervalSec),
      (_) => _checkHealth(),
    );
  }

  // ---------------------------------------------------------------------------
  // Upload flow
  // ---------------------------------------------------------------------------

  Future<void> _pickAndAnalyze() async {
    try {
      FilePickerResult? picked;
      try {
        picked = await FilePicker.platform.pickFiles(
          type: FileType.custom,
          allowedExtensions: ['wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac', 'webm', 'mp4', 'opus'],
          withData: true,
        );
      } catch (_) {
        picked = await FilePicker.platform.pickFiles(
          type: FileType.audio,
          withData: true,
        );
      }

      if (picked == null || picked.files.isEmpty) return;

      final file = picked.files.single;
      String? resolvedPath = file.path;

      if (resolvedPath == null || !File(resolvedPath).existsSync()) {
        if (file.bytes != null && file.bytes!.isNotEmpty) {
          final tmpDir = await getTemporaryDirectory();
          final ext = file.extension != null ? '.${file.extension}' : '.wav';
          final safeName = 'voiceguard_upload_${DateTime.now().millisecondsSinceEpoch}$ext';
          final tempFile = File('${tmpDir.path}/$safeName');
          await tempFile.writeAsBytes(file.bytes!);
          resolvedPath = tempFile.path;
        }
      }

      if (resolvedPath == null) {
        setState(() {
          _screen = _ScreenState.error;
          _errorMessage =
              'Could not access the selected file.\n\nPlease select a local audio file stored on your device.';
        });
        return;
      }

      setState(() {
        _activeFilename = file.name;
      });

      await _runPrediction(resolvedPath, originalFilename: file.name);
    } catch (e) {
      setState(() {
        _screen = _ScreenState.error;
        _errorMessage = 'Failed to open or read audio file:\n$e';
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Recording flow
  // ---------------------------------------------------------------------------

  Future<void> _startRecording() async {
    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) {
      _showPermissionDeniedDialog();
      return;
    }

    final tmpDir = await getTemporaryDirectory();
    _recordingPath =
        '${tmpDir.path}/voiceguard_rec_${DateTime.now().millisecondsSinceEpoch}.wav';

    await _recorder.start(
      const RecordConfig(encoder: AudioEncoder.wav, bitRate: 128000),
      path: _recordingPath!,
    );

    _recordingDuration = Duration.zero;
    _recordingTimer?.cancel();
    _recordingTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) {
        setState(() {
          _recordingDuration += const Duration(seconds: 1);
          _audioLevel = (0.25 + 0.45 * sin(_recordingDuration.inSeconds * 1.6).abs()).clamp(0.1, 0.95);
        });
      }
    });

    setState(() {
      _screen = _ScreenState.recording;
      _activeFilename = 'Microphone Recording.wav';
    });
  }

  Future<void> _stopRecordingAndAnalyze() async {
    _recordingTimer?.cancel();
    final path = await _recorder.stop();
    if (path == null || path.isEmpty) {
      setState(() {
        _screen = _ScreenState.error;
        _errorMessage = 'Recording failed — no audio was captured.';
      });
      return;
    }
    await _runPrediction(path, originalFilename: 'Microphone Recording.wav');
  }

  // ---------------------------------------------------------------------------
  // Prediction call (Complete-file analysis via POST /predict)
  // ---------------------------------------------------------------------------

  Future<void> _runPrediction(String filePath, {String? originalFilename}) async {
    setState(() {
      _screen = _ScreenState.loading;
      _predictResult = null;
      _slidingResult = null;
      _errorMessage = '';
      _expandedWindowIndex = null;
      _audioLevel = 0.55;
    });

    try {
      final result = await VoiceGuardApiClient.predict(
        filePath,
        originalFilename: originalFilename,
      );
      if (mounted) {
        setState(() {
          _screen = _ScreenState.result;
          _predictResult = result;
        });
      }
    } on BackendUnreachableException catch (e) {
      if (mounted) {
        setState(() {
          _screen = _ScreenState.error;
          _errorMessage =
              'Cannot reach the server (${e.detail}).\n\nMake sure the backend is running and the URL in Settings is correct.';
        });
      }
    } on UnsupportedFileException catch (e) {
      if (mounted) {
        setState(() {
          _screen = _ScreenState.error;
          _errorMessage = e.message ??
              'This file could not be analysed.\n\nPlease use a supported audio format (WAV, MP3, M4A, FLAC, OGG, AAC).';
        });
      }
    } on BackendErrorException catch (e) {
      if (mounted) {
        setState(() {
          _screen = _ScreenState.error;
          _errorMessage = e.detail.isNotEmpty
              ? 'Server error (HTTP ${e.statusCode}):\n${e.detail}'
              : 'The server returned an error (HTTP ${e.statusCode}).\n\nTry again or check backend logs.';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _screen = _ScreenState.error;
          _errorMessage = 'Unexpected error during analysis:\n$e';
        });
      }
    } finally {
      _deleteRecordingIfNeeded(filePath);
    }
  }

  void _deleteRecordingIfNeeded(String filePath) {
    if (_recordingPath != null && filePath == _recordingPath) {
      try {
        final f = File(filePath);
        if (f.existsSync()) {
          f.deleteSync();
        }
      } catch (_) {}
      _recordingPath = null;
    }
  }

  void _reset() {
    if (_recordingPath != null) {
      _deleteRecordingIfNeeded(_recordingPath!);
    }
    _recordingTimer?.cancel();
    setState(() {
      _screen = _ScreenState.idle;
      _predictResult = null;
      _slidingResult = null;
      _errorMessage = '';
      _activeFilename = '';
      _expandedWindowIndex = null;
      _recordingDuration = Duration.zero;
      _audioLevel = 0.0;
    });
  }

  void _showPermissionDeniedDialog() {
    showDialog<void>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: const Color(0xFF161B22),
        title: const Text('Microphone Access Required', style: TextStyle(color: Colors.white)),
        content: const Text(
          'VoiceGuard needs microphone access to record audio.\n\n'
          'Please grant permission in device settings and try again.',
          style: TextStyle(color: Colors.white70),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK', style: TextStyle(color: Color(0xFF58A6FF))),
          ),
        ],
      ),
    );
  }

  void _openLiveCallScreen() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const LiveCallScreen()),
    );
  }

  void _openSettingsDialog() {
    showDialog<void>(
      context: context,
      builder: (_) => const SettingsDialog(),
    );
  }

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    final bool isIdle = _screen == _ScreenState.idle;

    return Scaffold(
      backgroundColor: isIdle ? Colors.white : const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: isIdle ? const Color(0xFF1565C0) : const Color(0xFF161B22),
        foregroundColor: Colors.white,
        elevation: isIdle ? 2 : 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            if (Navigator.of(context).canPop()) {
              Navigator.of(context).pop();
            }
          },
        ),
        title: isIdle
            ? const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.shield_outlined, size: 22),
                  SizedBox(width: 8),
                  Flexible(
                    child: Text(
                      'VoiceGuard',
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        letterSpacing: 0.5,
                      ),
                    ),
                  ),
                ],
              )
            : const Text(
                'VoiceGuard File Analysis',
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 17,
                  letterSpacing: 0.3,
                ),
              ),
        actions: [
          IconButton(
            visualDensity: isIdle ? VisualDensity.standard : VisualDensity.compact,
            padding: EdgeInsets.symmetric(horizontal: isIdle ? 8 : 4),
            icon: Icon(
              Icons.settings,
              color: isIdle ? Colors.white : Colors.white70,
              size: isIdle ? 24 : 20,
            ),
            tooltip: 'Server Settings',
            onPressed: _openSettingsDialog,
          ),
          if (!isIdle)
            Padding(
              padding: const EdgeInsets.only(right: 12, left: 4),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
                decoration: BoxDecoration(
                  color: (_backendConnected
                          ? const Color(0xFF3FB950)
                          : const Color(0xFFF85149))
                      .withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(
                    color: (_backendConnected
                            ? const Color(0xFF3FB950)
                            : const Color(0xFFF85149))
                        .withValues(alpha: 0.35),
                    width: 1,
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 6,
                      height: 6,
                      decoration: BoxDecoration(
                        color: _backendConnected
                            ? const Color(0xFF3FB950)
                            : const Color(0xFFF85149),
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 4),
                    Text(
                      _backendConnected ? 'ENGINE' : 'OFFLINE',
                      style: TextStyle(
                        fontSize: 9,
                        fontWeight: FontWeight.w700,
                        color: _backendConnected
                            ? const Color(0xFF3FB950)
                            : const Color(0xFFF85149),
                        letterSpacing: 0.5,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
      bottomNavigationBar: _screen == _ScreenState.result
          ? SafeArea(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                decoration: BoxDecoration(
                  color: const Color(0xFF161B22),
                  border: Border(
                    top: BorderSide(
                      color: Colors.white.withValues(alpha: 0.08),
                    ),
                  ),
                ),
                child: SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF238636),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      elevation: 3,
                      textStyle: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 0.8,
                      ),
                    ),
                    onPressed: _reset,
                    icon: const Icon(Icons.refresh, size: 20),
                    label: const Text('SCREEN ANOTHER FILE'),
                  ),
                ),
              ),
            )
          : null,
      body: SafeArea(
        child: Padding(
          padding: EdgeInsets.symmetric(
            horizontal: isIdle ? 24 : 20,
            vertical: isIdle ? 32 : 16,
          ),
          child: switch (_screen) {
            _ScreenState.idle => _buildIdleView(),
            _ScreenState.recording => _buildRecordingView(),
            _ScreenState.loading => _buildLoadingView(),
            _ScreenState.result => _buildResultView(),
            _ScreenState.error => _buildErrorView(),
          },
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // IDLE view — matching the design with 3 options
  // ---------------------------------------------------------------------------

  Widget _buildIdleView() {
    return Center(
      child: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.shield, size: 64, color: Color(0xFF1565C0)),
            const SizedBox(height: 16),
            const Text(
              'VOICEGUARD',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.w900,
                color: Color(0xFF1A237E),
                letterSpacing: 3,
              ),
            ),
            const SizedBox(height: 6),
            const Text(
              'AI Voice Security System',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                color: Colors.black45,
                letterSpacing: 1,
              ),
            ),
            const SizedBox(height: 40),

            // Option 1 — Live Call Analysis
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
                    letterSpacing: 0.5,
                  ),
                ),
                onPressed: _openLiveCallScreen,
                icon: const Icon(Icons.mic, size: 24),
                label: const Text('LIVE CALL ANALYSIS'),
              ),
            ),
            const SizedBox(height: 16),

            // Option 2 — Analyze Audio File
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
                    letterSpacing: 0.5,
                  ),
                ),
                onPressed: _pickAndAnalyze,
                icon: const Icon(Icons.folder_open, size: 24),
                label: const Text('ANALYZE AUDIO FILE'),
              ),
            ),
            const SizedBox(height: 12),

            // Option 3 — Record & Analyze
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
                icon: const Icon(Icons.fiber_manual_record, size: 18, color: Color(0xFFC62828)),
                label: const Text('Record & Analyze'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Recording view
  // ---------------------------------------------------------------------------

  Widget _buildRecordingView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0xFFF85149).withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: const Color(0xFFF85149).withValues(alpha: 0.4)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  FadeTransition(
                    opacity: _pulseAnim,
                    child: Container(
                      width: 8,
                      height: 8,
                      decoration: const BoxDecoration(
                        color: Color(0xFFF85149),
                        shape: BoxShape.circle,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  const Text(
                    'RECORDING AUDIO',
                    style: TextStyle(
                      color: Color(0xFFF85149),
                      fontWeight: FontWeight.bold,
                      fontSize: 11,
                      letterSpacing: 1,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 32),

            FadeTransition(
              opacity: _pulseAnim,
              child: Container(
                width: 90,
                height: 90,
                decoration: BoxDecoration(
                  color: const Color(0xFFF85149).withValues(alpha: 0.2),
                  shape: BoxShape.circle,
                  border: Border.all(color: const Color(0xFFF85149), width: 2),
                ),
                child: const Icon(Icons.mic, color: Color(0xFFF85149), size: 44),
              ),
            ),
            const SizedBox(height: 24),

            Text(
              _formatDuration(_recordingDuration),
              style: const TextStyle(
                fontSize: 36,
                fontWeight: FontWeight.bold,
                fontFamily: 'monospace',
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Speak now. VoiceGuard will partition audio into 4s windows.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.white.withValues(alpha: 0.5), fontSize: 13),
            ),
            const SizedBox(height: 24),

            _buildAudioLevelBars(active: true),
            const SizedBox(height: 40),

            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFF85149),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  elevation: 3,
                  textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                onPressed: _stopRecordingAndAnalyze,
                icon: const Icon(Icons.stop),
                label: const Text('STOP & ANALYSE'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Loading view — live scanning layout replacing plain white spinner
  // ---------------------------------------------------------------------------

  Widget _buildLoadingView() {
    return SingleChildScrollView(
      child: Column(
        children: [
          // Status bar
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: const Color(0xFF58A6FF).withValues(alpha: 0.3),
              ),
            ),
            child: Row(
              children: [
                FadeTransition(
                  opacity: _pulseAnim,
                  child: Container(
                    width: 10,
                    height: 10,
                    decoration: const BoxDecoration(
                      color: Color(0xFF58A6FF),
                      shape: BoxShape.circle,
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                const Expanded(
                  child: Text(
                    'PROCESSING AUDIO FILE',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: Color(0xFF58A6FF),
                      fontWeight: FontWeight.w700,
                      fontSize: 12,
                      letterSpacing: 0.8,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                const SizedBox(
                  width: 14,
                  height: 14,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: Color(0xFF58A6FF),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // Audio visualizer bars
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'AUDIO INPUT WAVEFORM',
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    color: Colors.white.withValues(alpha: 0.4),
                    letterSpacing: 1,
                  ),
                ),
                const SizedBox(height: 10),
                _buildAudioLevelBars(active: true),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // Central scanning radar card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(vertical: 36, horizontal: 20),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: const Color(0xFF58A6FF).withValues(alpha: 0.2),
              ),
            ),
            child: Column(
              children: [
                const SizedBox(
                  width: 64,
                  height: 64,
                  child: CircularProgressIndicator(
                    color: Color(0xFF58A6FF),
                    strokeWidth: 3,
                  ),
                ),
                const SizedBox(height: 24),
                const Text(
                  'Analysing Audio Signal...',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _activeFilename.isNotEmpty ? 'File: $_activeFilename' : 'Processing audio capture...',
                  style: TextStyle(
                    fontSize: 12,
                    fontFamily: 'monospace',
                    color: Colors.white.withValues(alpha: 0.6),
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.05),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: const Text(
                    'Running deepfake acoustic representation inference',
                    style: TextStyle(
                      fontSize: 11,
                      color: Color(0xFF58A6FF),
                      fontWeight: FontWeight.w600,
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

  // ---------------------------------------------------------------------------
  // Result view router
  // ---------------------------------------------------------------------------

  Widget _buildResultView() {
    if (_predictResult != null) {
      return _buildPredictResultView(_predictResult!);
    }
    if (_slidingResult != null) {
      return _buildSlidingResultView(_slidingResult!);
    }
    return const SizedBox.shrink();
  }

  // ---------------------------------------------------------------------------
  // Complete-File Upload Result View (Single /predict analysis)
  // ---------------------------------------------------------------------------

  Widget _buildPredictResultView(PredictResult r) {
    final isLowRisk = r.canShowLowRisk;
    final scoreColor = isLowRisk
        ? const Color(0xFF3FB950)
        : (r.riskLevel == 'high'
            ? const Color(0xFFF85149)
            : (r.riskLevel == 'medium'
                ? const Color(0xFFD29922)
                : const Color(0xFF8B949E)));
    final spoofPct = (r.spoofScore * 100).toStringAsFixed(1);
    final confPct = (r.confidence * 100).toStringAsFixed(1);

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 1. Status Bar
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: scoreColor.withValues(alpha: 0.3),
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: scoreColor,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    isLowRisk
                        ? 'VOICE INTEGRITY VERIFIED'
                        : (r.riskLevel == 'high'
                            ? 'HIGH RISK DEEPFAKE DETECTED'
                            : (r.riskLevel == 'medium'
                                ? 'VERIFICATION REQUIRED'
                                : 'INSUFFICIENT EVIDENCE')),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: scoreColor,
                      fontWeight: FontWeight.w700,
                      fontSize: 11,
                      letterSpacing: 0.5,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 130),
                  child: Text(
                    _activeFilename.isNotEmpty ? _activeFilename : 'Audio File',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    textAlign: TextAlign.end,
                    style: const TextStyle(
                      color: Colors.white70,
                      fontWeight: FontWeight.w600,
                      fontSize: 11,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // 2. Risk Indicator
          RiskIndicator(
            riskLevel: r.riskLevel,
            action: r.action,
            decision: r.decision,
            canShowLowRisk: r.canShowLowRisk,
          ),
          const SizedBox(height: 16),

          // 3. Core Analysis Summary Card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: Text(
                        'VOICE INTEGRITY ASSESSMENT',
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          color: Colors.white.withValues(alpha: 0.5),
                          letterSpacing: 0.8,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: scoreColor.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(color: scoreColor.withValues(alpha: 0.4)),
                      ),
                      child: Text(
                        r.label.toUpperCase(),
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                          color: scoreColor,
                          letterSpacing: 0.5,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: _buildMetricTile(
                        label: 'SPOOF PROBABILITY',
                        value: '$spoofPct%',
                        color: scoreColor,
                        subValue: 'Raw: ${r.spoofScore.toStringAsFixed(4)}',
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _buildMetricTile(
                        label: 'CONFIDENCE',
                        value: '$confPct%',
                        color: Colors.white,
                        subValue: 'Class: ${r.label}',
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: _buildMetricTile(
                        label: 'DECISION',
                        value: r.decision.replaceAll('_', ' ').toUpperCase(),
                        color: scoreColor,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _buildMetricTile(
                        label: 'RECOMMENDED ACTION',
                        value: r.action.replaceAll('_', ' ').toUpperCase(),
                        color: scoreColor,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: _buildMetricTile(
                        label: 'SPEECH DETECTED',
                        value: r.speechDetected ? 'YES' : 'NO',
                        color: r.speechDetected ? const Color(0xFF3FB950) : const Color(0xFFF85149),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _buildMetricTile(
                        label: 'MODEL LOADED',
                        value: r.modelLoaded ? 'ACTIVE' : 'OFFLINE',
                        color: r.modelLoaded ? const Color(0xFF3FB950) : const Color(0xFFF85149),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // 4. Acoustic & Quality Telemetry Card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'ACOUSTIC & SIGNAL TELEMETRY',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: Colors.white.withValues(alpha: 0.5),
                    letterSpacing: 0.8,
                  ),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: _buildTelemetryRow(
                        icon: Icons.graphic_eq,
                        label: 'Quality Status',
                        value: r.audioQualityStatus.toUpperCase(),
                        valueColor: r.audioQualityStatus == 'acceptable'
                            ? const Color(0xFF3FB950)
                            : const Color(0xFFD29922),
                      ),
                    ),
                    Expanded(
                      child: _buildTelemetryRow(
                        icon: Icons.timer_outlined,
                        label: 'Audio Duration',
                        value: r.durationSeconds != null ? '${r.durationSeconds!.toStringAsFixed(1)}s' : 'N/A',
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: _buildTelemetryRow(
                        icon: Icons.volume_up_outlined,
                        label: 'RMS Energy',
                        value: r.rms != null ? r.rms!.toStringAsFixed(4) : 'N/A',
                      ),
                    ),
                    Expanded(
                      child: _buildTelemetryRow(
                        icon: Icons.speed,
                        label: 'Signal-to-Noise',
                        value: r.snrDb != null ? '${r.snrDb!.toStringAsFixed(1)} dB' : 'N/A',
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: _buildTelemetryRow(
                        icon: Icons.record_voice_over_outlined,
                        label: 'Voiced Ratio',
                        value: r.voicedRatio != null ? '${(r.voicedRatio! * 100).toStringAsFixed(1)}%' : 'N/A',
                      ),
                    ),
                    Expanded(
                      child: _buildTelemetryRow(
                        icon: Icons.memory,
                        label: 'Model Backend',
                        value: r.modelBackend.isNotEmpty ? r.modelBackend : 'wav2vec2',
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // 5. Model Backend & Score Type Card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'DETECTION PIPELINE & AUDIT METADATA',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: Colors.white.withValues(alpha: 0.5),
                    letterSpacing: 0.8,
                  ),
                ),
                const SizedBox(height: 12),
                _buildMetadataRow('Model Backend', r.modelBackend.isNotEmpty ? r.modelBackend : 'wav2vec2'),
                _buildMetadataRow('Score Type', r.scoreType.isNotEmpty ? r.scoreType : 'uncalibrated_softmax_score'),
                _buildMetadataRow('Model Version', r.modelVersion.isNotEmpty ? r.modelVersion : 'voiceguard-v1'),
                _buildMetadataRow('Threshold Version', r.thresholdVersion.isNotEmpty ? r.thresholdVersion : 'threshold-v2'),
                _buildMetadataRow('Engine Status', r.modelLoaded ? 'Model Loaded & Active' : 'Model Not Loaded'),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // 6. Security Reason Codes
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'SECURITY REASON CODES',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: Colors.white.withValues(alpha: 0.5),
                    letterSpacing: 0.8,
                  ),
                ),
                const SizedBox(height: 12),
                if (r.reasonCodes.isNotEmpty)
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: r.reasonCodes.map((rc) {
                      return Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                        decoration: BoxDecoration(
                          color: const Color(0xFF58A6FF).withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(
                            color: const Color(0xFF58A6FF).withValues(alpha: 0.3),
                          ),
                        ),
                        child: Text(
                          rc,
                          style: const TextStyle(
                            fontSize: 12,
                            color: Color(0xFF58A6FF),
                            fontFamily: 'monospace',
                          ),
                        ),
                      );
                    }).toList(),
                  )
                else
                  Text(
                    'No anomaly reason codes triggered (Standard Audio Profile)',
                    style: TextStyle(
                      fontSize: 13,
                      color: Colors.white.withValues(alpha: 0.6),
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // 7. Step-Up Verification Recommendation Card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(
                color: scoreColor.withValues(alpha: 0.35),
                width: 1.5,
              ),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(
                  isLowRisk
                      ? Icons.check_circle_outline
                      : (r.riskLevel == 'high'
                          ? Icons.block_outlined
                          : (r.riskLevel == 'medium'
                              ? Icons.warning_amber_rounded
                              : Icons.help_outline)),
                  color: scoreColor,
                  size: 24,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'STEP-UP VERIFICATION RECOMMENDATION',
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          color: scoreColor,
                          letterSpacing: 0.8,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        isLowRisk
                            ? 'Voice acoustic profile demonstrates high bonafide confidence. No secondary step-up verification required.'
                            : (r.riskLevel == 'high' || r.decision == 'action_held'
                                ? 'High deepfake probability detected. Action held. Mandatory out-of-band identity verification (e.g., video KYC or physical authorization) required before proceeding.'
                                : (r.riskLevel == 'medium' || r.decision == 'verification_required'
                                    ? 'Elevated risk detected. Secondary step-up verification required (e.g., SMS OTP or security challenge questions) before proceeding.'
                                    : 'Insufficient evidence to confirm voice authenticity. Standard identity verification required.')),
                        style: const TextStyle(
                          fontSize: 13,
                          color: Colors.white70,
                          height: 1.4,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // 8. Sensitive Transaction Card (Simulation)
          TransactionCard(
            riskLevel: r.riskLevel,
            decision: r.decision,
            reasonCodes: r.reasonCodes,
            evidenceWindows: 1,
            canShowLowRisk: r.canShowLowRisk,
          ),
          const SizedBox(height: 16),

          // 9. Disclaimer
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(
                color: Colors.white.withValues(alpha: 0.08),
              ),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.info_outline, size: 16, color: Colors.white.withValues(alpha: 0.4)),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Disclaimer: VoiceGuard provides machine-learning probabilistic evidence for deepfake anomaly detection. It does not authenticate caller identity or guarantee the absence of synthetic speech.',
                    style: TextStyle(
                      fontSize: 11,
                      color: Colors.white.withValues(alpha: 0.45),
                      height: 1.4,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  Widget _buildMetricTile({
    required String label,
    required String value,
    required Color color,
    String? subValue,
  }) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 9,
              fontWeight: FontWeight.w600,
              color: Colors.white.withValues(alpha: 0.45),
              letterSpacing: 0.5,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
          if (subValue != null) ...[
            const SizedBox(height: 2),
            Text(
              subValue,
              style: TextStyle(
                fontSize: 10,
                color: Colors.white.withValues(alpha: 0.35),
                fontFamily: 'monospace',
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildTelemetryRow({
    required IconData icon,
    required String label,
    required String value,
    Color? valueColor,
  }) {
    return Row(
      children: [
        Icon(icon, size: 16, color: Colors.white.withValues(alpha: 0.4)),
        const SizedBox(width: 8),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: TextStyle(
                  fontSize: 10,
                  color: Colors.white.withValues(alpha: 0.4),
                ),
              ),
              const SizedBox(height: 2),
              Text(
                value,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: valueColor ?? Colors.white,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildMetadataRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
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
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: Colors.white,
                fontFamily: 'monospace',
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Result view — full live call UI + 4-second sliding windows (Legacy / Test)
  // ---------------------------------------------------------------------------

  Widget _buildSlidingResultView(FileSlidingAnalysisResult r) {
    final scoreColor = r.canShowLowRisk
        ? const Color(0xFF3FB950)
        : _riskColor(r.maxRisk);
    final scorePct = r.maximumSpoofScore != null
        ? (r.maximumSpoofScore! * 100).toStringAsFixed(0)
        : (r.averageSpoofScore != null
            ? (r.averageSpoofScore! * 100).toStringAsFixed(0)
            : 'N/A');
    final emaPct = r.emaScore != null
        ? (r.emaScore! * 100).toStringAsFixed(0)
        : 'N/A';

    return SingleChildScrollView(
      child: Column(
        children: [
          // 1. File Status Bar
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: scoreColor.withValues(alpha: 0.3),
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: scoreColor,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    r.canShowLowRisk
                        ? 'ANALYSIS COMPLETE (LOW RISK)'
                        : (r.maxRisk == 'high'
                            ? 'HIGH RISK DETECTED'
                            : (r.maxRisk == 'medium'
                                ? 'VERIFICATION REQUIRED'
                                : 'INSUFFICIENT EVIDENCE')),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: scoreColor,
                      fontWeight: FontWeight.w700,
                      fontSize: 11,
                      letterSpacing: 0.5,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 150),
                  child: Text(
                    '${r.totalDurationSeconds.toStringAsFixed(1)}s (${r.windowCount} Windows)',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    textAlign: TextAlign.end,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                      fontFamily: 'monospace',
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // Fallback notice banner if single-file predict was used
          if (r.isFallbackPredict) ...[
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: const Color(0xFFD29922).withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                  color: const Color(0xFFD29922).withValues(alpha: 0.4),
                ),
              ),
              child: const Row(
                children: [
                  Icon(Icons.info_outline, color: Color(0xFFD29922), size: 18),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Sliding-window details unavailable; showing single-file analysis.',
                      style: TextStyle(
                        color: Color(0xFFD29922),
                        fontWeight: FontWeight.w600,
                        fontSize: 12,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
          ],

          // 2. Audio Level Visualization
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'AUDIO INPUT / WAVEFORM',
                      style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.w600,
                        color: Colors.white.withValues(alpha: 0.4),
                        letterSpacing: 1,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Flexible(
                      child: Text(
                        _activeFilename.isNotEmpty ? _activeFilename : 'Audio Clip',
                        overflow: TextOverflow.ellipsis,
                        textAlign: TextAlign.end,
                        style: TextStyle(
                          fontSize: 10,
                          fontFamily: 'monospace',
                          color: Colors.white.withValues(alpha: 0.5),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                _buildAudioLevelBars(active: false),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // 3. Voice Authenticity Risk Card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 20),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: scoreColor.withValues(alpha: 0.3),
                width: 1.5,
              ),
            ),
            child: Column(
              children: [
                Text(
                  'VOICE AUTHENTICITY RISK',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: Colors.white.withValues(alpha: 0.4),
                    letterSpacing: 1.5,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  scorePct == 'N/A' ? 'N/A' : '$scorePct%',
                  style: TextStyle(
                    fontSize: 48,
                    fontWeight: FontWeight.w900,
                    color: scoreColor,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  r.canShowLowRisk
                      ? 'Low Risk (Authentic)'
                      : _riskCategory(r.maxRisk),
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: scoreColor,
                  ),
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      'File Avg (EMA): ${emaPct == "N/A" ? "N/A" : "$emaPct%"}',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.white.withValues(alpha: 0.5),
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    if (r.persistenceTriggered) ...[
                      const SizedBox(width: 12),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: const Color(0xFFF85149).withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: const Text(
                          'PERSISTENT THREAT',
                          style: TextStyle(
                            fontSize: 9,
                            fontWeight: FontWeight.w700,
                            color: Color(0xFFF85149),
                            letterSpacing: 0.5,
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // 4. Call/File-level Risk Indicator Widget
          RiskIndicator(
            riskLevel: r.maxRisk,
            action: r.action,
            decision: r.decision,
            canShowLowRisk: r.canShowLowRisk,
          ),
          const SizedBox(height: 16),

          // 5. 4-SECOND SLIDING WINDOWS SECTION
          _buildSlidingWindowsSection(),
          const SizedBox(height: 16),

          // 6. Mock Transaction Card
          TransactionCard(
            riskLevel: r.maxRisk,
            decision: r.decision,
            reasonCodes: r.reasonCodes,
            evidenceWindows: r.windows.length,
            canShowLowRisk: r.canShowLowRisk,
          ),
          const SizedBox(height: 16),

          // 7. Live Dev Diagnostics Panel
          _buildDiagnosticsPanel(),
          const SizedBox(height: 16),

          // 8. Summary Card
          _buildSummaryCard(),
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // 4-Second Sliding Windows Section
  // ---------------------------------------------------------------------------

  Widget _buildSlidingWindowsSection() {
    final windows = _slidingResult?.windows ?? [];
    if (windows.isEmpty) return const SizedBox.shrink();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Row(
                children: [
                  Icon(Icons.view_timeline_outlined, size: 16, color: Color(0xFF58A6FF)),
                  SizedBox(width: 8),
                  Text(
                    '4-SECOND SLIDING WINDOWS',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                      letterSpacing: 0.8,
                    ),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: const Color(0xFF58A6FF).withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: const Color(0xFF58A6FF).withValues(alpha: 0.3),
                    width: 1,
                  ),
                ),
                child: Text(
                  '${windows.length} Windows (2s Stride)',
                  style: const TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF58A6FF),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            'Tap any window below to inspect detailed acoustic and inference telemetry.',
            style: TextStyle(
              fontSize: 11,
              color: Colors.white.withValues(alpha: 0.5),
            ),
          ),
          const SizedBox(height: 14),
          ...windows.map((w) {
            final isExpanded = _expandedWindowIndex == w.index;
            final dotColor = w.canShowLowRisk
                ? const Color(0xFF3FB950)
                : (w.riskLevel == 'high'
                    ? const Color(0xFFF85149)
                    : (w.riskLevel == 'medium'
                        ? const Color(0xFFD29922)
                        : const Color(0xFF8B949E)));
            final scorePct = w.spoofScore != null
                ? (w.spoofScore! * 100).toStringAsFixed(0)
                : 'N/A';

            return Container(
              margin: const EdgeInsets.only(bottom: 8),
              decoration: BoxDecoration(
                color: const Color(0xFF0D1117),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                  color: dotColor.withValues(alpha: isExpanded ? 0.6 : 0.25),
                  width: isExpanded ? 1.5 : 1,
                ),
              ),
              child: Column(
                children: [
                  InkWell(
                    borderRadius: BorderRadius.circular(10),
                    onTap: () {
                      setState(() {
                        _expandedWindowIndex = isExpanded ? null : w.index;
                      });
                    },
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                      child: Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: Colors.white.withValues(alpha: 0.06),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              w.timeRangeFormatted,
                              style: const TextStyle(
                                fontSize: 11,
                                fontFamily: 'monospace',
                                fontWeight: FontWeight.bold,
                                color: Colors.white70,
                              ),
                            ),
                          ),
                          const SizedBox(width: 10),
                          Container(
                            width: 8,
                            height: 8,
                            decoration: BoxDecoration(
                              color: dotColor,
                              shape: BoxShape.circle,
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              'W${(w.index + 1).toString().padLeft(2, '0')}: ${w.windowComplete ? _riskCategory(w.riskLevel) : "Incomplete (<4s)"}',
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w600,
                                color: dotColor,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            scorePct == 'N/A' ? 'N/A' : '$scorePct%',
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w900,
                              fontFamily: 'monospace',
                              color: dotColor,
                            ),
                          ),
                          const SizedBox(width: 6),
                          Icon(
                            isExpanded ? Icons.keyboard_arrow_up : Icons.keyboard_arrow_down,
                            size: 18,
                            color: Colors.white38,
                          ),
                        ],
                      ),
                    ),
                  ),
                  if (isExpanded) ...[
                    const Divider(height: 1, color: Colors.white12),
                    Padding(
                      padding: const EdgeInsets.all(12),
                      child: Column(
                        children: [
                          _telemetryRow('Window Index', 'WINDOW ${(w.index + 1).toString().padLeft(2, '0')}'),
                          _telemetryRow('Time Range', w.timeRangeFormatted),
                          _telemetryRow('Window Duration', '${w.durationSeconds.toStringAsFixed(1)}s (${w.windowComplete ? "Complete" : "Incomplete"})'),
                          _telemetryRow('Window Completeness', w.windowComplete ? 'COMPLETE (4.0s)' : 'INCOMPLETE (< 4.0s)'),
                          _telemetryRow('Decision', w.decision.toUpperCase()),
                          _telemetryRow('Action', w.action.toUpperCase()),
                          _telemetryRow('Label', w.label.toUpperCase()),
                          _telemetryRow('Spoof Score', w.spoofScore != null ? '${(w.spoofScore! * 100).toStringAsFixed(1)}%' : 'N/A'),
                          _telemetryRow('Confidence', w.confidence != null ? '${(w.confidence! * 100).toStringAsFixed(1)}%' : 'N/A'),
                          _telemetryRow('Risk Level', w.riskLevel.toUpperCase()),
                          _telemetryRow('Speech Detected', w.speechDetected ? 'YES' : 'NO'),
                          _telemetryRow('Quality Status', w.qualityStatus.toUpperCase(), isSuccess: w.qualityStatus == 'acceptable'),
                          _telemetryRow('Energy RMS', w.rms != null ? w.rms!.toStringAsFixed(4) : 'N/A'),
                          _telemetryRow('Signal-to-Noise', w.snrDb != null ? '${w.snrDb!.toStringAsFixed(1)} dB' : 'N/A'),
                          _telemetryRow('Voiced Speech Ratio', w.voicedRatio != null ? '${(w.voicedRatio! * 100).toStringAsFixed(1)}%' : 'N/A'),
                          _telemetryRow('Model Backend', w.modelBackend.isNotEmpty ? w.modelBackend : 'N/A'),
                          _telemetryRow('Score Type', w.scoreType.isNotEmpty ? w.scoreType : 'N/A'),
                          _telemetryRow('Model Version', w.modelVersion.isNotEmpty ? w.modelVersion : 'N/A'),
                          _telemetryRow('Threshold Version', w.thresholdVersion.isNotEmpty ? w.thresholdVersion : 'N/A'),
                          _telemetryRow('Reason Codes', w.reasonCodes.isEmpty ? 'none' : w.reasonCodes.join(', ')),
                        ],
                      ),
                    ),
                  ],
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _telemetryRow(String label, String value, {bool isSuccess = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 140,
            child: Text(
              label,
              style: TextStyle(
                fontSize: 11,
                color: Colors.white.withValues(alpha: 0.45),
                fontFamily: 'monospace',
              ),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              value,
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: isSuccess ? const Color(0xFF3FB950) : Colors.white.withValues(alpha: 0.85),
                fontFamily: 'monospace',
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Audio level bars
  // ---------------------------------------------------------------------------

  Widget _buildAudioLevelBars({bool active = true}) {
    return SizedBox(
      height: 32,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: List.generate(24, (i) {
          final wave = sin((i / 23) * pi);
          final restingHeight = 4.0 + wave * 4.0;
          final seed = (i * 7 + (_pulseCtrl.value * 10).toInt()) % 10;
          final variance = 0.4 + (seed / 10.0) * 0.6;
          final level = active ? (_audioLevel > 0.05 ? _audioLevel : (0.2 + wave * 0.3)) : 0.1;
          final barHeight = (level * variance * 32).clamp(restingHeight, 32.0);
          final barColor = level > 0.2
              ? const Color(0xFF58A6FF)
              : Colors.white.withValues(alpha: 0.2);

          return Container(
            width: 5,
            height: barHeight,
            decoration: BoxDecoration(
              color: barColor.withValues(alpha: 0.3 + level * 0.7),
              borderRadius: BorderRadius.circular(3),
            ),
          );
        }),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Diagnostics Panel (Dev Mode)
  // ---------------------------------------------------------------------------

  Widget _buildDiagnosticsPanel() {
    final r = _slidingResult!;
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.1)),
      ),
      child: ExpansionTile(
        initiallyExpanded: false,
        shape: const Border(),
        collapsedShape: const Border(),
        title: Row(
          children: [
            Icon(
              Icons.bug_report_outlined,
              size: 16,
              color: r.canShowLowRisk
                  ? const Color(0xFF3FB950)
                  : const Color(0xFFE3B341),
            ),
            const SizedBox(width: 8),
            const Text(
              'SLIDING WINDOW DIAGNOSTICS (DEV MODE)',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w700,
                color: Colors.white70,
                letterSpacing: 0.5,
              ),
            ),
          ],
        ),
        childrenPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        children: [
          _diagnosticRow('Window Stride Contract', '4000ms Full Window / 2000ms Stride'),
          _diagnosticRow('Audio Format Config', '$kLiveSampleRate Hz / 1ch / 16-bit PCM'),
          _diagnosticRow('Total Duration', '${r.totalDurationSeconds.toStringAsFixed(1)}s'),
          _diagnosticRow('Sliding Windows Count', '${r.windowCount} windows'),
          _diagnosticRow('Complete Windows Count', '${r.completeWindowCount} complete'),
          _diagnosticRow('High-Risk Windows', '${r.highRiskWindows}'),
          _diagnosticRow('Backend Connection', _backendConnected ? 'ONLINE (200 OK)' : 'OFFLINE / UNREACHABLE'),
          _diagnosticRow('Model Loaded Flag', r.modelLoaded ? 'TRUE (READY)' : 'FALSE (FAIL-CLOSED)'),
          _diagnosticRow('Model Backend', r.modelBackend.isNotEmpty ? r.modelBackend : 'wav2vec2'),
          _diagnosticRow('Score Type', r.scoreType.isNotEmpty ? r.scoreType : 'prob_fake'),
          _diagnosticRow('Threshold Version', r.thresholdVersion.isNotEmpty ? r.thresholdVersion : 'threshold-v2'),
          _diagnosticRow('canShowLowRisk Gate', '${r.canShowLowRisk}'),
          _diagnosticRow('Effective Decision', r.decision),
          _diagnosticRow('Effective Action', r.action),
          _diagnosticRow('Consecutive High Windows', '${r.consecutiveHighCount}'),
          _diagnosticRow(
            'Hold Status',
            r.persistenceTriggered
                ? 'ACTION HELD (Persistent Threat)'
                : 'NORMAL',
          ),
          _diagnosticRow('Active Reason Codes', r.reasonCodes.isEmpty ? 'none' : r.reasonCodes.join(', ')),
          const SizedBox(height: 8),
        ],
      ),
    );
  }

  Widget _diagnosticRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 150,
            child: Text(
              label,
              style: TextStyle(
                fontSize: 11,
                color: Colors.white.withValues(alpha: 0.5),
                fontFamily: 'monospace',
              ),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              value,
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: Colors.white.withValues(alpha: 0.85),
                fontFamily: 'monospace',
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Summary Card
  // ---------------------------------------------------------------------------

  Widget _buildSummaryCard() {
    final r = _slidingResult!;
    final maxColor = r.canShowLowRisk ? const Color(0xFF3FB950) : _riskColor(r.maxRisk);
    final recommendation = switch (r.maxRisk) {
      'high' => 'High synthetic anomaly detected across sliding windows. Verify caller independently before authorizing sensitive actions.',
      'medium' => 'Moderate synthetic anomaly detected. Secondary step-up verification is recommended.',
      'low' when r.canShowLowRisk => 'Acoustic parameters verified natural with low anomaly score across all 4s windows.',
      _ => 'Insufficient acoustic evidence to confirm authenticity. Exercise caution.',
    };

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: maxColor.withValues(alpha: 0.3), width: 1.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.summarize_rounded, size: 20, color: Color(0xFF58A6FF)),
              SizedBox(width: 8),
              Text(
                'AUDIO SCREENING SUMMARY',
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                  letterSpacing: 0.8,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _summaryRow('File Name', _activeFilename.isNotEmpty ? _activeFilename : 'Audio Clip'),
          const Divider(color: Colors.white12, height: 20),
          _summaryRow('Audio Duration', '${r.totalDurationSeconds.toStringAsFixed(1)}s'),
          const Divider(color: Colors.white12, height: 20),
          _summaryRow('Complete Windows', '${r.completeWindowCount} / ${r.windowCount} complete'),
          const Divider(color: Colors.white12, height: 20),
          _summaryRow('Window Length / Stride', '${r.windowLengthSeconds.toStringAsFixed(1)}s / ${r.strideSeconds.toStringAsFixed(1)}s'),
          const Divider(color: Colors.white12, height: 20),
          _summaryRow('Overall Decision', r.decision.toUpperCase(), valueColor: maxColor),
          const Divider(color: Colors.white12, height: 20),
          _summaryRow('Overall Action', r.action.toUpperCase(), valueColor: maxColor),
          const Divider(color: Colors.white12, height: 20),
          _summaryRow(
            'Maximum Spoof Score',
            r.maximumSpoofScore != null ? '${(r.maximumSpoofScore! * 100).toStringAsFixed(1)}%' : 'N/A',
          ),
          const Divider(color: Colors.white12, height: 20),
          _summaryRow(
            'Average / EMA Score',
            '${r.averageSpoofScore != null ? "${(r.averageSpoofScore! * 100).toStringAsFixed(1)}%" : "N/A"} / ${r.emaScore != null ? "${(r.emaScore! * 100).toStringAsFixed(1)}%" : "N/A"}',
          ),
          const Divider(color: Colors.white12, height: 20),
          _summaryRow(
            'High-Risk Windows',
            '${r.highRiskWindows}',
            valueColor: r.highRiskWindows > 0 ? const Color(0xFFF85149) : null,
          ),
          const Divider(color: Colors.white12, height: 20),
          _summaryRow(
            'Persistence Triggered',
            r.persistenceTriggered ? 'YES (ACTION HELD)' : 'NO',
            valueColor: r.persistenceTriggered ? const Color(0xFFF85149) : null,
          ),
          const SizedBox(height: 16),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.04),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              recommendation,
              style: TextStyle(
                fontSize: 12,
                color: Colors.white.withValues(alpha: 0.7),
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _summaryRow(String label, String value, {Color? valueColor}) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: TextStyle(
            color: Colors.white.withValues(alpha: 0.5),
            fontSize: 13,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            value,
            textAlign: TextAlign.end,
            overflow: TextOverflow.ellipsis,
            style: TextStyle(
              color: valueColor ?? Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 13,
              fontFamily: 'monospace',
            ),
          ),
        ),
      ],
    );
  }

  // ---------------------------------------------------------------------------
  // Error view
  // ---------------------------------------------------------------------------

  Widget _buildErrorView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline_rounded, size: 64, color: Color(0xFFF85149)),
            const SizedBox(height: 20),
            const Text(
              'Analysis Failed',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Color(0xFFF85149),
              ),
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFF161B22),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.white12),
              ),
              child: Text(
                _errorMessage,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.7),
                  height: 1.5,
                  fontSize: 13,
                ),
              ),
            ),
            const SizedBox(height: 32),
            ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF58A6FF),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              onPressed: _reset,
              icon: const Icon(Icons.refresh),
              label: const Text('Try Again'),
            ),
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  Color _riskColor(String level) {
    return switch (level) {
      'high' => const Color(0xFFF85149),
      'medium' => const Color(0xFFD29922),
      'unknown' => const Color(0xFF8B949E),
      _ => const Color(0xFF3FB950),
    };
  }

  String _riskCategory(String level) {
    return switch (level) {
      'high' => 'Critical Risk',
      'medium' => 'Medium Risk',
      'unknown' => 'Cannot Assess (Poor Audio)',
      _ => 'Low Risk',
    };
  }

  String _formatDuration(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }
}

// ---------------------------------------------------------------------------
// Settings Dialog — configurable server URL
// ---------------------------------------------------------------------------

class SettingsDialog extends StatefulWidget {
  const SettingsDialog({super.key});

  @override
  State<SettingsDialog> createState() => _SettingsDialogState();
}

class _SettingsDialogState extends State<SettingsDialog> {
  late TextEditingController _urlController;
  _ConnectionStatus _status = _ConnectionStatus.idle;

  @override
  void initState() {
    super.initState();
    _urlController = TextEditingController(text: BackendConfig.baseUrl);
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _testConnection() async {
    setState(() => _status = _ConnectionStatus.testing);
    final ok = await BackendConfig.setBaseUrl(_urlController.text.trim());
    if (!mounted) return;
    setState(() {
      _status = ok ? _ConnectionStatus.success : _ConnectionStatus.failure;
    });
  }

  Future<void> _save() async {
    final url = _urlController.text.trim();
    if (url.isEmpty) return;

    setState(() => _status = _ConnectionStatus.testing);
    final ok = await BackendConfig.setBaseUrl(url);
    if (!mounted) return;

    if (ok) {
      Navigator.of(context).pop();
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Server URL saved successfully.'),
          backgroundColor: Color(0xFF2E7D32),
        ),
      );
    } else {
      setState(() => _status = _ConnectionStatus.failure);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Row(
        children: [
          Icon(Icons.settings, size: 22, color: Color(0xFF1565C0)),
          SizedBox(width: 8),
          Expanded(child: Text('Server Settings')),
        ],
      ),
      content: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 400),
        child: SizedBox(
          width: double.maxFinite,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Backend URL',
                style: TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
              ),
              const SizedBox(height: 8),
              TextField(
                controller: _urlController,
                decoration: InputDecoration(
                  hintText: 'https://your-server:8443',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                  isDense: true,
                ),
                keyboardType: TextInputType.url,
              ),
              const SizedBox(height: 12),

              // Test Connection button
              SizedBox(
                width: double.infinity,
                child: OutlinedButton.icon(
                  onPressed: _status == _ConnectionStatus.testing
                      ? null
                      : _testConnection,
                  icon: _status == _ConnectionStatus.testing
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.wifi_find, size: 18),
                  label: const Text('Test Connection'),
                ),
              ),
              const SizedBox(height: 8),

              // Status chip
              if (_status == _ConnectionStatus.success)
                const Chip(
                  avatar: Icon(Icons.check_circle, color: Color(0xFF2E7D32), size: 18),
                  label: Text('Connected — model loaded'),
                  backgroundColor: Color(0xFFE8F5E9),
                ),
              if (_status == _ConnectionStatus.failure)
                const Chip(
                  avatar: Icon(Icons.error, color: Color(0xFFC62828), size: 18),
                  label: Text('Connection failed'),
                  backgroundColor: Color(0xFFFFEBEE),
                ),

              const SizedBox(height: 8),
              Text(
                'Default: $kDefaultBackendUrl',
                style: const TextStyle(fontSize: 11, color: Colors.black45),
              ),
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: _status == _ConnectionStatus.testing ? null : _save,
          child: const Text('Save'),
        ),
      ],
    );
  }
}

enum _ConnectionStatus { idle, testing, success, failure }