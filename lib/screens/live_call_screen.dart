/// LiveCallScreen — continuous live voice analysis with rolling risk assessment.
///
/// Architecture:
///   Microphone → PCM stream → rolling buffer → WAV chunks → /live/analyze
///   → detection score → EMA + persistence rule → call-level risk → UI
///
/// State machine:
///   IDLE → STARTING → ACTIVE → (ANALYZING ↔ ACTIVE) → ENDING → SUMMARY
///
/// Risk aggregation (two-layer):
///   LAYER 1: EMA = 0.3 * current_score + 0.7 * previous_EMA
///   LAYER 2: IF raw score >= 0.85 for 2 consecutive windows → HIGH/BLOCK
///
/// MVP limitation: This captures microphone audio, not cellular call audio.
/// Direct interception of arbitrary cellular call audio is outside the mobile
/// MVP and would require integration with supported VoIP/telephony infrastructure.
library;
import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:record/record.dart';

import '../api_client.dart';
import '../config.dart';
import '../models/live_analysis_result.dart';
import '../widgets/risk_indicator.dart';
import '../widgets/transaction_card.dart';

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------
enum _CallState { idle, starting, active, ending, summary, error }

// ---------------------------------------------------------------------------
// Risk history entry
// ---------------------------------------------------------------------------
class _RiskEntry {
  final Duration timestamp;
  final double detectionScore;
  final String riskLevel;
  final String action;

  const _RiskEntry({
    required this.timestamp,
    required this.detectionScore,
    required this.riskLevel,
    required this.action,
  });
}

// ---------------------------------------------------------------------------
// LiveCallScreen
// ---------------------------------------------------------------------------
class LiveCallScreen extends StatefulWidget {
  const LiveCallScreen({super.key});

  @override
  State<LiveCallScreen> createState() => _LiveCallScreenState();
}

class _LiveCallScreenState extends State<LiveCallScreen> {
  // -- state --
  _CallState _callState = _CallState.idle;
  String _errorMessage = '';

  // -- audio --
  final AudioRecorder _recorder = AudioRecorder();
  StreamSubscription<Uint8List>? _audioSub;
  final List<int> _pcmBuffer = []; // rolling PCM buffer

  // -- timers --
  Timer? _analysisTimer;
  Timer? _callTimer;
  Timer? _healthTimer;
  Duration _callDuration = Duration.zero;

  // -- backend --
  bool _backendConnected = false;
  bool _isAnalyzing = false; // prevents concurrent analysis requests

  // -- risk aggregation --
  double _emaScore = 0.0;
  double _currentScore = 0.0;
  String _callLevelRisk = 'low';
  String _callLevelAction = 'allow';
  int _consecutiveHighCount = 0; // for persistence rule
  bool _persistenceTriggered = false;

  // -- history --
  final List<_RiskEntry> _riskHistory = [];

  // -- summary --
  String _maxRisk = 'low';
  int _highRiskWindows = 0;

  // -- audio level visualization --
  double _audioLevel = 0.0; // 0.0–1.0 normalized level

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------
  @override
  void initState() {
    super.initState();
    _checkHealth();
  }

  @override
  void dispose() {
    _stopEverything();
    _recorder.dispose();
    super.dispose();
  }

  // -------------------------------------------------------------------------
  // Health check
  // -------------------------------------------------------------------------
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

  // -------------------------------------------------------------------------
  // Start call
  // -------------------------------------------------------------------------
  Future<void> _startCall() async {
    setState(() => _callState = _CallState.starting);

    // 1. Check microphone permission
    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) {
      setState(() {
        _callState = _CallState.error;
        _errorMessage = 'Microphone permission is required for live analysis.';
      });
      return;
    }

    // 2. Check backend connectivity
    await _checkHealth();
    if (!_backendConnected) {
      setState(() {
        _callState = _CallState.error;
        _errorMessage =
            'AI analysis engine is offline.\nSensitive actions are blocked until verification is restored.';
      });
      return;
    }

    // 3. Reset state
    _pcmBuffer.clear();
    _riskHistory.clear();
    _emaScore = 0.0;
    _currentScore = 0.0;
    _callLevelRisk = 'low';
    _callLevelAction = 'allow';
    _consecutiveHighCount = 0;
    _persistenceTriggered = false;
    _callDuration = Duration.zero;
    _maxRisk = 'low';
    _highRiskWindows = 0;
    _isAnalyzing = false;

    // 4. Start PCM audio stream
    try {
      const config = RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: kLiveSampleRate,
        numChannels: 1,
        // Enable noise suppression for cleaner analysis
        noiseSuppress: true,
      );
      final stream = await _recorder.startStream(config);

      _audioSub = stream.listen(
        _onAudioData,
        onError: (e) {
          if (mounted) {
            setState(() {
              _callState = _CallState.error;
              _errorMessage = 'Audio capture error: $e';
            });
          }
        },
      );
    } catch (e) {
      setState(() {
        _callState = _CallState.error;
        _errorMessage = 'Failed to start audio capture: $e';
      });
      return;
    }

    // 5. Start timers
    _callTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) {
        setState(() => _callDuration += const Duration(seconds: 1));
      }
    });

    _analysisTimer = Timer.periodic(
      Duration(seconds: kLiveAnalysisIntervalSec),
      (_) => _analyzeCurrentBuffer(),
    );

    _startHealthChecks();

    setState(() => _callState = _CallState.active);
  }

  // -------------------------------------------------------------------------
  // Audio data handler
  // -------------------------------------------------------------------------
  void _onAudioData(Uint8List data) {
    _pcmBuffer.addAll(data);

    // Keep only the last kLiveChunkDurationSec * sampleRate * 2 bytes
    // (16-bit PCM = 2 bytes per sample, mono)
    final maxBytes = kLiveChunkDurationSec * kLiveSampleRate * 2;
    if (_pcmBuffer.length > maxBytes * 2) {
      _pcmBuffer.removeRange(0, _pcmBuffer.length - maxBytes);
    }

    // Compute audio level for visualization from last chunk of data
    _updateAudioLevel(data);
  }

  void _updateAudioLevel(Uint8List data) {
    if (data.length < 2) return;

    // Read 16-bit PCM samples and compute RMS
    double sumSquares = 0;
    int sampleCount = 0;
    for (int i = 0; i < data.length - 1; i += 2) {
      // Little-endian 16-bit signed
      int sample = data[i] | (data[i + 1] << 8);
      if (sample >= 0x8000) sample -= 0x10000;
      final normalized = sample / 32768.0;
      sumSquares += normalized * normalized;
      sampleCount++;
    }

    if (sampleCount > 0) {
      final rms = sqrt(sumSquares / sampleCount);
      // Normalize to 0–1 with some headroom
      final level = (rms * 5.0).clamp(0.0, 1.0);
      if (mounted) setState(() => _audioLevel = level);
    }
  }

  // -------------------------------------------------------------------------
  // Analyze current buffer
  // -------------------------------------------------------------------------
  Future<void> _analyzeCurrentBuffer() async {
    // Prevent concurrent analysis requests
    if (_isAnalyzing) return;
    if (_callState != _CallState.active) return;

    // Need at least 1 second of audio (16000 samples * 2 bytes)
    final minBytes = kLiveSampleRate * 2;
    if (_pcmBuffer.length < minBytes) return;

    _isAnalyzing = true;

    try {
      // Take the last kLiveChunkDurationSec of buffer
      final targetBytes = kLiveChunkDurationSec * kLiveSampleRate * 2;
      final startIdx =
          _pcmBuffer.length > targetBytes ? _pcmBuffer.length - targetBytes : 0;
      final chunk = Uint8List.fromList(_pcmBuffer.sublist(startIdx));

      // Wrap PCM data as a WAV file
      final wavBytes = _wrapPcmAsWav(chunk, kLiveSampleRate, 1, 16);

      // Send to backend
      final result = await VoiceGuardApiClient.analyzeLiveChunk(wavBytes);

      if (!mounted || _callState != _CallState.active) return;

      _updateRisk(result);
    } on BackendUnreachableException {
      if (mounted) setState(() => _backendConnected = false);
    } on BackendErrorException {
      // Log but continue — don't crash the call for a single failed chunk
    } catch (_) {
      // Ignore transient errors and continue
    } finally {
      _isAnalyzing = false;
    }
  }

  // -------------------------------------------------------------------------
  // Risk aggregation (EMA + persistence rule)
  // -------------------------------------------------------------------------
  /// Two-layer risk aggregation:
  ///
  /// LAYER 1: Exponential Moving Average (EMA)
  ///   EMA = α * current_score + (1 - α) * previous_EMA
  ///   α = 0.3 → new evidence has 30% weight, history has 70% weight
  ///   Prevents one noisy chunk from immediately triggering HIGH.
  ///
  /// LAYER 2: Persistence rule for fast security response
  ///   IF raw detection_score >= 0.85 for 2 consecutive windows:
  ///       call_risk = HIGH, action = BLOCK
  ///   Ensures repeated strong evidence triggers HIGH within ~4 seconds.
  ///
  /// Combined logic:
  ///   IF persistence rule triggered → HIGH / BLOCK
  ///   ELSE use EMA thresholds:
  ///       < 0.40      → LOW / ALLOW
  ///       0.40–0.849  → MEDIUM / VERIFY
  ///       >= 0.85     → HIGH / BLOCK
  void _updateRisk(LiveAnalysisResult result) {
    const alpha = 0.3;
    const highThreshold = 0.85;
    const mediumThreshold = 0.40;

    _currentScore = result.detectionScore;

    // Layer 1: Update EMA
    if (_riskHistory.isEmpty) {
      _emaScore = _currentScore;
    } else {
      _emaScore = alpha * _currentScore + (1 - alpha) * _emaScore;
    }

    // Layer 2: Persistence rule
    if (_currentScore >= highThreshold) {
      _consecutiveHighCount++;
    } else {
      _consecutiveHighCount = 0;
    }
    _persistenceTriggered = _consecutiveHighCount >= 2;

    // Determine call-level risk
    String risk;
    String action;
    if (_persistenceTriggered) {
      // Persistence rule overrides EMA
      risk = 'high';
      action = 'block';
    } else if (_emaScore >= highThreshold) {
      risk = 'high';
      action = 'block';
    } else if (_emaScore >= mediumThreshold) {
      risk = 'medium';
      action = 'verify';
    } else {
      risk = 'low';
      action = 'allow';
    }

    // Track maximum risk
    if (_riskRank(risk) > _riskRank(_maxRisk)) {
      _maxRisk = risk;
    }
    if (risk == 'high') _highRiskWindows++;

    // Add to history
    _riskHistory.add(_RiskEntry(
      timestamp: _callDuration,
      detectionScore: _currentScore,
      riskLevel: risk,
      action: action,
    ));

    setState(() {
      _callLevelRisk = risk;
      _callLevelAction = action;
    });
  }

  int _riskRank(String level) {
    return switch (level) {
      'high' => 2,
      'medium' => 1,
      _ => 0,
    };
  }

  // -------------------------------------------------------------------------
  // End call
  // -------------------------------------------------------------------------
  void _endCall() {
    _stopEverything();
    setState(() => _callState = _CallState.summary);
  }

  void _stopEverything() {
    _audioSub?.cancel();
    _audioSub = null;
    _analysisTimer?.cancel();
    _analysisTimer = null;
    _callTimer?.cancel();
    _callTimer = null;
    _healthTimer?.cancel();
    _healthTimer = null;
    _recorder.stop();
    _pcmBuffer.clear();
  }

  // -------------------------------------------------------------------------
  // WAV header builder
  // -------------------------------------------------------------------------
  /// Wraps raw PCM bytes with a standard WAV header so librosa can read it.
  Uint8List _wrapPcmAsWav(
      Uint8List pcmData, int sampleRate, int numChannels, int bitsPerSample) {
    final byteRate = sampleRate * numChannels * (bitsPerSample ~/ 8);
    final blockAlign = numChannels * (bitsPerSample ~/ 8);
    final dataSize = pcmData.length;
    final fileSize = 36 + dataSize;

    final header = ByteData(44);
    // RIFF header
    header.setUint8(0, 0x52); // 'R'
    header.setUint8(1, 0x49); // 'I'
    header.setUint8(2, 0x46); // 'F'
    header.setUint8(3, 0x46); // 'F'
    header.setUint32(4, fileSize, Endian.little);
    header.setUint8(8, 0x57);  // 'W'
    header.setUint8(9, 0x41);  // 'A'
    header.setUint8(10, 0x56); // 'V'
    header.setUint8(11, 0x45); // 'E'
    // fmt sub-chunk
    header.setUint8(12, 0x66); // 'f'
    header.setUint8(13, 0x6D); // 'm'
    header.setUint8(14, 0x74); // 't'
    header.setUint8(15, 0x20); // ' '
    header.setUint32(16, 16, Endian.little); // sub-chunk size
    header.setUint16(20, 1, Endian.little); // audio format (PCM)
    header.setUint16(22, numChannels, Endian.little);
    header.setUint32(24, sampleRate, Endian.little);
    header.setUint32(28, byteRate, Endian.little);
    header.setUint16(32, blockAlign, Endian.little);
    header.setUint16(34, bitsPerSample, Endian.little);
    // data sub-chunk
    header.setUint8(36, 0x64); // 'd'
    header.setUint8(37, 0x61); // 'a'
    header.setUint8(38, 0x74); // 't'
    header.setUint8(39, 0x61); // 'a'
    header.setUint32(40, dataSize, Endian.little);

    final wav = Uint8List(44 + dataSize);
    wav.setAll(0, header.buffer.asUint8List());
    wav.setAll(44, pcmData);
    return wav;
  }

  // -------------------------------------------------------------------------
  // Format helpers
  // -------------------------------------------------------------------------
  String _formatDuration(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  String _riskEmoji(String level) {
    return switch (level) {
      'high' => '🔴',
      'medium' => '🟠',
      _ => '🟢',
    };
  }

  // -------------------------------------------------------------------------
  // Build
  // -------------------------------------------------------------------------
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF161B22),
        foregroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            if (_callState == _CallState.active) {
              _endCall();
            } else {
              Navigator.of(context).pop();
            }
          },
        ),
        title: const Row(
          children: [
            Icon(Icons.shield_outlined, size: 20, color: Color(0xFF58A6FF)),
            SizedBox(width: 8),
            Text(
              'VoiceGuard',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 18,
                letterSpacing: 0.5,
              ),
            ),
          ],
        ),
        actions: [
          // Backend status
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: _backendConnected
                        ? const Color(0xFF3FB950)
                        : const Color(0xFFF85149),
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 6),
                Text(
                  _backendConnected ? 'AI ENGINE' : 'OFFLINE',
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    color: _backendConnected
                        ? const Color(0xFF3FB950)
                        : const Color(0xFFF85149),
                    letterSpacing: 0.5,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
      body: SafeArea(
        child: switch (_callState) {
          _CallState.idle => _buildIdleView(),
          _CallState.starting => _buildStartingView(),
          _CallState.active => _buildActiveView(),
          _CallState.ending => _buildStartingView(),
          _CallState.summary => _buildSummaryView(),
          _CallState.error => _buildErrorView(),
        },
      ),
    );
  }

  // -------------------------------------------------------------------------
  // IDLE view
  // -------------------------------------------------------------------------
  Widget _buildIdleView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.mic_none_rounded, size: 72, color: Color(0xFF58A6FF)),
            const SizedBox(height: 20),
            const Text(
              'Live Call Voice\nIntegrity Analysis',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white,
                height: 1.3,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              'Continuously analyzes voice through the microphone\n'
              'to detect AI-generated or cloned speech in real time.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                color: Colors.white.withValues(alpha: 0.5),
                height: 1.5,
              ),
            ),
            const SizedBox(height: 40),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF238636),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 18),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  textStyle: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                onPressed: _startCall,
                icon: const Icon(Icons.phone, size: 22),
                label: const Text('START LIVE CALL'),
              ),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: const Color(0xFF161B22),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: Colors.white.withValues(alpha: 0.1),
                ),
              ),
              child: Text(
                'Place the phone near the speaker during a call.\n'
                'VoiceGuard will continuously analyze the voice.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.white.withValues(alpha: 0.4),
                  height: 1.5,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // -------------------------------------------------------------------------
  // STARTING / ENDING view
  // -------------------------------------------------------------------------
  Widget _buildStartingView() {
    return const Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          CircularProgressIndicator(color: Color(0xFF58A6FF), strokeWidth: 3),
          SizedBox(height: 20),
          Text(
            'Initializing...',
            style: TextStyle(
              color: Colors.white70,
              fontSize: 16,
            ),
          ),
        ],
      ),
    );
  }

  // -------------------------------------------------------------------------
  // ACTIVE view — main live call UI
  // -------------------------------------------------------------------------
  Widget _buildActiveView() {
    final scoreColor = RiskIndicator.riskColor(_callLevelRisk);
    final scorePct = (_currentScore * 100).toStringAsFixed(0);

    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      child: Column(
        children: [
          // -- Call status bar --
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: const Color(0xFFF85149).withValues(alpha: 0.3),
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 10,
                  height: 10,
                  decoration: const BoxDecoration(
                    color: Color(0xFFF85149),
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 10),
                const Text(
                  'CALL ACTIVE',
                  style: TextStyle(
                    color: Color(0xFFF85149),
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                    letterSpacing: 1,
                  ),
                ),
                const Spacer(),
                Text(
                  _formatDuration(_callDuration),
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 20,
                    fontFamily: 'monospace',
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // -- Audio level visualization --
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
                  'AUDIO INPUT',
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    color: Colors.white.withValues(alpha: 0.4),
                    letterSpacing: 1,
                  ),
                ),
                const SizedBox(height: 10),
                _buildAudioLevelBars(),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // -- Detection score --
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
                  'AI DETECTION SCORE',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: Colors.white.withValues(alpha: 0.4),
                    letterSpacing: 1.5,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _riskHistory.isEmpty ? '—' : '$scorePct%',
                  style: TextStyle(
                    fontSize: 48,
                    fontWeight: FontWeight.w900,
                    color: _riskHistory.isEmpty ? Colors.white30 : scoreColor,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  _riskHistory.isEmpty
                      ? 'Waiting for analysis...'
                      : 'AI Detection Score',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.white.withValues(alpha: 0.4),
                  ),
                ),
                if (_riskHistory.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  // EMA indicator
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'Call Avg: ${(_emaScore * 100).toStringAsFixed(0)}%',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.white.withValues(alpha: 0.5),
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      if (_persistenceTriggered) ...[
                        const SizedBox(width: 12),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 2,
                          ),
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
              ],
            ),
          ),
          const SizedBox(height: 16),

          // -- Call-level risk indicator --
          if (_riskHistory.isNotEmpty)
            RiskIndicator(
              riskLevel: _callLevelRisk,
              action: _callLevelAction,
            ),
          const SizedBox(height: 16),

          // -- Risk history --
          if (_riskHistory.isNotEmpty) _buildRiskHistory(),
          const SizedBox(height: 16),

          // -- Mock transaction --
          if (_riskHistory.isNotEmpty)
            TransactionCard(riskLevel: _callLevelRisk),
          const SizedBox(height: 24),

          // -- End call button --
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
                textStyle: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.bold,
                ),
              ),
              onPressed: _endCall,
              icon: const Icon(Icons.call_end),
              label: const Text('END CALL'),
            ),
          ),
          const SizedBox(height: 16),
        ],
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Audio level bars
  // -------------------------------------------------------------------------
  Widget _buildAudioLevelBars() {
    return SizedBox(
      height: 32,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: List.generate(24, (i) {
          // Create varied bar heights based on audio level + some visual variance
          final seed = (i * 7 + DateTime.now().millisecond) % 10;
          final variance = 0.3 + (seed / 10.0) * 0.7;
          final barHeight = (_audioLevel * variance * 32).clamp(2.0, 32.0);

          final barColor = _audioLevel > 0.1
              ? const Color(0xFF58A6FF)
              : Colors.white.withValues(alpha: 0.15);

          return Container(
            width: 6,
            height: barHeight,
            decoration: BoxDecoration(
              color: barColor.withValues(alpha: 0.3 + _audioLevel * 0.7),
              borderRadius: BorderRadius.circular(3),
            ),
          );
        }),
      ),
    );
  }

  // -------------------------------------------------------------------------
  // Risk history list
  // -------------------------------------------------------------------------
  Widget _buildRiskHistory() {
    final recentEntries =
        _riskHistory.length > 10 ? _riskHistory.sublist(_riskHistory.length - 10) : _riskHistory;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'CALL RISK HISTORY',
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w600,
              color: Colors.white.withValues(alpha: 0.4),
              letterSpacing: 1,
            ),
          ),
          const SizedBox(height: 10),
          ...recentEntries.map((e) {
            final scoreStr = (e.detectionScore * 100).toStringAsFixed(0);
            return Padding(
              padding: const EdgeInsets.symmetric(vertical: 3),
              child: Row(
                children: [
                  SizedBox(
                    width: 48,
                    child: Text(
                      _formatDuration(e.timestamp),
                      style: TextStyle(
                        fontSize: 12,
                        fontFamily: 'monospace',
                        color: Colors.white.withValues(alpha: 0.5),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    _riskEmoji(e.riskLevel),
                    style: const TextStyle(fontSize: 12),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    e.riskLevel.toUpperCase(),
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: RiskIndicator.riskColor(e.riskLevel),
                    ),
                  ),
                  const Spacer(),
                  Text(
                    '$scoreStr%',
                    style: TextStyle(
                      fontSize: 12,
                      fontFamily: 'monospace',
                      color: Colors.white.withValues(alpha: 0.5),
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  // -------------------------------------------------------------------------
  // SUMMARY view
  // -------------------------------------------------------------------------
  Widget _buildSummaryView() {
    final maxColor = RiskIndicator.riskColor(_maxRisk);
    final recommendation = switch (_maxRisk) {
      'high' =>
        'Verify the caller through an independent channel before permitting any sensitive action.',
      'medium' =>
        'Exercise caution. Additional identity verification is recommended.',
      _ => 'No significant anomalies detected during this call.',
    };

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          const SizedBox(height: 16),
          const Icon(Icons.summarize_rounded, size: 48, color: Color(0xFF58A6FF)),
          const SizedBox(height: 16),
          const Text(
            'CALL SUMMARY',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              color: Colors.white,
              letterSpacing: 1,
            ),
          ),
          const SizedBox(height: 24),

          // Summary card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: maxColor.withValues(alpha: 0.3),
                width: 1.5,
              ),
            ),
            child: Column(
              children: [
                _summaryRow('Duration', _formatDuration(_callDuration)),
                const Divider(color: Colors.white12, height: 24),
                _summaryRow('Analysis Windows', '${_riskHistory.length}'),
                const Divider(color: Colors.white12, height: 24),
                _summaryRow('Maximum Risk', _maxRisk.toUpperCase(),
                    valueColor: maxColor),
                const Divider(color: Colors.white12, height: 24),
                _summaryRow('High-Risk Windows', '$_highRiskWindows',
                    valueColor:
                        _highRiskWindows > 0 ? const Color(0xFFF85149) : null),
                const Divider(color: Colors.white12, height: 24),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      'Final Decision',
                      style: TextStyle(color: Colors.white54, fontSize: 14),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: maxColor,
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        _maxRisk == 'high'
                            ? '🛑 BLOCKED'
                            : _maxRisk == 'medium'
                                ? '⚠ VERIFY'
                                : '✅ ALLOWED',
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w700,
                          fontSize: 13,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // Recommendation
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'RECOMMENDATION',
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    color: Colors.white.withValues(alpha: 0.4),
                    letterSpacing: 1,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  recommendation,
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.white.withValues(alpha: 0.7),
                    height: 1.5,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // Back button
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                side: const BorderSide(color: Color(0xFF58A6FF), width: 1.5),
                foregroundColor: const Color(0xFF58A6FF),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              onPressed: () => Navigator.of(context).pop(),
              icon: const Icon(Icons.arrow_back),
              label: const Text(
                'BACK TO HOME',
                style: TextStyle(fontWeight: FontWeight.w600),
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
          style: const TextStyle(color: Colors.white54, fontSize: 14),
        ),
        Text(
          value,
          style: TextStyle(
            color: valueColor ?? Colors.white,
            fontWeight: FontWeight.w700,
            fontSize: 16,
          ),
        ),
      ],
    );
  }

  // -------------------------------------------------------------------------
  // ERROR view
  // -------------------------------------------------------------------------
  Widget _buildErrorView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: Color(0xFFF85149)),
            const SizedBox(height: 20),
            const Text(
              'Error',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Color(0xFFF85149),
              ),
            ),
            const SizedBox(height: 12),
            Text(
              _errorMessage,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.6),
                height: 1.5,
                fontSize: 14,
              ),
            ),
            const SizedBox(height: 12),
            if (!_backendConnected)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFFF85149).withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: const Color(0xFFF85149).withValues(alpha: 0.3),
                  ),
                ),
                child: const Text(
                  '⚠ AI ENGINE OFFLINE\n'
                  'Unable to verify voice.\n'
                  'Sensitive actions should not be permitted.',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 12,
                    color: Color(0xFFF85149),
                    height: 1.5,
                  ),
                ),
              ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF58A6FF),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              onPressed: () {
                setState(() {
                  _callState = _CallState.idle;
                  _errorMessage = '';
                });
              },
              icon: const Icon(Icons.refresh),
              label: const Text('Try Again'),
            ),
          ],
        ),
      ),
    );
  }
}
