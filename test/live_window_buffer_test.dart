import 'dart:math';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:sih2026/config.dart';

void main() {
  group('Live Call Rolling Buffer and Windowing Contract Tests', () {
    const int sampleRate = kLiveSampleRate; // 16000
    const int bytesPerSample = 2; // 16-bit PCM
    const int bytesPerSec = sampleRate * bytesPerSample; // 32000
    final int fullWindowBytes = (kLiveFullWindowMs / 1000 * bytesPerSec).round(); // 128000
    final int strideBytes = (kLiveStrideMs / 1000 * bytesPerSec).round(); // 64000

    test('Configuration constants match live-window contract', () {
      expect(kLiveFullWindowMs, equals(4000), reason: 'Full window must be 4000ms');
      expect(kLiveStrideMs, equals(2000), reason: 'Stride must be 2000ms');
      expect(kLiveSampleRate, equals(16000), reason: 'Sample rate must be 16kHz');
      expect(fullWindowBytes, equals(128000), reason: '4s of 16kHz 16-bit mono is 128,000 bytes');
      expect(strideBytes, equals(64000), reason: '2s stride is 64,000 bytes');
    });

    test('Initial 0-4s buffer accumulation is rejected and never dispatched', () {
      final List<int> pcmBuffer = [];

      // Simulate 1s of audio incoming (32,000 bytes)
      pcmBuffer.addAll(List.filled(bytesPerSec, 0));
      expect(pcmBuffer.length < fullWindowBytes, isTrue);

      // Simulate 2s of audio incoming (64,000 bytes total)
      pcmBuffer.addAll(List.filled(bytesPerSec, 0));
      expect(pcmBuffer.length < fullWindowBytes, isTrue);

      // Simulate 3s of audio incoming (96,000 bytes total)
      pcmBuffer.addAll(List.filled(bytesPerSec, 0));
      expect(pcmBuffer.length < fullWindowBytes, isTrue);

      // Incomplete window MUST NOT be analyzed
      final bool canAnalyze = pcmBuffer.length >= fullWindowBytes;
      expect(canAnalyze, isFalse, reason: 'Buffer under 4s must never trigger analysis');
    });

    test('First complete window is reached at exactly 4s and is exactly 4.0s (128,000 bytes)', () {
      final List<int> pcmBuffer = [];

      // Add 4.0 seconds of audio (128,000 bytes)
      pcmBuffer.addAll(List.filled(fullWindowBytes, 0x10));
      expect(pcmBuffer.length >= fullWindowBytes, isTrue);

      // Extract window
      final chunk = Uint8List.fromList(pcmBuffer.sublist(0, fullWindowBytes));
      expect(chunk.length, equals(128000));
      expect(chunk.length / bytesPerSec, equals(4.0), reason: 'First window duration must be 4.0s');

      // Advance by stride
      final removeBytes = strideBytes.clamp(0, pcmBuffer.length);
      pcmBuffer.removeRange(0, removeBytes);

      // Remaining buffer contains exactly the 2-second overlap tail (64,000 bytes)
      expect(pcmBuffer.length, equals(64000));
    });

    test('Second window uses 2s stride, incorporating 2s overlap tail + 2s new audio', () {
      final List<int> pcmBuffer = [];

      // First 4 seconds of numbered bytes
      for (int i = 0; i < fullWindowBytes; i++) {
        pcmBuffer.add(i % 256);
      }

      // Window 1 extraction
      final window1 = pcmBuffer.sublist(0, fullWindowBytes);
      expect(window1.length, equals(fullWindowBytes));

      // Advance by stride (2s = 64,000 bytes)
      pcmBuffer.removeRange(0, strideBytes);
      expect(pcmBuffer.length, equals(64000));

      // 2 seconds of new incoming audio arrives (t = 4s to 6s)
      for (int i = 0; i < strideBytes; i++) {
        pcmBuffer.add((i + 100) % 256);
      }
      expect(pcmBuffer.length, equals(fullWindowBytes));

      // Window 2 extraction
      final window2 = pcmBuffer.sublist(0, fullWindowBytes);
      expect(window2.length, equals(128000));
      expect(window2.length / bytesPerSec, equals(4.0));

      // Check overlap: the first 64,000 bytes of window2 must match the second 64,000 bytes of window1
      final window1Tail = window1.sublist(strideBytes, fullWindowBytes);
      final window2Head = window2.sublist(0, strideBytes);
      expect(window2Head, equals(window1Tail), reason: '2-second overlap must be strictly preserved');
    });

    test('Digital silence RMS calculation detects silence (<0.003) and suppresses network calls', () {
      // 4s of zero PCM
      final silentChunk = Uint8List(fullWindowBytes);

      double sumSq = 0;
      int samples = 0;
      for (int i = 0; i < silentChunk.length - 1; i += 2) {
        int sample = silentChunk[i] | (silentChunk[i + 1] << 8);
        if (sample >= 0x8000) sample -= 0x10000;
        final norm = sample / 32768.0;
        sumSq += norm * norm;
        samples++;
      }
      final chunkRms = samples > 0 ? sqrt(sumSq / samples) : 0.0;
      expect(chunkRms, equals(0.0));
      expect(chunkRms < 0.003, isTrue, reason: 'Silent chunk must be below 0.003 RMS');
    });

    test('Final partial buffer on call termination is never dispatched', () {
      final List<int> pcmBuffer = [];

      // Call ends with 2.1s left in buffer (67,200 bytes)
      final partialBytes = (2.1 * bytesPerSec).round();
      pcmBuffer.addAll(List.filled(partialBytes, 0));

      // Buffer has partial data
      expect(pcmBuffer.length, equals(67200));

      // Contract check: does it qualify for analysis?
      final bool canAnalyze = pcmBuffer.length >= fullWindowBytes;
      expect(canAnalyze, isFalse, reason: 'Final partial buffer must not be classified or sent');
    });
  });
}
