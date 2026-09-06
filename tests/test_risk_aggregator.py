"""
test_risk_aggregator.py — Unit tests for the risk aggregation state machine.
=============================================================================

Tests:
    - Initial state is NO_EVIDENCE
    - One high score does NOT hold action
    - Two consecutive high scores trigger ACTION_HELD
    - Silence after ACTION_HELD → INSUFFICIENT_EVIDENCE (not LOW_RISK)
    - Alternating high/low does not trigger persistence
    - Backend failure transitions appropriately
    - Recovery requires cooldown windows
    - Stale analysis handling
"""

import time
import pytest

from risk_aggregator import (
    RiskAggregator,
    CallDecision,
    CallAction,
    ClipEvidence,
)


def make_evidence(
    spoof_score: float = 0.1,
    speech_detected: bool = True,
    quality: str = "acceptable",
    label: str = "bonafide",
) -> ClipEvidence:
    return ClipEvidence(
        spoof_score=spoof_score,
        speech_detected=speech_detected,
        audio_quality_status=quality,
        label=label,
        timestamp=time.time(),
    )


class TestInitialState:
    """Aggregator starts in NO_EVIDENCE state."""

    def test_initial_state(self):
        agg = RiskAggregator()
        state = agg.state
        assert state.decision == CallDecision.NO_EVIDENCE
        assert state.action == CallAction.UNAVAILABLE
        assert state.total_windows == 0


class TestSingleHighScore:
    """One high-risk window must NOT hold action."""

    def test_one_high_score_does_not_hold(self):
        agg = RiskAggregator(persistent_high_windows=2)
        state = agg.update(make_evidence(spoof_score=0.95, label="spoof"))
        assert state.decision != CallDecision.ACTION_HELD
        # Should be VERIFICATION_REQUIRED at most
        assert state.decision in (
            CallDecision.VERIFICATION_REQUIRED,
            CallDecision.LOW_RISK,
        )


class TestPersistentHighScores:
    """Two consecutive high scores must trigger ACTION_HELD."""

    def test_two_consecutive_high(self):
        agg = RiskAggregator(
            persistent_high_windows=2,
            high_risk_threshold=0.85,
        )
        agg.update(make_evidence(spoof_score=0.90, label="spoof"))
        state = agg.update(make_evidence(spoof_score=0.92, label="spoof"))
        assert state.decision == CallDecision.ACTION_HELD
        assert state.action == CallAction.HOLD
        assert state.consecutive_high_windows >= 2

    def test_three_consecutive_high(self):
        agg = RiskAggregator(persistent_high_windows=2)
        agg.update(make_evidence(spoof_score=0.90, label="spoof"))
        agg.update(make_evidence(spoof_score=0.92, label="spoof"))
        state = agg.update(make_evidence(spoof_score=0.95, label="spoof"))
        assert state.decision == CallDecision.ACTION_HELD


class TestSilenceAfterActionHeld:
    """Silence after ACTION_HELD must NOT reset to LOW_RISK."""

    def test_silence_after_held_gives_insufficient(self):
        agg = RiskAggregator(persistent_high_windows=2)
        # Trigger ACTION_HELD
        agg.update(make_evidence(spoof_score=0.90, label="spoof"))
        agg.update(make_evidence(spoof_score=0.92, label="spoof"))
        assert agg.state.decision == CallDecision.ACTION_HELD

        # Send silence
        state = agg.update(make_evidence(
            spoof_score=0.0,
            speech_detected=False,
            quality="silent",
            label="unknown",
        ))
        assert state.decision == CallDecision.INSUFFICIENT_EVIDENCE
        assert state.decision != CallDecision.LOW_RISK

    def test_silence_never_allows(self):
        """Silence must never produce allow_with_caution after ACTION_HELD."""
        agg = RiskAggregator(persistent_high_windows=2)
        agg.update(make_evidence(spoof_score=0.90, label="spoof"))
        agg.update(make_evidence(spoof_score=0.92, label="spoof"))

        state = agg.update(make_evidence(
            spoof_score=0.0,
            speech_detected=False,
            quality="silent",
        ))
        assert state.action != CallAction.ALLOW_WITH_CAUTION


class TestAlternatingScores:
    """Alternating high/low should NOT trigger persistence."""

    def test_alternating_no_persistence(self):
        agg = RiskAggregator(persistent_high_windows=2)
        agg.update(make_evidence(spoof_score=0.90, label="spoof"))
        agg.update(make_evidence(spoof_score=0.10, label="bonafide"))
        agg.update(make_evidence(spoof_score=0.90, label="spoof"))
        state = agg.update(make_evidence(spoof_score=0.10, label="bonafide"))
        assert state.decision != CallDecision.ACTION_HELD


class TestBackendFailure:
    """Backend failure should produce appropriate state."""

    def test_backend_failure_from_no_evidence(self):
        agg = RiskAggregator()
        state = agg.record_backend_failure()
        assert state.decision == CallDecision.INSUFFICIENT_EVIDENCE
        assert "backend_unavailable" in state.reason_codes

    def test_backend_failure_after_held(self):
        agg = RiskAggregator(persistent_high_windows=2)
        agg.update(make_evidence(spoof_score=0.90, label="spoof"))
        agg.update(make_evidence(spoof_score=0.92, label="spoof"))
        assert agg.state.decision == CallDecision.ACTION_HELD

        state = agg.record_backend_failure()
        assert state.decision == CallDecision.INSUFFICIENT_EVIDENCE
        # Action should NOT be allow_with_caution
        assert state.action != CallAction.ALLOW_WITH_CAUTION


class TestCooldownRecovery:
    """Recovery from ACTION_HELD requires cooldown windows."""

    def test_recovery_requires_cooldown(self):
        agg = RiskAggregator(
            persistent_high_windows=2,
            cooldown_windows=3,
        )
        # Trigger ACTION_HELD
        agg.update(make_evidence(spoof_score=0.90, label="spoof"))
        agg.update(make_evidence(spoof_score=0.92, label="spoof"))
        assert agg.state.decision == CallDecision.ACTION_HELD

        # One normal window — should NOT recover to LOW_RISK
        state = agg.update(make_evidence(spoof_score=0.10, label="bonafide"))
        assert state.decision != CallDecision.LOW_RISK
        assert state.decision == CallDecision.VERIFICATION_REQUIRED

    def test_full_cooldown_recovery(self):
        agg = RiskAggregator(
            persistent_high_windows=2,
            cooldown_windows=2,
            medium_ema_threshold=0.40,
        )
        # Trigger ACTION_HELD
        agg.update(make_evidence(spoof_score=0.90, label="spoof"))
        agg.update(make_evidence(spoof_score=0.92, label="spoof"))

        # Send enough low-risk windows to complete cooldown
        for _ in range(5):
            state = agg.update(make_evidence(spoof_score=0.05, label="bonafide"))

        # After sufficient cooldown, should eventually reach LOW_RISK
        # (depends on EMA also settling)
        # The key assertion is that it's no longer ACTION_HELD
        assert state.decision != CallDecision.ACTION_HELD


class TestReset:
    """Reset should clear all state."""

    def test_reset(self):
        agg = RiskAggregator()
        agg.update(make_evidence(spoof_score=0.90, label="spoof"))
        agg.reset()
        state = agg.state
        assert state.decision == CallDecision.NO_EVIDENCE
        assert state.total_windows == 0
        assert state.ema_score == 0.0


class TestStaleAnalysis:
    """Stale analysis handling."""

    def test_stale_after_held(self):
        agg = RiskAggregator(persistent_high_windows=2)
        agg.update(make_evidence(spoof_score=0.90, label="spoof"))
        agg.update(make_evidence(spoof_score=0.92, label="spoof"))

        state = agg.record_stale_analysis()
        assert state.decision == CallDecision.INSUFFICIENT_EVIDENCE
        assert "analysis_stale" in state.reason_codes
