"""
risk_aggregator.py — Stateful call-level risk aggregation for VoiceGuard.
=========================================================================

Separates CLIP-LEVEL evidence (single audio chunk analysis) from
CALL-LEVEL decisions (aggregated over multiple windows).

The backend returns clip-level evidence. The aggregator (used by the
Flutter client or a session-aware server component) accumulates
evidence and produces call-level decisions.

State machine:
    NO_EVIDENCE
        → ANALYZING
        → LOW_RISK
        → VERIFICATION_REQUIRED
        → ACTION_HELD

Recovery from ACTION_HELD requires explicit safe conditions
(cooldown windows of acceptable evidence). Silence or backend
failure after ACTION_HELD transitions to INSUFFICIENT_EVIDENCE,
NOT to LOW_RISK.

    ACTION_HELD → silence → INSUFFICIENT_EVIDENCE  (correct)
    ACTION_HELD → silence → LOW_RISK               (NEVER)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CallDecision(str, Enum):
    """Call-level aggregated decision states."""
    NO_EVIDENCE = "no_evidence"
    ANALYZING = "analyzing"
    LOW_RISK = "low_risk"
    VERIFICATION_REQUIRED = "verification_required"
    ACTION_HELD = "action_held"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CallAction(str, Enum):
    """Recommended action for the current call state."""
    ALLOW_WITH_CAUTION = "allow_with_caution"
    VERIFY = "verify"
    HOLD = "hold"
    UNAVAILABLE = "unavailable"


@dataclass
class ClipEvidence:
    """Evidence from a single audio clip analysis."""
    spoof_score: float
    speech_detected: bool
    audio_quality_status: str  # "acceptable" | "poor" | "silent" | "invalid"
    label: str  # "bonafide" | "spoof" | "unknown"
    timestamp: float = field(default_factory=time.time)


@dataclass
class AggregatedState:
    """Snapshot of the current aggregated call-level state."""
    decision: CallDecision
    action: CallAction
    ema_score: float
    current_score: float
    total_windows: int
    consecutive_high_windows: int
    last_analysis_time: Optional[float]
    backend_fresh: bool
    audio_quality_status: str
    reason_codes: List[str]


class RiskAggregator:
    """
    Stateful risk aggregator for call-level decision making.

    Consumes clip-level evidence and produces call-level decisions
    using EMA smoothing, persistence rules, and cooldown logic.
    """

    def __init__(
        self,
        *,
        ema_alpha: float = 0.3,
        spoof_threshold: float = 0.30,
        high_risk_threshold: float = 0.85,
        medium_ema_threshold: float = 0.40,
        high_ema_threshold: float = 0.85,
        persistent_high_windows: int = 2,
        cooldown_windows: int = 3,
        staleness_seconds: float = 30.0,
    ):
        self._ema_alpha = ema_alpha
        self._spoof_threshold = spoof_threshold
        self._high_risk_threshold = high_risk_threshold
        self._medium_ema_threshold = medium_ema_threshold
        self._high_ema_threshold = high_ema_threshold
        self._persistent_high_windows = persistent_high_windows
        self._cooldown_windows = cooldown_windows
        self._staleness_seconds = staleness_seconds

        # State
        self._decision = CallDecision.NO_EVIDENCE
        self._ema_score: float = 0.0
        self._current_score: float = 0.0
        self._total_windows: int = 0
        self._consecutive_high: int = 0
        self._cooldown_remaining: int = 0
        self._was_action_held: bool = False
        self._last_analysis_time: Optional[float] = None
        self._backend_fresh: bool = True
        self._audio_quality_status: str = "unknown"
        self._reason_codes: List[str] = []

    def reset(self) -> None:
        """Reset all aggregation state for a new call."""
        self._decision = CallDecision.NO_EVIDENCE
        self._ema_score = 0.0
        self._current_score = 0.0
        self._total_windows = 0
        self._consecutive_high = 0
        self._cooldown_remaining = 0
        self._was_action_held = False
        self._last_analysis_time = None
        self._backend_fresh = True
        self._audio_quality_status = "unknown"
        self._reason_codes = []

    @property
    def state(self) -> AggregatedState:
        """Current aggregated state snapshot."""
        return AggregatedState(
            decision=self._decision,
            action=self._decision_to_action(self._decision),
            ema_score=round(self._ema_score, 6),
            current_score=round(self._current_score, 6),
            total_windows=self._total_windows,
            consecutive_high_windows=self._consecutive_high,
            last_analysis_time=self._last_analysis_time,
            backend_fresh=self._backend_fresh,
            audio_quality_status=self._audio_quality_status,
            reason_codes=list(self._reason_codes),
        )

    def record_backend_failure(self) -> AggregatedState:
        """Record a backend communication failure."""
        self._backend_fresh = False
        self._reason_codes = ["backend_unavailable"]

        if self._was_action_held or self._decision == CallDecision.ACTION_HELD:
            # ACTION_HELD + backend failure → INSUFFICIENT_EVIDENCE
            # Sensitive actions remain held.
            self._decision = CallDecision.INSUFFICIENT_EVIDENCE
        elif self._decision == CallDecision.NO_EVIDENCE:
            self._decision = CallDecision.INSUFFICIENT_EVIDENCE
        else:
            # Preserve current decision but mark as potentially stale
            self._reason_codes.append("analysis_stale")

        return self.state

    def record_stale_analysis(self) -> AggregatedState:
        """Mark the current analysis as stale (too old)."""
        self._backend_fresh = False
        self._reason_codes = ["analysis_stale"]

        if self._was_action_held or self._decision == CallDecision.ACTION_HELD:
            self._decision = CallDecision.INSUFFICIENT_EVIDENCE
        elif self._total_windows == 0:
            self._decision = CallDecision.INSUFFICIENT_EVIDENCE

        return self.state

    def update(self, evidence: ClipEvidence) -> AggregatedState:
        """
        Process a new clip-level evidence and update call-level decision.

        Parameters
        ----------
        evidence : ClipEvidence
            Evidence from a single audio clip analysis.

        Returns
        -------
        AggregatedState
            Updated call-level state.
        """
        self._last_analysis_time = evidence.timestamp
        self._backend_fresh = True
        self._audio_quality_status = evidence.audio_quality_status
        self._reason_codes = []

        # --- Handle insufficient evidence at clip level ---
        if not evidence.speech_detected:
            self._reason_codes.append("no_speech")
            return self._handle_no_speech()

        if evidence.audio_quality_status in ("silent", "invalid"):
            self._reason_codes.append("audio_quality_poor")
            return self._handle_no_speech()

        if evidence.audio_quality_status == "poor":
            self._reason_codes.append("audio_quality_poor")
            # Poor quality but has some speech — process with caution
            # Don't count toward persistence, but update EMA conservatively

        # --- Update scores ---
        self._current_score = evidence.spoof_score
        self._total_windows += 1

        # EMA update
        if self._total_windows == 1:
            self._ema_score = self._current_score
        else:
            self._ema_score = (
                self._ema_alpha * self._current_score
                + (1 - self._ema_alpha) * self._ema_score
            )

        # --- Persistence rule ---
        if self._current_score >= self._high_risk_threshold:
            self._consecutive_high += 1
            self._reason_codes.append("elevated_spoof_score")
        else:
            self._consecutive_high = 0

        persistence_triggered = self._consecutive_high >= self._persistent_high_windows
        if persistence_triggered:
            self._reason_codes.append("persistent_high_spoof_score")

        # --- Cooldown logic ---
        if self._cooldown_remaining > 0:
            if self._current_score < self._medium_ema_threshold:
                self._cooldown_remaining -= 1
                if self._cooldown_remaining == 0:
                    # Cooldown complete — allow recovery
                    self._was_action_held = False
                else:
                    self._reason_codes.append("cooldown_active")
            else:
                # Reset cooldown if evidence rises again
                self._cooldown_remaining = self._cooldown_windows
                self._reason_codes.append("cooldown_reset")

        # --- Determine decision ---
        if persistence_triggered:
            self._decision = CallDecision.ACTION_HELD
            self._was_action_held = True
            self._cooldown_remaining = self._cooldown_windows
        elif self._was_action_held and self._cooldown_remaining > 0:
            # Still in recovery from ACTION_HELD
            self._decision = CallDecision.VERIFICATION_REQUIRED
            self._reason_codes.append("recovering_from_action_held")
        elif self._ema_score >= self._high_ema_threshold:
            self._decision = CallDecision.VERIFICATION_REQUIRED
            self._reason_codes.append("elevated_spoof_score")
        elif self._ema_score >= self._medium_ema_threshold:
            self._decision = CallDecision.VERIFICATION_REQUIRED
            self._reason_codes.append("independent_verification_required")
        else:
            self._decision = CallDecision.LOW_RISK

        if self._decision == CallDecision.ANALYZING:
            self._decision = CallDecision.LOW_RISK

        return self.state

    def _handle_no_speech(self) -> AggregatedState:
        """Handle a clip with no speech detected."""
        # Do NOT reset consecutive high count for silence
        # Do NOT escalate risk for silence
        # If we were in ACTION_HELD, transition to INSUFFICIENT_EVIDENCE
        if self._was_action_held or self._decision == CallDecision.ACTION_HELD:
            self._decision = CallDecision.INSUFFICIENT_EVIDENCE
        elif self._total_windows == 0:
            self._decision = CallDecision.INSUFFICIENT_EVIDENCE
        else:
            # Keep previous decision but note insufficient evidence
            self._decision = CallDecision.INSUFFICIENT_EVIDENCE

        return self.state

    @staticmethod
    def _decision_to_action(decision: CallDecision) -> CallAction:
        """Map a decision state to a recommended action."""
        return {
            CallDecision.NO_EVIDENCE: CallAction.UNAVAILABLE,
            CallDecision.ANALYZING: CallAction.VERIFY,
            CallDecision.LOW_RISK: CallAction.ALLOW_WITH_CAUTION,
            CallDecision.VERIFICATION_REQUIRED: CallAction.VERIFY,
            CallDecision.ACTION_HELD: CallAction.HOLD,
            CallDecision.INSUFFICIENT_EVIDENCE: CallAction.VERIFY,
        }.get(decision, CallAction.VERIFY)
