"""
test_api_schema.py — API schema contract tests.
=================================================

Ensures the backend response schema matches what the Flutter client
expects to parse. Tests use the Pydantic models directly without
requiring a running backend or model.

Tests:
    - DetectionResponse has all required fields
    - All valid decision values are accepted
    - All valid action values are accepted
    - Response schema is compatible with Flutter LiveAnalysisResult.fromJson
    - Threshold is identical in config
"""

import pytest

from voiceguard_config import load_config


class TestResponseSchema:
    """Verify the Pydantic response model has all required fields."""

    def test_detection_response_fields(self):
        from backend import DetectionResponse
        fields = DetectionResponse.model_fields
        required_fields = [
            "decision", "action", "label", "spoof_score", "confidence",
            "risk_level", "risk_percentage", "speech_detected",
            "audio_quality", "evidence", "reason_codes",
            "model_version", "threshold_version", "request_id",
        ]
        for field_name in required_fields:
            assert field_name in fields, f"Missing field: {field_name}"

    def test_valid_decisions(self):
        from backend import DetectionResponse
        valid_decisions = [
            "low_risk", "verification_required", "action_held", "insufficient_evidence"
        ]
        for decision in valid_decisions:
            resp = DetectionResponse(
                decision=decision,
                action="verify",
                label="unknown",
                spoof_score=0.5,
                confidence=0.5,
                risk_level="medium",
                risk_percentage=50.0,
            )
            assert resp.decision == decision

    def test_valid_actions(self):
        from backend import DetectionResponse
        valid_actions = [
            "allow_with_caution", "verify", "hold", "unavailable"
        ]
        for action in valid_actions:
            resp = DetectionResponse(
                decision="low_risk",
                action=action,
                label="bonafide",
                spoof_score=0.1,
                confidence=0.9,
                risk_level="low",
                risk_percentage=10.0,
            )
            assert resp.action == action

    def test_health_response_fields(self):
        from backend import HealthResponse
        resp = HealthResponse(status="ok", model_loaded=True)
        assert resp.status == "ok"
        assert resp.model_loaded is True

    def test_readiness_response_fields(self):
        from backend import ReadinessResponse
        resp = ReadinessResponse(
            status="ready",
            model_loaded=True,
            model_version="voiceguard-v1",
            threshold_version="threshold-v2",
            device="cpu",
            demo_mode=False,
        )
        assert resp.model_version == "voiceguard-v1"
        assert resp.demo_mode is False


class TestConfigConsistency:
    """Verify configuration is consistent across components."""

    def test_threshold_from_config(self):
        """Threshold must come from the unified config, not hardcoded."""
        cfg = load_config()
        assert cfg.thresholds.spoof_threshold == 0.30
        assert cfg.thresholds.high_risk_threshold == 0.85
        assert cfg.thresholds.version == "threshold-v2"

    def test_model_version_from_config(self):
        cfg = load_config()
        assert cfg.model.version == "voiceguard-v1"
        assert cfg.model.name == "garystafford/wav2vec2-deepfake-voice-detector"

    def test_secure_defaults(self):
        """Secure mode must be the default."""
        cfg = load_config()
        assert cfg.security.demo_mode is False

    def test_sample_rate_consistent(self):
        cfg = load_config()
        assert cfg.audio.sample_rate == 16000


class TestFlutterCompatibility:
    """Verify response dict is parseable by Flutter LiveAnalysisResult.fromJson."""

    def test_response_dict_has_flutter_fields(self):
        """The serialized response must contain all fields that Flutter expects."""
        from backend import DetectionResponse, AudioQualityResponse, EvidenceResponse

        resp = DetectionResponse(
            decision="low_risk",
            action="allow_with_caution",
            label="bonafide",
            spoof_score=0.1,
            confidence=0.9,
            risk_level="low",
            risk_percentage=10.0,
            speech_detected=True,
            audio_quality=AudioQualityResponse(
                status="acceptable",
                duration_seconds=3.0,
                rms=0.15,
            ),
            evidence=EvidenceResponse(),
            reason_codes=[],
            model_version="voiceguard-v1",
            threshold_version="threshold-v2",
            request_id="test-123",
        )

        d = resp.model_dump()

        # Fields that Flutter LiveAnalysisResult.fromJson must find
        assert "decision" in d
        assert "action" in d
        assert "label" in d
        assert "spoof_score" in d
        assert "confidence" in d
        assert "risk_level" in d
        assert "speech_detected" in d
        assert "reason_codes" in d
        assert "model_version" in d
        assert "audio_quality" in d
        assert isinstance(d["audio_quality"], dict)
        assert "status" in d["audio_quality"]
