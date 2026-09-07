"""
voiceguard_config.py — Typed configuration loader for VoiceGuard.
================================================================

Loads config/model_config.yaml and exposes a frozen, typed configuration
object used by backend.py, evaluation scripts, tests, and risk_aggregator.py.

DEPENDENCY CHAIN:
    config/model_config.yaml  →  voiceguard_config.py  →  backend.py
                                                        →  evaluation scripts
                                                        →  tests
                                                        →  documentation

No independent threshold constants may exist outside this chain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThresholdConfig:
    version: str
    spoof_threshold: float
    high_risk_threshold: float


@dataclass(frozen=True)
class BackendModelConfig:
    backend: str
    model_id: str
    revision: str
    version: str
    score_type: str
    thresholds: ThresholdConfig
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None


@dataclass(frozen=True)
class ModelConfig:
    """Legacy model config structure for backward-compatibility."""
    name: str
    revision: str
    version: str


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    minimum_duration_seconds: float
    maximum_duration_seconds: float
    allowed_extensions: tuple[str, ...]


@dataclass(frozen=True)
class RiskAggregationConfig:
    ema_alpha: float
    persistent_high_windows: int
    cooldown_windows: int
    medium_ema_threshold: float
    high_ema_threshold: float


@dataclass(frozen=True)
class AudioQualityConfig:
    rms_silence_threshold: float
    minimum_voiced_ratio: float
    maximum_clipping_ratio: float
    poor_snr_threshold_db: float


@dataclass(frozen=True)
class ServerConfig:
    maximum_upload_bytes: int
    maximum_concurrency: int
    request_timeout_seconds: int
    live_request_timeout_seconds: int
    rate_limit_requests_per_minute: int


@dataclass(frozen=True)
class SecurityConfig:
    demo_mode: bool
    allowed_cors_origins: tuple[str, ...]


@dataclass(frozen=True)
class VoiceGuardConfig:
    """Root configuration object. Immutable after construction."""
    active_model: str
    models: Dict[str, BackendModelConfig]
    model: ModelConfig
    audio: AudioConfig
    thresholds: ThresholdConfig
    risk_aggregation: RiskAggregationConfig
    audio_quality: AudioQualityConfig
    server: ServerConfig
    security: SecurityConfig

    @property
    def active_backend(self) -> BackendModelConfig:
        """Return the configuration for the active model backend."""
        if self.active_model in self.models:
            return self.models[self.active_model]
        raise KeyError(
            f"Active model '{self.active_model}' not found in registered models: "
            f"{list(self.models.keys())}"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "model_config.yaml",
)

_cached_config: Optional[VoiceGuardConfig] = None


def load_config(config_path: Optional[str] = None) -> VoiceGuardConfig:
    """
    Load and parse config/model_config.yaml into a typed VoiceGuardConfig.

    Uses environment variable VOICEGUARD_CONFIG_PATH if set, otherwise
    falls back to the default path relative to this file.

    Results are cached — subsequent calls return the same object.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    path = config_path or os.environ.get("VOICEGUARD_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"VoiceGuard configuration file not found: {path}\n"
            f"Expected at: {_DEFAULT_CONFIG_PATH}"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid configuration file: {path}")

    # Active model selection (supports env var override for testing)
    active_model = os.environ.get("VOICEGUARD_ACTIVE_MODEL") or raw.get("active_model", "wav2vec2")

    # Parse models registry
    models_dict: Dict[str, BackendModelConfig] = {}
    models_raw = raw.get("models", {})
    for m_key, m_val in models_raw.items():
        th_raw = m_val.get("thresholds", {})
        th_obj = ThresholdConfig(
            version=th_raw.get("version", "threshold-v2"),
            spoof_threshold=float(th_raw.get("spoof_threshold", 0.30)),
            high_risk_threshold=float(th_raw.get("high_risk_threshold", 0.85)),
        )
        models_dict[m_key] = BackendModelConfig(
            backend=m_val.get("backend", m_key),
            model_id=m_val.get("model_id", ""),
            revision=m_val.get("revision", "main"),
            version=m_val.get("version", ""),
            score_type=m_val.get("score_type", "uncalibrated_softmax_score"),
            thresholds=th_obj,
            checkpoint_path=m_val.get("checkpoint_path"),
            checkpoint_sha256=m_val.get("checkpoint_sha256"),
        )

    # Resolve active model thresholds & info
    if active_model in models_dict:
        active_backend_cfg = models_dict[active_model]
        active_thresholds = active_backend_cfg.thresholds
        active_model_name = active_backend_cfg.model_id
        active_revision = active_backend_cfg.revision
        active_version = active_backend_cfg.version
    else:
        # Fallback to legacy sections
        legacy_th = raw.get("thresholds", {})
        active_thresholds = ThresholdConfig(
            version=legacy_th.get("version", "threshold-v2"),
            spoof_threshold=float(legacy_th.get("spoof_threshold", 0.30)),
            high_risk_threshold=float(legacy_th.get("high_risk_threshold", 0.85)),
        )
        legacy_m = raw.get("model", {})
        active_model_name = legacy_m.get("name", "garystafford/wav2vec2-deepfake-voice-detector")
        active_revision = legacy_m.get("revision", "main")
        active_version = legacy_m.get("version", "voiceguard-v1")

    audio_raw = raw.get("audio", {})
    risk_raw = raw.get("risk_aggregation", {})
    quality_raw = raw.get("audio_quality", {})
    server_raw = raw.get("server", {})
    security_raw = raw.get("security", {})

    # Allow environment variable override for demo_mode
    demo_mode_env = os.environ.get("VOICEGUARD_DEMO_MODE", "").strip().lower()
    if demo_mode_env in ("1", "true", "yes"):
        demo_mode = True
    elif demo_mode_env in ("0", "false", "no"):
        demo_mode = False
    else:
        demo_mode = security_raw.get("demo_mode", False)

    config = VoiceGuardConfig(
        active_model=active_model,
        models=models_dict,
        model=ModelConfig(
            name=active_model_name,
            revision=active_revision,
            version=active_version,
        ),
        audio=AudioConfig(
            sample_rate=audio_raw.get("sample_rate", 16000),
            minimum_duration_seconds=float(audio_raw.get("minimum_duration_seconds", 0.5)),
            maximum_duration_seconds=float(audio_raw.get("maximum_duration_seconds", 60.0)),
            allowed_extensions=tuple(audio_raw.get("allowed_extensions", [".wav", ".flac"])),
        ),
        thresholds=active_thresholds,
        risk_aggregation=RiskAggregationConfig(
            ema_alpha=float(risk_raw.get("ema_alpha", 0.3)),
            persistent_high_windows=int(risk_raw.get("persistent_high_windows", 2)),
            cooldown_windows=int(risk_raw.get("cooldown_windows", 3)),
            medium_ema_threshold=float(risk_raw.get("medium_ema_threshold", 0.40)),
            high_ema_threshold=float(risk_raw.get("high_ema_threshold", 0.85)),
        ),
        audio_quality=AudioQualityConfig(
            rms_silence_threshold=float(quality_raw.get("rms_silence_threshold", 0.003)),
            minimum_voiced_ratio=float(quality_raw.get("minimum_voiced_ratio", 0.10)),
            maximum_clipping_ratio=float(quality_raw.get("maximum_clipping_ratio", 0.05)),
            poor_snr_threshold_db=float(quality_raw.get("poor_snr_threshold_db", 5.0)),
        ),
        server=ServerConfig(
            maximum_upload_bytes=int(server_raw.get("maximum_upload_bytes", 10_485_760)),
            maximum_concurrency=int(server_raw.get("maximum_concurrency", 4)),
            request_timeout_seconds=int(server_raw.get("request_timeout_seconds", 30)),
            live_request_timeout_seconds=int(server_raw.get("live_request_timeout_seconds", 15)),
            rate_limit_requests_per_minute=int(server_raw.get("rate_limit_requests_per_minute", 60)),
        ),
        security=SecurityConfig(
            demo_mode=demo_mode,
            allowed_cors_origins=tuple(security_raw.get("allowed_cors_origins", [])),
        ),
    )

    _cached_config = config
    return config


def reset_config_cache() -> None:
    """Clear the cached configuration. Used by tests."""
    global _cached_config
    _cached_config = None


# Alias for compatibility
get_config = load_config
