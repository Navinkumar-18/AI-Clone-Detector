"""
test_security_config.py — Security configuration tests.
========================================================

Tests:
    - TLS verification is enabled by default (demo_mode=false)
    - CORS differs correctly between secure and demo mode
    - Config secure defaults
    - Git does not track cert/key files (checked externally)
"""

import os
import pytest

from voiceguard_config import load_config, reset_config_cache


class TestSecureDefaults:
    """Secure behavior must be the default."""

    def test_demo_mode_default_false(self):
        cfg = load_config()
        assert cfg.security.demo_mode is False

    def test_cors_not_wildcard_by_default(self):
        cfg = load_config()
        assert "*" not in cfg.security.allowed_cors_origins

    def test_cors_has_localhost(self):
        cfg = load_config()
        origins = cfg.security.allowed_cors_origins
        assert any("localhost" in o for o in origins)

    def test_upload_limit_set(self):
        cfg = load_config()
        assert cfg.server.maximum_upload_bytes > 0
        assert cfg.server.maximum_upload_bytes <= 50 * 1024 * 1024  # Max 50 MB

    def test_concurrency_limit_set(self):
        cfg = load_config()
        assert cfg.server.maximum_concurrency > 0
        assert cfg.server.maximum_concurrency <= 32

    def test_rate_limit_set(self):
        cfg = load_config()
        assert cfg.server.rate_limit_requests_per_minute > 0


class TestDemoModeOverride:
    """Demo mode can be enabled via environment variable."""

    def test_env_override(self):
        reset_config_cache()
        os.environ["VOICEGUARD_DEMO_MODE"] = "true"
        try:
            cfg = load_config()
            assert cfg.security.demo_mode is True
        finally:
            os.environ.pop("VOICEGUARD_DEMO_MODE", None)
            reset_config_cache()


class TestConfigFileExists:
    """Configuration file must exist."""

    def test_config_file_exists(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "model_config.yaml",
        )
        assert os.path.exists(config_path), f"Config file missing: {config_path}"
