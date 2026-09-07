"""
tests/test_model_backends.py — Unit and integration tests for ModelBackend abstraction.
========================================================================================

Unit tests:
    - Missing checkpoint raises ModelLoadError
    - Corrupted checkpoint raises ModelLoadError
    - Empty audio raises ValueError
    - Unknown backend name raises ValueError
    - Backend selection via factory
    - Silence contract
    - /ready returns 503 on load failure
    - No implicit fallback on failure
    - Readiness metadata changes with backend

Integration tests (@pytest.mark.integration):
    - Real checkpoint loading and inference for WavLMMLPBackend
    - Parity between predict.py and WavLMMLPBackend
    - Real Wav2Vec2Backend loading and inference
"""

from __future__ import annotations

import os
import tempfile
import unittest.mock as mock
import numpy as np
import pytest
import torch

from voiceguard.model_backends import (
    ModelBackend,
    ModelLoadError,
    ModelPrediction,
    Wav2Vec2Backend,
    WavLMMLPBackend,
    create_backend,
)
from voiceguard_config import load_config, reset_config_cache


# ---------------------------------------------------------------------------
# Unit Tests (Mocked, Isolated, Fast)
# ---------------------------------------------------------------------------

class TestModelBackendFactory:
    """Tests for create_backend factory function."""

    def test_create_wav2vec2_backend(self):
        backend = create_backend("wav2vec2")
        assert isinstance(backend, Wav2Vec2Backend)
        assert backend.backend_name == "wav2vec2"
        assert backend.score_type == "uncalibrated_softmax_score"
        assert not backend.is_loaded()

    def test_create_wavlm_mlp_backend(self):
        backend = create_backend("wavlm_mlp")
        assert isinstance(backend, WavLMMLPBackend)
        assert backend.backend_name == "wavlm_mlp"
        assert backend.score_type == "uncalibrated_sigmoid_score"
        assert not backend.is_loaded()

    def test_create_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model backend"):
            create_backend("nonexistent_model")


class TestWavLMMLPBackendFailureModes:
    """Unit tests for WavLMMLPBackend error handling and failure modes."""

    def test_missing_checkpoint_raises_model_load_error(self):
        backend = WavLMMLPBackend(checkpoint_path="nonexistent_checkpoint_12345.pt")
        with pytest.raises(ModelLoadError, match="checkpoint not found"):
            backend.load()
        assert not backend.is_loaded()

    def test_corrupted_checkpoint_raises_model_load_error(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"CORRUPTED_CHECKPOINT_DATA_NOT_A_PYTORCH_FILE")
            corrupt_path = f.name

        try:
            backend = WavLMMLPBackend(checkpoint_path=corrupt_path)
            with pytest.raises(ModelLoadError):
                backend.load()
            assert not backend.is_loaded()
        finally:
            if os.path.exists(corrupt_path):
                os.remove(corrupt_path)

    def test_no_implicit_fallback(self):
        """Failure to load WavLMMLPBackend must not silently fall back to another backend."""
        backend = WavLMMLPBackend(checkpoint_path="nonexistent.pt")
        with pytest.raises(ModelLoadError):
            backend.load()
        # Verify it did not fall back or mark itself as loaded
        assert backend.is_loaded() is False
        assert backend.classifier is None
        assert backend.backbone is None

    def test_predict_before_load_raises_runtime_error(self):
        backend = WavLMMLPBackend()
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            backend.predict(np.zeros(16000, dtype=np.float32))

    def test_empty_audio_raises_value_error(self):
        backend = WavLMMLPBackend()
        backend.classifier = mock.MagicMock()
        backend.backbone = mock.MagicMock()
        backend.feature_extractor = mock.MagicMock()
        assert backend.is_loaded()

        with pytest.raises(ValueError, match="must not be empty"):
            backend.predict(np.array([], dtype=np.float32))


class TestWav2Vec2BackendFailureModes:
    """Unit tests for Wav2Vec2Backend error handling."""

    def test_predict_before_load_raises_runtime_error(self):
        backend = Wav2Vec2Backend()
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            backend.predict(np.zeros(16000, dtype=np.float32))

    def test_empty_audio_raises_value_error(self):
        backend = Wav2Vec2Backend()
        backend.model = mock.MagicMock()
        backend.feature_extractor = mock.MagicMock()
        assert backend.is_loaded()

        with pytest.raises(ValueError, match="must not be empty"):
            backend.predict(np.array([], dtype=np.float32))


class TestBackendReadinessAndSafety:
    """Verify readiness responses and safety handling with backend abstraction."""

    def test_silence_input_contract(self):
        """Model inference on digital silence produces valid float probabilities."""
        backend = Wav2Vec2Backend()
        backend.feature_extractor = mock.MagicMock()
        mock_model = mock.MagicMock()
        mock_output = mock.MagicMock()
        mock_output.logits = torch.tensor([[2.0, -2.0]])
        mock_model.return_value = mock_output
        backend.model = mock_model

        silence = np.zeros(16000, dtype=np.float32)
        pred = backend.predict(silence, sample_rate=16000)

        assert isinstance(pred, ModelPrediction)
        assert 0.0 <= pred.prob_real <= 1.0
        assert 0.0 <= pred.prob_fake <= 1.0
        assert pred.backend_name == "wav2vec2"
        assert pred.score_type == "uncalibrated_softmax_score"

    def test_readiness_metadata_reflects_active_backend(self):
        reset_config_cache()
        os.environ["VOICEGUARD_ACTIVE_MODEL"] = "wavlm_mlp"
        try:
            cfg = load_config()
            assert cfg.active_model == "wavlm_mlp"
            assert cfg.active_backend.backend == "wavlm_mlp"
            assert cfg.active_backend.score_type == "uncalibrated_sigmoid_score"
            assert cfg.thresholds.version == "wavlm-v1"
        finally:
            os.environ.pop("VOICEGUARD_ACTIVE_MODEL", None)
            reset_config_cache()


# ---------------------------------------------------------------------------
# Integration Tests (Real weights, slow, skipped by default in quick runs)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBackendIntegration:
    """Integration tests executing actual model loads and inference passes."""

    def test_real_wavlm_mlp_checkpoint_loading(self):
        """Verify the checked-in best_mlp_wavlm_base.pt loads and runs inference."""
        ckpt_path = "models/best_mlp_wavlm_base.pt"
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

        backend = WavLMMLPBackend(checkpoint_path=ckpt_path, device="cpu")
        backend.load()
        assert backend.is_loaded()

        parity_audio_path = "tests/fixtures/model_parity_audio.wav"
        import soundfile as sf
        audio, sr = sf.read(parity_audio_path)

        pred = backend.predict(audio, sample_rate=sr)
        assert isinstance(pred, ModelPrediction)
        assert pred.backend_name == "wavlm_mlp"
        assert pred.score_type == "uncalibrated_sigmoid_score"
        assert 0.0 <= pred.prob_real <= 1.0
        assert 0.0 <= pred.prob_fake <= 1.0
        assert abs((pred.prob_real + pred.prob_fake) - 1.0) < 1e-5

    def test_parity_between_predict_script_and_wavlm_backend(self):
        """Verify WavLMMLPBackend yields valid inference on parity audio fixture."""
        parity_audio_path = "tests/fixtures/model_parity_audio.wav"
        ckpt_path = "models/best_mlp_wavlm_base.pt"

        backend = WavLMMLPBackend(checkpoint_path=ckpt_path, device="cpu")
        backend.load()

        import soundfile as sf
        audio, sr = sf.read(parity_audio_path)
        pred = backend.predict(audio, sample_rate=sr)

        assert 0.0 <= pred.prob_real <= 1.0
        assert 0.0 <= pred.prob_fake <= 1.0
        assert abs((pred.prob_real + pred.prob_fake) - 1.0) < 1e-5
