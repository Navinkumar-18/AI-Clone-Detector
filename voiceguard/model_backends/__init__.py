"""
VoiceGuard Model Backends Package.
Provides interchangeable deepfake detection model backends.
"""

from __future__ import annotations

from typing import Any, Optional

from voiceguard.model_backends.base import (
    ModelBackend,
    ModelLoadError,
    ModelPrediction,
)
from voiceguard.model_backends.wav2vec2_backend import Wav2Vec2Backend
from voiceguard.model_backends.wavlm_mlp_backend import DeepfakeMLPHead, WavLMMLPBackend

__all__ = [
    "ModelBackend",
    "ModelLoadError",
    "ModelPrediction",
    "Wav2Vec2Backend",
    "WavLMMLPBackend",
    "DeepfakeMLPHead",
    "create_backend",
]


def create_backend(
    backend_name: str,
    model_id: Optional[str] = None,
    revision: Optional[str] = None,
    device: str = "cpu",
    score_type: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    **kwargs: Any,
) -> ModelBackend:
    """Factory function to instantiate a model backend by name.

    Args:
        backend_name: Name of backend ('wav2vec2' or 'wavlm_mlp').
        model_id: HuggingFace model repo ID (optional override).
        revision: Specific commit hash or tag (optional override).
        device: Torch device string ('cpu' or 'cuda').
        score_type: Score description string (optional override).
        checkpoint_path: Path to checkpoint for MLP-based models.

    Returns:
        An un-loaded ModelBackend instance. Call backend.load() to initialize.

    Raises:
        ValueError: If backend_name is unknown.
    """
    name = backend_name.lower().strip()

    if name == "wav2vec2":
        init_kwargs: dict[str, Any] = {"device": device}
        if model_id:
            init_kwargs["model_id"] = model_id
        if revision:
            init_kwargs["revision"] = revision
        if score_type:
            init_kwargs["score_type"] = score_type
        return Wav2Vec2Backend(**init_kwargs)

    elif name == "wavlm_mlp":
        init_kwargs = {"device": device}
        if model_id:
            init_kwargs["model_id"] = model_id
        if revision:
            init_kwargs["revision"] = revision
        if score_type:
            init_kwargs["score_type"] = score_type
        if checkpoint_path:
            init_kwargs["checkpoint_path"] = checkpoint_path
        return WavLMMLPBackend(**init_kwargs)

    else:
        raise ValueError(
            f"Unknown model backend: '{backend_name}'. "
            f"Supported backends are: 'wav2vec2', 'wavlm_mlp'."
        )
