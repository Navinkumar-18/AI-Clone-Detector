"""
Base abstractions for VoiceGuard model backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


class ModelLoadError(Exception):
    """Raised when a model fails to load (missing files, network errors, corruption)."""
    pass


@dataclass(frozen=True)
class ModelPrediction:
    """Standardized output structure for all model backends."""
    prob_real: float
    prob_fake: float
    raw_score: float
    score_type: str  # "uncalibrated_softmax_score" | "uncalibrated_sigmoid_score"
    backend_name: str
    inference_time_ms: float


class ModelBackend(ABC):
    """Abstract base class for audio deepfake detection model backends."""

    def __init__(
        self,
        backend_name: str,
        model_id: str,
        revision: Optional[str] = None,
        device: str = "cpu",
        score_type: str = "uncalibrated_softmax_score",
    ) -> None:
        self.backend_name = backend_name
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.score_type = score_type

    @abstractmethod
    def load(self) -> None:
        """Load the model and feature extractor into memory.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model and feature extractor are loaded and ready."""
        pass

    @abstractmethod
    def predict(self, audio: np.ndarray, sample_rate: int = 16_000) -> ModelPrediction:
        """Run inference on an audio waveform.

        Args:
            audio: 1D float32 numpy array of audio samples normalized to [-1.0, 1.0].
            sample_rate: Sample rate of the audio (default: 16,000 Hz).

        Returns:
            ModelPrediction containing probabilities, raw score, and timing.

        Raises:
            RuntimeError: If the model is not loaded before inference.
            ValueError: If audio is invalid or empty.
        """
        pass
