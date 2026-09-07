"""
Wav2Vec2 deepfake detection backend.
Uses HuggingFace AutoModelForAudioClassification.
"""

from __future__ import annotations

import time
from typing import Optional
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from voiceguard.model_backends.base import ModelBackend, ModelLoadError, ModelPrediction


class Wav2Vec2Backend(ModelBackend):
    """Production backend using Wav2Vec2 audio classification."""

    def __init__(
        self,
        backend_name: str = "wav2vec2",
        model_id: str = "garystafford/wav2vec2-deepfake-voice-detector",
        revision: Optional[str] = "c66306024a7ede0be291e9c4558b37634782dc4e",
        device: str = "cpu",
        score_type: str = "uncalibrated_softmax_score",
    ) -> None:
        super().__init__(
            backend_name=backend_name,
            model_id=model_id,
            revision=revision,
            device=device,
            score_type=score_type,
        )
        self.feature_extractor: Optional[AutoFeatureExtractor] = None
        self.model: Optional[AutoModelForAudioClassification] = None

    def load(self) -> None:
        """Load feature extractor and classification model.

        Raises:
            ModelLoadError: If loading fails.
        """
        try:
            kwargs = {}
            if self.revision:
                kwargs["revision"] = self.revision

            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_id, **kwargs
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_id, **kwargs
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            self.feature_extractor = None
            self.model = None
            raise ModelLoadError(
                f"Failed to load Wav2Vec2 model '{self.model_id}' (revision: {self.revision}): {exc}"
            ) from exc

    def is_loaded(self) -> bool:
        """Return True if model and feature extractor are available."""
        return self.model is not None and self.feature_extractor is not None

    def predict(self, audio: np.ndarray, sample_rate: int = 16_000) -> ModelPrediction:
        """Run synchronous inference on audio.

        Args:
            audio: 1D float32 numpy array.
            sample_rate: Audio sample rate in Hz.

        Returns:
            ModelPrediction with softmax probabilities and timing.
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Call load() before predict().")
        if audio is None or len(audio) == 0:
            raise ValueError("Audio waveform must not be empty.")

        t0 = time.perf_counter()

        audio = np.asarray(audio, dtype=np.float32)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        prob_real = float(probs[0].item())
        prob_fake = float(probs[1].item())
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ModelPrediction(
            prob_real=round(prob_real, 6),
            prob_fake=round(prob_fake, 6),
            raw_score=round(prob_fake, 6),
            score_type=self.score_type,
            backend_name=self.backend_name,
            inference_time_ms=round(latency_ms, 2),
        )
