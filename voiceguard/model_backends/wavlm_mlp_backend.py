"""
WavLM-base + MLP classifier backend.
Uses HuggingFace WavLMModel backbone with custom 2-layer MLP classifier head.
"""

from __future__ import annotations

import os
import time
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, WavLMModel

from voiceguard.model_backends.base import ModelBackend, ModelLoadError, ModelPrediction


class DeepfakeMLPHead(nn.Module):
    """
    Lightweight 2-Layer MLP Classifier Head.
    Architecture: Linear(input_dim -> hidden_dim) -> ReLU -> Dropout(0.3) -> Linear(hidden_dim -> 1)
    Outputs raw scalar logit (unbounded). Positive = bonafide, negative = spoof.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class WavLMMLPBackend(ModelBackend):
    """Backend combining frozen WavLM-base backbone with fine-tuned MLP head."""

    def __init__(
        self,
        backend_name: str = "wavlm_mlp",
        model_id: str = "microsoft/wavlm-base",
        revision: Optional[str] = "efa81aae7ff777e464159e0f877d54eac5b84f81",
        checkpoint_path: str = "models/best_mlp_wavlm_base.pt",
        device: str = "cpu",
        score_type: str = "uncalibrated_sigmoid_score",
    ) -> None:
        super().__init__(
            backend_name=backend_name,
            model_id=model_id,
            revision=revision,
            device=device,
            score_type=score_type,
        )
        self.checkpoint_path = checkpoint_path
        self.feature_extractor: Optional[AutoFeatureExtractor] = None
        self.backbone: Optional[WavLMModel] = None
        self.classifier: Optional[DeepfakeMLPHead] = None
        self.saved_threshold: Optional[float] = None

    def load(self) -> None:
        """Load WavLM backbone and MLP checkpoint.

        Raises:
            ModelLoadError: If backbone or checkpoint loading fails.
        """
        if not os.path.exists(self.checkpoint_path):
            raise ModelLoadError(
                f"MLP checkpoint not found at '{self.checkpoint_path}'. "
                "Ensure models/best_mlp_wavlm_base.pt exists."
            )

        try:
            # 1. Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if "model_state_dict" not in checkpoint:
                raise KeyError("Checkpoint missing required 'model_state_dict' key.")

            input_dim = checkpoint.get("input_dim", 768)
            hidden_dim = checkpoint.get("hidden_dim", 128)
            self.saved_threshold = checkpoint.get("threshold", None)

            # 2. Instantiate MLP head
            self.classifier = DeepfakeMLPHead(input_dim=input_dim, hidden_dim=hidden_dim)
            self.classifier.load_state_dict(checkpoint["model_state_dict"])
            self.classifier.to(self.device)
            self.classifier.eval()

            # 3. Load feature extractor & frozen WavLM backbone
            kwargs = {}
            if self.revision:
                kwargs["revision"] = self.revision

            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_id, **kwargs
            )
            self.backbone = WavLMModel.from_pretrained(self.model_id, **kwargs)
            self.backbone.to(self.device)
            self.backbone.eval()

            # Freeze backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

        except Exception as exc:
            self.feature_extractor = None
            self.backbone = None
            self.classifier = None
            raise ModelLoadError(
                f"Failed to load WavLM+MLP backend (backbone: {self.model_id}, "
                f"checkpoint: {self.checkpoint_path}): {exc}"
            ) from exc

    def is_loaded(self) -> bool:
        """Return True if all components are loaded and ready."""
        return (
            self.feature_extractor is not None
            and self.backbone is not None
            and self.classifier is not None
        )

    def predict(self, audio: np.ndarray, sample_rate: int = 16_000) -> ModelPrediction:
        """Run synchronous inference on audio.

        Args:
            audio: 1D float32 numpy array.
            sample_rate: Audio sample rate in Hz.

        Returns:
            ModelPrediction with sigmoid-derived probabilities and timing.
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
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.backbone(input_values)
            hidden_states = outputs.last_hidden_state  # shape: (1, seq_len, 768)
            mean_pooled = torch.mean(hidden_states, dim=1)  # shape: (1, 768)
            logit = self.classifier(mean_pooled)  # shape: (1,)
            bonafide_prob = float(torch.sigmoid(logit).item())

        prob_real = bonafide_prob
        prob_fake = 1.0 - bonafide_prob
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ModelPrediction(
            prob_real=round(prob_real, 6),
            prob_fake=round(prob_fake, 6),
            raw_score=round(prob_fake, 6),
            score_type=self.score_type,
            backend_name=self.backend_name,
            inference_time_ms=round(latency_ms, 2),
        )
