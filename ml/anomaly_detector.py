"""
Inference coordinator for Combat Anomaly Detection (Phase 4).
Loads the trained CombatAutoencoder checkpoint once, evaluates live combat windows
at controlled gameplay intervals, and flags unusual combat patterns relative to reference training data.

CRITICAL ATML NOTE:
This module performs statistical anomaly detection on feature distributions.
It does NOT detect cheating or malicious behavior, and must never automatically
punish or damage the player.
"""

import os
import numpy as np
import torch

from ml.dataset import NUMERICAL_FEATURE_NAMES, FeatureScaler
from ml.autoencoder import CombatAutoencoder
from ml.features import extract_window_features


class CombatAnomalyDetector:
    """
    Safely executes autoencoder reconstruction error inference at controlled gameplay intervals.
    Guarantees zero-crash failover if the model is absent or corrupted.
    """

    def __init__(self, model_path: str = "saved_models/autoencoder_checkpoint.pt"):
        self.model_path = model_path
        self.model: CombatAutoencoder | None = None
        self.scaler: FeatureScaler | None = None
        self.threshold: float = 0.0
        self.feature_names: list[str] = list(NUMERICAL_FEATURE_NAMES)
        self.is_loaded = False

        self._load_model()

    @property
    def is_available(self) -> bool:
        """Returns True if model weights and anomaly threshold are ready for inference."""
        return self.is_loaded

    def _load_model(self) -> bool:
        """Safely loads model weights, scaler, feature schema, and validated threshold."""
        if not os.path.exists(self.model_path):
            self.is_loaded = False
            return False

        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            self.scaler = FeatureScaler.from_dict(checkpoint["scaler"])
            self.threshold = float(checkpoint.get("threshold", 0.10))

            in_features = checkpoint.get("in_features", 11)
            latent_dim = checkpoint.get("latent_dim", 4)
            hidden_dim = checkpoint.get("hidden_dim", 8)

            self.model = CombatAutoencoder(
                in_features=in_features,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self.feature_names = checkpoint.get("feature_names", list(NUMERICAL_FEATURE_NAMES))
            if len(self.feature_names) != in_features:
                raise ValueError(f"Feature count mismatch: {len(self.feature_names)} vs {in_features}")

            self.is_loaded = True
            return True
        except Exception:
            # Safe failover: game runs without anomaly detector
            self.is_loaded = False
            self.model = None
            self.scaler = None
            self.threshold = 0.0
            return False

    def detect_anomaly(self, recent_events: list[dict]) -> tuple[bool, float, float]:
        """
        Calculates reconstruction error on the latest window of recent gameplay events.
        
        Returns:
            tuple (is_anomaly, reconstruction_error, threshold)
            where:
              is_anomaly: bool (True if reconstruction_error > threshold)
              reconstruction_error: float (non-negative MSE reconstruction error)
              threshold: float (validated statistical decision boundary)
        """
        if not self.is_loaded or self.model is None or self.scaler is None or len(recent_events) < 4:
            return (False, 0.0, self.threshold)

        try:
            # 1. Extract feature window from recent events
            windows = extract_window_features(recent_events, window_size_sec=5.0, stride_sec=5.0)
            if not windows:
                return (False, 0.0, self.threshold)

            latest_window = windows[-1]
            raw_vector = np.array([float(latest_window.get(col, 0.0)) for col in self.feature_names], dtype=np.float32).reshape(1, -1)
            raw_vector = np.nan_to_num(raw_vector, nan=0.0, posinf=999.0, neginf=-999.0)

            # 2. Preprocess with training scaler
            norm_vector = self.scaler.transform(raw_vector)
            t_input = torch.tensor(norm_vector, dtype=torch.float32)

            # 3. Compute reconstruction error
            error_tensor = self.model.compute_reconstruction_error(t_input)
            recon_error = float(round(float(error_tensor.item()), 4))

            is_anomaly = recon_error > self.threshold
            return (is_anomaly, recon_error, self.threshold)

        except Exception:
            # Zero-crash guarantee: return nominal status
            return (False, 0.0, self.threshold)
