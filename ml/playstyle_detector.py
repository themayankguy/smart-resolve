"""
Inference coordinator for Playstyle Similarity (Phase 3).
Loads the trained Siamese checkpoint once, compares live combat windows against
reference archetype prototypes, and provides safe, low-confidence fallback.
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F

from ml.dataset import NUMERICAL_FEATURE_NAMES, FeatureScaler
from ml.siamese import SiameseNet
from ml.features import extract_window_features


class PlaystyleDetector:
    """
    Safely executes playstyle similarity inference at controlled gameplay intervals.
    Guarantees zero-crash failover to heuristic baseline if model is absent or errors occur.
    """

    # Fallback threshold: Similarity score = 1 / (1 + distance).
    # With Contrastive Loss margin m = 1.0, similar pairs have distance <= 1.0 (Similarity >= 0.50).
    # A threshold of 0.45 corresponds to distance d_min <= 1.22 (slightly beyond the margin).
    # If the closest archetype prototype is further than 1.22 distance units away in embedding
    # space, the confidence is deemed insufficient and the enemy AI defaults to neutral "Balanced".
    CONFIDENCE_THRESHOLD = 0.45

    def __init__(self, model_path: str = "saved_models/siamese_checkpoint.pt"):
        self.model_path = model_path
        self.model: SiameseNet | None = None
        self.scaler: FeatureScaler | None = None
        self.class_prototypes: dict[str, torch.Tensor] = {}
        self.feature_names: list[str] = list(NUMERICAL_FEATURE_NAMES)
        self.is_loaded = False

        self._load_model()

    @property
    def is_available(self) -> bool:
        """Returns True if model weights and prototypes are ready for inference."""
        return self.is_loaded

    def _load_model(self) -> bool:
        """Safely loads model weights, scaler, feature schema, and prototype centroids."""
        if not os.path.exists(self.model_path):
            self.is_loaded = False
            return False

        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            self.scaler = FeatureScaler.from_dict(checkpoint["scaler"])

            in_features = checkpoint.get("in_features", 11)
            embedding_dim = checkpoint.get("embedding_dim", 8)

            self.model = SiameseNet(
                in_features=in_features,
                embedding_dim=embedding_dim,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            # Load canonical feature ordering saved during training
            self.feature_names = checkpoint.get("feature_names", list(NUMERICAL_FEATURE_NAMES))
            if len(self.feature_names) != in_features:
                raise ValueError(f"Feature names count ({len(self.feature_names)}) mismatch with in_features ({in_features})")

            # Load archetype prototype embeddings
            self.class_prototypes = {}
            for name, vec in checkpoint.get("class_prototypes", {}).items():
                self.class_prototypes[name] = torch.tensor(vec, dtype=torch.float32)

            self.is_loaded = True
            return True
        except Exception:
            # Safe failover
            self.is_loaded = False
            self.model = None
            self.scaler = None
            self.class_prototypes = {}
            return False

    def predict_playstyle(self, recent_events: list[dict]) -> tuple[str, float]:
        """
        Infers player combat style from recent events using the frozen Siamese encoder.
        
        Returns:
            tuple (predicted_style, similarity_score)
            where predicted_style is in ('Aggressive', 'Evasive', 'Balanced')
            and similarity_score is in [0.0, 1.0].
            
        MATHEMATICAL DEFINITION & NON-PROBABILITY DISCLAIMER:
        The similarity score is computed as:
            similarity = 1 / (1 + min_k ||z - c_k||_2)
        where z is the 8D Siamese embedding of the input window and c_k is the
        centroid prototype of style k in embedding space.
        
        CRITICAL ATML RESEARCH NOTE:
        This similarity score is an uncalibrated metric distance score, NOT a
        calibrated Bayesian posterior probability. It does not integrate prior
        class distributions, does not sum to 1 across styles, and must not be
        described or treated as a formal statistical probability.
        """
        # Fallback if model is not loaded or too few events
        if not self.is_loaded or self.model is None or self.scaler is None or not self.class_prototypes or len(recent_events) < 4:
            return ("Balanced", 0.0)

        try:
            # 1. Extract feature window from recent events
            windows = extract_window_features(recent_events, window_size_sec=5.0, stride_sec=5.0)
            if not windows:
                return ("Balanced", 0.0)

            latest_window = windows[-1]
            raw_vector = np.array([float(latest_window.get(col, 0.0)) for col in self.feature_names], dtype=np.float32).reshape(1, -1)

            # Sanitize any non-finite entries
            raw_vector = np.nan_to_num(raw_vector, nan=0.0, posinf=999.0, neginf=-999.0)

            # 2. Scale with exact training parameters
            norm_vector = self.scaler.transform(raw_vector)
            t_input = torch.tensor(norm_vector, dtype=torch.float32)

            # 3. Pass through Siamese shared encoder
            with torch.no_grad():
                z = self.model.forward_one(t_input)  # [1, 8]

            # 4. Compare Euclidean distance against archetype prototypes
            best_style = "Balanced"
            min_dist = float("inf")

            for style_name, proto_tensor in self.class_prototypes.items():
                dist = float(F.pairwise_distance(z, proto_tensor.unsqueeze(0)).item())
                if dist < min_dist:
                    min_dist = dist
                    best_style = style_name

            # Similarity measure (uncalibrated metric distance function)
            similarity = round(1.0 / (1.0 + max(0.0, min_dist)), 3)

            # Low confidence threshold check
            if similarity < self.CONFIDENCE_THRESHOLD:
                return ("Balanced", similarity)

            return (best_style, similarity)

        except Exception:
            # Zero-crash guarantee: return default neutral state
            return ("Balanced", 0.0)
