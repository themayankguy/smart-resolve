"""
Runtime inference coordinator for Next-Action Prediction (Phase 5).
Loads the trained ActionTransformer checkpoint once, extracts recent action token sequences
from gameplay telemetry, and predicts the player's anticipated next combat action.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

from ml.transformer_model import ActionTransformer, ACTION_VOCAB, ACTION_NAMES
from ml.dataset import extract_session_action_sequence


class ActionPredictor:
    """
    Safely executes next-action prediction at controlled gameplay intervals.
    Provides failover to nominal baseline if weights are missing or corrupt.
    """

    CONFIDENCE_THRESHOLD = 0.45  # Below this threshold, predictions are disregarded as unconfident

    def __init__(self, model_path: str = "saved_models/transformer_checkpoint.pt"):
        self.model_path = model_path
        self.model: ActionTransformer | None = None
        self.config: dict = {}
        self.max_seq_len: int = 16
        self.is_loaded: bool = False

        self._load_model()

    @property
    def is_available(self) -> bool:
        """Returns True if the model weights are loaded and ready for inference."""
        return self.is_loaded and self.model is not None

    def _load_model(self) -> bool:
        """Safely loads model weights and configuration."""
        if not os.path.exists(self.model_path):
            self.is_loaded = False
            return False

        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            self.config = checkpoint.get("config", {})
            self.max_seq_len = self.config.get("max_seq_len", 16)

            self.model = ActionTransformer(
                vocab_size=self.config.get("vocab_size", len(ACTION_VOCAB)),
                d_model=self.config.get("d_model", 32),
                nhead=self.config.get("nhead", 4),
                num_layers=self.config.get("num_layers", 2),
                dim_feedforward=self.config.get("dim_feedforward", 64),
                max_seq_len=self.max_seq_len,
                dropout=self.config.get("dropout", 0.1),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self.is_loaded = True
            return True
        except Exception:
            self.is_loaded = False
            self.model = None
            return False

    def predict_next_action(self, recent_events: list[dict]) -> tuple[str, float]:
        """
        Extracts recent chronological action sequence and forecasts the next action.
        
        Returns:
            tuple (predicted_action_name, confidence)
            where:
              predicted_action_name: str in ("IDLE", "MOVE", "ATTACK", "DODGE", "HURT", "NONE")
              confidence: float in [0.0, 1.0] representing normalized Softmax output.
              
        CALIBRATION DISCLAIMER:
        The confidence score is a normalized Softmax score across action classes.
        It is NOT a calibrated Bayesian probability and should not be interpreted
        as a verified empirical forecast likelihood without formal Platt/temperature scaling.
        """
        if not self.is_loaded or self.model is None or len(recent_events) < 4:
            return ("NONE", 0.0)

        try:
            # 1. Extract chronological action sequence from recent events
            tokens = extract_session_action_sequence(recent_events)
            if not tokens:
                return ("NONE", 0.0)

            # 2. Query model
            _, action_name, confidence, _ = self.model.predict_next(tokens)

            # 3. Confidence threshold filter
            if confidence < self.CONFIDENCE_THRESHOLD:
                return ("NONE", confidence)

            return (action_name, round(confidence, 3))

        except Exception:
            # Zero-crash fallback
            return ("NONE", 0.0)
