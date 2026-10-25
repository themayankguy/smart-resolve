"""
Room Threat Predictor for Neural Dungeon (Phase 6).
Loads trained RoomThreatGCN checkpoint, builds dungeon graph representations,
executes spectral graph convolution inference, and associates predictions with room IDs.
Includes failover resilience against missing or corrupted model weights.
"""

import os
import logging
import torch
import torch.nn.functional as F

from game.dungeon import Dungeon
from ml.gcn_model import (
    RoomThreatGCN,
    THREAT_CLASSES,
    THREAT_TO_ID,
    ID_TO_THREAT,
    build_normalized_adjacency,
)
from ml.gcn_dataset import (
    FEATURE_NAMES,
    GraphFeatureScaler,
    extract_dungeon_graph,
)

logger = logging.getLogger("RoomThreatPredictor")


class RoomThreatPredictor:
    """
    Dedicated runtime coordinator for GCN-based room threat classification.
    Executes graph inference once upon floor generation and caches predictions.
    """

    def __init__(self, model_path: str = "saved_models/gcn_checkpoint_controlled.pt"):
        if model_path == "saved_models/gcn_checkpoint_controlled.pt" and not os.path.exists(model_path):
            if os.path.exists("saved_models/gcn_checkpoint.pt"):
                model_path = "saved_models/gcn_checkpoint.pt"
        self.model_path = model_path
        self.model: RoomThreatGCN | None = None
        self.scaler: GraphFeatureScaler | None = None
        self.feature_names: list[str] | None = None
        self.config_name: str = "unknown"
        self.is_available: bool = False
        self.cached_threats: dict[int, dict] = {}

        self._load_model()

    def _load_model(self) -> None:
        """Loads GCN checkpoint and feature scaler with failover protection."""
        if not os.path.exists(self.model_path):
            logger.warning("GCN checkpoint '%s' not found. Using heuristic fallback.", self.model_path)
            self.is_available = False
            return

        try:
            checkpoint = torch.load(self.model_path, map_location="cpu")
            cfg = checkpoint["model_config"]
            scaler_dict = checkpoint["scaler"]

            self.scaler = GraphFeatureScaler.from_dict(scaler_dict)
            self.feature_names = checkpoint.get("feature_names", None)
            self.config_name = checkpoint.get("config_name", "controlled_5feat" if len(self.feature_names or []) == 5 else "original_7feat")
            self.model = RoomThreatGCN(
                in_features=cfg["in_features"],
                hidden_dim=cfg["hidden_dim"],
                num_classes=cfg["num_classes"],
                dropout=cfg.get("dropout", 0.1),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.is_available = True
            logger.info("RoomThreatGCN checkpoint loaded successfully from '%s' (%s).", self.model_path, self.config_name)
        except Exception as e:
            logger.error("Failed to load GCN checkpoint from '%s': %s. Using heuristic fallback.", self.model_path, e)
            self.model = None
            self.scaler = None
            self.is_available = False

    def predict_dungeon_threats(self, dungeon: Dungeon) -> dict[int, dict]:
        """
        Extracts dungeon room graph, executes GCN message-passing inference,
        and returns room-ID indexed threat predictions.

        Args:
            dungeon: Active Dungeon instance with generated rooms and corridors.

        Returns:
            dict mapping room_id (int) -> {
                "threat": "LOW" | "MEDIUM" | "HIGH",
                "confidence": float,
                "probabilities": dict[str, float],
                "is_fallback": bool
            }
        """
        rooms = dungeon.rooms
        if not rooms:
            self.cached_threats = {}
            return self.cached_threats

        # Fallback if model is unavailable
        if not self.is_available or self.model is None or self.scaler is None:
            fallback_threats = {}
            for r in rooms:
                if r.room_type == "spawn":
                    t_name = "LOW"
                    conf = 1.0
                elif r.room_type == "exit":
                    t_name = "HIGH"
                    conf = 0.8
                else:
                    t_name = "MEDIUM"
                    conf = 0.5

                fallback_threats[r.id] = {
                    "threat": t_name,
                    "confidence": conf,
                    "probabilities": {name: (1.0 if name == t_name else 0.0) for name in THREAT_CLASSES},
                    "is_fallback": True,
                }
            self.cached_threats = fallback_threats
            return self.cached_threats

        try:
            # 1. Extract graph structure
            graph_data = extract_dungeon_graph(dungeon, feature_names=self.feature_names)
            raw_features = graph_data["raw_features"]
            room_ids = graph_data["room_ids"]
            num_nodes = graph_data["num_nodes"]
            edges = graph_data["edges"]

            # 2. Standardize features
            norm_features = self.scaler.transform(raw_features)
            x = torch.from_numpy(norm_features).float()

            # 3. Build normalized adjacency A_hat with self-loops
            norm_adj = build_normalized_adjacency(num_nodes, edges)

            # 4. GCN Inference
            with torch.no_grad():
                logits = self.model(x, norm_adj)
                probs = F.softmax(logits, dim=-1)
                confidences, pred_classes = torch.max(probs, dim=-1)

            results = {}
            for idx, r_id in enumerate(room_ids):
                cid = pred_classes[idx].item()
                conf = confidences[idx].item()
                cname = ID_TO_THREAT.get(cid, "MEDIUM")

                prob_dict = {
                    name: round(float(probs[idx, THREAT_TO_ID[name]].item()), 4)
                    for name in THREAT_CLASSES
                }

                results[r_id] = {
                    "threat": cname,
                    "confidence": round(float(conf), 4),
                    "probabilities": prob_dict,
                    "is_fallback": False,
                }

            self.cached_threats = results
            return results

        except Exception as e:
            logger.error("Error during GCN dungeon threat inference: %s. Using heuristic fallback.", e)
            fallback_threats = {}
            for r in rooms:
                fallback_threats[r.id] = {
                    "threat": "LOW" if r.room_type == "spawn" else "MEDIUM",
                    "confidence": 0.5,
                    "probabilities": {name: (1.0 / 3.0) for name in THREAT_CLASSES},
                    "is_fallback": True,
                }
            self.cached_threats = fallback_threats
            return self.cached_threats

    def get_threat_for_room(self, room_id: int) -> dict:
        """Retrieves cached threat prediction for a room ID with safe default."""
        return self.cached_threats.get(
            room_id,
            {
                "threat": "UNKNOWN",
                "confidence": 0.0,
                "probabilities": {name: 0.0 for name in THREAT_CLASSES},
                "is_fallback": True,
            },
        )
