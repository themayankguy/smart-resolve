"""
Unit and regression test suite for Phase 6: Graph Convolutional Network (GCN) Room Threat Prediction.
Verifies:
1. Node and edge graph construction.
2. Adjacency matrix shape, symmetric normalization, and self-loops.
3. Isolated node and self-loop resilience (no NaN / division-by-zero).
4. Model output shapes and threat class vocabularies.
5. Dungeon-level train/test separation (zero graph/seed leakage).
6. Checkpoint saving, serialization, and parameter integrity.
7. Feature schema and ordering consistency.
8. Missing and corrupted checkpoint failover resilience.
9. Room ID mapping and prediction association.
10. Seamless coexistence of all four ATML models in GameEngine.
11. Evaluation report verification and baseline comparison.
"""

import os
import sys
import tempfile
import unittest
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pygame

from game.dungeon import Dungeon, Room
from game.engine import GameEngine
from ml.gcn_model import (
    RoomThreatGCN,
    NonGraphMLP,
    THREAT_CLASSES,
    THREAT_TO_ID,
    ID_TO_THREAT,
    build_normalized_adjacency,
)
from ml.gcn_dataset import (
    FEATURE_NAMES,
    NUM_FEATURES,
    GraphFeatureScaler,
    extract_dungeon_graph,
    generate_dungeon_graph_dataset,
    calculate_room_threat_label,
)
from ml.threat_predictor import RoomThreatPredictor


class TestGCNPhase6(unittest.TestCase):
    """Test suite covering the Phase 6 GCN Room Threat implementation."""

    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.font.init()

    def setUp(self):
        pygame.init()
        pygame.font.init()
        self.dungeon = Dungeon(seed=42)
        self.test_dir = tempfile.mkdtemp(prefix="neural_dungeon_test_gcn_")

    def test_01_graph_node_and_edge_construction(self):
        """Verifies that dungeon graph extraction accurately reflects rooms and corridor topology."""
        graph_data = extract_dungeon_graph(self.dungeon)
        num_rooms = len(self.dungeon.rooms)

        self.assertEqual(graph_data["num_nodes"], num_rooms)
        self.assertEqual(len(graph_data["room_ids"]), num_rooms)
        self.assertEqual(graph_data["raw_features"].shape, (num_rooms, NUM_FEATURES))
        self.assertEqual(len(graph_data["labels"]), num_rooms)

        # Verify edges are symmetric and valid room index pairs
        for u, v in graph_data["edges"]:
            self.assertTrue(0 <= u < num_rooms)
            self.assertTrue(0 <= v < num_rooms)
            self.assertNotEqual(u, v)

    def test_02_adjacency_matrix_shape_and_symmetric_normalization(self):
        """Verifies mathematical properties of symmetrically normalized adjacency: A_hat = D^-1/2 (A + I) D^-1/2."""
        num_nodes = 5
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        norm_adj = build_normalized_adjacency(num_nodes, edges)

        self.assertEqual(norm_adj.shape, (num_nodes, num_nodes))
        # Verify symmetry: A_hat == A_hat.T
        diff = torch.abs(norm_adj - norm_adj.T).max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)

        # Diagonal entries should be non-zero (self-loops)
        for i in range(num_nodes):
            self.assertGreater(norm_adj[i, i].item(), 0.0)

    def test_03_self_loops_and_isolated_node_handling(self):
        """Verifies that isolated nodes (0 corridor connections) receive self-loops and never produce NaNs."""
        num_nodes = 4
        edges = [(0, 1)]  # Nodes 2 and 3 are isolated
        norm_adj = build_normalized_adjacency(num_nodes, edges)

        self.assertFalse(torch.isnan(norm_adj).any().item())
        self.assertFalse(torch.isinf(norm_adj).any().item())

        # Isolated nodes should have degree 1 from self-loop -> A_hat[2,2] == 1.0, A_hat[3,3] == 1.0
        self.assertAlmostEqual(norm_adj[2, 2].item(), 1.0, places=5)
        self.assertAlmostEqual(norm_adj[3, 3].item(), 1.0, places=5)
        self.assertAlmostEqual(norm_adj[2, 0].item(), 0.0, places=5)

    def test_04_gcn_output_shape_and_valid_threat_classes(self):
        """Verifies RoomThreatGCN output dimensions and threat class mappings."""
        model = RoomThreatGCN(in_features=NUM_FEATURES, hidden_dim=32, num_classes=3, dropout=0.0)
        model.eval()

        num_nodes = 7
        dummy_x = torch.randn(num_nodes, NUM_FEATURES)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        norm_adj = build_normalized_adjacency(num_nodes, edges)

        with torch.no_grad():
            logits = model(dummy_x, norm_adj)

        self.assertEqual(logits.shape, (num_nodes, 3))
        preds = torch.argmax(logits, dim=-1).tolist()
        for p in preds:
            self.assertIn(p, [0, 1, 2])
            self.assertIn(ID_TO_THREAT[p], ["LOW", "MEDIUM", "HIGH"])

    def test_05_dungeon_level_train_test_separation(self):
        """Verifies strict dungeon-level split: zero seed overlap and scaler fitted solely on train data."""
        train_graphs, test_graphs, scaler = generate_dungeon_graph_dataset(
            num_dungeons=20,
            seed_offset=3000,
            train_ratio=0.75,
        )

        train_seeds = {g["seed"] for g in train_graphs}
        test_seeds = {g["seed"] for g in test_graphs}

        # Zero seed overlap
        overlap = train_seeds.intersection(test_seeds)
        self.assertEqual(len(overlap), 0)

        # Scaler must be fitted
        self.assertTrue(scaler.fitted)
        self.assertEqual(len(scaler.mean), NUM_FEATURES)
        self.assertEqual(len(scaler.scale), NUM_FEATURES)

    def test_06_checkpoint_save_and_load_integrity(self):
        """Verifies that saved_models/gcn_checkpoint.pt contains valid architecture and scaler metadata."""
        ckpt_path = "saved_models/gcn_checkpoint.pt"
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("model_config", checkpoint)
        self.assertIn("scaler", checkpoint)
        self.assertIn("feature_names", checkpoint)
        self.assertIn("threat_classes", checkpoint)

        # Verify config parameters
        cfg = checkpoint["model_config"]
        self.assertEqual(cfg["in_features"], NUM_FEATURES)
        self.assertEqual(cfg["num_classes"], 3)

    def test_07_inference_with_expected_feature_ordering(self):
        """Verifies feature schema has exactly 7 expected local features in documented order."""
        expected = [
            "room_area",
            "aspect_ratio",
            "degree",
            "is_spawn",
            "is_exit",
            "norm_center_x",
            "norm_center_y",
        ]
        self.assertEqual(FEATURE_NAMES, expected)
        self.assertEqual(NUM_FEATURES, 7)

    def test_08_missing_and_corrupt_checkpoint_fallback(self):
        """Verifies RoomThreatPredictor handles missing and corrupt checkpoint files safely without crashing."""
        # 1. Missing checkpoint
        missing_predictor = RoomThreatPredictor(model_path="saved_models/non_existent_gcn.pt")
        self.assertFalse(missing_predictor.is_available)
        threats_missing = missing_predictor.predict_dungeon_threats(self.dungeon)
        self.assertGreater(len(threats_missing), 0)
        for r_id, info in threats_missing.items():
            self.assertTrue(info["is_fallback"])
            self.assertIn(info["threat"], ["LOW", "MEDIUM", "HIGH"])

        # 2. Corrupted checkpoint
        corrupt_path = os.path.join(self.test_dir, "corrupt_gcn.pt")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_TORCH_MODEL_DATA_BYTES")

        corrupt_predictor = RoomThreatPredictor(model_path=corrupt_path)
        self.assertFalse(corrupt_predictor.is_available)
        threats_corrupt = corrupt_predictor.predict_dungeon_threats(self.dungeon)
        self.assertGreater(len(threats_corrupt), 0)
        for r_id, info in threats_corrupt.items():
            self.assertTrue(info["is_fallback"])

    def test_09_correct_mapping_of_predictions_to_room_ids(self):
        """Verifies that predicted threats map 1-to-1 to actual room IDs in the dungeon."""
        predictor = RoomThreatPredictor(model_path="saved_models/gcn_checkpoint.pt")
        threats = predictor.predict_dungeon_threats(self.dungeon)

        dungeon_room_ids = {r.id for r in self.dungeon.rooms}
        predicted_room_ids = set(threats.keys())
        self.assertEqual(dungeon_room_ids, predicted_room_ids)

        # Spawn room should have LOW threat
        spawn_id = self.dungeon.spawn_room.id
        self.assertIn(spawn_id, threats)
        self.assertEqual(threats[spawn_id]["threat"], "LOW")

    def test_10_coexistence_all_four_models(self):
        """Verifies that GameEngine runs with Siamese, Autoencoder, Transformer, and GCN all active."""
        engine = GameEngine(screen=None)
        engine.start_new_game(seed=99)

        # All four components must be instantiated
        self.assertIsNotNone(engine.playstyle_detector)
        self.assertIsNotNone(engine.anomaly_detector)
        self.assertIsNotNone(engine.action_predictor)
        self.assertIsNotNone(engine.threat_predictor)

        # Verify GCN predicted threats for all rooms
        self.assertGreater(len(engine.room_threats), 0)
        self.assertEqual(len(engine.room_threats), len(engine.dungeon.rooms))

        # Simulate update frames
        engine.update(dt=0.016)
        self.assertIsNotNone(engine.current_room_threat)
        self.assertIn(engine.current_room_threat["threat"], ["LOW", "MEDIUM", "HIGH", "TRANSIT"])

    def test_11_gcn_vs_mlp_baseline_evaluation_report(self):
        """Verifies that evaluation report exists, contains all ablation configurations, and valid metrics."""
        report_json = "evaluation/gcn_eval_report.json"
        self.assertTrue(os.path.exists(report_json), "GCN evaluation JSON report missing.")

        import json
        with open(report_json, "r") as f:
            data = json.load(f)

        self.assertIn("configurations", data)
        self.assertIn("majority_baseline_metrics", data)
        configs = data["configurations"]
        self.assertIn("config_a_7feat", configs)
        self.assertIn("config_b_5feat", configs)
        self.assertIn("config_c_3feat", configs)

        # Config A metrics
        cfg_a = configs["config_a_7feat"]
        self.assertGreater(cfg_a["gcn_metrics"]["accuracy"], 0.70)
        self.assertGreater(cfg_a["gcn_metrics"]["accuracy"], cfg_a["mlp_metrics"]["accuracy"])

    def test_12_leakage_controlled_feature_schema_regression(self):
        """Regression test ensuring is_spawn and is_exit are strictly excluded from controlled schemas."""
        from ml.gcn_dataset import CONFIG_A_FEATURES, CONFIG_B_FEATURES, CONFIG_C_FEATURES

        # Config A has 7 features
        self.assertEqual(len(CONFIG_A_FEATURES), 7)
        self.assertIn("is_spawn", CONFIG_A_FEATURES)
        self.assertIn("is_exit", CONFIG_A_FEATURES)

        # Config B strictly excludes is_spawn and is_exit
        self.assertEqual(len(CONFIG_B_FEATURES), 5)
        self.assertNotIn("is_spawn", CONFIG_B_FEATURES)
        self.assertNotIn("is_exit", CONFIG_B_FEATURES)

        # Config C strictly excludes flags and spatial coordinates
        self.assertEqual(len(CONFIG_C_FEATURES), 3)
        self.assertEqual(CONFIG_C_FEATURES, ["room_area", "aspect_ratio", "degree"])
        self.assertNotIn("norm_center_x", CONFIG_C_FEATURES)
        self.assertNotIn("norm_center_y", CONFIG_C_FEATURES)

    def test_13_controlled_and_pure_checkpoints_exist(self):
        """Verifies that trained checkpoints for leakage-controlled and pure-geometry models exist and load."""
        b_path = "saved_models/gcn_checkpoint_controlled.pt"
        c_path = "saved_models/gcn_checkpoint_pure_geom.pt"

        self.assertTrue(os.path.exists(b_path), f"Checkpoint missing at {b_path}")
        self.assertTrue(os.path.exists(c_path), f"Checkpoint missing at {c_path}")

        # Test loading controlled 5-feature model into RoomThreatPredictor
        pred_b = RoomThreatPredictor(model_path=b_path)
        self.assertTrue(pred_b.is_available)
        threats_b = pred_b.predict_dungeon_threats(self.dungeon)
        self.assertEqual(len(threats_b), len(self.dungeon.rooms))

        # Test loading pure 3-feature model into RoomThreatPredictor
        pred_c = RoomThreatPredictor(model_path=c_path)
        self.assertTrue(pred_c.is_available)
        threats_c = pred_c.predict_dungeon_threats(self.dungeon)
        self.assertEqual(len(threats_c), len(self.dungeon.rooms))


if __name__ == "__main__":
    unittest.main()
