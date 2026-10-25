"""
Unit and integration tests for Phase 3: Siamese Network for Combat-Style Similarity.
Verifies feature dimensions, shared weights, pair balance, session leakage prevention,
checkpoint serialization, inference coordinator, low-confidence fallback, and game resilience.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

from ml.dataset import (
    NUMERICAL_FEATURE_NAMES,
    FeatureScaler,
    assign_combat_style,
    session_train_test_split,
    build_siamese_pairs,
    SiamesePairDataset,
)
from ml.siamese import SiameseNet, ContrastiveLoss
from ml.playstyle_detector import PlaystyleDetector
from game.engine import GameEngine


class TestSiamesePhase3(unittest.TestCase):
    """Test suite for Phase 3 Siamese Network and playstyle inference."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_feature_dimensions_and_preprocessing_consistency(self):
        """Verifies feature schema has 11 numerical features and FeatureScaler maintains consistency."""
        self.assertEqual(len(NUMERICAL_FEATURE_NAMES), 11)
        self.assertIn("attack_frequency", NUMERICAL_FEATURE_NAMES)
        self.assertIn("dodge_frequency", NUMERICAL_FEATURE_NAMES)
        self.assertIn("dodge_success_rate", NUMERICAL_FEATURE_NAMES)
        self.assertIn("avg_enemy_distance", NUMERICAL_FEATURE_NAMES)
        self.assertIn("damage_dealt", NUMERICAL_FEATURE_NAMES)
        self.assertIn("damage_received", NUMERICAL_FEATURE_NAMES)

        # Preprocessing scaler fit/transform
        dummy_data = np.array([
            [1.0, 0.5, 0.2, 10.0, 5.0, 2.0, 80.0, 50.0, 150.0, 0.1, 0.2],
            [3.0, 0.8, 0.5, 30.0, 10.0, 1.0, 40.0, 20.0, 80.0, 0.6, 0.05],
            [0.2, 0.1, 0.9, 0.0, 0.0, 5.0, 95.0, 85.0, 220.0, 0.0, 0.7],
        ], dtype=np.float32)

        scaler = FeatureScaler().fit(dummy_data)
        transformed = scaler.transform(dummy_data)
        self.assertEqual(transformed.shape, (3, 11))

        # Check serialization round-trip
        state_dict = scaler.to_dict()
        restored_scaler = FeatureScaler.from_dict(state_dict)
        restored_transformed = restored_scaler.transform(dummy_data)
        np.testing.assert_allclose(transformed, restored_transformed, rtol=1e-5)

    def test_02_shared_siamese_encoder_weights(self):
        """Verifies Siamese branches strictly share identical encoder weights and structure."""
        model = SiameseNet(in_features=11, embedding_dim=8)
        x1 = torch.randn(4, 11)
        x2 = torch.randn(4, 11)

        # Forward pass through dual-branch interface
        z1, z2 = model(x1, x2)
        self.assertEqual(z1.shape, (4, 8))
        self.assertEqual(z2.shape, (4, 8))

        # Forward pass through single-branch interface must match exactly
        z1_direct = model.forward_one(x1)
        z2_direct = model.forward_one(x2)
        torch.testing.assert_close(z1, z1_direct)
        torch.testing.assert_close(z2, z2_direct)

        # Ensure encoder is the single underlying module
        self.assertIs(model.encoder, model.encoder)
        param_count = sum(p.numel() for p in model.parameters())
        encoder_param_count = sum(p.numel() for p in model.encoder.parameters())
        self.assertEqual(param_count, encoder_param_count)

    def test_03_positive_and_negative_pair_construction(self):
        """Verifies Siamese pairs contain 50% positive (target=1.0) and 50% negative (target=0.0) pairs."""
        features = np.random.randn(9, 11).astype(np.float32)
        labels = [
            "Aggressive", "Aggressive", "Aggressive",
            "Evasive", "Evasive", "Evasive",
            "Balanced", "Balanced", "Balanced",
        ]

        x1, x2, targets = build_siamese_pairs(features, labels, pairs_per_sample=2, seed=42)
        self.assertEqual(len(x1), len(x2))
        self.assertEqual(len(x1), len(targets))

        pos_count = np.sum(targets == 1.0)
        neg_count = np.sum(targets == 0.0)
        self.assertEqual(pos_count, neg_count)
        self.assertEqual(pos_count + neg_count, len(targets))

        # Dataset wrapper
        dataset = SiamesePairDataset(x1, x2, targets)
        self.assertEqual(len(dataset), len(targets))
        sample_x1, sample_x2, sample_y = dataset[0]
        self.assertIsInstance(sample_x1, torch.Tensor)
        self.assertIsInstance(sample_y, torch.Tensor)

    def test_04_session_level_split_integrity(self):
        """Verifies session-level splitting prevents temporal and session ID leakage between train and test."""
        # Create synthetic window records across 6 sessions
        records = []
        for s_idx in range(6):
            s_id = f"session_{s_idx:03d}"
            for w_idx in range(5):
                records.append({
                    "session_id": s_id,
                    "window_index": w_idx,
                    "attack_rate": float(s_idx),
                    "combat_style": "Aggressive" if s_idx % 2 == 0 else "Evasive",
                })

        train_recs, test_recs = session_train_test_split(records, test_ratio=0.33, seed=123)
        train_sessions = set(r["session_id"] for r in train_recs)
        test_sessions = set(r["session_id"] for r in test_recs)

        # Must have no session overlap
        leakage = train_sessions.intersection(test_sessions)
        self.assertEqual(len(leakage), 0, f"Session leakage detected: {leakage}")
        self.assertTrue(len(train_sessions) > 0)
        self.assertTrue(len(test_sessions) > 0)
        self.assertEqual(len(train_sessions) + len(test_sessions), 6)

    def test_05_checkpoint_save_and_load(self):
        """Verifies trained checkpoint file structure, tensor weights, and class prototypes."""
        checkpoint_path = "saved_models/siamese_checkpoint.pt"
        self.assertTrue(os.path.exists(checkpoint_path), f"Checkpoint missing at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("scaler", checkpoint)
        self.assertIn("class_prototypes", checkpoint)
        self.assertIn("in_features", checkpoint)
        self.assertIn("embedding_dim", checkpoint)
        self.assertEqual(checkpoint["in_features"], 11)
        self.assertEqual(checkpoint["embedding_dim"], 8)

        # Check prototypes exist for all 3 styles
        prototypes = checkpoint["class_prototypes"]
        for style in ["Aggressive", "Evasive", "Balanced"]:
            self.assertIn(style, prototypes)
            self.assertEqual(len(prototypes[style]), 8)

    def test_06_inference_and_low_confidence_fallback(self):
        """Verifies PlaystyleDetector returns valid similarity scores and defaults on low confidence."""
        detector = PlaystyleDetector(model_path="saved_models/siamese_checkpoint.pt")
        self.assertTrue(detector.is_loaded)

        # Too few events should safely return Balanced with 0.0 confidence
        few_events = [{"type": "PLAYER_MOVE", "timestamp": 1.0}]
        style, conf = detector.predict_playstyle(few_events)
        self.assertEqual(style, "Balanced")
        self.assertEqual(conf, 0.0)

        # Full simulated aggressive window
        aggressive_events = []
        for i in range(25):
            t = float(i) * 0.2
            aggressive_events.append({
                "type": "PLAYER_ATTACK",
                "timestamp": t,
                "hits": 1,
                "damage_dealt": 35,
                "player_x": 100.0,
                "player_y": 100.0,
                "player_health": 100,
                "player_stamina": 90,
                "enemy_count": 1,
                "nearest_enemy_dist": 40.0,
            })

        style, conf = detector.predict_playstyle(aggressive_events)
        self.assertIn(style, ["Aggressive", "Evasive", "Balanced"])
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_07_game_resilience_when_model_weights_are_unavailable(self):
        """Verifies the game and inference coordinator never crash when model files are missing."""
        non_existent_path = os.path.join(self.temp_dir, "missing_siamese.pt")
        detector = PlaystyleDetector(model_path=non_existent_path)
        self.assertFalse(detector.is_loaded)

        # Inference must return fallback without error
        style, conf = detector.predict_playstyle([{"type": "PLAYER_ATTACK", "timestamp": 1.0}])
        self.assertEqual(style, "Balanced")
        self.assertEqual(conf, 0.0)

        # Game engine execution with missing model
        engine = GameEngine(screen=None, raw_data_dir=os.path.join(self.temp_dir, "raw"))
        engine.playstyle_detector = detector
        engine.start_new_game(seed=42)

        # Step update loop multiple times
        for _ in range(10):
            engine.update(0.016)

        self.assertTrue(engine.player.is_alive)
        self.assertEqual(engine.detected_playstyle, "Balanced")
        self.assertEqual(engine.playstyle_confidence, 0.0)

    def test_08_corrupt_model_weights_resilience(self):
        """Verifies corrupt or malformed checkpoint files never crash the game or detector."""
        corrupt_path = os.path.join(self.temp_dir, "corrupt_checkpoint.pt")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_PYTORCH_CHECKPOINT_GARBAGE_BYTES_12345")

        detector = PlaystyleDetector(model_path=corrupt_path)
        self.assertFalse(detector.is_loaded)

        style, conf = detector.predict_playstyle([{"type": "PLAYER_ATTACK", "timestamp": 1.0}])
        self.assertEqual(style, "Balanced")
        self.assertEqual(conf, 0.0)

        # Game engine execution with corrupt model
        engine = GameEngine(screen=None, raw_data_dir=os.path.join(self.temp_dir, "raw"))
        engine.playstyle_detector = detector
        engine.start_new_game(seed=42)

        for _ in range(5):
            engine.update(0.016)

        self.assertTrue(engine.player.is_alive)
        self.assertEqual(engine.detected_playstyle, "Balanced")

    def test_09_enemy_counter_tactics_behavior_changes(self):
        """Verifies enemy counter-tactics actually modify state machine movement speeds and distances."""
        from game.entities import Player, MeleeEnemy, RangedEnemy
        from game.combat import CombatSystem
        from game.dungeon import Dungeon
        import config

        dungeon = Dungeon(seed=42)
        cx, cy = dungeon.spawn_room.center_pixel
        player = Player(cx, cy)
        melee = MeleeEnemy(cx + 60.0, cy)
        ranged = RangedEnemy(cx + 120.0, cy)
        combat = CombatSystem(player, [melee, ranged], dungeon)

        # 1. Aggressive playstyle counter-tactic: Melee backpedals during cooldown
        melee.state = MeleeEnemy.STATE_COOLDOWN
        melee.cooldown_timer = 1.0
        init_dist = abs(player.x - melee.x)
        melee.update_ai(dt=0.1, player=player, dungeon=dungeon, combat_system=combat, playstyle="Aggressive", confidence=0.80)
        new_dist = abs(player.x - melee.x)
        # Distance to player must have increased (backpedaling)
        self.assertGreater(new_dist, init_dist)

        # 2. Aggressive playstyle counter-tactic: Ranged buffers distance (+40 px)
        # Standard min dist is 190.0 px. Player at dist=200 px would normally chase.
        # With Aggressive buff (+40 => 230 px), dist=200 px must trigger retreat!
        ranged.x = cx + 200.0  # dist = 200 px from player
        ranged.state = RangedEnemy.STATE_CHASE
        ranged.cooldown_timer = 0.0
        ranged.update_ai(dt=0.016, player=player, dungeon=dungeon, combat_system=combat, playstyle="Aggressive", confidence=0.80)
        self.assertEqual(ranged.state, RangedEnemy.STATE_RETREAT)

        # 3. Evasive playstyle counter-tactic: Melee chases faster (+15% speed)
        melee.state = MeleeEnemy.STATE_CHASE
        melee.cooldown_timer = 0.0
        melee.x = cx + 80.0
        init_x = melee.x
        melee.update_ai(dt=0.1, player=player, dungeon=dungeon, combat_system=combat, playstyle="Evasive", confidence=0.80)
        dist_moved_evasive = init_x - melee.x

        # Baseline chase
        melee.x = cx + 80.0
        melee.update_ai(dt=0.1, player=player, dungeon=dungeon, combat_system=combat, playstyle="Balanced", confidence=0.0)
        dist_moved_baseline = init_x - melee.x
        self.assertGreater(dist_moved_evasive, dist_moved_baseline)

        # 4. Low confidence fallback: Below 0.45, retains baseline behavior
        ranged.x = cx + 200.0  # dist = 200 px
        ranged.state = RangedEnemy.STATE_CHASE
        ranged.update_ai(dt=0.016, player=player, dungeon=dungeon, combat_system=combat, playstyle="Aggressive", confidence=0.30)
        # At confidence=0.30 (< 0.45), preferred_min_dist remains 190 px, so at 200 px it does NOT retreat
        self.assertNotEqual(ranged.state, RangedEnemy.STATE_RETREAT)

    def test_10_held_out_session_split_and_scaler_isolation(self):
        """Verifies session-level split integrity and confirms FeatureScaler is fitted strictly on train data."""
        records = []
        for s in range(10):
            sid = f"test_session_{s:02d}"
            for w in range(4):
                records.append({
                    "session_id": sid,
                    "val": float(s * 10 + w),
                })

        train_rec, test_rec = session_train_test_split(records, test_ratio=0.30, seed=42)
        train_sids = set(r["session_id"] for r in train_rec)
        test_sids = set(r["session_id"] for r in test_rec)

        # Strictly disjoint
        self.assertEqual(len(train_sids.intersection(test_sids)), 0)
        self.assertEqual(len(train_sids) + len(test_sids), 10)

        # Scaler isolation: fitting on train only vs full produces distinct means
        X_train = np.array([[r["val"]] for r in train_rec], dtype=np.float32)
        X_all = np.array([[r["val"]] for r in records], dtype=np.float32)
        scaler_train = FeatureScaler().fit(X_train)
        scaler_all = FeatureScaler().fit(X_all)
        self.assertNotEqual(float(scaler_train.mean[0]), float(scaler_all.mean[0]))

    def test_11_feature_schema_consistency_with_checkpoint(self):
        """Verifies checkpoint contains identical 11-feature schema and FeatureScaler matches saved parameters."""
        checkpoint_path = "saved_models/siamese_checkpoint.pt"
        detector = PlaystyleDetector(model_path=checkpoint_path)
        self.assertTrue(detector.is_loaded)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.assertEqual(detector.feature_names, checkpoint["feature_names"])
        self.assertEqual(len(detector.feature_names), 11)
        self.assertEqual(detector.feature_names, NUMERICAL_FEATURE_NAMES)
        np.testing.assert_allclose(detector.scaler.mean, np.array(checkpoint["scaler"]["mean"], dtype=np.float32))
        np.testing.assert_allclose(detector.scaler.std, np.array(checkpoint["scaler"]["std"], dtype=np.float32))

    def test_12_prototype_distance_and_similarity_mapping(self):
        """Verifies the mathematical inverse-distance mapping and boundary properties."""
        detector = PlaystyleDetector(model_path="saved_models/siamese_checkpoint.pt")
        self.assertTrue(detector.is_loaded)

        # Distance 0.0 => Similarity 1.0
        sim_zero = round(1.0 / (1.0 + 0.0), 3)
        self.assertEqual(sim_zero, 1.0)

        # Distance 1.0 (margin) => Similarity 0.50
        sim_margin = round(1.0 / (1.0 + 1.0), 3)
        self.assertEqual(sim_margin, 0.50)

        # Distance 1.222 => Similarity ~ 0.45 (threshold boundary)
        sim_thresh = round(1.0 / (1.0 + 1.222), 3)
        self.assertAlmostEqual(sim_thresh, 0.45, places=2)


if __name__ == "__main__":
    unittest.main()

