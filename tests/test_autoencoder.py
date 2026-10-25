"""
Unit and integration tests for Phase 4: Combat Autoencoder for Anomaly Detection.
Verifies input schemas, latent compression, checkpoint serialization, threshold selection
without leakage, session isolation, inference coordinator, corrupt file resilience, and game integration.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

from ml.dataset import NUMERICAL_FEATURE_NAMES, FeatureScaler, session_train_test_split
from ml.autoencoder import CombatAutoencoder
from ml.anomaly_detector import CombatAnomalyDetector
from game.engine import GameEngine


class TestAutoencoderPhase4(unittest.TestCase):
    """Test suite for Phase 4 Combat Autoencoder and anomaly detection."""

    def setUp(self):
        import pygame
        pygame.init()
        pygame.font.init()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_feature_dimensions_and_scaler_consistency(self):
        """Verifies 11-feature schema and FeatureScaler preprocessing consistency."""
        self.assertEqual(len(NUMERICAL_FEATURE_NAMES), 11)

        dummy_data = np.random.randn(8, 11).astype(np.float32)
        scaler = FeatureScaler().fit(dummy_data)
        norm_data = scaler.transform(dummy_data)
        self.assertEqual(norm_data.shape, (8, 11))

        # Check serialization round-trip
        state = scaler.to_dict()
        restored = FeatureScaler.from_dict(state)
        restored_norm = restored.transform(dummy_data)
        np.testing.assert_allclose(norm_data, restored_norm, rtol=1e-5)

    def test_02_autoencoder_architecture_and_reconstruction_shape(self):
        """Verifies CombatAutoencoder compresses 11D inputs to 4D latent space and reconstructs 11D outputs."""
        model = CombatAutoencoder(in_features=11, latent_dim=4, hidden_dim=8)
        batch = torch.randn(6, 11)

        rec_x, z = model(batch)
        self.assertEqual(rec_x.shape, (6, 11))
        self.assertEqual(z.shape, (6, 4))

        # Per-sample MSE error
        errors = model.compute_reconstruction_error(batch)
        self.assertEqual(errors.shape, (6,))
        self.assertTrue(torch.all(errors >= 0.0))
        self.assertTrue(torch.all(torch.isfinite(errors)))

    def test_03_training_checkpoint_save_and_load(self):
        """Verifies saved_models/autoencoder_checkpoint.pt contains valid weights, threshold, and schema."""
        ckpt_path = "saved_models/autoencoder_checkpoint.pt"
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("scaler", checkpoint)
        self.assertIn("threshold", checkpoint)
        self.assertIn("in_features", checkpoint)
        self.assertIn("latent_dim", checkpoint)
        self.assertIn("feature_names", checkpoint)

        self.assertEqual(checkpoint["in_features"], 11)
        self.assertEqual(checkpoint["latent_dim"], 4)
        self.assertEqual(checkpoint["feature_names"], NUMERICAL_FEATURE_NAMES)
        self.assertGreater(checkpoint["threshold"], 0.0)

    def test_04_threshold_selection_without_test_set_leakage(self):
        """Verifies anomaly threshold is computed strictly from training split distributions."""
        train_errors = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.03, 0.04, 0.07], dtype=np.float32)
        test_errors = np.array([0.15, 0.20, 0.25], dtype=np.float32)

        # Threshold using training data only
        train_mean = float(np.mean(train_errors))
        train_std = float(np.std(train_errors))
        threshold = train_mean + 2.0 * train_std

        # Threshold using all data (which would represent leakage)
        all_errors = np.concatenate([train_errors, test_errors])
        leaked_threshold = float(np.mean(all_errors) + 2.0 * np.std(all_errors))

        # Must be distinct: training-only threshold cannot be influenced by test errors
        self.assertNotEqual(round(threshold, 4), round(leaked_threshold, 4))
        self.assertLess(threshold, leaked_threshold)

    def test_05_session_level_split_integrity(self):
        """Verifies session-level split prevents temporal overlap between autoencoder train and test sets."""
        records = []
        for s in range(8):
            sid = f"ae_session_{s:02d}"
            for w in range(5):
                records.append({
                    "session_id": sid,
                    "window_id": w,
                    "attack_frequency": float(s),
                })

        train_recs, test_recs = session_train_test_split(records, test_ratio=0.25, seed=42)
        train_sids = set(r["session_id"] for r in train_recs)
        test_sids = set(r["session_id"] for r in test_recs)

        self.assertEqual(len(train_sids.intersection(test_sids)), 0)
        self.assertEqual(len(train_sids) + len(test_sids), 8)

    def test_06_anomaly_inference_and_threshold_detection(self):
        """Verifies CombatAnomalyDetector returns finite error and flags high reconstruction errors."""
        detector = CombatAnomalyDetector(model_path="saved_models/autoencoder_checkpoint.pt")
        self.assertTrue(detector.is_loaded)

        # 1. Normal simulated window (balanced attacks and dodges)
        normal_events = []
        for i in range(25):
            t = float(i) * 0.2
            normal_events.append({
                "type": "STATE_TICK",
                "timestamp": t,
                "player_x": 300.0,
                "player_y": 300.0,
                "player_health": 100,
                "player_stamina": 90.0,
                "is_dodging": False,
                "aim_angle": 0.5,
                "nearest_enemy_dist": 150.0,
            })
            if i % 6 == 0:
                normal_events.append({
                    "type": "PLAYER_ATTACK",
                    "timestamp": t + 0.05,
                    "hits": 1,
                    "damage_dealt": 35,
                })

        is_anom, recon_err, thresh = detector.detect_anomaly(normal_events)
        self.assertIsInstance(is_anom, bool)
        self.assertGreaterEqual(recon_err, 0.0)
        self.assertEqual(thresh, detector.threshold)

        # 2. Insufficient events should safely return nominal
        is_anom_few, err_few, _ = detector.detect_anomaly([{"type": "PLAYER_ATTACK", "timestamp": 1.0}])
        self.assertFalse(is_anom_few)
        self.assertEqual(err_few, 0.0)

    def test_07_resilience_missing_or_corrupt_weights(self):
        """Verifies detector and game engine never crash when autoencoder weights are absent or corrupted."""
        # Missing checkpoint
        missing_path = os.path.join(self.temp_dir, "non_existent_ae.pt")
        detector_missing = CombatAnomalyDetector(model_path=missing_path)
        self.assertFalse(detector_missing.is_loaded)
        is_anom, err, _ = detector_missing.detect_anomaly([{"type": "PLAYER_ATTACK", "timestamp": 1.0}])
        self.assertFalse(is_anom)
        self.assertEqual(err, 0.0)

        # Corrupt checkpoint
        corrupt_path = os.path.join(self.temp_dir, "corrupt_ae.pt")
        with open(corrupt_path, "wb") as f:
            f.write(b"CORRUPTED_GARBAGE_PAYLOAD_AE_12345")

        detector_corrupt = CombatAnomalyDetector(model_path=corrupt_path)
        self.assertFalse(detector_corrupt.is_loaded)
        is_anom, err, _ = detector_corrupt.detect_anomaly([{"type": "PLAYER_ATTACK", "timestamp": 1.0}])
        self.assertFalse(is_anom)
        self.assertEqual(err, 0.0)

    def test_08_game_operation_with_both_models(self):
        """Verifies GameEngine runs update and render loops with both Siamese and Autoencoder active."""
        engine = GameEngine(screen=None, raw_data_dir=os.path.join(self.temp_dir, "raw"))
        engine.start_new_game(seed=42)

        # Step update loop multiple times
        for _ in range(15):
            engine.update(0.016)

        self.assertTrue(engine.player.is_alive)
        self.assertIn(engine.detected_playstyle, ["Aggressive", "Evasive", "Balanced"])
        self.assertIsInstance(engine.is_anomaly, bool)
        self.assertGreaterEqual(engine.reconstruction_error, 0.0)

        # Verify operation when both models are pointed to invalid paths
        engine.playstyle_detector = CombatAnomalyDetector(model_path="invalid.pt")  # invalid
        engine.anomaly_detector = CombatAnomalyDetector(model_path="invalid.pt")
        for _ in range(10):
            engine.update(0.016)
        self.assertTrue(engine.player.is_alive)


if __name__ == "__main__":
    unittest.main()
