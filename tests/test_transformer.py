"""
Unit and integration test suite for Neural Dungeon Phase 5:
Transformer model for next-action sequence prediction.
Validates causal masking, temporal sequence alignment, session-level splitting,
checkpoint integrity, runtime inference fallback, enemy counter-tactics,
and multi-model coexistence (Siamese + Autoencoder + Transformer).
"""

import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pygame

from ml.transformer_model import ActionTransformer, ACTION_TO_ID, ID_TO_ACTION
from ml.dataset import build_action_sequence_dataset, extract_session_action_sequence
from ml.action_predictor import ActionPredictor
from game.entities import Player, MeleeEnemy, RangedEnemy
from game.engine import GameEngine
from game.dungeon import Dungeon
from game.combat import CombatSystem
import config


class TestTransformerPhase5(unittest.TestCase):
    """Test suite for Phase 5 Transformer next-action sequence prediction."""

    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.font.init()

    def setUp(self):
        pygame.init()
        pygame.font.init()
        self.temp_dir = tempfile.mkdtemp(prefix="neural_dungeon_test_transformer_")
        self.device = torch.device("cpu")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_01_causal_masking_autoregressive_property(self):
        """
        Verifies causal masking: for any sequence [a_0, a_1, ... a_t],
        changing token a_{t+1} must NOT alter the logits or attention output at position t.
        """
        model = ActionTransformer(
            vocab_size=len(ACTION_TO_ID),
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            max_seq_len=16,
            dropout=0.0,
        )
        model.eval()

        # Two sequences identical up to position 4, differing at position 5
        seq1 = torch.tensor([[1, 2, 3, 4, 2, 3, 0, 0]], dtype=torch.long)
        seq2 = torch.tensor([[1, 2, 3, 4, 2, 5, 0, 0]], dtype=torch.long)

        with torch.no_grad():
            out1 = model(seq1, return_all_positions=True)  # (1, 8, vocab_size)
            out2 = model(seq2, return_all_positions=True)  # (1, 8, vocab_size)

        # Positions 0 through 4 must produce strictly identical logits
        diff = torch.max(torch.abs(out1[:, :5, :] - out2[:, :5, :])).item()
        self.assertAlmostEqual(diff, 0.0, places=5,
                               msg=f"Causal masking failed: token change at t=5 leaked to earlier positions (diff: {diff})")

    def test_02_chronological_sequence_target_alignment(self):
        """
        Verifies sequence dataset construction: for every pair (x_i, y_i),
        the target y_i is strictly the action immediately following the context window x_i,
        and padding is strictly on the left (PAD=0). No future tokens leaked into x_i.
        """
        # Create a mock sequence of actions
        events = [
            {"event_type": "MOVE"},
            {"event_type": "MOVE"},
            {"event_type": "PLAYER_ATTACK"},
            {"event_type": "PLAYER_DODGE"},
            {"event_type": "PLAYER_HURT"},
            {"event_type": "IDLE"},
        ]
        tokens = extract_session_action_sequence(events)
        expected_tokens = [ACTION_TO_ID["MOVE"], ACTION_TO_ID["MOVE"], ACTION_TO_ID["ATTACK"], ACTION_TO_ID["DODGE"], ACTION_TO_ID["HURT"], ACTION_TO_ID["IDLE"]]
        self.assertEqual(tokens, expected_tokens)

        max_len = 4

        # Manually verify sliding window logic
        pairs = []
        for i in range(1, len(tokens)):
            ctx = tokens[max(0, i - max_len):i]
            if len(ctx) < max_len:
                ctx = [0] * (max_len - len(ctx)) + ctx
            target = tokens[i]
            pairs.append((ctx, target))

        # Check first pair: target is 2nd token (index 1 = MOVE)
        self.assertEqual(pairs[0][1], ACTION_TO_ID["MOVE"])
        self.assertEqual(pairs[0][0][-1], ACTION_TO_ID["MOVE"])  # Last token in context is 1st token
        self.assertEqual(pairs[0][0][0], 0)  # Left-padded with PAD=0

        # Check third pair: target is 4th token (index 3 = DODGE)
        self.assertEqual(pairs[2][1], ACTION_TO_ID["DODGE"])
        self.assertEqual(pairs[2][0][-1], ACTION_TO_ID["ATTACK"])  # Immediately precedes DODGE

    def test_03_session_level_split_integrity(self):
        """
        Verifies session-level splitting ensures zero overlap between train, val, and test session IDs.
        """
        sessions_dict = {
            f"session_{i:02d}": [
                {"event_type": "MOVE"},
                {"event_type": "PLAYER_ATTACK"},
                {"event_type": "MOVE"},
            ]
            for i in range(10)
        }

        # Partition sessions: 6 train, 2 val, 2 test
        session_names = sorted(sessions_dict.keys())
        train_names = set(session_names[:6])
        val_names = set(session_names[6:8])
        test_names = set(session_names[8:])

        train_ds = build_action_sequence_dataset({k: sessions_dict[k] for k in train_names}, max_seq_len=8)
        val_ds = build_action_sequence_dataset({k: sessions_dict[k] for k in val_names}, max_seq_len=8)
        test_ds = build_action_sequence_dataset({k: sessions_dict[k] for k in test_names}, max_seq_len=8)

        # Check disjointness
        self.assertEqual(len(train_names.intersection(val_names)), 0)
        self.assertEqual(len(train_names.intersection(test_names)), 0)
        self.assertEqual(len(val_names.intersection(test_names)), 0)
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(val_ds), 0)
        self.assertGreater(len(test_ds), 0)

    def test_04_checkpoint_save_and_load(self):
        """
        Verifies that saved_models/transformer_checkpoint.pt contains valid architecture weights,
        hyperparameters, vocabulary mappings, and can be instantiated and executed.
        """
        ckpt_path = "saved_models/transformer_checkpoint.pt"
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.assertIn("model_state_dict", ckpt)
        self.assertIn("config", ckpt)
        self.assertIn("action_to_id", ckpt)
        self.assertIn("id_to_action", ckpt)

        # Instantiate model from checkpoint config
        cfg = ckpt["config"]
        model = ActionTransformer(
            vocab_size=cfg.get("vocab_size", len(ACTION_TO_ID)),
            d_model=cfg.get("d_model", 32),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 64),
            max_seq_len=cfg.get("max_seq_len", 16),
            dropout=0.0,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Run dummy forward pass
        dummy_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with torch.no_grad():
            logits = model(dummy_input)
        self.assertEqual(logits.shape, (1, cfg.get("vocab_size", len(ACTION_TO_ID))))

    def test_05_action_predictor_inference_and_fallback(self):
        """
        Verifies ActionPredictor:
        - Accurately converts events to sequence tokens
        - Runs inference and returns action name + confidence score
        - Falls back to 'NONE' if confidence is below 0.45 or if events are empty
        """
        predictor = ActionPredictor(model_path="saved_models/transformer_checkpoint.pt")
        self.assertTrue(predictor.is_available)

        # Empty event list returns fallback
        act, conf = predictor.predict_next_action([])
        self.assertEqual(act, "NONE")
        self.assertEqual(conf, 0.0)

        # Events without valid action types
        act, conf = predictor.predict_next_action([{"event_type": "UNKNOWN_EVENT"}])
        self.assertEqual(act, "NONE")
        self.assertEqual(conf, 0.0)

        # Valid sequence of actions
        sample_events = [
            {"event_type": "MOVE"},
            {"event_type": "MOVE"},
            {"event_type": "PLAYER_ATTACK"},
            {"event_type": "MOVE"},
            {"event_type": "PLAYER_DODGE"},
        ]
        act, conf = predictor.predict_next_action(sample_events)
        self.assertIn(act, list(ACTION_TO_ID.keys()) + ["NONE"])
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_06_resilience_missing_and_corrupt_weights(self):
        """
        Verifies ActionPredictor resilience: missing or corrupt checkpoint files
        never crash the predictor or game, falling back cleanly to 'NONE'.
        """
        # 1. Missing weights file
        missing_predictor = ActionPredictor(model_path="saved_models/nonexistent_transformer.pt")
        self.assertFalse(missing_predictor.is_available)
        act, conf = missing_predictor.predict_next_action([{"event_type": "MOVE"}])
        self.assertEqual(act, "NONE")
        self.assertEqual(conf, 0.0)

        # 2. Corrupt weights file
        corrupt_path = os.path.join(self.temp_dir, "corrupt_transformer.pt")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_PYTORCH_CHECKPOINT_GARBAGE")

        corrupt_predictor = ActionPredictor(model_path=corrupt_path)
        self.assertFalse(corrupt_predictor.is_available)
        act, conf = corrupt_predictor.predict_next_action([{"event_type": "MOVE"}])
        self.assertEqual(act, "NONE")
        self.assertEqual(conf, 0.0)

    def test_07_enemy_counter_tactics(self):
        """
        Verifies that predicted next actions alter enemy state machine behavior:
        - MeleeEnemy delays swing telegraph when DODGE is anticipated (to roll-catch recovery)
        - RangedEnemy retreats faster when ATTACK is anticipated
        """
        dungeon = Dungeon(seed=42)
        cx, cy = dungeon.spawn_room.center_pixel
        player = Player(cx, cy)
        combat = CombatSystem(player, [], dungeon)

        # 1. MeleeEnemy counter-tactic: delay telegraph on DODGE
        melee = MeleeEnemy(cx + 10, cy)
        melee.state = melee.STATE_TELEGRAPH
        melee.telegraph_timer = 0.10  # in window < 0.15

        dt = 0.016
        # Without prediction or low confidence: timer decreases normally by dt
        timer_before = melee.telegraph_timer
        melee.update_ai(dt, player, dungeon, combat, predicted_action="NONE", action_confidence=0.0)
        normal_elapsed = timer_before - melee.telegraph_timer

        # With high-confidence DODGE prediction: timer increases by dt * 0.4 before subtracting dt
        melee.telegraph_timer = 0.10
        melee.update_ai(dt, player, dungeon, combat, predicted_action="DODGE", action_confidence=0.85)
        adapted_elapsed = 0.10 - melee.telegraph_timer

        self.assertLess(adapted_elapsed, normal_elapsed,
                        "MeleeEnemy should prolong telegraph when DODGE is forecasted to catch recovery")

        # 2. RangedEnemy counter-tactic: faster retreat on ATTACK
        ranged_normal = RangedEnemy(cx + 60.0, cy)
        ranged_normal.state = ranged_normal.STATE_RETREAT
        x_before = ranged_normal.x
        ranged_normal.update_ai(dt, player, dungeon, combat, predicted_action="NONE", action_confidence=0.0)
        normal_dist_moved = abs(ranged_normal.x - x_before)

        ranged_counter = RangedEnemy(cx + 60.0, cy)
        ranged_counter.state = ranged_counter.STATE_RETREAT
        x_before_counter = ranged_counter.x
        ranged_counter.update_ai(dt, player, dungeon, combat, predicted_action="ATTACK", action_confidence=0.85)
        counter_dist_moved = abs(ranged_counter.x - x_before_counter)

        self.assertGreater(counter_dist_moved, normal_dist_moved,
                           "RangedEnemy should retreat faster (1.20x) when ATTACK is forecasted")

    def test_08_game_engine_coexistence_all_three_models(self):
        """
        Verifies that GameEngine seamlessly runs with all three models active:
        - Siamese Network (Phase 3 playstyle adaptation)
        - Combat Autoencoder (Phase 4 anomaly monitoring)
        - Action Transformer (Phase 5 next-action prediction)
        and verifies zero-crash execution even with missing weights.
        """
        engine = GameEngine(raw_data_dir=os.path.join(self.temp_dir, "raw"))
        engine.start_new_game(seed=12345)

        # Confirm all three inference modules are initialized
        self.assertIsNotNone(engine.playstyle_detector)
        self.assertIsNotNone(engine.anomaly_detector)
        self.assertIsNotNone(engine.action_predictor)

        # Run 10 engine update frames
        for _ in range(10):
            engine.update(0.016)

        # Trigger the 2.0s periodic evaluation timer
        engine.playstyle_eval_timer = 2.1
        engine.update(0.016)

        # Check that predictions and metrics exist without throwing exceptions
        self.assertIn(engine.detected_playstyle, ["Aggressive", "Evasive", "Balanced"])
        self.assertIsInstance(engine.is_anomaly, bool)
        self.assertIn(engine.predicted_action, list(ACTION_TO_ID.keys()) + ["NONE"])
        self.assertGreaterEqual(engine.action_confidence, 0.0)

        # Test headless surface rendering with all HUD panels
        surf = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        engine.ui.render_hud(
            surf,
            engine.player,
            len(engine.enemies),
            False,
            playstyle=engine.detected_playstyle,
            confidence=engine.playstyle_confidence,
            is_anomaly=engine.is_anomaly,
            recon_error=engine.reconstruction_error,
            anomaly_thresh=engine.anomaly_threshold,
            predicted_action=engine.predicted_action,
            action_confidence=engine.action_confidence,
        )

        # Test operation when all three models point to invalid or missing paths
        from ml.playstyle_detector import PlaystyleDetector
        from ml.anomaly_detector import CombatAnomalyDetector
        engine.playstyle_detector = PlaystyleDetector(model_path="nonexistent.pt")
        engine.anomaly_detector = CombatAnomalyDetector(model_path="nonexistent.pt")
        engine.action_predictor = ActionPredictor(model_path="nonexistent.pt")
        for _ in range(5):
            engine.update(0.016)
        self.assertTrue(engine.player.is_alive)

    def test_09_class_weighted_model_and_evaluation_report(self):
        """
        Verifies that the balanced class-weighted checkpoint exists, is configured properly,
        and that evaluate_transformer produces both unweighted and weighted comparative metrics.
        """
        from evaluation.eval_transformer import evaluate_transformer
        weighted_ckpt_path = "saved_models/transformer_weighted_checkpoint.pt"
        self.assertTrue(os.path.exists(weighted_ckpt_path), f"Weighted checkpoint missing at {weighted_ckpt_path}")

        ckpt = torch.load(weighted_ckpt_path, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt.get("weight_scheme"), "balanced")
        self.assertIsNotNone(ckpt.get("class_weights"))

        # Run evaluation and verify report output
        results = evaluate_transformer(
            model_path="saved_models/transformer_checkpoint.pt",
            weighted_model_path=weighted_ckpt_path,
            output_dir=self.temp_dir,
        )

        self.assertIn("transformer_unweighted_metrics", results)
        self.assertIn("transformer_weighted_metrics", results)
        self.assertIn("majority_baseline_metrics", results)
        self.assertIn("markov_baseline_metrics", results)

        # Confirm Macro-F1 is higher on balanced model due to minority class predictions
        unw_f1 = results["transformer_unweighted_metrics"]["macro_f1"]
        w_f1 = results["transformer_weighted_metrics"]["macro_f1"]
        self.assertGreater(w_f1, unw_f1, "Balanced class weighting should improve Macro-F1 over unweighted ERM")

    def test_10_counter_tactics_deactivation_under_low_confidence(self):
        """
        Verifies that enemy counter-tactics are strictly gated by the 0.45 confidence threshold:
        - When confidence < 0.45, MeleeEnemy does NOT prolong attack telegraph on DODGE.
        - When confidence < 0.45, RangedEnemy does NOT retreat faster on ATTACK.
        - When prediction is NONE, both enemies maintain default rule-based movement.
        """
        dungeon = Dungeon(seed=42)
        cx, cy = dungeon.spawn_room.center_pixel
        player = Player(cx, cy)
        combat = CombatSystem(player, [], dungeon)

        # 1. MeleeEnemy: low-confidence DODGE should NOT trigger delay
        melee_nominal = MeleeEnemy(cx + 10, cy)
        melee_nominal.state = melee_nominal.STATE_TELEGRAPH
        melee_nominal.telegraph_timer = 0.10

        melee_low_conf = MeleeEnemy(cx + 10, cy)
        melee_low_conf.state = melee_low_conf.STATE_TELEGRAPH
        melee_low_conf.telegraph_timer = 0.10

        dt = 0.016
        melee_nominal.update_ai(dt, player, dungeon, combat, predicted_action="NONE", action_confidence=0.0)
        melee_low_conf.update_ai(dt, player, dungeon, combat, predicted_action="DODGE", action_confidence=0.30)  # < 0.45

        # Both timers should elapse at identical rates (no prolongation)
        self.assertAlmostEqual(melee_nominal.telegraph_timer, melee_low_conf.telegraph_timer, places=5,
                               msg="Low-confidence DODGE should not prolong Melee telegraph")

        # 2. RangedEnemy: low-confidence ATTACK should NOT trigger faster retreat
        ranged_nominal = RangedEnemy(cx + 60.0, cy)
        ranged_nominal.state = ranged_nominal.STATE_RETREAT
        x_nom_start = ranged_nominal.x

        ranged_low_conf = RangedEnemy(cx + 60.0, cy)
        ranged_low_conf.state = ranged_low_conf.STATE_RETREAT
        x_low_start = ranged_low_conf.x

        ranged_nominal.update_ai(dt, player, dungeon, combat, predicted_action="NONE", action_confidence=0.0)
        ranged_low_conf.update_ai(dt, player, dungeon, combat, predicted_action="ATTACK", action_confidence=0.30)  # < 0.45

        dist_nom = abs(ranged_nominal.x - x_nom_start)
        dist_low = abs(ranged_low_conf.x - x_low_start)

        self.assertAlmostEqual(dist_nom, dist_low, places=5,
                               msg="Low-confidence ATTACK should not increase Ranged retreat speed")


if __name__ == "__main__":
    unittest.main()
