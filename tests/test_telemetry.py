"""
Comprehensive test suite for Phase 2 & Phase 2.1: Gameplay Telemetry & Feature Extraction.
Tests schema validation, event-summary consistency, authoritative aggregation,
synthetic vs genuine data separation, outcome integrity guards, and pipeline idempotence.
"""

import os
import json
import shutil
import tempfile
import unittest

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import config
from game.dungeon import Dungeon
from game.entities import Player, MeleeEnemy, RangedEnemy
from game.combat import CombatSystem
from game.engine import GameEngine
from game.telemetry import TelemetryManager
from ml.features import (
    load_session_events,
    extract_window_features,
    extract_session_style_vector,
    process_raw_sessions,
    ACTION_VOCAB,
)


class TestTelemetryPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.font.init()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.telemetry = TelemetryManager(raw_data_dir=self.test_dir, session_type="human")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_01_schema_validation(self):
        """Verifies schema validation enforces bounds and sanitizes missing/invalid values."""
        self.telemetry.start_session(dungeon_seed=123)

        dirty_event = {
            "event_type": "TEST_EVENT",
            "timestamp": -5.0,
            "player_x": 120.456,
            "player_y": 340.789,
            "player_health": 999,
            "player_stamina": -50.0,
            "room_id": 2,
        }
        clean = self.telemetry.validate_event(dirty_event)

        self.assertEqual(clean["session_id"], self.telemetry.session_id)
        self.assertEqual(clean["timestamp"], 0.0)
        self.assertEqual(clean["player_health"], config.PLAYER_MAX_HEALTH)
        self.assertEqual(clean["player_stamina"], 0.0)
        self.assertEqual(clean["player_x"], 120.46)
        self.assertEqual(clean["player_y"], 340.79)
        self.assertEqual(clean["room_id"], 2)

    def test_02_session_separation_and_summaries(self):
        """Verifies separate sessions generate distinct files and append to summaries.jsonl."""
        s1_id = self.telemetry.start_session(dungeon_seed=10)
        self.telemetry.sample_state(0.5, Player(100, 100), [], Dungeon(seed=10))
        summary1 = self.telemetry.end_session("DEFEAT")

        self.assertIsNotNone(summary1)
        self.assertEqual(summary1["session_id"], s1_id)
        self.assertEqual(summary1["final_outcome"], "DEFEAT")

        s2_id = self.telemetry.start_session(dungeon_seed=20)
        self.assertNotEqual(s1_id, s2_id)
        self.telemetry.sample_state(0.5, Player(100, 100), [], Dungeon(seed=20))
        summary2 = self.telemetry.end_session("DEFEAT")

        self.assertEqual(summary2["session_id"], s2_id)

        # Verify files exist on disk
        f1 = os.path.join(self.test_dir, f"{s1_id}.jsonl")
        f2 = os.path.join(self.test_dir, f"{s2_id}.jsonl")
        summaries_file = os.path.join(self.test_dir, "summaries.jsonl")

        self.assertTrue(os.path.exists(f1))
        self.assertTrue(os.path.exists(f2))
        self.assertTrue(os.path.exists(summaries_file))

        with open(summaries_file, "r", encoding="utf-8") as sf:
            lines = sf.readlines()
            self.assertEqual(len(lines), 2)
            rec1 = json.loads(lines[0])
            rec2 = json.loads(lines[1])
            self.assertEqual(rec1["session_id"], s1_id)
            self.assertEqual(rec2["session_id"], s2_id)

    def test_03_combat_telemetry_events(self):
        """Verifies combat actions (attacks, evasions, damage, kills) generate telemetry events."""
        self.telemetry.start_session(dungeon_seed=42)
        dungeon = Dungeon(seed=42)
        player = Player(150, 150)
        enemy = MeleeEnemy(180, 150)
        combat = CombatSystem(player, [enemy], dungeon, telemetry=self.telemetry)

        # 1. Player attack
        player.aim_angle = 0.0
        combat.execute_player_melee()
        attack_events = [e for e in self.telemetry.events if e["event_type"] == "PLAYER_ATTACK"]
        self.assertEqual(len(attack_events), 1)
        self.assertEqual(attack_events[0]["hits"], 1)

        # 2. Player dodge evasion
        player.start_dodge((1.0, 0.0))
        combat.apply_player_damage(20, (0.0, 0.0), enemy_type="melee")
        evasion_events = [e for e in self.telemetry.events if e["event_type"] == "DODGED_ATTACK"]
        self.assertEqual(len(evasion_events), 1)

        # 3. Enemy attack execution
        combat.record_enemy_attack(enemy, "melee", damage=config.MELEE_ENEMY_DAMAGE)
        enemy_atk_events = [e for e in self.telemetry.events if e["event_type"] == "ENEMY_ATTACK"]
        self.assertEqual(len(enemy_atk_events), 1)

        # 4. Kill enemy with lethal strike
        enemy.health = 10
        player.is_dodging = False
        player.attack_cooldown = 0.0
        player.stamina = 100.0
        combat.execute_player_melee()
        kill_events = [e for e in self.telemetry.events if e["event_type"] == "ENEMY_KILLED"]
        self.assertGreaterEqual(len(kill_events), 1)

    def test_04_feature_extraction(self):
        """Verifies window feature extraction calculates combat metrics, rates, and token sequences."""
        events = [
            {"session_id": "session_test", "timestamp": 0.0, "event_type": "SESSION_START", "player_x": 100, "player_y": 100, "player_health": 100, "player_stamina": 100.0, "room_id": 0},
            {"session_id": "session_test", "timestamp": 1.0, "event_type": "STATE_TICK", "player_x": 110, "player_y": 100, "player_health": 100, "player_stamina": 90.0, "room_id": 0, "aim_angle": 0.0, "is_dodging": False, "nearest_enemy_dist": 200.0},
            {"session_id": "session_test", "timestamp": 2.0, "event_type": "PLAYER_ATTACK", "player_x": 120, "player_y": 100, "player_health": 100, "player_stamina": 75.0, "room_id": 0, "hits": 1, "damage_dealt": 35},
            {"session_id": "session_test", "timestamp": 3.0, "event_type": "PLAYER_DODGE", "player_x": 130, "player_y": 100, "player_health": 100, "player_stamina": 50.0, "room_id": 0, "success": True},
            {"session_id": "session_test", "timestamp": 3.2, "event_type": "DODGED_ATTACK", "player_x": 140, "player_y": 100, "player_health": 100, "player_stamina": 50.0, "room_id": 0, "damage_avoided": 20},
            {"session_id": "session_test", "timestamp": 4.0, "event_type": "PLAYER_HURT", "player_x": 150, "player_y": 100, "player_health": 80, "player_stamina": 60.0, "room_id": 0, "damage": 20},
            {"session_id": "session_test", "timestamp": 6.0, "event_type": "SESSION_END", "timestamp": 6.0, "outcome": "VICTORY"},
        ]

        windows = extract_window_features(events, window_size_sec=5.0, stride_sec=2.5, seq_len=4)
        self.assertGreater(len(windows), 0)

        first_window = windows[0]
        self.assertEqual(first_window["session_id"], "session_test")
        self.assertEqual(first_window["session_outcome"], "VICTORY")
        self.assertEqual(first_window["damage_dealt"], 35)
        self.assertEqual(first_window["damage_received"], 20)
        self.assertEqual(first_window["block_frequency"], 0.0)
        self.assertGreater(first_window["attack_frequency"], 0.0)
        self.assertGreater(first_window["dodge_frequency"], 0.0)
        self.assertEqual(first_window["dodge_success_rate"], 1.0)

        tokens = [int(t) for t in first_window["action_tokens"].split(",")]
        self.assertEqual(len(tokens), 4)
        self.assertIn(ACTION_VOCAB["ATTACK"], tokens)
        self.assertIn(ACTION_VOCAB["DODGE"], tokens)
        self.assertIn(ACTION_VOCAB["HURT"], tokens)

    def test_05_style_vector_and_batch_processing(self):
        """Verifies session style vector aggregation and batch directory CSV processing."""
        s1_file = os.path.join(self.test_dir, "session_test_1.jsonl")
        events1 = [
            {"session_id": "session_test_1", "timestamp": 0.0, "event_type": "SESSION_START", "player_x": 0, "player_y": 0, "player_health": 100, "player_stamina": 100.0, "room_id": 0},
            {"session_id": "session_test_1", "timestamp": 1.0, "event_type": "PLAYER_ATTACK", "hits": 1, "damage_dealt": 35},
            {"session_id": "session_test_1", "timestamp": 2.0, "event_type": "PLAYER_ATTACK", "hits": 0, "damage_dealt": 0},
            {"session_id": "session_test_1", "timestamp": 3.0, "event_type": "SESSION_END", "timestamp": 3.0, "outcome": "DEFEAT"},
        ]
        with open(s1_file, "w", encoding="utf-8") as f:
            for ev in events1:
                f.write(json.dumps(ev) + "\n")

        style = extract_session_style_vector(events1)
        self.assertEqual(style["session_id"], "session_test_1")
        self.assertEqual(style["accuracy"], 0.5)

        proc_dir = os.path.join(self.test_dir, "processed")
        count, rows = process_raw_sessions(raw_dir=self.test_dir, processed_dir=proc_dir)

        self.assertEqual(count, 1)
        self.assertGreater(rows, 0)
        self.assertTrue(os.path.exists(os.path.join(proc_dir, "all_features.csv")))

    # --- Phase 2.1 Audit & Regression Tests ---

    def test_06_hit_and_damage_summary_consistency(self):
        """Regression test for Issue 1: Verifies attack hits and damage match session summary exactly."""
        self.telemetry.start_session(dungeon_seed=101)
        # Record hit dealing 35 damage
        self.telemetry.record_event("PLAYER_ATTACK", hits=1, damage_dealt=35)
        self.telemetry.sample_state(1.0, Player(100, 100), [], Dungeon(seed=101))
        summary = self.telemetry.end_session("DEFEAT")

        self.assertEqual(summary["attacks_swung"], 1)
        self.assertEqual(summary["attacks_hit"], 1)
        self.assertEqual(summary["damage_dealt"], 35)
        self.assertEqual(summary["accuracy"], 1.0)

    def test_07_attack_miss_accuracy(self):
        """Regression test: Verifies swung attack with 0 hits does not increment hits or damage."""
        self.telemetry.start_session(dungeon_seed=102)
        # Swing and miss
        self.telemetry.record_event("PLAYER_ATTACK", hits=0, damage_dealt=0)
        self.telemetry.sample_state(1.0, Player(100, 100), [], Dungeon(seed=102))
        summary = self.telemetry.end_session("DEFEAT")

        self.assertEqual(summary["attacks_swung"], 1)
        self.assertEqual(summary["attacks_hit"], 0)
        self.assertEqual(summary["damage_dealt"], 0)
        self.assertEqual(summary["accuracy"], 0.0)

    def test_08_enemy_death_count(self):
        """Regression test: Verifies defeated-enemy counter accurately tallies slain enemies."""
        self.telemetry.start_session(dungeon_seed=103)
        self.telemetry.record_event("ENEMY_KILLED", enemy_type="melee")
        self.telemetry.record_event("ENEMY_KILLED", enemy_type="ranged")
        self.telemetry.sample_state(1.0, Player(100, 100), [], Dungeon(seed=103))
        summary = self.telemetry.end_session("DEFEAT")

        self.assertEqual(summary["enemies_defeated"], 2)

    def test_09_incomplete_session_not_falsely_marked_victory(self):
        """Regression test for Issue 2: Incomplete human session cannot falsely claim VICTORY."""
        # Genuine human session ends without defeating any enemies
        self.telemetry.start_session(dungeon_seed=104, session_type="human")
        self.telemetry.sample_state(0.5, Player(100, 100), [], Dungeon(seed=104))
        summary = self.telemetry.end_session("VICTORY")

        # Must be guarded and downgraded to ABANDONED
        self.assertEqual(summary["final_outcome"], "ABANDONED")

    def test_10_processing_idempotence(self):
        """Regression test: Re-running feature processing multiple times never duplicates rows."""
        s_file = os.path.join(self.test_dir, "session_idem_test.jsonl")
        events = [
            {"session_id": "session_idem_test", "timestamp": 0.0, "event_type": "SESSION_START", "player_x": 0, "player_y": 0, "player_health": 100, "player_stamina": 100.0, "room_id": 0},
            {"session_id": "session_idem_test", "timestamp": 1.0, "event_type": "PLAYER_ATTACK", "hits": 1, "damage_dealt": 35},
            {"session_id": "session_idem_test", "timestamp": 3.0, "event_type": "SESSION_END", "timestamp": 3.0, "outcome": "DEFEAT"},
        ]
        with open(s_file, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        proc_dir = os.path.join(self.test_dir, "proc_idem")
        c1, r1 = process_raw_sessions(raw_dir=self.test_dir, processed_dir=proc_dir)
        c2, r2 = process_raw_sessions(raw_dir=self.test_dir, processed_dir=proc_dir)

        self.assertEqual(c1, c2)
        self.assertEqual(r1, r2)

    def test_11_synthetic_vs_genuine_separation(self):
        """Regression test: Synthetic sessions are tagged, prefixed, and excluded from human datasets by default."""
        synth_mgr = TelemetryManager(raw_data_dir=self.test_dir, session_type="synthetic")
        sid = synth_mgr.start_session()
        self.assertTrue(sid.startswith("synthetic_session_"))

        synth_mgr.sample_state(1.0, Player(100, 100), [], Dungeon(seed=55))
        summary = synth_mgr.end_session("VICTORY")
        self.assertEqual(summary["session_type"], "synthetic")

        # Process without synthetic flag: should process 0 sessions
        proc_dir = os.path.join(self.test_dir, "proc_synth")
        human_count, human_rows = process_raw_sessions(raw_dir=self.test_dir, processed_dir=proc_dir, include_synthetic=False)
        self.assertEqual(human_count, 0)
        self.assertEqual(human_rows, 0)

        # Process with include_synthetic=True: includes the synthetic session
        synth_count, synth_rows = process_raw_sessions(raw_dir=self.test_dir, processed_dir=proc_dir, include_synthetic=True)
        self.assertEqual(synth_count, 1)
        self.assertGreater(synth_rows, 0)


if __name__ == "__main__":
    unittest.main()
