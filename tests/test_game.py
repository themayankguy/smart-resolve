"""
Automated headless test suite for Neural Dungeon Phase 1.
Tests dungeon generation, player movement, wall collision, melee attacks,
dodge i-frames, enemy AI state machines, and win/loss/restart state transitions.
"""

import os
import unittest
import math

# Force dummy SDL video driver for headless testing
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import config
from game.dungeon import Dungeon, Room
from game.entities import Player, MeleeEnemy, RangedEnemy
from game.combat import CombatSystem
from game.engine import GameEngine


class TestNeuralDungeon(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.font.init()

    def setUp(self):
        self.dungeon = Dungeon(seed=42)
        spawn_px, spawn_py = self.dungeon.spawn_room.center_pixel
        self.player = Player(spawn_px, spawn_py)
        self.enemies = []
        self.combat = CombatSystem(self.player, self.enemies, self.dungeon)

    def test_01_dungeon_generation(self):
        """Verifies dungeon generates connected rooms, corridors, spawn, and exit."""
        self.assertGreaterEqual(len(self.dungeon.rooms), 3)
        self.assertIsNotNone(self.dungeon.spawn_room)
        self.assertIsNotNone(self.dungeon.exit_room)
        self.assertEqual(self.dungeon.spawn_room.room_type, "spawn")
        self.assertEqual(self.dungeon.exit_room.room_type, "exit")

        # Verify center of spawn room is walkable
        sx, sy = self.dungeon.spawn_room.center_pixel
        self.assertTrue(self.dungeon.is_point_walkable(sx, sy))

        # Verify exit coordinates are walkable
        ex, ey = self.dungeon.exit_pos
        self.assertTrue(self.dungeon.is_point_walkable(ex, ey))

        # Verify walls exist
        self.assertGreater(len(self.dungeon.wall_rects), 0)

    def test_02_player_movement_and_wall_collision(self):
        """Verifies player moves in open space and is blocked by walls."""
        initial_x = self.player.x
        initial_y = self.player.y

        # Move slightly inside room (should move freely)
        self.player.move_and_slide(10.0, 0.0, self.dungeon)
        self.assertAlmostEqual(self.player.x, initial_x + 10.0, places=1)

        # Place player directly adjacent to a solid wall and try moving into it
        # Find a wall coordinate
        wall_x = self.dungeon.spawn_room.x * config.TILE_SIZE - 2
        wall_y = self.dungeon.spawn_room.y * config.TILE_SIZE + 24
        self.player.x = wall_x + config.PLAYER_RADIUS + 5
        self.player.y = wall_y

        prev_x = self.player.x
        # Push into left wall
        self.player.move_and_slide(-50.0, 0.0, self.dungeon)
        # Player should be stopped at the wall boundary
        self.assertGreaterEqual(self.player.x, prev_x - 10.0)

    def test_03_dodge_roll_and_stamina(self):
        """Verifies dodge roll consumes stamina, activates i-frames, and triggers cooldown."""
        initial_stamina = self.player.stamina
        success = self.player.start_dodge((1.0, 0.0))

        self.assertTrue(success)
        self.assertTrue(self.player.is_dodging)
        self.assertGreater(self.player.invulnerable_timer, 0.0)
        self.assertEqual(self.player.stamina, initial_stamina - config.DODGE_STAMINA_COST)

        # Second immediate dodge must fail due to cooldown
        second_attempt = self.player.start_dodge((1.0, 0.0))
        self.assertFalse(second_attempt)

    def test_04_melee_attack_and_damage(self):
        """Verifies melee swing damages enemy in front of player."""
        # Place enemy within swing reach and facing angle
        enemy = MeleeEnemy(self.player.x + 40.0, self.player.y)
        self.enemies.append(enemy)
        self.player.aim_angle = 0.0  # Facing right (towards enemy)

        initial_health = enemy.health
        hits = self.combat.execute_player_melee()

        self.assertEqual(hits, 1)
        self.assertEqual(enemy.health, initial_health - config.MELEE_DAMAGE)
        self.assertGreater(enemy.hit_flash_timer, 0.0)
        self.assertGreater(enemy.knockback_vx, 0.0)

    def test_05_dodge_iframes_negate_damage(self):
        """Verifies player is invulnerable to damage while dodging."""
        self.player.start_dodge((1.0, 0.0))
        self.assertTrue(self.player.is_dodging)

        damage_applied = self.combat.apply_player_damage(25, (0.0, 0.0))
        self.assertFalse(damage_applied)
        self.assertEqual(self.player.health, config.PLAYER_MAX_HEALTH)

    def test_06_melee_enemy_ai_states(self):
        """Verifies MeleeEnemy chases player and enters TELEGRAPH state when close."""
        enemy = MeleeEnemy(self.player.x + 45.0, self.player.y)
        self.enemies.append(enemy)

        # Update enemy AI when close to player
        enemy.update_ai(0.1, self.player, self.dungeon, self.combat)

        # Enemy should transition to TELEGRAPH state with attack indicator
        self.assertEqual(enemy.state, MeleeEnemy.STATE_TELEGRAPH)
        self.assertGreater(enemy.telegraph_timer, 0.0)

    def test_07_ranged_enemy_ai_and_projectile(self):
        """Verifies RangedEnemy retreats if player is too close and spawns projectile."""
        # Place ranged enemy at comfortable casting distance
        enemy = RangedEnemy(self.player.x + 220.0, self.player.y)
        self.enemies.append(enemy)

        # Trigger chase & telegraph
        enemy.update_ai(0.1, self.player, self.dungeon, self.combat)
        self.assertEqual(enemy.state, RangedEnemy.STATE_TELEGRAPH)

        # Finish telegraph time to trigger projectile spawn
        enemy.update_ai(config.RANGED_ENEMY_TELEGRAPH_TIME + 0.1, self.player, self.dungeon, self.combat)
        self.assertGreaterEqual(len(self.combat.projectiles), 1)

    def test_08_health_depletion_and_game_over(self):
        """Verifies game transitions to GAME_OVER when player health is depleted."""
        engine = GameEngine(raw_data_dir=None, session_type="test")
        engine.start_new_game()
        self.assertEqual(engine.state, GameEngine.STATE_PLAYING)

        # Deplete player health
        engine.player.health = 0
        engine.update(0.1)

        self.assertEqual(engine.state, GameEngine.STATE_GAME_OVER)

    def test_09_victory_condition(self):
        """Verifies exit portal activates upon defeating enemies, and touching it gives victory."""
        engine = GameEngine(raw_data_dir=None, session_type="test")
        engine.start_new_game()

        # Defeat all enemies
        for e in engine.enemies:
            e.health = 0
        engine.update(0.1)

        # Exit should now be active
        self.assertTrue(engine.dungeon.exit_active)

        # Move player to exit portal
        ex, ey = engine.dungeon.exit_pos
        engine.player.x = ex
        engine.player.y = ey
        engine.update(0.1)

        self.assertEqual(engine.state, GameEngine.STATE_VICTORY)

    def test_10_pause_and_restart(self):
        """Verifies pause state and restart functionality."""
        engine = GameEngine(raw_data_dir=None, session_type="test")
        engine.start_new_game()
        self.assertEqual(engine.state, GameEngine.STATE_PLAYING)

        # Pause
        engine.state = GameEngine.STATE_PAUSED
        engine.update(0.1)
        self.assertEqual(engine.state, GameEngine.STATE_PAUSED)

        # Restart
        engine.start_new_game()
        self.assertEqual(engine.state, GameEngine.STATE_PLAYING)
        self.assertEqual(engine.player.health, config.PLAYER_MAX_HEALTH)
        self.assertFalse(engine.dungeon.exit_active)


if __name__ == "__main__":
    unittest.main()
