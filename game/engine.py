"""
Game engine and master loop coordinator for Neural Dungeon.
Manages state transitions, camera lerping, input events, entity updates, and rendering.
"""

import math
import random
import pygame
import config
from game.dungeon import Dungeon
from game.entities import Player, MeleeEnemy, RangedEnemy
from game.combat import CombatSystem
from game.ui import UIManager
from game.telemetry import TelemetryManager
from ml.playstyle_detector import PlaystyleDetector
from ml.anomaly_detector import CombatAnomalyDetector
from ml.action_predictor import ActionPredictor
from ml.threat_predictor import RoomThreatPredictor


class GameEngine:
    """Master engine managing states, entities, camera, and the main game loop."""

    STATE_MENU = "menu"
    STATE_PLAYING = "playing"
    STATE_PAUSED = "paused"
    STATE_GAME_OVER = "game_over"
    STATE_VICTORY = "victory"

    def __init__(self, screen: pygame.Surface | None = None, raw_data_dir: str | None = "data/raw", session_type: str = "human"):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = self.STATE_MENU
        self.session_type = session_type

        # Visual camera (world coordinates centered on player)
        self.camera_x = 0.0
        self.camera_y = 0.0

        # UI Manager
        self.ui = UIManager()

        # Telemetry Pipeline
        self.telemetry = TelemetryManager(raw_data_dir=raw_data_dir, session_type=session_type)

        # Core systems (initialized in start_new_game)
        self.dungeon: Dungeon | None = None
        self.player: Player | None = None
        self.enemies: list[MeleeEnemy | RangedEnemy] = []
        self.combat: CombatSystem | None = None

        # Statistics for run summary
        self.elapsed_time = 0.0

        # Phase 3: Siamese Playstyle Similarity Detection
        self.playstyle_detector = PlaystyleDetector(model_path="saved_models/siamese_checkpoint.pt")
        self.detected_playstyle = "Balanced"
        self.playstyle_confidence = 0.0
        self.playstyle_eval_timer = 0.0

        # Phase 4: Combat Autoencoder Anomaly Detection
        self.anomaly_detector = CombatAnomalyDetector(model_path="saved_models/autoencoder_checkpoint.pt")
        self.is_anomaly = False
        self.reconstruction_error = 0.0
        self.anomaly_threshold = 0.0

        # Phase 5: Transformer Next-Action Prediction
        self.action_predictor = ActionPredictor(model_path="saved_models/transformer_checkpoint.pt")
        self.predicted_action = "NONE"
        self.action_confidence = 0.0

        # Phase 6: GCN Room Threat Prediction (Defaults to Leakage-Controlled 5-Feature Model)
        self.threat_predictor = RoomThreatPredictor(model_path="saved_models/gcn_checkpoint_controlled.pt")
        self.room_threats: dict[int, dict] = {}
        self.current_room_id: int | None = None
        self.current_room_threat: dict = {"threat": "UNKNOWN", "confidence": 0.0, "is_fallback": True}

    def start_new_game(self, seed: int | None = None, session_type: str | None = None) -> None:
        """Initializes a fresh dungeon floor, player, and enemy roster."""
        # Ensure previous session is finalized
        if self.telemetry and self.telemetry.session_active:
            self.telemetry.end_session("ABANDONED")

        # Start telemetry tracking for new session
        if self.telemetry:
            self.telemetry.start_session(dungeon_seed=seed, session_type=session_type or self.session_type)

        self.dungeon = Dungeon(seed=seed)
        # Phase 6: Predict room threats across the dungeon floor using GCN
        self.room_threats = self.threat_predictor.predict_dungeon_threats(self.dungeon)
        self.dungeon.room_threats = self.room_threats

        # Place player at center of spawn room
        spawn_px, spawn_py = self.dungeon.spawn_room.center_pixel
        self.player = Player(spawn_px, spawn_py)

        # Center camera immediately on player
        self.camera_x = self.player.x - config.SCREEN_WIDTH / 2
        self.camera_y = self.player.y - config.SCREEN_HEIGHT / 2

        # Spawn enemies in non-spawn rooms
        self.enemies = []
        for room in self.dungeon.rooms:
            if room.room_type == "spawn":
                continue

            cx, cy = room.center_pixel
            # Spawn 1 Melee and 1 Ranged enemy in each combat room
            m_enemy = MeleeEnemy(cx - 30.0, cy)
            r_enemy = RangedEnemy(cx + 30.0, cy)
            self.enemies.extend([m_enemy, r_enemy])

        # Initialize combat manager with telemetry hook
        self.combat = CombatSystem(self.player, self.enemies, self.dungeon, telemetry=self.telemetry)

        self.elapsed_time = 0.0
        self.detected_playstyle = "Balanced"
        self.playstyle_confidence = 0.0
        self.playstyle_eval_timer = 0.0
        self.is_anomaly = False
        self.reconstruction_error = 0.0
        self.predicted_action = "NONE"
        self.action_confidence = 0.0
        self.state = self.STATE_PLAYING

    def get_run_stats(self) -> dict:
        """Returns statistics snapshot for Game Over or Victory screens."""
        return {
            "time_sec": self.elapsed_time,
            "defeated": self.player.enemies_defeated if self.player else 0,
            "damage_dealt": self.player.damage_dealt if self.player else 0,
            "damage_taken": self.player.damage_taken if self.player else 0,
            "dodges": self.player.dodges_performed if self.player else 0,
        }

    def handle_input_events(self) -> None:
        """Processes keyboard, mouse, and UI button clicks."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            # --- UI Button Interactions based on current state ---
            if self.state == self.STATE_MENU:
                if self.ui.show_controls_overlay:
                    if event.type == pygame.KEYDOWN and event.key in (pygame.K_h, pygame.K_ESCAPE):
                        self.ui.show_controls_overlay = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        self.ui.show_controls_overlay = False
                    continue

                for btn in self.ui.menu_buttons:
                    action = btn.handle_event(event)
                    if action == "start":
                        self.start_new_game()
                    elif action == "controls":
                        self.ui.show_controls_overlay = True
                    elif action == "quit":
                        self.running = False

            elif self.state == self.STATE_PAUSED:
                for btn in self.ui.pause_buttons:
                    action = btn.handle_event(event)
                    if action == "resume":
                        self.state = self.STATE_PLAYING
                    elif action == "restart":
                        self.start_new_game()
                    elif action == "menu":
                        if self.telemetry.session_active:
                            self.telemetry.end_session("ABANDONED")
                        self.state = self.STATE_MENU

                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_p):
                    self.state = self.STATE_PLAYING

            elif self.state == self.STATE_GAME_OVER:
                for btn in self.ui.game_over_buttons:
                    action = btn.handle_event(event)
                    if action == "restart":
                        self.start_new_game()
                    elif action == "menu":
                        self.state = self.STATE_MENU

            elif self.state == self.STATE_VICTORY:
                for btn in self.ui.victory_buttons:
                    action = btn.handle_event(event)
                    if action == "restart":
                        self.start_new_game()
                    elif action == "menu":
                        self.state = self.STATE_MENU

            elif self.state == self.STATE_PLAYING:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_p):
                        self.state = self.STATE_PAUSED
                    elif event.key == pygame.K_r:
                        self.start_new_game()
                    elif event.key == pygame.K_h:
                        self.ui.show_controls_overlay = not self.ui.show_controls_overlay
                    elif event.key == pygame.K_SPACE:
                        # Spacebar triggers dodge roll
                        keys = pygame.key.get_pressed()
                        mx = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_a] or keys[pygame.K_LEFT])
                        my = float(keys[pygame.K_s] or keys[pygame.K_DOWN]) - float(keys[pygame.K_w] or keys[pygame.K_UP])
                        self._trigger_player_dodge(mx, my)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # Left click = Melee attack swing
                        self.combat.execute_player_melee()
                    elif event.button == 3:
                        # Right click = Dodge roll
                        keys = pygame.key.get_pressed()
                        mx = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_a] or keys[pygame.K_LEFT])
                        my = float(keys[pygame.K_s] or keys[pygame.K_DOWN]) - float(keys[pygame.K_w] or keys[pygame.K_UP])
                        self._trigger_player_dodge(mx, my)

    def _trigger_player_dodge(self, mx: float, my: float) -> None:
        """Executes player dodge and logs telemetry."""
        if not self.player or not self.dungeon:
            return
        success = self.player.start_dodge((mx, my))
        if self.telemetry:
            room = self.dungeon.get_room_at(self.player.x, self.player.y)
            room_id = room.id if room else -1
            self.telemetry.record_event(
                "PLAYER_DODGE",
                player_x=self.player.x,
                player_y=self.player.y,
                player_health=self.player.health,
                player_stamina=self.player.stamina,
                room_id=room_id,
                move_dir=self.player.dodge_dir,
                success=success,
            )

    def update(self, dt: float) -> None:
        """Main update step for simulation physics, enemies, and win/loss checks."""
        if self.state != self.STATE_PLAYING or not self.player or not self.dungeon:
            return

        self.elapsed_time += dt

        # Read continuous movement inputs
        keys = pygame.key.get_pressed()
        move_x = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_a] or keys[pygame.K_LEFT])
        move_y = float(keys[pygame.K_s] or keys[pygame.K_DOWN]) - float(keys[pygame.K_w] or keys[pygame.K_UP])

        # Mouse world position for aiming
        mouse_screen_x, mouse_screen_y = pygame.mouse.get_pos()
        mouse_world = (mouse_screen_x + self.camera_x, mouse_screen_y + self.camera_y)

        # Update player
        self.player.update(dt, self.dungeon, (move_x, move_y), mouse_world)

        # Periodic telemetry state sampling (5 Hz)
        if self.telemetry:
            self.telemetry.sample_state(dt, self.player, self.enemies, self.dungeon)

        # Check death
        if not self.player.is_alive:
            self.state = self.STATE_GAME_OVER
            if self.telemetry:
                self.telemetry.end_session("DEFEAT")
            return

        # Siamese playstyle detection & Autoencoder anomaly monitoring at controlled interval (2.0s)
        self.playstyle_eval_timer += dt
        if self.playstyle_eval_timer >= 2.0:
            self.playstyle_eval_timer = 0.0
            if self.telemetry and self.telemetry.events:
                # 1. Siamese Playstyle Inference
                style, conf = self.playstyle_detector.predict_playstyle(self.telemetry.events[-40:])
                self.detected_playstyle = style
                self.playstyle_confidence = conf

                # 2. Independent Autoencoder Anomaly Detection
                is_anom, err, thresh = self.anomaly_detector.detect_anomaly(self.telemetry.events[-40:])
                self.is_anomaly = is_anom
                self.reconstruction_error = err
                self.anomaly_threshold = thresh

                # 3. Independent Transformer Next-Action Prediction
                act_name, act_conf = self.action_predictor.predict_next_action(self.telemetry.events[-40:])
                self.predicted_action = act_name
                self.action_confidence = act_conf

        # Update enemies AI with Siamese playstyle & Transformer action adaptation
        for enemy in self.enemies:
            if enemy.is_alive:
                enemy.update_ai(
                    dt,
                    self.player,
                    self.dungeon,
                    self.combat,
                    playstyle=self.detected_playstyle,
                    confidence=self.playstyle_confidence,
                    predicted_action=self.predicted_action,
                    action_confidence=self.action_confidence,
                )

        # Update combat projectiles, particles, and slashes
        self.combat.update(dt)

        # Check objective (Exit portal activates when all enemies are slain)
        alive_enemies = sum(1 for e in self.enemies if e.is_alive)
        if alive_enemies == 0:
            self.dungeon.exit_active = True

        # Check victory condition (player touching active exit portal)
        if self.dungeon.exit_active:
            ex, ey = self.dungeon.exit_pos
            if math.hypot(self.player.x - ex, self.player.y - ey) < (config.TILE_SIZE * 0.75):
                self.state = self.STATE_VICTORY
                if self.telemetry:
                    self.telemetry.end_session("VICTORY")
                return

        # Update current room and threat context for HUD (Phase 6 GCN)
        if self.dungeon and self.player:
            cur_r = self.dungeon.get_room_at(self.player.x, self.player.y)
            if cur_r:
                self.current_room_id = cur_r.id
                self.current_room_threat = self.threat_predictor.get_threat_for_room(cur_r.id)
            else:
                self.current_room_id = None
                self.current_room_threat = {"threat": "TRANSIT", "confidence": 1.0, "is_fallback": False}

        # Smooth camera lerp toward player
        target_cam_x = self.player.x - config.SCREEN_WIDTH / 2
        target_cam_y = self.player.y - config.SCREEN_HEIGHT / 2
        self.camera_x += (target_cam_x - self.camera_x) * config.CAMERA_LERP
        self.camera_y += (target_cam_y - self.camera_y) * config.CAMERA_LERP

    def render(self) -> None:
        """Renders the world, entities, HUD, or overlay screens onto self.screen."""
        if not self.screen:
            return

        cam_offset = (self.camera_x, self.camera_y)

        if self.state == self.STATE_MENU:
            self.ui.render_main_menu(self.screen)

        elif self.state in (self.STATE_PLAYING, self.STATE_PAUSED, self.STATE_GAME_OVER, self.STATE_VICTORY):
            self.screen.fill(config.COLOR_BG)

            # 1. Dungeon floor and walls
            self.dungeon.render(self.screen, cam_offset)

            # 2. Enemies
            for enemy in self.enemies:
                if enemy.is_alive:
                    enemy.render(self.screen, cam_offset)

            # 3. Player
            if self.player:
                self.player.render(self.screen, cam_offset)

            # 4. Combat slashes, projectiles, particles, damage text
            self.combat.render(self.screen, cam_offset)

            # 5. Minimap and HUD
            alive_enemies = sum(1 for e in self.enemies if e.is_alive)
            self.dungeon.render_minimap(self.screen, (self.player.x, self.player.y), self.enemies)
            self.ui.render_hud(
                self.screen,
                self.player,
                alive_enemies,
                self.dungeon.exit_active,
                playstyle=self.detected_playstyle,
                confidence=self.playstyle_confidence,
                is_anomaly=self.is_anomaly,
                recon_error=self.reconstruction_error,
                anomaly_thresh=self.anomaly_threshold,
                predicted_action=self.predicted_action,
                action_confidence=self.action_confidence,
                room_threat=self.current_room_threat,
                room_id=self.current_room_id,
            )

            # 6. Overlays
            if self.ui.show_controls_overlay and self.state == self.STATE_PLAYING:
                self.ui.render_controls_modal(self.screen)
            elif self.state == self.STATE_PAUSED:
                self.ui.render_pause(self.screen)
            elif self.state == self.STATE_GAME_OVER:
                self.ui.render_game_over(self.screen, self.get_run_stats())
            elif self.state == self.STATE_VICTORY:
                self.ui.render_victory(self.screen, self.get_run_stats())

        pygame.display.flip()

    def run(self) -> None:
        """Standard game execution loop."""
        while self.running:
            dt = self.clock.tick(config.TARGET_FPS) / 1000.0
            dt = min(dt, 0.05)  # Cap delta-time to prevent physics spikes

            self.handle_input_events()
            self.update(dt)
            self.render()

        if self.telemetry and self.telemetry.session_active:
            self.telemetry.end_session("ABANDONED")

        pygame.quit()
