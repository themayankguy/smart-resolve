"""
User interface, HUD, menus, and overlays for Neural Dungeon.
Includes Health/Stamina bars, Main Menu, Pause, Victory, and Game Over screens.
"""

import pygame
import config


class Button:
    """Interactive button with hover states and callback action."""

    def __init__(self, rect: tuple[int, int, int, int], text: str, font: pygame.font.Font, action_id: str):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.action_id = action_id
        self.is_hovered = False

    def handle_event(self, event: pygame.event.Event) -> str | None:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return self.action_id
        return None

    def render(self, surface: pygame.Surface) -> None:
        bg_color = (65, 50, 80) if self.is_hovered else (38, 30, 48)
        border_color = config.COLOR_GOLD if self.is_hovered else (80, 70, 95)
        text_color = (255, 255, 255) if self.is_hovered else (210, 205, 220)

        pygame.draw.rect(surface, bg_color, self.rect, border_radius=6)
        pygame.draw.rect(surface, border_color, self.rect, width=2, border_radius=6)

        txt_surf = self.font.render(self.text, True, text_color)
        tx = self.rect.centerx - txt_surf.get_width() // 2
        ty = self.rect.centery - txt_surf.get_height() // 2
        surface.blit(txt_surf, (tx, ty))


class UIManager:
    """Manages all game UI states: HUD, Main Menu, Pause, Defeat, Victory."""

    def __init__(self, screen_width: int = config.SCREEN_WIDTH, screen_height: int = config.SCREEN_HEIGHT):
        self.w = screen_width
        self.h = screen_height

        # Fonts
        self.title_font = pygame.font.SysFont("Georgia", 42, bold=True)
        self.header_font = pygame.font.SysFont("Georgia", 28, bold=True)
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 13)

        self.show_controls_overlay = False

        # Menu Buttons
        btn_w, btn_h = 240, 50
        cx = self.w // 2 - btn_w // 2
        self.menu_buttons = [
            Button((cx, 340, btn_w, btn_h), "START EXPEDITION", self.font, "start"),
            Button((cx, 410, btn_w, btn_h), "CONTROLS GUIDE", self.font, "controls"),
            Button((cx, 480, btn_w, btn_h), "QUIT GAME", self.font, "quit"),
        ]

        self.pause_buttons = [
            Button((cx, 280, btn_w, btn_h), "RESUME", self.font, "resume"),
            Button((cx, 350, btn_w, btn_h), "RESTART", self.font, "restart"),
            Button((cx, 420, btn_w, btn_h), "MAIN MENU", self.font, "menu"),
        ]

        self.game_over_buttons = [
            Button((cx, 400, btn_w, btn_h), "TRY AGAIN", self.font, "restart"),
            Button((cx, 470, btn_w, btn_h), "MAIN MENU", self.font, "menu"),
        ]

        self.victory_buttons = [
            Button((cx, 420, btn_w, btn_h), "NEXT EXPEDITION", self.font, "restart"),
            Button((cx, 490, btn_w, btn_h), "MAIN MENU", self.font, "menu"),
        ]

    def render_hud(
        self,
        surface: pygame.Surface,
        player,
        enemies_remaining: int,
        exit_active: bool,
        playstyle: str = "Balanced",
        confidence: float = 0.0,
        is_anomaly: bool = False,
        recon_error: float = 0.0,
        anomaly_thresh: float = 0.0,
        predicted_action: str = "NONE",
        action_confidence: float = 0.0,
        room_threat: dict | None = None,
        room_id: int | None = None,
    ) -> None:
        """Renders player Health/Stamina bars, objectives, Siamese playstyle, Autoencoder anomaly status, Transformer next-action forecast, GCN room threat, and controls reminder."""
        # --- Health Bar ---
        bar_x, bar_y = 25, 25
        bar_w, bar_h = 220, 20

        # Health background
        pygame.draw.rect(surface, config.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        hp_pct = max(0.0, player.health / player.max_health)
        pygame.draw.rect(surface, config.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_w * hp_pct), bar_h), border_radius=4)
        pygame.draw.rect(surface, (120, 30, 30), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=4)

        hp_txt = self.small_font.render(f"HP: {player.health} / {player.max_health}", True, (255, 255, 255))
        surface.blit(hp_txt, (bar_x + 8, bar_y + 2))

        # --- Stamina Bar ---
        st_y = bar_y + 28
        pygame.draw.rect(surface, config.COLOR_STAMINA_BG, (bar_x, st_y, bar_w, 14), border_radius=3)
        st_pct = max(0.0, player.stamina / player.max_stamina)
        pygame.draw.rect(surface, config.COLOR_STAMINA_BAR, (bar_x, st_y, int(bar_w * st_pct), 14), border_radius=3)
        pygame.draw.rect(surface, (140, 90, 20), (bar_x, st_y, bar_w, 14), 1, border_radius=3)

        st_txt = self.small_font.render(f"STAMINA: {int(player.stamina)} / {int(player.max_stamina)}", True, (255, 255, 220))
        surface.blit(st_txt, (bar_x + 8, st_y))

        # --- Objective Panel ---
        obj_panel = pygame.Rect(25, 80, 250, 48)
        pygame.draw.rect(surface, (20, 16, 26, 200), obj_panel, border_radius=5)
        pygame.draw.rect(surface, (60, 50, 75), obj_panel, 1, border_radius=5)

        if not exit_active:
            obj_txt = self.small_font.render(f"OBJECTIVE: Slay Dungeon Foes ({enemies_remaining} left)", True, (230, 180, 100))
            sub_txt = self.small_font.render("Portal unlocks when dungeon is purged.", True, (170, 160, 180))
        else:
            obj_txt = self.small_font.render("OBJECTIVE: Reach the Golden Exit Portal!", True, (100, 240, 140))
            sub_txt = self.small_font.render("Sanctum unlocked — step into the portal.", True, (241, 196, 15))

        surface.blit(obj_txt, (obj_panel.x + 10, obj_panel.y + 6))
        surface.blit(sub_txt, (obj_panel.x + 10, obj_panel.y + 26))

        # --- AI Perception Badge (Phase 3 Siamese Network) ---
        ai_panel = pygame.Rect(25, 136, 250, 38)
        pygame.draw.rect(surface, (20, 16, 26, 200), ai_panel, border_radius=5)
        pygame.draw.rect(surface, (80, 70, 105), ai_panel, 1, border_radius=5)

        style_color = (255, 100, 100) if playstyle == "Aggressive" else ((100, 200, 255) if playstyle == "Evasive" else (220, 220, 220))
        ai_txt = self.small_font.render(f"AI ADAPTATION: {playstyle}", True, style_color)
        conf_txt = self.small_font.render(f"Siamese Similarity: {confidence:.2f} (Thresh: 0.45)", True, (160, 160, 180))
        surface.blit(ai_txt, (ai_panel.x + 10, ai_panel.y + 4))
        surface.blit(conf_txt, (ai_panel.x + 10, ai_panel.y + 20))

        # --- Anomaly Indicator (Phase 4 Combat Autoencoder) ---
        anom_panel = pygame.Rect(25, 180, 250, 38)
        pygame.draw.rect(surface, (20, 16, 26, 200), anom_panel, border_radius=5)

        if is_anomaly:
            pygame.draw.rect(surface, (220, 140, 20), anom_panel, 1, border_radius=5)
            anom_title = self.small_font.render("COMBAT REGIME: UNUSUAL", True, (243, 156, 18))
            anom_sub = self.small_font.render(f"Recon MSE: {recon_error:.3f} > {anomaly_thresh:.3f}", True, (255, 200, 120))
        else:
            pygame.draw.rect(surface, (50, 65, 85), anom_panel, 1, border_radius=5)
            anom_title = self.small_font.render("COMBAT REGIME: NOMINAL", True, (130, 210, 160))
            anom_sub = self.small_font.render(f"Recon MSE: {recon_error:.3f} (Thresh: {anomaly_thresh:.3f})", True, (150, 160, 175))

        surface.blit(anom_title, (anom_panel.x + 10, anom_panel.y + 4))
        surface.blit(anom_sub, (anom_panel.x + 10, anom_panel.y + 20))

        # --- Next-Action Forecast (Phase 5 Transformer) ---
        act_panel = pygame.Rect(25, 224, 250, 38)
        pygame.draw.rect(surface, (20, 16, 26, 200), act_panel, border_radius=5)

        if predicted_action != "NONE" and action_confidence >= 0.45:
            pygame.draw.rect(surface, (100, 70, 140), act_panel, 1, border_radius=5)
            act_color = (241, 196, 15) if predicted_action == "DODGE" else ((231, 76, 60) if predicted_action == "ATTACK" else (100, 200, 255))
            act_title = self.small_font.render(f"ACTION FORECAST: {predicted_action}", True, act_color)
            act_sub = self.small_font.render(f"Transformer Conf: {action_confidence:.2f} (Thresh: 0.45)", True, (190, 175, 215))
        else:
            pygame.draw.rect(surface, (50, 50, 70), act_panel, 1, border_radius=5)
            act_title = self.small_font.render("ACTION FORECAST: NONE", True, (140, 140, 160))
            act_sub = self.small_font.render(f"Transformer Conf: {action_confidence:.2f} (< 0.45)", True, (130, 130, 150))

        surface.blit(act_title, (act_panel.x + 10, act_panel.y + 4))
        surface.blit(act_sub, (act_panel.x + 10, act_panel.y + 20))

        # --- Room Threat Assessment (Phase 6 GCN) ---
        threat_panel = pygame.Rect(25, 268, 250, 38)
        pygame.draw.rect(surface, (20, 16, 26, 200), threat_panel, border_radius=5)

        t_data = room_threat or {}
        threat_cat = t_data.get("threat", "UNKNOWN")
        threat_conf = t_data.get("confidence", 0.0)

        if threat_cat == "LOW":
            t_color = (46, 204, 113)  # Green
            border_color = (35, 120, 70)
        elif threat_cat == "MEDIUM":
            t_color = (241, 196, 15)  # Amber
            border_color = (140, 110, 20)
        elif threat_cat == "HIGH":
            t_color = (231, 76, 60)   # Red
            border_color = (150, 40, 40)
        elif threat_cat == "TRANSIT":
            t_color = (175, 160, 200) # Purple/gray
            border_color = (70, 60, 85)
        else:
            t_color = (160, 160, 160)
            border_color = (60, 60, 60)

        pygame.draw.rect(surface, border_color, threat_panel, 1, border_radius=5)
        threat_title = self.small_font.render(f"ROOM THREAT: {threat_cat}", True, t_color)

        is_fallback = t_data.get("is_fallback", False)
        if is_fallback:
            threat_sub = self.small_font.render(f"GCN Threat: Fallback Heuristic", True, (160, 150, 170))
        else:
            threat_sub = self.small_font.render(f"GCN Threat Conf: {threat_conf:.2f} (Controlled)", True, (170, 165, 180))

        surface.blit(threat_title, (threat_panel.x + 10, threat_panel.y + 4))
        surface.blit(threat_sub, (threat_panel.x + 10, threat_panel.y + 20))

        # --- Bottom Controls Prompt ---
        tip_surf = self.small_font.render("WASD: Move | Left-Click: Melee Swing | Space/Right-Click: Dodge Roll | ESC: Pause | H: Controls", True, (150, 140, 165))
        surface.blit(tip_surf, (self.w // 2 - tip_surf.get_width() // 2, self.h - 26))

    def render_controls_modal(self, surface: pygame.Surface) -> None:
        """Renders detailed controls pop-up overlay."""
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill((10, 8, 14, 230))
        surface.blit(overlay, (0, 0))

        box = pygame.Rect(self.w // 2 - 280, self.h // 2 - 200, 560, 400)
        pygame.draw.rect(surface, (28, 22, 36), box, border_radius=10)
        pygame.draw.rect(surface, config.COLOR_GOLD, box, 2, border_radius=10)

        title = self.header_font.render("COMBAT CONTROLS GUIDE", True, config.COLOR_GOLD)
        surface.blit(title, (box.centerx - title.get_width() // 2, box.y + 25))

        lines = [
            ("WASD / Arrow Keys", "Move player character (smooth wall sliding)"),
            ("Mouse Cursor", "Aim directional weapon swing and facing angle"),
            ("Left Mouse Click", "Melee Sword Slash (cleaves enemies in swing arc)"),
            ("Space / Right Click", "Dodge Roll (i-frames: immune to damage, costs stamina)"),
            ("ESC / P", "Pause or resume current expedition"),
            ("R", "Quick restart run"),
            ("H", "Toggle this controls guide overlay"),
        ]

        y_off = box.y + 80
        for key, desc in lines:
            k_surf = self.font.render(key, True, (140, 200, 255))
            d_surf = self.small_font.render(desc, True, (210, 205, 220))
            surface.blit(k_surf, (box.x + 35, y_off))
            surface.blit(d_surf, (box.x + 210, y_off + 2))
            y_off += 38

        close_txt = self.small_font.render("Press 'H' or Click anywhere to close", True, (160, 150, 175))
        surface.blit(close_txt, (box.centerx - close_txt.get_width() // 2, box.bottom - 35))

    def render_main_menu(self, surface: pygame.Surface) -> None:
        """Renders dark-fantasy title screen with buttons."""
        surface.fill(config.COLOR_BG)

        # Title decoration
        title = self.title_font.render("NEURAL DUNGEON", True, config.COLOR_GOLD)
        subtitle = self.header_font.render("Advanced Techniques in Machine Learning (ATML)", True, (180, 175, 200))
        flavor = self.small_font.render("A 2D Dark-Fantasy Research Crawler & Metric Testbed", True, (140, 130, 155))

        surface.blit(title, (self.w // 2 - title.get_width() // 2, 130))
        surface.blit(subtitle, (self.w // 2 - subtitle.get_width() // 2, 195))
        surface.blit(flavor, (self.w // 2 - flavor.get_width() // 2, 235))

        # Render menu buttons
        for btn in self.menu_buttons:
            btn.render(surface)

        if self.show_controls_overlay:
            self.render_controls_modal(surface)

    def render_pause(self, surface: pygame.Surface) -> None:
        """Renders pause translucent overlay."""
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill((10, 8, 14, 190))
        surface.blit(overlay, (0, 0))

        title = self.header_font.render("EXPEDITION PAUSED", True, config.COLOR_GOLD)
        surface.blit(title, (self.w // 2 - title.get_width() // 2, 190))

        for btn in self.pause_buttons:
            btn.render(surface)

    def render_game_over(self, surface: pygame.Surface, stats: dict) -> None:
        """Renders defeat screen with run statistics."""
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill((20, 5, 8, 225))
        surface.blit(overlay, (0, 0))

        title = self.title_font.render("YOU HAVE FALLEN", True, (230, 60, 60))
        sub = self.small_font.render("The dungeon claimed your soul.", True, (190, 150, 160))
        surface.blit(title, (self.w // 2 - title.get_width() // 2, 130))
        surface.blit(sub, (self.w // 2 - sub.get_width() // 2, 190))

        # Stats summary box
        box = pygame.Rect(self.w // 2 - 220, 225, 440, 140)
        pygame.draw.rect(surface, (25, 18, 24), box, border_radius=8)
        pygame.draw.rect(surface, (90, 40, 45), box, 1, border_radius=8)

        stats_items = [
            f"Time Survived: {stats.get('time_sec', 0):.1f}s",
            f"Enemies Defeated: {stats.get('defeated', 0)}",
            f"Damage Dealt: {stats.get('damage_dealt', 0)}",
            f"Damage Taken: {stats.get('damage_taken', 0)}",
            f"Dodges Performed: {stats.get('dodges', 0)}",
        ]
        sy = box.y + 16
        for line in stats_items:
            t_surf = self.small_font.render(line, True, (220, 210, 220))
            surface.blit(t_surf, (box.x + 25, sy))
            sy += 24

        for btn in self.game_over_buttons:
            btn.render(surface)

    def render_victory(self, surface: pygame.Surface, stats: dict) -> None:
        """Renders victory screen when player escapes through portal."""
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill((8, 18, 12, 225))
        surface.blit(overlay, (0, 0))

        title = self.title_font.render("DUNGEON PURGED — SANCTUM ESCAPED!", True, config.COLOR_GOLD)
        sub = self.small_font.render("You conquered all dungeon threats and claimed mastery.", True, (160, 220, 180))
        surface.blit(title, (self.w // 2 - title.get_width() // 2, 140))
        surface.blit(sub, (self.w // 2 - sub.get_width() // 2, 205))

        # Stats summary box
        box = pygame.Rect(self.w // 2 - 220, 245, 440, 140)
        pygame.draw.rect(surface, (18, 30, 24), box, border_radius=8)
        pygame.draw.rect(surface, (45, 95, 60), box, 1, border_radius=8)

        stats_items = [
            f"Clear Time: {stats.get('time_sec', 0):.1f}s",
            f"Enemies Defeated: {stats.get('defeated', 0)} (100% Cleared)",
            f"Total Damage Dealt: {stats.get('damage_dealt', 0)}",
            f"Damage Sustained: {stats.get('damage_taken', 0)}",
            f"Dodges Executed: {stats.get('dodges', 0)}",
        ]
        sy = box.y + 16
        for line in stats_items:
            t_surf = self.small_font.render(line, True, (210, 240, 220))
            surface.blit(t_surf, (box.x + 25, sy))
            sy += 24

        for btn in self.victory_buttons:
            btn.render(surface)
