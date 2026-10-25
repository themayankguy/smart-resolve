"""
Entity system for Neural Dungeon.
Defines Player, MeleeEnemy, RangedEnemy, smooth wall-sliding collision, and state machines.
"""

import math
import random
import pygame
import config


class Entity:
    """Base class for any active game character with collision and health."""

    def __init__(self, x: float, y: float, radius: float, health: int, speed: float):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.health = health
        self.max_health = health
        self.speed = float(speed)

        # Hit flash & invulnerability
        self.hit_flash_timer = 0.0
        self.invulnerable_timer = 0.0

        # Knockback
        self.knockback_vx = 0.0
        self.knockback_vy = 0.0

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    @property
    def is_invulnerable(self) -> bool:
        return self.invulnerable_timer > 0.0

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            int(self.radius * 2),
            int(self.radius * 2),
        )

    def take_damage(self, amount: int, knockback: tuple[float, float] = (0.0, 0.0)) -> bool:
        """Applies damage and knockback if not invulnerable. Returns True if hit registered."""
        if self.invulnerable_timer > 0.0 or not self.is_alive:
            return False

        self.health = max(0, self.health - amount)
        self.hit_flash_timer = 0.15
        self.knockback_vx += knockback[0]
        self.knockback_vy += knockback[1]
        return True

    def move_and_slide(self, dx: float, dy: float, dungeon) -> None:
        """
        Moves the entity by (dx, dy) with separate X/Y axis checks.
        Enables smooth sliding along dungeon walls without getting stuck.
        """
        # --- Move X axis ---
        self.x += dx
        rect_x = self.rect
        for wall in dungeon.get_nearby_walls(rect_x):
            if rect_x.colliderect(wall):
                if dx > 0:
                    self.x = wall.left - self.radius
                elif dx < 0:
                    self.x = wall.right + self.radius
                rect_x = self.rect

        # --- Move Y axis ---
        self.y += dy
        rect_y = self.rect
        for wall in dungeon.get_nearby_walls(rect_y):
            if rect_y.colliderect(wall):
                if dy > 0:
                    self.y = wall.top - self.radius
                elif dy < 0:
                    self.y = wall.bottom + self.radius
                rect_y = self.rect

    def update_physics(self, dt: float, dungeon) -> None:
        """Applies knockback friction and updates flash timers."""
        if self.hit_flash_timer > 0:
            self.hit_flash_timer = max(0.0, self.hit_flash_timer - dt)
        if self.invulnerable_timer > 0:
            self.invulnerable_timer = max(0.0, self.invulnerable_timer - dt)

        # Apply knockback with sliding
        if abs(self.knockback_vx) > 1.0 or abs(self.knockback_vy) > 1.0:
            self.move_and_slide(self.knockback_vx * dt, self.knockback_vy * dt, dungeon)
            friction = math.exp(-8.0 * dt)
            self.knockback_vx *= friction
            self.knockback_vy *= friction
        else:
            self.knockback_vx = 0.0
            self.knockback_vy = 0.0


class Player(Entity):
    """
    Player character with WASD movement, dodge roll i-frames, stamina,
    melee aiming, and performance telemetry tracking.
    """

    def __init__(self, x: float, y: float):
        super().__init__(
            x=x,
            y=y,
            radius=config.PLAYER_RADIUS,
            health=config.PLAYER_MAX_HEALTH,
            speed=config.PLAYER_SPEED,
        )
        self.stamina = float(config.PLAYER_MAX_STAMINA)
        self.max_stamina = float(config.PLAYER_MAX_STAMINA)

        # Dodge state
        self.is_dodging = False
        self.dodge_timer = 0.0
        self.dodge_cooldown = 0.0
        self.dodge_dir = (1.0, 0.0)

        # Attack state
        self.attack_cooldown = 0.0
        self.aim_angle = 0.0

        # Run statistics (for Game Over / Victory review)
        self.enemies_defeated = 0
        self.damage_dealt = 0
        self.damage_taken = 0
        self.dodges_performed = 0
        self.attacks_swung = 0

    def start_dodge(self, move_dir: tuple[float, float]) -> bool:
        """Attempts to execute a dodge roll. Returns True if successful."""
        if self.is_dodging or self.dodge_cooldown > 0 or self.stamina < config.DODGE_STAMINA_COST:
            return False

        # Determine roll direction
        mx, my = move_dir
        mag = math.hypot(mx, my)
        if mag > 0.01:
            self.dodge_dir = (mx / mag, my / mag)
        else:
            # Roll towards aim direction if stationary
            self.dodge_dir = (math.cos(self.aim_angle), math.sin(self.aim_angle))

        self.stamina -= config.DODGE_STAMINA_COST
        self.is_dodging = True
        self.dodge_timer = config.DODGE_DURATION
        self.dodge_cooldown = config.DODGE_COOLDOWN
        self.invulnerable_timer = config.DODGE_DURATION
        self.dodges_performed += 1
        return True

    def can_attack(self) -> bool:
        return self.attack_cooldown <= 0.0 and self.stamina >= config.MELEE_STAMINA_COST and not self.is_dodging

    def trigger_attack(self) -> bool:
        """Deducts stamina and initiates attack cooldown. Returns True if swing triggered."""
        if not self.can_attack():
            return False

        self.stamina -= config.MELEE_STAMINA_COST
        self.attack_cooldown = config.MELEE_COOLDOWN
        self.attacks_swung += 1
        return True

    def update(self, dt: float, dungeon, move_input: tuple[float, float], mouse_world_pos: tuple[float, float]) -> None:
        """Updates player movement, timers, and stamina regeneration."""
        # Aim orientation
        dx = mouse_world_pos[0] - self.x
        dy = mouse_world_pos[1] - self.y
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            self.aim_angle = math.atan2(dy, dx)

        # Timers
        if self.dodge_cooldown > 0:
            self.dodge_cooldown = max(0.0, self.dodge_cooldown - dt)
        if self.attack_cooldown > 0:
            self.attack_cooldown = max(0.0, self.attack_cooldown - dt)

        # Dodge execution or standard movement
        if self.is_dodging:
            self.dodge_timer -= dt
            move_x = self.dodge_dir[0] * config.DODGE_SPEED * dt
            move_y = self.dodge_dir[1] * config.DODGE_SPEED * dt
            self.move_and_slide(move_x, move_y, dungeon)

            if self.dodge_timer <= 0:
                self.is_dodging = False
        else:
            # Regenerate stamina when not dodging
            self.stamina = min(self.max_stamina, self.stamina + config.PLAYER_STAMINA_REGEN * dt)

            # Normal directional movement
            mx, my = move_input
            mag = math.hypot(mx, my)
            if mag > 0.01:
                norm_x = mx / mag
                norm_y = my / mag
                self.move_and_slide(norm_x * self.speed * dt, norm_y * self.speed * dt, dungeon)

        # Physics & knockback
        self.update_physics(dt, dungeon)

    def render(self, surface: pygame.Surface, camera_offset: tuple[float, float]) -> None:
        """Draws the dark-fantasy armored player character."""
        screen_x = int(self.x - camera_offset[0])
        screen_y = int(self.y - camera_offset[1])

        # Ghost trail during dodge roll
        if self.is_dodging:
            ghost_surf = pygame.Surface((int(self.radius * 2.4), int(self.radius * 2.4)), pygame.SRCALPHA)
            pygame.draw.circle(
                ghost_surf,
                (*config.COLOR_PLAYER_DODGE, 120),
                (int(self.radius * 1.2), int(self.radius * 1.2)),
                int(self.radius),
            )
            surface.blit(ghost_surf, (screen_x - int(self.radius * 1.2), screen_y - int(self.radius * 1.2)))

        # Character body (Flashing white when damaged)
        body_color = (255, 255, 255) if self.hit_flash_timer > 0 else config.COLOR_PLAYER
        pygame.draw.circle(surface, (20, 20, 25), (screen_x, screen_y + 3), int(self.radius * 0.9))  # Shadow
        pygame.draw.circle(surface, body_color, (screen_x, screen_y), int(self.radius))
        pygame.draw.circle(surface, (20, 40, 70), (screen_x, screen_y), int(self.radius), 2)  # Armor rim

        # Glowing visor pointing toward aim angle
        visor_dist = self.radius * 0.6
        vx = screen_x + math.cos(self.aim_angle) * visor_dist
        vy = screen_y + math.sin(self.aim_angle) * visor_dist
        pygame.draw.circle(surface, config.COLOR_PLAYER_VISOR, (int(vx), int(vy)), 4)

        # Sword indicator / hilt
        hilt_dist = self.radius * 1.15
        hx = screen_x + math.cos(self.aim_angle + 0.5) * hilt_dist
        hy = screen_y + math.sin(self.aim_angle + 0.5) * hilt_dist
        pygame.draw.circle(surface, (220, 220, 240), (int(hx), int(hy)), 3)


class Enemy(Entity):
    """Base class for dungeon monsters with health bars and telegraph indicators."""

    STATE_IDLE = "idle"
    STATE_CHASE = "chase"
    STATE_TELEGRAPH = "telegraph"
    STATE_COOLDOWN = "cooldown"
    STATE_RETREAT = "retreat"

    def __init__(self, x: float, y: float, radius: float, health: int, speed: float, enemy_type: str):
        super().__init__(x=x, y=y, radius=radius, health=health, speed=speed)
        self.enemy_type = enemy_type
        self.state = self.STATE_IDLE
        self.telegraph_timer = 0.0
        self.cooldown_timer = 0.0
        self.facing_angle = random.uniform(0, 2 * math.pi)

    def render_health_bar(self, surface: pygame.Surface, screen_x: int, screen_y: int) -> None:
        """Renders mini health bar above the enemy."""
        bar_w = int(self.radius * 2.2)
        bar_h = 4
        bar_x = screen_x - bar_w // 2
        bar_y = screen_y - int(self.radius) - 10

        # Background
        pygame.draw.rect(surface, config.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
        # Health fill
        fill_w = int(bar_w * (self.health / self.max_health))
        if fill_w > 0:
            pygame.draw.rect(surface, config.COLOR_HEALTH_BAR, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(surface, (0, 0, 0), (bar_x, bar_y, bar_w, bar_h), 1)


class MeleeEnemy(Enemy):
    """
    Skeleton Warrior: Pursues player when in range, telegraphs a swing with
    a visible charge ring, lunges forward, and deals melee damage.
    """

    def __init__(self, x: float, y: float):
        super().__init__(
            x=x,
            y=y,
            radius=config.MELEE_ENEMY_RADIUS,
            health=config.MELEE_ENEMY_HEALTH,
            speed=config.MELEE_ENEMY_SPEED,
            enemy_type="melee",
        )

    def update_ai(
        self,
        dt: float,
        player: Player,
        dungeon,
        combat_system,
        playstyle: str = "Balanced",
        confidence: float = 0.0,
        predicted_action: str = "NONE",
        action_confidence: float = 0.0,
    ) -> None:
        """State machine for melee behavior with Siamese playstyle & Transformer action adaptation."""
        dist = math.hypot(player.x - self.x, player.y - self.y)
        dx = player.x - self.x
        dy = player.y - self.y
        if dist > 0.01:
            self.facing_angle = math.atan2(dy, dx)

        if self.cooldown_timer > 0:
            self.cooldown_timer = max(0.0, self.cooldown_timer - dt)

        # Siamese playstyle counter-tactics
        speed_mult = 1.0
        if confidence >= 0.45:
            if playstyle == "Evasive":
                speed_mult = 1.15  # Pursue faster against evasive kiters
            elif playstyle == "Aggressive":
                speed_mult = 0.90  # Cautious spacing against aggressive rushers

        # State transitions
        if self.state == self.STATE_IDLE:
            if dist < config.MELEE_ENEMY_AGGRO_RANGE:
                self.state = self.STATE_CHASE

        if self.state == self.STATE_CHASE:
            if dist > config.MELEE_ENEMY_AGGRO_RANGE * 1.3:
                self.state = self.STATE_IDLE
            elif dist <= config.MELEE_ENEMY_ATTACK_RANGE and self.cooldown_timer <= 0:
                # Start attack telegraph
                self.state = self.STATE_TELEGRAPH
                self.telegraph_timer = config.MELEE_ENEMY_TELEGRAPH_TIME
            else:
                # Move towards player
                norm_x = dx / dist
                norm_y = dy / dist
                self.move_and_slide(norm_x * self.speed * speed_mult * dt, norm_y * self.speed * speed_mult * dt, dungeon)

        elif self.state == self.STATE_TELEGRAPH:
            # Stand ground and charge attack
            # Transformer counter-tactic: If DODGE is anticipated, hold swing slightly to catch dodge recovery
            if action_confidence >= 0.45 and predicted_action == "DODGE" and self.telegraph_timer < 0.15:
                self.telegraph_timer += dt * 0.4

            self.telegraph_timer -= dt
            if self.telegraph_timer <= 0:
                # Deliver strike!
                self.state = self.STATE_COOLDOWN
                self.cooldown_timer = config.MELEE_ENEMY_ATTACK_COOLDOWN

                # Forward lunge
                lunge_dist = 20.0
                self.move_and_slide(math.cos(self.facing_angle) * lunge_dist, math.sin(self.facing_angle) * lunge_dist, dungeon)
                combat_system.record_enemy_attack(self, "melee", config.MELEE_ENEMY_DAMAGE)

                # Check if player hit by lunge attack
                curr_dist = math.hypot(player.x - self.x, player.y - self.y)
                if curr_dist <= config.MELEE_ENEMY_ATTACK_RANGE + 15.0:
                    knock_dir = (math.cos(self.facing_angle) * 350.0, math.sin(self.facing_angle) * 350.0)
                    combat_system.apply_player_damage(config.MELEE_ENEMY_DAMAGE, knock_dir, enemy_type="melee")

        elif self.state == self.STATE_COOLDOWN:
            if self.cooldown_timer <= 0:
                self.state = self.STATE_CHASE
            else:
                if confidence >= 0.45 and playstyle == "Aggressive":
                    # Backpedal defensively during cooldown against aggressive player
                    norm_x = -dx / dist
                    norm_y = -dy / dist
                    self.move_and_slide(norm_x * self.speed * 0.45 * dt, norm_y * self.speed * 0.45 * dt, dungeon)
                elif dist > config.MELEE_ENEMY_ATTACK_RANGE * 1.1:
                    norm_x = dx / dist
                    norm_y = dy / dist
                    self.move_and_slide(norm_x * self.speed * 0.4 * dt, norm_y * self.speed * 0.4 * dt, dungeon)

        self.update_physics(dt, dungeon)

    def render(self, surface: pygame.Surface, camera_offset: tuple[float, float]) -> None:
        screen_x = int(self.x - camera_offset[0])
        screen_y = int(self.y - camera_offset[1])

        # Attack telegraph ring
        if self.state == self.STATE_TELEGRAPH:
            progress = 1.0 - (self.telegraph_timer / config.MELEE_ENEMY_TELEGRAPH_TIME)
            tele_radius = int(self.radius + (config.MELEE_ENEMY_ATTACK_RANGE - self.radius) * progress)
            tele_surf = pygame.Surface((tele_radius * 2 + 4, tele_radius * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(
                tele_surf,
                (*config.COLOR_TELEGRAPH, int(100 + 140 * progress)),
                (tele_radius + 2, tele_radius + 2),
                tele_radius,
                2,
            )
            surface.blit(tele_surf, (screen_x - tele_radius - 2, screen_y - tele_radius - 2))

        # Body (Skeleton bone color or red hit flash)
        color = (255, 255, 255) if self.hit_flash_timer > 0 else (205, 195, 185)
        pygame.draw.circle(surface, (15, 12, 16), (screen_x, screen_y + 3), int(self.radius * 0.85))  # Shadow
        pygame.draw.circle(surface, color, (screen_x, screen_y), int(self.radius))
        pygame.draw.circle(surface, (60, 50, 45), (screen_x, screen_y), int(self.radius), 2)

        # Red glowing eye sockets
        eye_dist = self.radius * 0.5
        e1x = screen_x + math.cos(self.facing_angle - 0.35) * eye_dist
        e1y = screen_y + math.sin(self.facing_angle - 0.35) * eye_dist
        e2x = screen_x + math.cos(self.facing_angle + 0.35) * eye_dist
        e2y = screen_y + math.sin(self.facing_angle + 0.35) * eye_dist
        pygame.draw.circle(surface, (230, 40, 40), (int(e1x), int(e1y)), 2)
        pygame.draw.circle(surface, (230, 40, 40), (int(e2x), int(e2y)), 2)

        # Health bar
        self.render_health_bar(surface, screen_x, screen_y)


class RangedEnemy(Enemy):
    """
    Shadow Cultist: Kites the player to maintain range, charges dark energy
    with a purple telegraph, and launches arcane projectiles.
    """

    def __init__(self, x: float, y: float):
        super().__init__(
            x=x,
            y=y,
            radius=config.RANGED_ENEMY_RADIUS,
            health=config.RANGED_ENEMY_HEALTH,
            speed=config.RANGED_ENEMY_SPEED,
            enemy_type="ranged",
        )

    def update_ai(
        self,
        dt: float,
        player: Player,
        dungeon,
        combat_system,
        playstyle: str = "Balanced",
        confidence: float = 0.0,
        predicted_action: str = "NONE",
        action_confidence: float = 0.0,
    ) -> None:
        """State machine for ranged kiting and projectile attacks with Siamese & Transformer adaptation."""
        dist = math.hypot(player.x - self.x, player.y - self.y)
        dx = player.x - self.x
        dy = player.y - self.y
        if dist > 0.01:
            self.facing_angle = math.atan2(dy, dx)

        if self.cooldown_timer > 0:
            self.cooldown_timer = max(0.0, self.cooldown_timer - dt)

        # Adaptive distance buffer against aggressive rushers
        preferred_min_dist = config.RANGED_ENEMY_PREFERRED_MIN_DIST
        if confidence >= 0.45 and playstyle == "Aggressive":
            preferred_min_dist += 40.0  # Back away sooner from aggressive rushers

        # Behavior logic
        if self.state == self.STATE_IDLE:
            if dist < config.RANGED_ENEMY_AGGRO_RANGE:
                self.state = self.STATE_CHASE

        if self.state == self.STATE_CHASE:
            # If player is too close, kite / retreat
            if dist < preferred_min_dist:
                self.state = self.STATE_RETREAT
            # If in optimal firing window and cooldown ready, telegraph spell
            elif dist <= config.RANGED_ENEMY_PREFERRED_MAX_DIST and self.cooldown_timer <= 0:
                self.state = self.STATE_TELEGRAPH
                self.telegraph_timer = config.RANGED_ENEMY_TELEGRAPH_TIME
            elif dist > config.RANGED_ENEMY_PREFERRED_MAX_DIST:
                # Approach player to get into casting range
                norm_x = dx / dist
                norm_y = dy / dist
                self.move_and_slide(norm_x * self.speed * dt, norm_y * self.speed * dt, dungeon)

        elif self.state == self.STATE_RETREAT:
            # Move away from player
            # Transformer counter-tactic: Swiftly backpedal away from forecasted melee swing
            retreat_speed_mult = 1.0
            if action_confidence >= 0.45 and predicted_action == "ATTACK" and dist < 170.0:
                retreat_speed_mult = 1.20

            if dist < preferred_min_dist * 1.25:
                norm_x = -dx / dist
                norm_y = -dy / dist
                self.move_and_slide(norm_x * self.speed * retreat_speed_mult * dt, norm_y * self.speed * retreat_speed_mult * dt, dungeon)
            else:
                self.state = self.STATE_CHASE

        elif self.state == self.STATE_TELEGRAPH:
            # Transformer counter-tactic: delay firing if player is predicted to dodge into i-frames
            if action_confidence >= 0.45 and predicted_action == "DODGE" and self.telegraph_timer < 0.18:
                self.telegraph_timer += dt * 0.4

            self.telegraph_timer -= dt
            if self.telegraph_timer <= 0:
                # Fire projectile towards player's position
                self.state = self.STATE_COOLDOWN
                self.cooldown_timer = config.RANGED_ENEMY_ATTACK_COOLDOWN

                aim_dx = player.x - self.x
                aim_dy = player.y - self.y
                aim_dist = math.hypot(aim_dx, aim_dy)
                if aim_dist > 0.01:
                    vx = (aim_dx / aim_dist) * config.PROJECTILE_SPEED
                    vy = (aim_dy / aim_dist) * config.PROJECTILE_SPEED
                    spawn_x = self.x + (aim_dx / aim_dist) * (self.radius + 6.0)
                    spawn_y = self.y + (aim_dy / aim_dist) * (self.radius + 6.0)
                    combat_system.spawn_projectile(spawn_x, spawn_y, vx, vy, config.RANGED_ENEMY_DAMAGE)
                    combat_system.record_enemy_attack(self, "ranged", config.RANGED_ENEMY_DAMAGE)

        elif self.state == self.STATE_COOLDOWN:
            if self.cooldown_timer <= 0:
                self.state = self.STATE_CHASE
            else:
                # Reposition sideways or away while waiting for spell charge
                if dist < config.RANGED_ENEMY_PREFERRED_MIN_DIST:
                    norm_x = -dx / dist
                    norm_y = -dy / dist
                    self.move_and_slide(norm_x * self.speed * 0.7 * dt, norm_y * self.speed * 0.7 * dt, dungeon)

        self.update_physics(dt, dungeon)

    def render(self, surface: pygame.Surface, camera_offset: tuple[float, float]) -> None:
        screen_x = int(self.x - camera_offset[0])
        screen_y = int(self.y - camera_offset[1])

        # Arcane telegraph orb
        if self.state == self.STATE_TELEGRAPH:
            progress = 1.0 - (self.telegraph_timer / config.RANGED_ENEMY_TELEGRAPH_TIME)
            orb_r = int(4 + 8 * progress)
            orb_x = screen_x + int(math.cos(self.facing_angle) * (self.radius + 8))
            orb_y = screen_y + int(math.sin(self.facing_angle) * (self.radius + 8))
            pygame.draw.circle(surface, (230, 100, 255), (orb_x, orb_y), orb_r)
            pygame.draw.circle(surface, (255, 230, 255), (orb_x, orb_y), max(2, orb_r // 2))

        # Cultist robe body
        color = (255, 255, 255) if self.hit_flash_timer > 0 else config.COLOR_RANGED_ENEMY
        pygame.draw.circle(surface, (15, 10, 18), (screen_x, screen_y + 3), int(self.radius * 0.85))  # Shadow
        pygame.draw.circle(surface, color, (screen_x, screen_y), int(self.radius))
        pygame.draw.circle(surface, (70, 30, 90), (screen_x, screen_y), int(self.radius), 2)

        # Hood and mystic gem
        gem_dist = self.radius * 0.65
        gx = screen_x + math.cos(self.facing_angle) * gem_dist
        gy = screen_y + math.sin(self.facing_angle) * gem_dist
        pygame.draw.circle(surface, (235, 130, 255), (int(gx), int(gy)), 3)

        # Health bar
        self.render_health_bar(surface, screen_x, screen_y)
