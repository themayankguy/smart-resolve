"""
Combat system for Neural Dungeon.
Handles melee swing arc hitboxes, ranged projectiles, collision detection,
floating damage indicators, particles, and telemetry logging hooks.
"""

import math
import random
import pygame
import config


class Projectile:
    """Arcane projectile fired by ranged enemies with wall/player collision."""

    def __init__(self, x: float, y: float, vx: float, vy: float, damage: int):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.damage = damage
        self.radius = config.PROJECTILE_RADIUS
        self.lifetime = config.PROJECTILE_LIFETIME
        self.is_alive = True

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            int(self.radius * 2),
            int(self.radius * 2),
        )

    def update(self, dt: float, dungeon) -> None:
        """Updates position and checks collision with dungeon walls."""
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.is_alive = False
            return

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Wall collision
        prect = self.rect
        for wall in dungeon.get_nearby_walls(prect):
            if prect.colliderect(wall):
                self.is_alive = False
                return

    def render(self, surface: pygame.Surface, camera_offset: tuple[float, float]) -> None:
        sx = int(self.x - camera_offset[0])
        sy = int(self.y - camera_offset[1])

        # Outer glow and core
        pygame.draw.circle(surface, config.COLOR_PROJECTILE, (sx, sy), int(self.radius * 1.5), 1)
        pygame.draw.circle(surface, (235, 140, 255), (sx, sy), int(self.radius))
        pygame.draw.circle(surface, (255, 255, 255), (sx, sy), max(2, int(self.radius * 0.45)))


class FloatingText:
    """Damage and status indicator floating above entities."""

    def __init__(self, text: str, x: float, y: float, color: tuple[int, int, int]):
        self.text = text
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.lifetime = 0.75
        self.max_lifetime = 0.75
        self.vy = -35.0  # Drift upward

    def update(self, dt: float) -> bool:
        self.lifetime -= dt
        self.y += self.vy * dt
        return self.lifetime > 0

    def render(self, surface: pygame.Surface, camera_offset: tuple[float, float], font: pygame.font.Font) -> None:
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        text_surf = font.render(self.text, True, self.color)
        text_surf.set_alpha(alpha)
        sx = int(self.x - camera_offset[0] - text_surf.get_width() / 2)
        sy = int(self.y - camera_offset[1])
        surface.blit(text_surf, (sx, sy))


class Particle:
    """Hit spark, dust, or blood splatter particle."""

    def __init__(self, x: float, y: float, vx: float, vy: float, color: tuple[int, int, int], lifetime: float, size: int):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size

    def update(self, dt: float) -> bool:
        self.lifetime -= dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.92
        self.vy *= 0.92
        return self.lifetime > 0

    def render(self, surface: pygame.Surface, camera_offset: tuple[float, float]) -> None:
        sx = int(self.x - camera_offset[0])
        sy = int(self.y - camera_offset[1])
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        curr_size = max(1, int(self.size * (self.lifetime / self.max_lifetime)))
        p_surf = pygame.Surface((curr_size * 2, curr_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(p_surf, (*self.color, alpha), (curr_size, curr_size), curr_size)
        surface.blit(p_surf, (sx - curr_size, sy - curr_size))


class SlashVisual:
    """Visual arc displaying the player's sword swing."""

    def __init__(self, cx: float, cy: float, angle: float, arc_deg: float, radius: float):
        self.cx = float(cx)
        self.cy = float(cy)
        self.angle = angle
        self.arc_rad = math.radians(arc_deg)
        self.radius = radius
        self.lifetime = 0.18
        self.max_lifetime = 0.18

    def update(self, dt: float) -> bool:
        self.lifetime -= dt
        return self.lifetime > 0

    def render(self, surface: pygame.Surface, camera_offset: tuple[float, float]) -> None:
        sx = int(self.cx - camera_offset[0])
        sy = int(self.cy - camera_offset[1])
        alpha = int(255 * (self.lifetime / self.max_lifetime))

        # Draw arc segments
        steps = 14
        half_arc = self.arc_rad / 2
        pts = []
        for i in range(steps + 1):
            theta = (self.angle - half_arc) + (self.arc_rad * i / steps)
            px = sx + math.cos(theta) * self.radius
            py = sy + math.sin(theta) * self.radius
            pts.append((px, py))

        if len(pts) > 1:
            arc_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            pygame.draw.lines(arc_surf, (220, 240, 255, alpha), False, pts, 4)
            pygame.draw.lines(arc_surf, (140, 200, 255, int(alpha * 0.6)), False, pts, 8)
            surface.blit(arc_surf, (0, 0))


class CombatSystem:
    """
    Coordinates attacks, projectiles, damage calculations, and telemetry hooks.
    """

    def __init__(self, player, enemies: list, dungeon, telemetry=None):
        self.player = player
        self.enemies = enemies
        self.dungeon = dungeon
        self.telemetry = telemetry

        self.projectiles: list[Projectile] = []
        self.floating_texts: list[FloatingText] = []
        self.particles: list[Particle] = []
        self.slashes: list[SlashVisual] = []

        # Font for damage numbers
        self.font = pygame.font.SysFont("Arial", 14, bold=True)

        # Telemetry event log (fallback buffer)
        self.telemetry_buffer: list[dict] = []

    def spawn_projectile(self, x: float, y: float, vx: float, vy: float, damage: int) -> None:
        self.projectiles.append(Projectile(x, y, vx, vy, damage))

    def spawn_particles(self, x: float, y: float, count: int, color: tuple[int, int, int], speed: float = 120.0) -> None:
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            spd = random.uniform(speed * 0.3, speed)
            vx = math.cos(angle) * spd
            vy = math.sin(angle) * spd
            lifetime = random.uniform(0.2, 0.45)
            self.particles.append(Particle(x, y, vx, vy, color, lifetime, random.randint(2, 4)))

    def apply_player_damage(self, amount: int, knockback: tuple[float, float], enemy_type: str = "melee") -> bool:
        """Inflicts damage on player, factoring in dodge i-frames and telemetry."""
        room = self.dungeon.get_room_at(self.player.x, self.player.y) if self.dungeon else None
        room_id = room.id if room else -1

        if self.player.is_dodging or self.player.invulnerable_timer > 0:
            self.floating_texts.append(FloatingText("DODGED!", self.player.x, self.player.y - 20, (140, 210, 255)))
            if self.telemetry:
                self.telemetry.record_event(
                    "DODGED_ATTACK",
                    player_x=self.player.x,
                    player_y=self.player.y,
                    player_health=self.player.health,
                    player_stamina=self.player.stamina,
                    room_id=room_id,
                    enemy_type=enemy_type,
                    damage_avoided=amount,
                )
            return False

        registered = self.player.take_damage(amount, knockback)
        if registered:
            self.player.damage_taken += amount
            self.floating_texts.append(FloatingText(f"-{amount}", self.player.x, self.player.y - 20, (255, 75, 75)))
            self.spawn_particles(self.player.x, self.player.y, 8, (200, 40, 40), speed=160.0)

            # Telemetry record
            if self.telemetry:
                self.telemetry.record_event(
                    "PLAYER_HURT",
                    player_x=self.player.x,
                    player_y=self.player.y,
                    player_health=self.player.health,
                    player_stamina=self.player.stamina,
                    room_id=room_id,
                    damage=amount,
                    enemy_type=enemy_type,
                )
            self.record_telemetry("PLAYER_HURT", damage=amount, enemy_type=enemy_type)
        return registered

    def execute_player_melee(self) -> int:
        """
        Executes a melee swing arc facing the player's aim angle.
        Damages and knocks back any enemies within range and cone.
        Returns number of enemies hit.
        """
        if not self.player.trigger_attack():
            return 0

        room = self.dungeon.get_room_at(self.player.x, self.player.y) if self.dungeon else None
        room_id = room.id if room else -1

        # Spawn visual slash arc
        self.slashes.append(
            SlashVisual(
                self.player.x,
                self.player.y,
                self.player.aim_angle,
                config.MELEE_ARC_DEG,
                config.MELEE_RANGE,
            )
        )

        half_arc_rad = math.radians(config.MELEE_ARC_DEG / 2)
        hits = 0

        for enemy in self.enemies:
            if not enemy.is_alive:
                continue

            dx = enemy.x - self.player.x
            dy = enemy.y - self.player.y
            dist = math.hypot(dx, dy)

            # Check distance (within swing reach)
            if dist <= config.MELEE_RANGE + enemy.radius:
                # Check angle in swing sector
                enemy_angle = math.atan2(dy, dx)
                angle_diff = (enemy_angle - self.player.aim_angle + math.pi) % (2 * math.pi) - math.pi

                if abs(angle_diff) <= half_arc_rad:
                    # Enemy is within swing cone!
                    knock_mag = 280.0
                    knock_dir = (math.cos(enemy_angle) * knock_mag, math.sin(enemy_angle) * knock_mag)
                    hit_registered = enemy.take_damage(config.MELEE_DAMAGE, knock_dir)

                    if hit_registered:
                        hits += 1
                        self.player.damage_dealt += config.MELEE_DAMAGE
                        self.floating_texts.append(
                            FloatingText(f"-{config.MELEE_DAMAGE}", enemy.x, enemy.y - 20, (255, 230, 100))
                        )
                        self.spawn_particles(enemy.x, enemy.y, 10, (240, 180, 50), speed=140.0)

                        if not enemy.is_alive:
                            self.player.enemies_defeated += 1
                            if self.telemetry:
                                self.telemetry.record_event(
                                    "ENEMY_KILLED",
                                    player_x=self.player.x,
                                    player_y=self.player.y,
                                    player_health=self.player.health,
                                    player_stamina=self.player.stamina,
                                    room_id=room_id,
                                    enemy_type=enemy.enemy_type,
                                )
                            self.spawn_particles(enemy.x, enemy.y, 20, (180, 60, 60), speed=200.0)

        # Telemetry record
        if self.telemetry:
            self.telemetry.record_event(
                "PLAYER_ATTACK",
                player_x=self.player.x,
                player_y=self.player.y,
                player_health=self.player.health,
                player_stamina=self.player.stamina,
                room_id=room_id,
                aim_angle=self.player.aim_angle,
                hits=hits,
                damage_dealt=hits * config.MELEE_DAMAGE,
            )
        self.record_telemetry("PLAYER_ATTACK", hits=hits)
        return hits

    def record_enemy_attack(self, enemy, attack_type: str, damage: int) -> None:
        """Logs an enemy attack execution (melee lunge or ranged projectile)."""
        if self.telemetry:
            room = self.dungeon.get_room_at(enemy.x, enemy.y) if self.dungeon else None
            room_id = room.id if room else -1
            dist = math.hypot(enemy.x - self.player.x, enemy.y - self.player.y)
            self.telemetry.record_event(
                "ENEMY_ATTACK",
                player_x=self.player.x,
                player_y=self.player.y,
                player_health=self.player.health,
                player_stamina=self.player.stamina,
                room_id=room_id,
                enemy_type=enemy.enemy_type,
                attack_type=attack_type,
                damage_intended=damage,
                distance_to_player=round(dist, 2),
            )

    def record_telemetry(self, event_type: str, **kwargs) -> None:
        """Stores a telemetry snapshot hook for future ATML dataset collection."""
        snapshot = {
            "event": event_type,
            "player_x": round(self.player.x, 2),
            "player_y": round(self.player.y, 2),
            "health": self.player.health,
            "stamina": round(self.player.stamina, 2),
            "is_dodging": self.player.is_dodging,
            "aim_angle": round(self.player.aim_angle, 3),
            "alive_enemies": sum(1 for e in self.enemies if e.is_alive),
            **kwargs,
        }
        self.telemetry_buffer.append(snapshot)
        if len(self.telemetry_buffer) > 500:
            self.telemetry_buffer.pop(0)

    def update(self, dt: float) -> None:
        """Updates projectiles, texts, particles, and slashes."""
        # Update projectiles
        for proj in self.projectiles:
            if not proj.is_alive:
                continue
            proj.update(dt, self.dungeon)

            # Check collision with player
            pdx = self.player.x - proj.x
            pdy = self.player.y - proj.y
            pdist = math.hypot(pdx, pdy)
            if pdist <= self.player.radius + proj.radius:
                proj.is_alive = False
                knock = (proj.vx * 0.4, proj.vy * 0.4)
                self.apply_player_damage(proj.damage, knock, enemy_type="ranged")

        # Prune dead projectiles
        self.projectiles = [p for p in self.projectiles if p.is_alive]

        # Update floating texts
        self.floating_texts = [ft for ft in self.floating_texts if ft.update(dt)]

        # Update particles
        self.particles = [pt for pt in self.particles if pt.update(dt)]

        # Update slashes
        self.slashes = [s for s in self.slashes if s.update(dt)]

    def render(self, surface: pygame.Surface, camera_offset: tuple[float, float]) -> None:
        """Draws all active slashes, projectiles, particles, and floating text."""
        for s in self.slashes:
            s.render(surface, camera_offset)

        for proj in self.projectiles:
            proj.render(surface, camera_offset)

        for pt in self.particles:
            pt.render(surface, camera_offset)

        for ft in self.floating_texts:
            ft.render(surface, camera_offset, self.font)
