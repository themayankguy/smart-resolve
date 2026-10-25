"""
Dungeon procedural generation, room connectivity graph, and tile map management.
Builds top-down stone dungeons with connected corridors, collision masks, and minimap data.
"""

import math
import random
import pygame
import config


class Room:
    """Represents an individual dungeon room with bounding box and graph connectivity."""

    def __init__(self, room_id: int, x: int, y: int, w: int, h: int, room_type: str = "combat"):
        self.id = room_id
        self.x = x  # Tile coordinates
        self.y = y
        self.w = w
        self.h = h
        self.room_type = room_type  # 'spawn', 'combat', 'treasury', 'boss', 'exit'
        self.neighbors: list[int] = []  # Neighboring room IDs for graph representation (GCN ready)
        self.cleared = (room_type == "spawn")
        self.enemies_spawned = False

    @property
    def center_tile(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def center_pixel(self) -> tuple[float, float]:
        cx, cy = self.center_tile
        return ((cx + 0.5) * config.TILE_SIZE, (cy + 0.5) * config.TILE_SIZE)

    @property
    def rect_tiles(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.w, self.h)

    def contains_point(self, world_x: float, world_y: float) -> bool:
        tx = int(world_x // config.TILE_SIZE)
        ty = int(world_y // config.TILE_SIZE)
        return self.x <= tx < self.x + self.w and self.y <= ty < self.y + self.h

    def intersects(self, other: "Room", padding: int = 1) -> bool:
        """Returns True if this room overlaps with another room plus padding."""
        return (
            self.x - padding < other.x + other.w
            and self.x + self.w + padding > other.x
            and self.y - padding < other.y + other.h
            and self.y + self.h + padding > other.y
        )


class Dungeon:
    """
    Manages the 2D grid representation of the dungeon.
    Tiles:
        0 = WALKABLE_FLOOR
        1 = SOLID_WALL
        2 = EXIT_PORTAL
    """

    TILE_FLOOR = 0
    TILE_WALL = 1
    TILE_EXIT = 2

    def __init__(self, width: int = config.DUNGEON_WIDTH_TILES, height: int = config.DUNGEON_HEIGHT_TILES, seed: int | None = None):
        self.width = width
        self.height = height
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self.rooms: list[Room] = []
        self.grid = [[self.TILE_WALL for _ in range(self.width)] for _ in range(self.height)]
        self.spawn_room: Room | None = None
        self.exit_room: Room | None = None
        self.exit_pos: tuple[float, float] = (0.0, 0.0)
        self.exit_active: bool = False

        self.wall_rects: list[pygame.Rect] = []
        self.generate()

    def generate(self, target_rooms: int = 7) -> None:
        """Generates rooms and connects them with L-shaped corridors."""
        self.rooms.clear()
        self.grid = [[self.TILE_WALL for _ in range(self.width)] for _ in range(self.height)]

        attempts = 0
        max_attempts = 150
        room_id = 0

        # 1. Carve Non-Overlapping Rooms
        while len(self.rooms) < target_rooms and attempts < max_attempts:
            attempts += 1
            w = random.randint(7, 11)
            h = random.randint(7, 10)
            x = random.randint(2, self.width - w - 2)
            y = random.randint(2, self.height - h - 2)

            new_room = Room(room_id, x, y, w, h)
            if any(new_room.intersects(r, padding=2) for r in self.rooms):
                continue

            self.rooms.append(new_room)
            room_id += 1

        # Fallback if too few rooms
        if len(self.rooms) < 3:
            # Deterministic standard layout
            self.rooms.clear()
            self.rooms.append(Room(0, 3, 3, 8, 8, "spawn"))
            self.rooms.append(Room(1, 16, 5, 9, 8, "combat"))
            self.rooms.append(Room(2, 30, 8, 10, 9, "combat"))
            self.rooms.append(Room(3, 18, 20, 10, 8, "combat"))
            self.rooms.append(Room(4, 32, 22, 9, 9, "exit"))

        # Designate Spawn and Exit rooms (farthest apart)
        self.spawn_room = self.rooms[0]
        self.spawn_room.room_type = "spawn"
        self.spawn_room.cleared = True

        # Find room farthest from spawn for exit
        sx, sy = self.spawn_room.center_tile
        best_dist = -1.0
        exit_cand = self.rooms[-1]
        for r in self.rooms[1:]:
            rx, ry = r.center_tile
            dist = math.hypot(rx - sx, ry - sy)
            if dist > best_dist:
                best_dist = dist
                exit_cand = r

        self.exit_room = exit_cand
        self.exit_room.room_type = "exit"

        # Carve room floors
        for r in self.rooms:
            for ty in range(r.y, r.y + r.h):
                for tx in range(r.x, r.x + r.w):
                    self.grid[ty][tx] = self.TILE_FLOOR

        # 2. Connect Rooms using Minimum Spanning Tree / Chain
        connected = [self.rooms[0]]
        unconnected = list(self.rooms[1:])

        while unconnected:
            # Find closest pair
            best_pair = (None, None)
            min_dist = float("inf")
            for r1 in connected:
                c1x, c1y = r1.center_tile
                for r2 in unconnected:
                    c2x, c2y = r2.center_tile
                    d = math.hypot(c1x - c2x, c1y - c2y)
                    if d < min_dist:
                        min_dist = d
                        best_pair = (r1, r2)

            r1, r2 = best_pair
            self._carve_corridor(r1.center_tile, r2.center_tile)
            r1.neighbors.append(r2.id)
            r2.neighbors.append(r1.id)
            connected.append(r2)
            unconnected.remove(r2)

        # Add 1 or 2 extra cyclic connections for interesting navigation loops
        if len(self.rooms) >= 4:
            r_a = self.rooms[1]
            r_b = self.rooms[3]
            if r_b.id not in r_a.neighbors:
                self._carve_corridor(r_a.center_tile, r_b.center_tile)
                r_a.neighbors.append(r_b.id)
                r_b.neighbors.append(r_a.id)

        # Set exit portal location at center of exit room
        ex, ey = self.exit_room.center_tile
        self.grid[ey][ex] = self.TILE_EXIT
        self.exit_pos = ((ex + 0.5) * config.TILE_SIZE, (ey + 0.5) * config.TILE_SIZE)

        # 3. Precompute Wall Rectangles for Fast Collision
        self._build_wall_collision_cache()

    def _carve_corridor(self, p1: tuple[int, int], p2: tuple[int, int], width: int = 2) -> None:
        """Carves a 2-tile wide corridor between two tile centers."""
        x1, y1 = p1
        x2, y2 = p2

        # Horizontal then vertical
        min_x, max_x = min(x1, x2), max(x1, x2)
        for x in range(min_x, max_x + 1):
            for offset in range(width):
                ty = y1 + offset
                if 0 <= ty < self.height and 0 <= x < self.width:
                    self.grid[ty][x] = self.TILE_FLOOR

        min_y, max_y = min(y1, y2), max(y1, y2)
        for y in range(min_y, max_y + 1):
            for offset in range(width):
                tx = x2 + offset
                if 0 <= y < self.height and 0 <= tx < self.width:
                    self.grid[y][tx] = self.TILE_FLOOR

    def _build_wall_collision_cache(self) -> None:
        """Caches wall bounding rects for collision detection."""
        self.wall_rects = []
        ts = config.TILE_SIZE
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == self.TILE_WALL:
                    self.wall_rects.append(pygame.Rect(x * ts, y * ts, ts, ts))

    def is_wall_at_tile(self, tx: int, ty: int) -> bool:
        if 0 <= tx < self.width and 0 <= ty < self.height:
            return self.grid[ty][tx] == self.TILE_WALL
        return True

    def is_point_walkable(self, world_x: float, world_y: float) -> bool:
        tx = int(world_x // config.TILE_SIZE)
        ty = int(world_y // config.TILE_SIZE)
        if 0 <= tx < self.width and 0 <= ty < self.height:
            return self.grid[ty][tx] != self.TILE_WALL
        return False

    def get_nearby_walls(self, rect: pygame.Rect) -> list[pygame.Rect]:
        """Returns only wall rects in the immediate vicinity of an entity rect."""
        ts = config.TILE_SIZE
        min_tx = max(0, rect.left // ts - 1)
        max_tx = min(self.width - 1, rect.right // ts + 1)
        min_ty = max(0, rect.top // ts - 1)
        max_ty = min(self.height - 1, rect.bottom // ts + 1)

        nearby = []
        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                if self.grid[ty][tx] == self.TILE_WALL:
                    nearby.append(pygame.Rect(tx * ts, ty * ts, ts, ts))
        return nearby

    def get_room_at(self, world_x: float, world_y: float) -> Room | None:
        """Returns the Room containing the world coordinates, or None if in a corridor."""
        for room in self.rooms:
            if room.contains_point(world_x, world_y):
                return room
        return None

    def render(self, surface: pygame.Surface, camera_offset: tuple[float, float]) -> None:
        """Draws visible dungeon tiles within camera viewport."""
        cam_x, cam_y = camera_offset
        ts = config.TILE_SIZE
        screen_w, screen_h = surface.get_size()

        # Visible tile bounds
        start_tx = max(0, int(cam_x // ts))
        end_tx = min(self.width, int((cam_x + screen_w) // ts) + 2)
        start_ty = max(0, int(cam_y // ts))
        end_ty = min(self.height, int((cam_y + screen_h) // ts) + 2)

        for ty in range(start_ty, end_ty):
            for tx in range(start_tx, end_tx):
                tile_type = self.grid[ty][tx]
                screen_pos_x = tx * ts - cam_x
                screen_pos_y = ty * ts - cam_y

                tile_rect = pygame.Rect(screen_pos_x, screen_pos_y, ts, ts)

                if tile_type == self.TILE_WALL:
                    # Dark stone wall with bevel
                    pygame.draw.rect(surface, config.COLOR_WALL, tile_rect)
                    # Wall top lip
                    top_rect = pygame.Rect(screen_pos_x, screen_pos_y, ts, 6)
                    pygame.draw.rect(surface, config.COLOR_WALL_TOP, top_rect)
                    pygame.draw.rect(surface, config.COLOR_WALL_OUTLINE, tile_rect, 1)

                elif tile_type == self.TILE_FLOOR or tile_type == self.TILE_EXIT:
                    # Checker/slate floor pattern
                    color = config.COLOR_FLOOR if (tx + ty) % 2 == 0 else config.COLOR_FLOOR_ALT
                    pygame.draw.rect(surface, color, tile_rect)
                    pygame.draw.rect(surface, config.COLOR_FLOOR_GRID, tile_rect, 1)

                    if tile_type == self.TILE_EXIT:
                        # Mystical portal rune
                        portal_color = config.COLOR_EXIT_ACTIVE if self.exit_active else config.COLOR_EXIT_LOCKED
                        center = (int(screen_pos_x + ts / 2), int(screen_pos_y + ts / 2))
                        pygame.draw.circle(surface, portal_color, center, int(ts * 0.42), 3)
                        pygame.draw.circle(surface, portal_color, center, int(ts * 0.22), 2)
                        # Portal status text
                        txt = "EXIT" if self.exit_active else "LOCKED"
                        font = pygame.font.SysFont("Arial", 11, bold=True)
                        t_surf = font.render(txt, True, portal_color)
                        surface.blit(t_surf, (center[0] - t_surf.get_width() // 2, center[1] - t_surf.get_height() // 2))

    def render_minimap(self, surface: pygame.Surface, player_pos: tuple[float, float], enemies: list, size: tuple[int, int] = (180, 140)) -> None:
        """Renders a real-time minimap in the top corner."""
        w, h = size
        map_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        map_surf.fill((12, 10, 16, 200))  # Semi-transparent dark background
        pygame.draw.rect(map_surf, config.COLOR_WALL_TOP, (0, 0, w, h), 2)

        scale_x = (w - 10) / self.width
        scale_y = (h - 10) / self.height
        ts = config.TILE_SIZE

        # Draw walkable floor tiles on minimap
        for ty in range(self.height):
            for tx in range(self.width):
                if self.grid[ty][tx] != self.TILE_WALL:
                    mx = int(5 + tx * scale_x)
                    my = int(5 + ty * scale_y)
                    color = (65, 60, 75) if self.grid[ty][tx] == self.TILE_FLOOR else config.COLOR_EXIT_ACTIVE
                    pygame.draw.rect(map_surf, color, (mx, my, max(2, int(scale_x)), max(2, int(scale_y))))

        # Draw room threat markers (Phase 6 GCN)
        room_threats = getattr(self, "room_threats", {})
        for r in self.rooms:
            cx, cy = r.center_tile
            rmx = int(5 + cx * scale_x)
            rmy = int(5 + cy * scale_y)
            t_info = room_threats.get(r.id, {})
            t_cat = t_info.get("threat", "UNKNOWN")
            if t_cat == "LOW":
                t_dot_color = (46, 204, 113)
            elif t_cat == "MEDIUM":
                t_dot_color = (241, 196, 15)
            elif t_cat == "HIGH":
                t_dot_color = (231, 76, 60)
            else:
                t_dot_color = (120, 115, 130)
            pygame.draw.circle(map_surf, t_dot_color, (rmx, rmy), 3)

        # Draw enemies
        for e in enemies:
            if e.is_alive:
                ex = int(5 + (e.x / ts) * scale_x)
                ey = int(5 + (e.y / ts) * scale_y)
                color = config.COLOR_MELEE_ENEMY if e.enemy_type == "melee" else config.COLOR_RANGED_ENEMY
                pygame.draw.circle(map_surf, color, (ex, ey), 2)

        # Draw player
        px = int(5 + (player_pos[0] / ts) * scale_x)
        py = int(5 + (player_pos[1] / ts) * scale_y)
        pygame.draw.circle(map_surf, (80, 220, 120), (px, py), 3)

        # Blit minimap onto top-right of main surface
        dest_x = surface.get_width() - w - 20
        dest_y = 20
        surface.blit(map_surf, (dest_x, dest_y))
