"""
Telemetry logging and session management for Neural Dungeon.
Records structured, timestamped gameplay events and writes validated JSONL session datasets.
Guarantees event-summary consistency, outcome integrity, and separates human from synthetic data.
"""

import os
import json
import time
import math
import uuid
from datetime import datetime
import config


class TelemetryManager:
    """
    Manages session lifecycle, runtime event buffering, schema validation,
    and structured disk persistence for ATML dataset pipelines.
    """

    SAMPLE_INTERVAL = 0.2  # 5 Hz periodic state sampling

    def __init__(self, raw_data_dir: str | None = "data/raw", session_type: str = "human"):
        self.raw_data_dir = raw_data_dir
        if self.raw_data_dir:
            os.makedirs(self.raw_data_dir, exist_ok=True)

        self.session_type = session_type  # 'human', 'synthetic', 'test'
        self.session_id: str = ""
        self.dungeon_seed: int | None = None
        self.start_epoch: float = 0.0
        self.session_active: bool = False
        self.elapsed_time: float = 0.0
        self.sample_timer: float = 0.0

        # Memory buffer for current session events (Single Source of Truth)
        self.events: list[dict] = []

    def start_session(self, dungeon_seed: int | None = None, session_type: str | None = None) -> str:
        """Initializes a brand-new recording session with a unique session ID."""
        if self.session_active:
            self.end_session("ABANDONED")

        if session_type is not None:
            self.session_type = session_type

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_hash = uuid.uuid4().hex[:6]
        prefix = "synthetic_session" if self.session_type in ("synthetic", "test") else "session"
        self.session_id = f"{prefix}_{now_str}_{short_hash}"

        self.dungeon_seed = dungeon_seed
        self.start_epoch = time.time()
        self.elapsed_time = 0.0
        self.sample_timer = 0.0
        self.session_active = True
        self.events = []

        # Record initial SESSION_START event
        self.record_event(
            event_type="SESSION_START",
            player_x=0.0,
            player_y=0.0,
            player_health=config.PLAYER_MAX_HEALTH,
            player_stamina=float(config.PLAYER_MAX_STAMINA),
            room_id=0,
            session_type=self.session_type,
            details={"seed": self.dungeon_seed},
        )
        return self.session_id

    def validate_event(self, event: dict) -> dict:
        """
        Validates required schema fields, enforces numeric bounds,
        and sanitizes invalid or missing values safely.
        """
        sanitized = dict(event)

        # Enforce required metadata
        sanitized["session_id"] = self.session_id
        sanitized["session_type"] = self.session_type
        sanitized["timestamp"] = max(0.0, round(float(event.get("timestamp", self.elapsed_time)), 3))
        sanitized["event_type"] = str(event.get("event_type", "UNKNOWN"))

        # Numeric bounds
        sanitized["player_x"] = round(float(event.get("player_x", 0.0)), 2)
        sanitized["player_y"] = round(float(event.get("player_y", 0.0)), 2)
        sanitized["player_health"] = max(0, min(config.PLAYER_MAX_HEALTH, int(event.get("player_health", 0))))
        sanitized["player_stamina"] = max(0.0, min(float(config.PLAYER_MAX_STAMINA), round(float(event.get("player_stamina", 0.0)), 2)))
        sanitized["room_id"] = int(event.get("room_id", -1))

        # Safe defaults for optional combat metrics
        if "move_dir" in sanitized and sanitized["move_dir"] is not None:
            md = sanitized["move_dir"]
            sanitized["move_dir"] = [round(float(md[0]), 3), round(float(md[1]), 3)]
        if "aim_angle" in sanitized and sanitized["aim_angle"] is not None:
            sanitized["aim_angle"] = round(float(sanitized["aim_angle"]), 3)
        if "nearest_enemy_dist" in sanitized:
            ned = sanitized["nearest_enemy_dist"]
            sanitized["nearest_enemy_dist"] = round(float(ned), 2) if ned is not None else None
        if "nearest_enemy_type" in sanitized:
            sanitized["nearest_enemy_type"] = str(sanitized["nearest_enemy_type"]) if sanitized["nearest_enemy_type"] else None
        if "hits" in sanitized:
            sanitized["hits"] = max(0, int(sanitized["hits"]))
        if "damage_dealt" in sanitized:
            sanitized["damage_dealt"] = max(0, int(sanitized["damage_dealt"]))
        if "damage" in sanitized:
            sanitized["damage"] = max(0, int(sanitized["damage"]))

        return sanitized

    def record_event(self, event_type: str, **kwargs) -> None:
        """Creates, validates, and appends a structured event to the session buffer."""
        if not self.session_active:
            return

        raw_event = {
            "event_type": event_type,
            "timestamp": self.elapsed_time,
            **kwargs,
        }
        valid_event = self.validate_event(raw_event)
        self.events.append(valid_event)

    def sample_state(self, dt: float, player, enemies: list, dungeon) -> None:
        """
        Periodic sampling loop (5 Hz) to record continuous player trajectories
        and enemy distances without bloating disk I/O.
        """
        if not self.session_active or not player or not dungeon:
            return

        self.elapsed_time += dt
        self.sample_timer += dt

        if self.sample_timer >= self.SAMPLE_INTERVAL:
            self.sample_timer = 0.0

            # Calculate nearest active enemy distance & type
            nearest_dist = None
            nearest_type = None
            min_dist = float("inf")
            for e in enemies:
                if e.is_alive:
                    d = math.hypot(e.x - player.x, e.y - player.y)
                    if d < min_dist:
                        min_dist = d
                        nearest_dist = d
                        nearest_type = e.enemy_type

            # Determine current room
            room = dungeon.get_room_at(player.x, player.y)
            room_id = room.id if room else -1

            self.record_event(
                event_type="STATE_TICK",
                player_x=player.x,
                player_y=player.y,
                player_health=player.health,
                player_stamina=player.stamina,
                room_id=room_id,
                aim_angle=player.aim_angle,
                is_dodging=player.is_dodging,
                nearest_enemy_dist=nearest_dist,
                nearest_enemy_type=nearest_type,
                alive_enemies=sum(1 for e in enemies if e.is_alive),
            )

    @property
    def summary_stats(self) -> dict:
        """Real-time authoritative summary dictionary derived from recorded events."""
        return self._compile_authoritative_summary("IN_PROGRESS", round(self.elapsed_time, 2))

    def _compile_authoritative_summary(self, verified_outcome: str, duration: float) -> dict:
        """
        Calculates session metrics directly from the immutable event stream.
        Guarantees 100% mathematical consistency with raw recorded events.
        """
        # Attacks
        attacks = [e for e in self.events if e.get("event_type") == "PLAYER_ATTACK"]
        attacks_swung = len(attacks)
        attacks_hit = sum(1 for e in attacks if e.get("hits", 0) > 0)
        damage_dealt = sum(e.get("damage_dealt", 0) for e in attacks)

        # Damage received & evasions
        hurts = [e for e in self.events if e.get("event_type") == "PLAYER_HURT"]
        damage_taken = sum(e.get("damage", 0) for e in hurts)

        dodges = [e for e in self.events if e.get("event_type") == "PLAYER_DODGE"]
        dodges_performed = sum(1 for e in dodges if e.get("success", False))

        evasions = [e for e in self.events if e.get("event_type") == "DODGED_ATTACK"]
        successful_evasions = len(evasions)

        # Enemies defeated
        kills = [e for e in self.events if e.get("event_type") == "ENEMY_KILLED"]
        enemies_defeated = len(kills)

        accuracy = round(attacks_hit / max(1, attacks_swung), 3) if attacks_swung > 0 else 0.0

        return {
            "session_id": self.session_id,
            "session_type": self.session_type,
            "dungeon_seed": self.dungeon_seed,
            "start_time_iso": datetime.fromtimestamp(self.start_epoch).isoformat(),
            "duration_sec": duration,
            "event_count": len(self.events),
            "final_outcome": verified_outcome,
            "enemies_defeated": enemies_defeated,
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
            "attacks_swung": attacks_swung,
            "attacks_hit": attacks_hit,
            "accuracy": accuracy,
            "dodges_performed": dodges_performed,
            "successful_evasions": successful_evasions,
        }

    def end_session(self, outcome: str) -> dict | None:
        """
        Finalizes the active session, validates the outcome against game state facts,
        writes events to JSONL, appends to summaries.jsonl, and returns the summary dict.
        """
        if not self.session_active:
            return None

        self.session_active = False
        duration = round(self.elapsed_time, 2)

        # Outcome validation guard:
        # A human session cannot claim VICTORY without having defeated enemies
        verified_outcome = outcome
        if self.session_type == "human" and outcome == "VICTORY":
            kills = sum(1 for e in self.events if e.get("event_type") == "ENEMY_KILLED")
            if kills == 0:
                verified_outcome = "ABANDONED"

        # Check for empty/aborted session
        action_events = [e for e in self.events if e.get("event_type") not in ("SESSION_START", "SESSION_END")]
        if not action_events and duration < 0.2:
            verified_outcome = "ABANDONED_EMPTY"

        # Record terminal SESSION_END event
        end_event = {
            "event_type": "SESSION_END",
            "timestamp": duration,
            "player_x": self.events[-1]["player_x"] if self.events else 0.0,
            "player_y": self.events[-1]["player_y"] if self.events else 0.0,
            "player_health": self.events[-1]["player_health"] if self.events else 0,
            "player_stamina": self.events[-1]["player_stamina"] if self.events else 0.0,
            "room_id": self.events[-1]["room_id"] if self.events else -1,
            "outcome": verified_outcome,
        }
        self.events.append(self.validate_event(end_event))

        # Compile authoritative summary from event stream
        summary = self._compile_authoritative_summary(verified_outcome, duration)

        # Persist to disk if raw_data_dir is configured
        if self.raw_data_dir:
            # 1. Write session raw events to JSON Lines file
            session_file = os.path.join(self.raw_data_dir, f"{self.session_id}.jsonl")
            with open(session_file, "w", encoding="utf-8") as f:
                for ev in self.events:
                    f.write(json.dumps(ev) + "\n")

            # 2. Append summary to master summaries log
            summaries_file = os.path.join(self.raw_data_dir, "summaries.jsonl")
            with open(summaries_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")

        return summary
