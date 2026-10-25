"""
Feature extraction pipeline for Neural Dungeon.
Transforms raw timestamped gameplay telemetry event streams into structured,
windowed feature vectors and tabular datasets for ATML models (Siamese, Autoencoder, Transformer).
"""

import os
import csv
import json
import math
from typing import Any

# Action vocabulary mapping for Transformer sequence prediction
ACTION_VOCAB = {
    "PAD": 0,
    "IDLE": 1,
    "MOVE": 2,
    "ATTACK": 3,
    "DODGE": 4,
    "HURT": 5,
}


def load_session_events(file_path: str) -> list[dict]:
    """Reads a session JSONL file and returns a list of event dictionaries."""
    events = []
    if not os.path.exists(file_path):
        return events

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def extract_window_features(
    events: list[dict],
    window_size_sec: float = 5.0,
    stride_sec: float = 2.5,
    seq_len: int = 8,
) -> list[dict]:
    """
    Computes rolling window combat features across a list of session events.
    
    Args:
        events: List of raw telemetry event dicts from a single session.
        window_size_sec: Duration in seconds of each feature calculation window.
        stride_sec: Step size in seconds to advance the window forward.
        seq_len: Fixed sequence length for recent action tokens.
        
    Returns:
        List of feature dictionaries, one per valid sliding window.
    """
    if not events:
        return []

    session_id = events[0].get("session_id", "unknown_session")
    session_outcome = "IN_PROGRESS"

    # Identify final session outcome if available
    for ev in reversed(events):
        if ev.get("event_type") == "SESSION_END":
            session_outcome = ev.get("outcome", "IN_PROGRESS")
            break

    # Find total session duration or slice duration
    min_timestamp = min(ev.get("timestamp", 0.0) for ev in events)
    max_timestamp = max(ev.get("timestamp", 0.0) for ev in events)
    if (max_timestamp - min_timestamp) < 0.5:
        # Session or slice too short for windowing
        return []

    windows = []
    t_start = min_timestamp

    while t_start < max_timestamp:
        t_end = t_start + window_size_sec
        # Collect events within [t_start, t_end)
        w_events = [e for e in events if t_start <= e.get("timestamp", 0.0) < t_end]

        if not w_events:
            t_start += stride_sec
            continue

        actual_duration = max(0.1, t_end - t_start)

        # 1. Combat Action Frequencies
        attacks = [e for e in w_events if e.get("event_type") == "PLAYER_ATTACK"]
        dodges = [e for e in w_events if e.get("event_type") == "PLAYER_DODGE"]
        evasions = [e for e in w_events if e.get("event_type") == "DODGED_ATTACK"]
        hurts = [e for e in w_events if e.get("event_type") == "PLAYER_HURT"]

        attack_count = len(attacks)
        dodge_count = len(dodges)
        evasion_count = len(evasions)

        attack_freq = round(attack_count / actual_duration, 3)
        dodge_freq = round(dodge_count / actual_duration, 3)
        dodge_success_rate = round(evasion_count / max(1, dodge_count), 3)

        # Blocking is not part of the game mechanics
        block_freq = 0.0

        # 2. Movement & Direction Changes
        ticks = [e for e in w_events if e.get("event_type") == "STATE_TICK"]
        moving_ticks = 0
        direction_changes = 0
        last_angle = None

        for t in ticks:
            # Check if player was dodging or moving
            if t.get("is_dodging", False):
                moving_ticks += 1

            angle = t.get("aim_angle")
            if angle is not None:
                if last_angle is not None:
                    # Angular deviation in degrees
                    diff = abs((angle - last_angle + math.pi) % (2 * math.pi) - math.pi)
                    if diff > math.radians(45.0):
                        direction_changes += 1
                last_angle = angle

        movement_freq = round(moving_ticks / max(1, len(ticks)), 3) if ticks else 0.0

        # 3. Health & Stamina Averages
        health_samples = [e.get("player_health", 0) for e in ticks]
        stamina_samples = [e.get("player_stamina", 0.0) for e in ticks]

        avg_health = round(sum(health_samples) / max(1, len(health_samples)), 2) if health_samples else 100.0
        avg_stamina = round(sum(stamina_samples) / max(1, len(stamina_samples)), 2) if stamina_samples else 100.0

        # 4. Enemy Distance
        dist_samples = [e.get("nearest_enemy_dist") for e in ticks if e.get("nearest_enemy_dist") is not None]
        avg_enemy_dist = round(sum(dist_samples) / max(1, len(dist_samples)), 2) if dist_samples else 999.0

        # 5. Damage Metrics
        damage_dealt = sum(e.get("damage_dealt", 0) for e in attacks)
        damage_received = sum(e.get("damage", 0) for e in hurts)

        # 6. Recent Action Token Sequence (for Transformer input)
        action_names = []
        action_tokens = []
        for e in w_events:
            etype = e.get("event_type")
            if etype == "PLAYER_ATTACK":
                action_names.append("ATTACK")
                action_tokens.append(ACTION_VOCAB["ATTACK"])
            elif etype == "PLAYER_DODGE":
                action_names.append("DODGE")
                action_tokens.append(ACTION_VOCAB["DODGE"])
            elif etype == "PLAYER_HURT":
                action_names.append("HURT")
                action_tokens.append(ACTION_VOCAB["HURT"])
            elif etype == "STATE_TICK" and e.get("is_dodging", False):
                action_names.append("MOVE")
                action_tokens.append(ACTION_VOCAB["MOVE"])

        # Take last seq_len tokens with padding
        recent_names = action_names[-seq_len:]
        recent_tokens = action_tokens[-seq_len:]
        if len(recent_tokens) < seq_len:
            pad_count = seq_len - len(recent_tokens)
            recent_tokens = [ACTION_VOCAB["PAD"]] * pad_count + recent_tokens

        session_type = events[0].get("session_type", "synthetic" if "synthetic" in session_id else "human")

        feature_row = {
            "session_id": session_id,
            "session_type": session_type,
            "window_start": round(t_start, 2),
            "window_end": round(t_end, 2),
            "duration_sec": round(actual_duration, 2),
            "attack_frequency": attack_freq,
            "dodge_frequency": dodge_freq,
            "dodge_success_rate": dodge_success_rate,
            "block_frequency": block_freq,
            "movement_frequency": movement_freq,
            "direction_changes": direction_changes,
            "avg_player_health": avg_health,
            "avg_player_stamina": avg_stamina,
            "avg_enemy_distance": avg_enemy_dist,
            "damage_dealt": damage_dealt,
            "damage_received": damage_received,
            "recent_actions": "|".join(recent_names) if recent_names else "NONE",
            "action_tokens": ",".join(map(str, recent_tokens)),
            "session_outcome": session_outcome,
        }
        windows.append(feature_row)
        t_start += stride_sec

    return windows


def extract_session_style_vector(events: list[dict]) -> dict:
    """
    Extracts an aggregated combat playstyle vector for Siamese network metric learning.
    Measures aggression, kiting distance, dodge reliance, and combat efficiency.
    """
    if not events:
        return {}

    session_id = events[0].get("session_id", "unknown_session")
    session_type = events[0].get("session_type", "synthetic" if "synthetic" in session_id else "human")
    duration = max(1.0, max(e.get("timestamp", 0.0) for e in events))

    attacks = [e for e in events if e.get("event_type") == "PLAYER_ATTACK"]
    dodges = [e for e in events if e.get("event_type") == "PLAYER_DODGE"]
    hurts = [e for e in events if e.get("event_type") == "PLAYER_HURT"]
    ticks = [e for e in events if e.get("event_type") == "STATE_TICK"]

    total_swung = len(attacks)
    total_hit = sum(1 for e in attacks if e.get("hits", 0) > 0)
    total_damage_dealt = sum(e.get("damage_dealt", 0) for e in attacks)
    total_damage_taken = sum(e.get("damage", 0) for e in hurts)

    attacks_per_min = (total_swung / duration) * 60.0
    dodges_per_min = (len(dodges) / duration) * 60.0

    dists = [e.get("nearest_enemy_dist") for e in ticks if e.get("nearest_enemy_dist") is not None]
    mean_dist = sum(dists) / max(1, len(dists)) if dists else 300.0

    return {
        "session_id": session_id,
        "session_type": session_type,
        "duration_sec": round(duration, 2),
        "aggression_index": round(attacks_per_min / max(1.0, dodges_per_min + 1.0), 3),
        "dodge_reliance": round(dodges_per_min, 3),
        "accuracy": round(total_hit / max(1, total_swung), 3),
        "damage_ratio": round(total_damage_dealt / max(1.0, float(total_damage_taken)), 3),
        "mean_combat_distance": round(mean_dist, 2),
    }


def process_raw_sessions(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    include_synthetic: bool = False,
) -> tuple[int, int]:
    """
    Reads all session JSONL files in raw_dir, computes window features,
    and writes them into processed CSV datasets.
    
    Args:
        raw_dir: Path to directory containing raw .jsonl session files.
        processed_dir: Path to directory where processed CSV files are written.
        include_synthetic: If False, ignores synthetic/test sessions to protect training data.
        
    Returns:
        tuple (sessions_processed_count, total_feature_rows_generated)
    """
    os.makedirs(processed_dir, exist_ok=True)
    if not os.path.exists(raw_dir):
        return (0, 0)

    all_files = [f for f in os.listdir(raw_dir) if f.endswith(".jsonl") and not f.startswith("summaries")]
    if include_synthetic:
        session_files = [f for f in all_files if f.startswith("session_") or f.startswith("synthetic_session_")]
    else:
        # Genuine human player sessions only
        session_files = [f for f in all_files if f.startswith("session_")]

    if not session_files:
        return (0, 0)

    all_rows = []
    processed_count = 0

    fieldnames = [
        "session_id",
        "session_type",
        "window_start",
        "window_end",
        "duration_sec",
        "attack_frequency",
        "dodge_frequency",
        "dodge_success_rate",
        "block_frequency",
        "movement_frequency",
        "direction_changes",
        "avg_player_health",
        "avg_player_stamina",
        "avg_enemy_distance",
        "damage_dealt",
        "damage_received",
        "recent_actions",
        "action_tokens",
        "session_outcome",
    ]

    for fname in sorted(session_files):
        fpath = os.path.join(raw_dir, fname)
        events = load_session_events(fpath)
        features = extract_window_features(events)
        if not features:
            continue

        processed_count += 1
        all_rows.extend(features)

        # Write per-session processed CSV
        session_out = os.path.join(processed_dir, f"features_{fname.replace('.jsonl', '.csv')}")
        with open(session_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(features)

    # Write aggregated dataset
    if all_rows:
        all_out = os.path.join(processed_dir, "all_features.csv")
        with open(all_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    return (processed_count, len(all_rows))


if __name__ == "__main__":
    count, rows = process_raw_sessions()
    print(f"[ML Pipeline] Processed {count} sessions -> Generated {rows} feature window rows in data/processed/")
