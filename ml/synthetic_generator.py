"""
Synthetic Gameplay Telemetry Generator for Neural Dungeon Development.

IMPORTANT PROVENANCE & FACULTY VIVA NOTICE:
This module generates strictly synthetic, scripted telemetry data for model architecture
prototyping and offline unit test pipelines. These data are NOT genuine human player telemetry.
Real-player validation remains an outstanding milestone.
"""

import os
import json
import random
import config


def generate_synthetic_session(
    archetype: str,
    session_id: str,
    duration_sec: float = 20.0,
    seed: int = 42,
) -> list[dict]:
    """
    Simulates a synthetic combat trajectory for a specific combat archetype.
    
    Archetypes:
    - 'Aggressive': High attack cadence, low distance, high damage dealt, minimal dodging.
    - 'Evasive': Frequent dodging, high enemy distance, kiting movement, occasional strikes.
    - 'Balanced': Moderate attack and dodge cadence, mixed distances.
    """
    rng = random.Random(seed)
    events = []

    # 1. Start event
    events.append({
        "event_type": "SESSION_START",
        "timestamp": 0.0,
        "player_x": 300.0,
        "player_y": 300.0,
        "player_health": config.PLAYER_MAX_HEALTH,
        "player_stamina": float(config.PLAYER_MAX_STAMINA),
        "room_id": 1,
        "session_id": session_id,
        "session_type": "synthetic",
        "details": {"archetype": archetype, "synthetic": True},
    })

    t = 0.2
    hp = 100
    stamina = 100.0

    while t < duration_sec:
        # Distance and pacing based on archetype
        if archetype == "Aggressive":
            dist = rng.uniform(40.0, 130.0)
            attack_prob = 0.40
            dodge_prob = 0.08
        elif archetype == "Evasive":
            dist = rng.uniform(220.0, 360.0)
            attack_prob = 0.08
            dodge_prob = 0.35
        elif archetype == "Anomalous":
            # Atypical combat behavior: stationary spinning, takes damage without attacking or dodging
            dist = rng.uniform(10.0, 90.0)
            attack_prob = 0.0
            dodge_prob = 0.0
        else:  # Balanced
            dist = rng.uniform(110.0, 210.0)
            attack_prob = 0.22
            dodge_prob = 0.20

        # Movement tick
        aim_angle = rng.uniform(-3.14, 3.14)
        events.append({
            "event_type": "STATE_TICK",
            "timestamp": round(t, 3),
            "player_x": round(300.0 + rng.uniform(-80, 80), 2),
            "player_y": round(300.0 + rng.uniform(-80, 80), 2),
            "player_health": hp,
            "player_stamina": round(stamina, 2),
            "room_id": 1,
            "aim_angle": round(aim_angle, 3),
            "is_dodging": False,
            "nearest_enemy_dist": round(dist, 2),
            "nearest_enemy_type": "melee",
            "alive_enemies": 4,
            "session_id": session_id,
            "session_type": "synthetic",
        })

        # Attack event?
        if rng.random() < attack_prob:
            hits = 1 if rng.random() < 0.85 else 0
            dmg = config.MELEE_DAMAGE if hits else 0
            events.append({
                "event_type": "PLAYER_ATTACK",
                "timestamp": round(t + 0.05, 3),
                "player_x": 300.0,
                "player_y": 300.0,
                "player_health": hp,
                "player_stamina": max(0.0, round(stamina - 15, 2)),
                "room_id": 1,
                "aim_angle": round(aim_angle, 3),
                "hits": hits,
                "damage_dealt": dmg,
                "session_id": session_id,
                "session_type": "synthetic",
            })

        # Dodge event?
        if rng.random() < dodge_prob:
            events.append({
                "event_type": "PLAYER_DODGE",
                "timestamp": round(t + 0.1, 3),
                "player_x": 300.0,
                "player_y": 300.0,
                "player_health": hp,
                "player_stamina": max(0.0, round(stamina - 28, 2)),
                "room_id": 1,
                "move_dir": [1.0, 0.0],
                "success": True,
                "session_id": session_id,
                "session_type": "synthetic",
            })
            if rng.random() < 0.7:
                events.append({
                    "event_type": "DODGED_ATTACK",
                    "timestamp": round(t + 0.12, 3),
                    "player_x": 300.0,
                    "player_y": 300.0,
                    "player_health": hp,
                    "player_stamina": round(stamina, 2),
                    "room_id": 1,
                    "damage_avoided": 20,
                    "enemy_type": "melee",
                    "session_id": session_id,
                    "session_type": "synthetic",
                })

        # For Anomalous sessions: passive damage taken (e.g. standing still taking hits)
        if archetype == "Anomalous" and rng.random() < 0.30:
            hp = max(15, hp - 18)
            events.append({
                "event_type": "PLAYER_HURT",
                "timestamp": round(t + 0.15, 3),
                "player_x": 300.0,
                "player_y": 300.0,
                "player_health": hp,
                "player_stamina": round(stamina, 2),
                "room_id": 1,
                "damage": 18,
                "enemy_type": "melee",
                "session_id": session_id,
                "session_type": "synthetic",
            })

        t += 0.2
        stamina = min(100.0, stamina + 5.0)

    # End event
    events.append({
        "event_type": "SESSION_END",
        "timestamp": round(duration_sec, 2),
        "player_x": 300.0,
        "player_y": 300.0,
        "player_health": hp,
        "player_stamina": stamina,
        "room_id": 1,
        "outcome": "DEFEAT",
        "session_id": session_id,
        "session_type": "synthetic",
    })

    return events


def generate_synthetic_dataset(output_dir: str = "data/synthetic", sessions_per_style: int = 5, seed: int = 42) -> list[str]:
    """
    Generates balanced synthetic sessions for development and offline training.
    Files are saved as data/synthetic/synthetic_session_<style>_<id>.jsonl
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    archetypes = ["Aggressive", "Evasive", "Balanced"]

    session_idx = 1
    for arch in archetypes:
        for i in range(sessions_per_style):
            s_seed = seed + session_idx * 17
            sid = f"synthetic_session_{arch.lower()}_{i+1:02d}_{s_seed}"
            events = generate_synthetic_session(arch, sid, duration_sec=20.0, seed=s_seed)

            filepath = os.path.join(output_dir, f"{sid}.jsonl")
            with open(filepath, "w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")

            generated_files.append(filepath)
            session_idx += 1

    return generated_files


def generate_anomalous_dataset(output_dir: str = "data/synthetic", session_count: int = 5, seed: int = 999) -> list[str]:
    """
    Generates synthetic anomalous combat sessions displaying atypical behaviors
    (e.g., passive damage taking, non-engagement, zero attacks or dodges) for evaluation.
    Files are saved as data/synthetic/synthetic_session_anomalous_<id>.jsonl
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    for i in range(session_count):
        s_seed = seed + i * 31
        sid = f"synthetic_session_anomalous_{i+1:02d}_{s_seed}"
        events = generate_synthetic_session("Anomalous", sid, duration_sec=20.0, seed=s_seed)

        filepath = os.path.join(output_dir, f"{sid}.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        generated_files.append(filepath)

    return generated_files


if __name__ == "__main__":
    files = generate_synthetic_dataset()
    print(f"[Synthetic Data] Generated {len(files)} synthetic sessions in data/synthetic/")
