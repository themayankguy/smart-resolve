"""
Dataset utilities, heuristic labeling, session-level splitting,
feature scaling, and pair construction for the Siamese Network.
"""

import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np

# Canonical ordered list of 11 numerical input features
NUMERICAL_FEATURE_NAMES = [
    "attack_frequency",       # Attacks per second in window
    "dodge_frequency",        # Dodges per second in window
    "dodge_success_rate",     # Ratio of damage avoided to total dodges [0.0, 1.0]
    "block_frequency",        # 0.0 (blocking is not present in game mechanics)
    "movement_frequency",     # Fraction of window spent actively moving [0.0, 1.0]
    "direction_changes",      # Count of angular changes > 45 degrees
    "avg_player_health",      # Mean HP in window [0.0, 100.0]
    "avg_player_stamina",     # Mean stamina in window [0.0, 100.0]
    "avg_enemy_distance",     # Mean distance to nearest foe (pixels)
    "damage_dealt",           # Total damage inflicted in window
    "damage_received",        # Total damage sustained in window
]

# Style classes for metric learning prototypes
STYLE_CLASSES = ["Aggressive", "Evasive", "Balanced"]


def assign_combat_style(feature_dict: dict) -> str:
    """
    Assigns a provisional combat-style label based on documented heuristic thresholds.
    
    IMPORTANT ATML FACULTY NOTE:
    These heuristic labels serve strictly as rule-based proxy categories for metric
    learning supervision and development testing, NOT as ground-truth psychological
    or definitive behavioral categories.
    
    Heuristic Rules:
    - Aggressive: High attack cadence (>= 0.50 atk/s) and close proximity (<= 160 px).
    - Evasive: High dodge frequency (>= 0.35 dodge/s) or maintaining distance (>= 220 px) with low attacks.
    - Balanced: Moderate combat pacing with balanced defense and offense.
    """
    atk_freq = float(feature_dict.get("attack_frequency", 0.0))
    dodge_freq = float(feature_dict.get("dodge_frequency", 0.0))
    dist = float(feature_dict.get("avg_enemy_distance", 300.0))
    dmg_dealt = float(feature_dict.get("damage_dealt", 0.0))

    if atk_freq >= 0.45 and (dist <= 170.0 or dmg_dealt >= 35.0):
        return "Aggressive"
    elif (dodge_freq >= 0.30 or dist >= 220.0) and atk_freq < 0.35:
        return "Evasive"
    else:
        return "Balanced"


class FeatureScaler:
    """
    StandardScaler (z-score normalization) fitted strictly on the training set.
    Preserves exact mean and standard deviation parameters for deterministic inference.
    """

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero on constant columns (e.g., block_frequency)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("FeatureScaler must be fitted before transforming data.")
        return (X - self.mean) / self.std

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else [],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureScaler":
        scaler = cls()
        scaler.mean = np.array(data["mean"], dtype=np.float32)
        scaler.std = np.array(data["std"], dtype=np.float32)
        return scaler


def session_train_test_split(
    window_records: list[dict],
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Partitions window records by session ID to prevent data leakage between overlapping windows.
    """
    rng = random.Random(seed)
    unique_sessions = sorted(list(set(r["session_id"] for r in window_records)))
    rng.shuffle(unique_sessions)

    test_count = max(1, int(len(unique_sessions) * test_ratio))
    test_session_ids = set(unique_sessions[:test_count])
    train_session_ids = set(unique_sessions[test_count:])

    train_records = [r for r in window_records if r["session_id"] in train_session_ids]
    test_records = [r for r in window_records if r["session_id"] in test_session_ids]

    return train_records, test_records


def build_siamese_pairs(
    vectors: np.ndarray,
    labels: list[str],
    pairs_per_sample: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs balanced positive (label=1.0) and negative (label=0.0) pairs.
    
    Returns:
        tuple (X1, X2, targets) where targets are 1.0 for same style, 0.0 for different style.
    """
    rng = random.Random(seed)
    n = len(vectors)
    if n < 2:
        return np.empty((0, vectors.shape[1])), np.empty((0, vectors.shape[1])), np.empty((0,))

    label_to_indices = {}
    for idx, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(idx)

    pairs_x1 = []
    pairs_x2 = []
    targets = []

    for i in range(n):
        curr_label = labels[i]
        same_class = [j for j in label_to_indices[curr_label] if j != i]
        diff_class = [j for lbl, idxs in label_to_indices.items() if lbl != curr_label for j in idxs]

        # Generate positive pairs
        if same_class:
            for _ in range(pairs_per_sample):
                pos_idx = rng.choice(same_class)
                pairs_x1.append(vectors[i])
                pairs_x2.append(vectors[pos_idx])
                targets.append(1.0)

        # Generate negative pairs
        if diff_class:
            for _ in range(pairs_per_sample):
                neg_idx = rng.choice(diff_class)
                pairs_x1.append(vectors[i])
                pairs_x2.append(vectors[neg_idx])
                targets.append(0.0)

    return np.array(pairs_x1, dtype=np.float32), np.array(pairs_x2, dtype=np.float32), np.array(targets, dtype=np.float32)


class SiamesePairDataset(Dataset):
    """PyTorch Dataset yielding (x1, x2, label)."""

    def __init__(self, x1: np.ndarray, x2: np.ndarray, targets: np.ndarray):
        self.x1 = torch.tensor(x1, dtype=torch.float32)
        self.x2 = torch.tensor(x2, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x1[idx], self.x2[idx], self.targets[idx]


def extract_session_action_sequence(events: list[dict]) -> list[int]:
    """
    Extracts a chronological integer token sequence representing discrete player actions:
    IDLE (1), MOVE (2), ATTACK (3), DODGE (4), HURT (5).
    """
    from ml.transformer_model import ACTION_VOCAB
    sorted_events = sorted(events, key=lambda e: e.get("timestamp", 0.0))
    sequence = []
    last_x = None
    last_y = None

    for e in sorted_events:
        etype = e.get("event_type")
        if etype in ("PLAYER_ATTACK", "ATTACK"):
            sequence.append(ACTION_VOCAB["ATTACK"])
        elif etype in ("PLAYER_DODGE", "DODGE"):
            sequence.append(ACTION_VOCAB["DODGE"])
        elif etype in ("PLAYER_HURT", "HURT"):
            sequence.append(ACTION_VOCAB["HURT"])
        elif etype == "MOVE":
            sequence.append(ACTION_VOCAB["MOVE"])
        elif etype == "IDLE":
            sequence.append(ACTION_VOCAB["IDLE"])
        elif etype == "STATE_TICK":
            px = e.get("player_x", 0.0)
            py = e.get("player_y", 0.0)
            if e.get("is_dodging", False):
                sequence.append(ACTION_VOCAB["DODGE"])
            elif last_x is not None and last_y is not None:
                dx = px - last_x
                dy = py - last_y
                if (dx * dx + dy * dy) > 1.0:
                    sequence.append(ACTION_VOCAB["MOVE"])
                else:
                    sequence.append(ACTION_VOCAB["IDLE"])
            else:
                sequence.append(ACTION_VOCAB["IDLE"])
            last_x = px
            last_y = py

    return sequence


def build_action_sequence_dataset(
    session_files: list[str] | dict[str, list[dict]],
    context_len: int = 16,
    max_seq_len: int | None = None,
) -> list[dict]:
    """
    Constructs chronological (input_seq, target, session_id) records from session logs or event dicts.
    Targets strictly succeed input context to preserve causal temporal ordering.
    """
    from ml.features import load_session_events
    from ml.transformer_model import ACTION_VOCAB

    if max_seq_len is not None:
        context_len = max_seq_len

    records = []
    if isinstance(session_files, dict):
        items = session_files.items()
        is_dict = True
    else:
        items = [(fpath, fpath) for fpath in session_files]
        is_dict = False

    for sid_or_path, val in items:
        if is_dict:
            events = val
            sid = sid_or_path
        else:
            events = load_session_events(val)
            if not events:
                continue
            sid = events[0].get("session_id", os.path.basename(val))

        seq = extract_session_action_sequence(events)
        if len(seq) < 2:
            continue

        for t in range(1, len(seq)):
            context_start = max(0, t - context_len)
            history = seq[context_start:t]
            if len(history) < context_len:
                pad_count = context_len - len(history)
                input_seq = [ACTION_VOCAB["PAD"]] * pad_count + history
            else:
                input_seq = history

            target = seq[t]
            records.append({
                "session_id": sid,
                "input_seq": input_seq,
                "target": target,
            })

    return records

