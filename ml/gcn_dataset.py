"""
Dungeon Graph Representation, Feature Extraction, and Dataset Pipeline.
Converts procedural Dungeon objects into node feature matrices, adjacency graphs,
and heuristic threat targets without graph-level or target leakage.

Feature Schema (9 Architectural & Topological Features):
    0. room_area: Width * Height in tiles (measure of combat maneuvering space)
    1. aspect_ratio: Width / Height (room geometry elongation)
    2. degree: Number of traversable corridor connections (graph connectivity)
    3. hop_distance_from_spawn: Shortest-path graph hop distance from spawn room (BFS)
    4. euclidean_distance_from_spawn: Spatial center-to-center distance from spawn
    5. is_spawn: Binary indicator (1.0 for spawn room, 0.0 otherwise)
    6. is_exit: Binary indicator (1.0 for exit sanctum, 0.0 otherwise)
    7. norm_center_x: Room center X coordinate normalized by dungeon width
    8. norm_center_y: Room center Y coordinate normalized by dungeon height

Target Threat Categories:
    - LOW (0): Safe rooms (e.g., spawn room, 0 enemies)
    - MEDIUM (1): Standard combat rooms with open maneuver space
    - HIGH (2): Cramped combat rooms (high congestion) or exit sanctum

Target Leakage Prevention:
    Direct combat counts (e.g., enemy_count, melee_count, ranged_count) are strictly
    EXCLUDED from the input feature vector. The GCN must infer threat from room
    geometry, graph position, and corridor network topology.
"""

import math
from collections import deque
import numpy as np
import torch
import config
from game.dungeon import Dungeon, Room
from ml.gcn_model import THREAT_CLASSES, THREAT_TO_ID, build_normalized_adjacency

CONFIG_A_FEATURES = [
    "room_area",
    "aspect_ratio",
    "degree",
    "is_spawn",
    "is_exit",
    "norm_center_x",
    "norm_center_y",
]

CONFIG_B_FEATURES = [
    "room_area",
    "aspect_ratio",
    "degree",
    "norm_center_x",
    "norm_center_y",
]

CONFIG_C_FEATURES = [
    "room_area",
    "aspect_ratio",
    "degree",
]

FEATURE_SETS = {
    "config_a_7feat": CONFIG_A_FEATURES,
    "config_b_5feat": CONFIG_B_FEATURES,
    "config_c_3feat": CONFIG_C_FEATURES,
}

FEATURE_NAMES = CONFIG_A_FEATURES
NUM_FEATURES = len(FEATURE_NAMES)


def compute_shortest_path_hops(rooms: list[Room], spawn_id: int) -> dict[int, int]:
    """
    Computes shortest-path graph hop distance from the spawn room to all rooms using BFS.
    Used exclusively for objective ground-truth threat target assignment, NOT input features.
    """
    adj_list: dict[int, list[int]] = {r.id: list(r.neighbors) for r in rooms}
    hops: dict[int, int] = {r.id: -1 for r in rooms}
    hops[spawn_id] = 0

    queue = deque([spawn_id])
    while queue:
        curr = queue.popleft()
        curr_dist = hops[curr]
        for neighbor in adj_list.get(curr, []):
            if hops[neighbor] == -1:
                hops[neighbor] = curr_dist + 1
                queue.append(neighbor)

    # Fallback for disconnected components (if any)
    max_h = max((h for h in hops.values() if h != -1), default=0)
    for r_id in hops:
        if hops[r_id] == -1:
            hops[r_id] = max_h + 1

    return hops


def calculate_room_threat_label(room: Room, hop_dist: int, is_exit: bool, exit_adjacent: bool = False) -> tuple[str, int, float]:
    """
    Assigns a principled heuristic threat category to a room based on combat mechanics.

    Ground-Truth Heuristic Formulation:
        - Spawn room: 0 enemies, guaranteed safe zone -> LOW threat (Score = 0.0)
        - Exit sanctum: Climax room with exit portal requirement -> HIGH threat (Score = 3.5)
        - Combat rooms:
            Base encounter: 1.0 (Standard 1 Melee + 1 Ranged)
            Congestion factor: +1.0 if area < 72 tiles (cramped space restricts dodging)
            Topological depth / flank risk:
                +1.0 if hop_dist >= 3 or exit_adjacent (deep dungeon progression or boss neighbor)

        Discretization thresholds:
            Score <= 1.0  -> LOW
            Score == 2.0  -> MEDIUM
            Score >= 3.0  -> HIGH

    Returns:
        tuple of (class_name, class_id, continuous_score)
    """
    if room.room_type == "spawn":
        return "LOW", THREAT_TO_ID["LOW"], 0.0

    if room.room_type == "exit" or is_exit:
        return "HIGH", THREAT_TO_ID["HIGH"], 3.5

    area = room.w * room.h
    score = 1.0  # Base combat room with standard enemies

    # Cramped combat room: high collision / restricted evasion
    if area < 72:
        score += 1.0

    # Deep progression or adjacent to exit sanctum (flank hazard)
    if hop_dist >= 3 or exit_adjacent:
        score += 1.0

    if score <= 1.0:
        cls_name = "LOW"
    elif score == 2.0:
        cls_name = "MEDIUM"
    else:
        cls_name = "HIGH"

    return cls_name, THREAT_TO_ID[cls_name], float(score)


def extract_dungeon_graph(dungeon: Dungeon, feature_names: list[str] | None = None) -> dict:
    """
    Extracts node features, edges, and threat targets from a Dungeon instance.

    Args:
        dungeon: Procedural Dungeon instance.
        feature_names: List of feature names to extract (defaults to FEATURE_NAMES).

    Returns dict with:
        - num_nodes: int N
        - room_ids: list[int]
        - raw_features: np.ndarray of shape (N, len(feature_names))
        - edges: list[tuple[int, int]]
        - labels: list[int] (0, 1, or 2)
        - label_names: list[str]
        - threat_scores: list[float]
        - feature_names: list[str]
    """
    active_features = feature_names if feature_names is not None else FEATURE_NAMES
    rooms = dungeon.rooms
    num_nodes = len(rooms)
    spawn_room = dungeon.spawn_room or rooms[0]
    exit_room = dungeon.exit_room or rooms[-1]
    spawn_id = spawn_room.id

    hops = compute_shortest_path_hops(rooms, spawn_id)
    sx, sy = spawn_room.center_tile

    raw_features = []
    labels = []
    label_names = []
    threat_scores = []
    room_ids = []

    # Map room.id to contiguous 0-indexed position [0, N-1]
    id_to_idx = {r.id: idx for idx, r in enumerate(rooms)}

    for r in rooms:
        room_ids.append(r.id)
        area = float(r.w * r.h)
        aspect = float(r.w) / max(1.0, float(r.h))
        degree = float(len(r.neighbors))
        h_dist = float(hops[r.id])

        cx, cy = r.center_tile
        is_spawn = 1.0 if r.id == spawn_id else 0.0
        is_exit = 1.0 if r.id == exit_room.id else 0.0
        exit_adjacent = exit_room.id in r.neighbors

        norm_cx = float(cx) / float(dungeon.width)
        norm_cy = float(cy) / float(dungeon.height)

        feat_map = {
            "room_area": area,
            "aspect_ratio": aspect,
            "degree": degree,
            "is_spawn": is_spawn,
            "is_exit": is_exit,
            "norm_center_x": norm_cx,
            "norm_center_y": norm_cy,
        }
        feat_vector = [feat_map[f] for f in active_features]
        raw_features.append(feat_vector)

        cls_name, cls_id, score = calculate_room_threat_label(r, int(h_dist), is_exit > 0.5, exit_adjacent)
        labels.append(cls_id)
        label_names.append(cls_name)
        threat_scores.append(score)

    # Collect unique undirected edges using contiguous 0-indexed indices
    edges_set = set()
    for r in rooms:
        u = id_to_idx[r.id]
        for n_id in r.neighbors:
            if n_id in id_to_idx:
                v = id_to_idx[n_id]
                if u != v:
                    edge = (min(u, v), max(u, v))
                    edges_set.add(edge)

    edges = list(edges_set)

    return {
        "num_nodes": num_nodes,
        "room_ids": room_ids,
        "raw_features": np.array(raw_features, dtype=np.float32),
        "edges": edges,
        "labels": labels,
        "label_names": label_names,
        "threat_scores": threat_scores,
    }


class GraphFeatureScaler:
    """
    StandardScaler fitted strictly on training dungeon node features.
    Prevents distribution leakage across train and test dungeon graphs.
    """

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.fitted: bool = False

    def fit(self, all_features: np.ndarray) -> "GraphFeatureScaler":
        """Computes feature-wise mean and standard deviation."""
        self.mean = np.mean(all_features, axis=0)
        self.scale = np.std(all_features, axis=0)
        # Avoid division by zero on static features
        self.scale[self.scale < 1e-6] = 1.0
        self.fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Standardizes features using fitted training statistics."""
        if not self.fitted or self.mean is None or self.scale is None:
            raise RuntimeError("GraphFeatureScaler must be fitted before transforming.")
        return (features - self.mean) / self.scale

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist() if self.mean is not None else [],
            "scale": self.scale.tolist() if self.scale is not None else [],
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphFeatureScaler":
        scaler = cls()
        scaler.mean = np.array(data.get("mean", []), dtype=np.float32)
        scaler.scale = np.array(data.get("scale", []), dtype=np.float32)
        scaler.fitted = data.get("fitted", False)
        return scaler


def generate_dungeon_graph_dataset(
    num_dungeons: int = 150,
    seed_offset: int = 2000,
    train_ratio: float = 0.8,
    feature_names: list[str] | None = None,
) -> tuple[list[dict], list[dict], GraphFeatureScaler]:
    """
    Generates a procedural dungeon dataset split cleanly by generation seed.

    Ensures zero graph-level leakage: all rooms from a specific dungeon belong
    exclusively to either the training split or the test split.

    Args:
        num_dungeons: Total procedural dungeons to generate.
        seed_offset: Starting seed for deterministic procedural generation.
        train_ratio: Fraction of dungeons assigned to train split.
        feature_names: List of feature names to extract (defaults to FEATURE_NAMES).

    Returns:
        (train_graphs, test_graphs, fitted_scaler)
    """
    all_graphs = []
    num_train = int(num_dungeons * train_ratio)

    for i in range(num_dungeons):
        seed = seed_offset + i
        dungeon = Dungeon(seed=seed)
        graph_data = extract_dungeon_graph(dungeon, feature_names=feature_names)
        graph_data["seed"] = seed
        all_graphs.append(graph_data)

    train_graphs = all_graphs[:num_train]
    test_graphs = all_graphs[num_train:]

    # Stack training features to fit scaler strictly on train split
    train_features_list = [g["raw_features"] for g in train_graphs]
    all_train_features = np.vstack(train_features_list)

    scaler = GraphFeatureScaler()
    scaler.fit(all_train_features)

    # Standardize both splits using training scaler
    for g in train_graphs:
        g["features"] = scaler.transform(g["raw_features"])
    for g in test_graphs:
        g["features"] = scaler.transform(g["raw_features"])

    return train_graphs, test_graphs, scaler
