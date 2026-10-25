"""
Training pipeline for the Room Threat Graph Convolutional Network (Phase 6).
Trains both the RoomThreatGCN and the NonGraphMLP baseline on seed-partitioned
procedural dungeon graphs and saves full checkpoints with scaler metadata.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.gcn_model import (
    RoomThreatGCN,
    NonGraphMLP,
    THREAT_CLASSES,
    THREAT_TO_ID,
    ID_TO_THREAT,
    build_normalized_adjacency,
)
from ml.gcn_dataset import (
    FEATURE_NAMES,
    NUM_FEATURES,
    generate_dungeon_graph_dataset,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch_gcn(
    model: RoomThreatGCN,
    graphs: list[dict],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Trains RoomThreatGCN for one epoch over all training dungeon graphs."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_nodes = 0

    # Shuffle dungeon graphs each epoch
    indices = list(range(len(graphs)))
    random.shuffle(indices)

    for idx in indices:
        g = graphs[idx]
        x = torch.from_numpy(g["features"]).float().to(device)
        norm_adj = build_normalized_adjacency(g["num_nodes"], g["edges"], device=device)
        y = torch.tensor(g["labels"], dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(x, norm_adj)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * g["num_nodes"]
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y).sum().item()
        total_nodes += g["num_nodes"]

    avg_loss = total_loss / max(1, total_nodes)
    accuracy = correct / max(1, total_nodes)
    return avg_loss, accuracy


def train_epoch_mlp(
    model: NonGraphMLP,
    graphs: list[dict],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Trains NonGraphMLP baseline for one epoch over all training dungeon graphs."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_nodes = 0

    indices = list(range(len(graphs)))
    random.shuffle(indices)

    for idx in indices:
        g = graphs[idx]
        x = torch.from_numpy(g["features"]).float().to(device)
        y = torch.tensor(g["labels"], dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * g["num_nodes"]
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y).sum().item()
        total_nodes += g["num_nodes"]

    avg_loss = total_loss / max(1, total_nodes)
    accuracy = correct / max(1, total_nodes)
    return avg_loss, accuracy


def train_configuration(
    config_name: str,
    feature_names: list[str],
    num_dungeons: int = 150,
    seed_offset: int = 2000,
    epochs: int = 60,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    hidden_dim: int = 32,
    dropout: float = 0.1,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "saved_models",
) -> dict:
    """Trains GCN and MLP baseline for a specific feature configuration."""
    set_seed(42)
    os.makedirs(save_dir, exist_ok=True)

    if config_name == "config_a_7feat":
        gcn_ckpt_path = os.path.join(save_dir, "gcn_checkpoint.pt")
        mlp_ckpt_path = os.path.join(save_dir, "gcn_baseline_mlp_checkpoint.pt")
    elif config_name == "config_b_5feat":
        gcn_ckpt_path = os.path.join(save_dir, "gcn_checkpoint_controlled.pt")
        mlp_ckpt_path = os.path.join(save_dir, "gcn_baseline_mlp_controlled.pt")
    elif config_name == "config_c_3feat":
        gcn_ckpt_path = os.path.join(save_dir, "gcn_checkpoint_pure_geom.pt")
        mlp_ckpt_path = os.path.join(save_dir, "gcn_baseline_mlp_pure_geom.pt")
    else:
        gcn_ckpt_path = os.path.join(save_dir, f"gcn_{config_name}.pt")
        mlp_ckpt_path = os.path.join(save_dir, f"mlp_{config_name}.pt")

    print(f"\n{'=' * 70}")
    print(f"TRAINING CONFIGURATION: {config_name} ({len(feature_names)} features)")
    print(f"Features: {feature_names}")
    print(f"{'=' * 70}")

    train_graphs, test_graphs, scaler = generate_dungeon_graph_dataset(
        num_dungeons=num_dungeons,
        seed_offset=seed_offset,
        train_ratio=0.8,
        feature_names=feature_names,
    )

    train_rooms = sum(g["num_nodes"] for g in train_graphs)
    test_rooms = sum(g["num_nodes"] for g in test_graphs)

    all_train_labels = []
    for g in train_graphs:
        all_train_labels.extend(g["labels"])

    class_counts = {name: all_train_labels.count(cid) for name, cid in THREAT_TO_ID.items()}
    total_samples = len(all_train_labels)
    num_classes = len(THREAT_CLASSES)
    class_weights = []
    for name in THREAT_CLASSES:
        c_count = max(1, class_counts[name])
        w = total_samples / (num_classes * c_count)
        class_weights.append(w)

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    in_features = len(feature_names)

    # 1. Train RoomThreatGCN
    print("--- Training RoomThreatGCN ---")
    gcn_model = RoomThreatGCN(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)
    gcn_optimizer = optim.AdamW(gcn_model.parameters(), lr=lr, weight_decay=weight_decay)
    gcn_scheduler = optim.lr_scheduler.CosineAnnealingLR(gcn_optimizer, T_max=epochs, eta_min=1e-4)

    for epoch in range(1, epochs + 1):
        loss, acc = train_epoch_gcn(gcn_model, train_graphs, gcn_optimizer, criterion, device)
        gcn_scheduler.step()
        if epoch % 20 == 0 or epoch == epochs:
            print(f"GCN Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss:.4f} | Train Acc: {acc * 100:.2f}%")

    # 2. Train NonGraphMLP Baseline
    print("--- Training NonGraphMLP Baseline ---")
    mlp_model = NonGraphMLP(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)
    mlp_optimizer = optim.AdamW(mlp_model.parameters(), lr=lr, weight_decay=weight_decay)
    mlp_scheduler = optim.lr_scheduler.CosineAnnealingLR(mlp_optimizer, T_max=epochs, eta_min=1e-4)

    for epoch in range(1, epochs + 1):
        loss_mlp, acc_mlp = train_epoch_mlp(mlp_model, train_graphs, mlp_optimizer, criterion, device)
        mlp_scheduler.step()
        if epoch % 20 == 0 or epoch == epochs:
            print(f"MLP Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss_mlp:.4f} | Train Acc: {acc_mlp * 100:.2f}%")

    # Save Checkpoints
    gcn_checkpoint = {
        "model_state_dict": gcn_model.state_dict(),
        "model_config": {
            "in_features": in_features,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "dropout": dropout,
        },
        "config_name": config_name,
        "feature_names": feature_names,
        "scaler": scaler.to_dict(),
        "threat_classes": THREAT_CLASSES,
        "threat_to_id": THREAT_TO_ID,
        "id_to_threat": ID_TO_THREAT,
        "class_weights": class_weights,
        "train_dungeons_count": len(train_graphs),
        "train_rooms_count": train_rooms,
        "test_dungeons_count": len(test_graphs),
        "test_rooms_count": test_rooms,
    }
    torch.save(gcn_checkpoint, gcn_ckpt_path)

    mlp_checkpoint = {
        "model_state_dict": mlp_model.state_dict(),
        "model_config": {
            "in_features": in_features,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "dropout": dropout,
        },
        "config_name": config_name,
        "feature_names": feature_names,
        "scaler": scaler.to_dict(),
        "threat_classes": THREAT_CLASSES,
        "threat_to_id": THREAT_TO_ID,
        "id_to_threat": ID_TO_THREAT,
    }
    torch.save(mlp_checkpoint, mlp_ckpt_path)
    print(f"Checkpoints saved: {gcn_ckpt_path} and {mlp_ckpt_path}")

    return {
        "config_name": config_name,
        "gcn_checkpoint_path": gcn_ckpt_path,
        "mlp_checkpoint_path": mlp_ckpt_path,
        "feature_names": feature_names,
    }


def train_models(
    num_dungeons: int = 150,
    seed_offset: int = 2000,
    epochs: int = 60,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    hidden_dim: int = 32,
    dropout: float = 0.1,
    device_name: str | None = None,
    save_dir: str = "saved_models",
    config_selection: str = "all",
) -> dict:
    """
    Main training routine for GCN and MLP baselines across feature configurations.
    """
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    from ml.gcn_dataset import CONFIG_A_FEATURES, CONFIG_B_FEATURES, CONFIG_C_FEATURES

    configs_to_train = []
    if config_selection in ("all", "a", "config_a_7feat"):
        configs_to_train.append(("config_a_7feat", CONFIG_A_FEATURES))
    if config_selection in ("all", "b", "config_b_5feat"):
        configs_to_train.append(("config_b_5feat", CONFIG_B_FEATURES))
    if config_selection in ("all", "c", "config_c_3feat"):
        configs_to_train.append(("config_c_3feat", CONFIG_C_FEATURES))

    results = {}
    for c_name, f_names in configs_to_train:
        res = train_configuration(
            config_name=c_name,
            feature_names=f_names,
            num_dungeons=num_dungeons,
            seed_offset=seed_offset,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=device,
            save_dir=save_dir,
        )
        results[c_name] = res

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GCN for Dungeon Room Threat Prediction.")
    parser.add_argument("--dungeons", type=int, default=150, help="Total procedural dungeons to generate.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimensions.")
    parser.add_argument("--config", type=str, default="all", choices=["all", "a", "b", "c"], help="Feature configuration to train.")
    args = parser.parse_args()

    train_models(
        num_dungeons=args.dungeons,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        config_selection=args.config,
    )
