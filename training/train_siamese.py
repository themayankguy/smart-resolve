"""
Offline training script for the Siamese Network (Phase 3).
Trains metric embeddings with Contrastive Loss, computes archetype prototypes,
and saves model weights, feature schema, and scaler parameters to saved_models/siamese_checkpoint.pt.
"""

import os
import json
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ml.features import load_session_events, extract_window_features
from ml.dataset import (
    NUMERICAL_FEATURE_NAMES,
    assign_combat_style,
    FeatureScaler,
    session_train_test_split,
    build_siamese_pairs,
    SiamesePairDataset,
)
from ml.siamese import SiameseNet, ContrastiveLoss
from ml.synthetic_generator import generate_synthetic_dataset


def train_siamese(
    data_dir: str = "data/raw",
    synthetic_dir: str = "data/synthetic",
    model_save_path: str = "saved_models/siamese_checkpoint.pt",
    epochs: int = 35,
    batch_size: int = 32,
    learning_rate: float = 0.003,
    margin: float = 1.0,
    seed: int = 42,
) -> dict:
    """Trains the Siamese Network with session-level splitting and Contrastive Loss."""
    # Set reproducible seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Determine Data Source & Provenance
    raw_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("session_") and f.endswith(".jsonl")
    ] if os.path.exists(data_dir) else []

    if len(raw_files) >= 4:
        session_files = raw_files
        provenance = "human_gameplay"
        print(f"[Train] Found {len(session_files)} genuine human sessions in {data_dir}.")
    else:
        # Use synthetic development path
        provenance = "synthetic_development"
        if not os.path.exists(synthetic_dir) or len(os.listdir(synthetic_dir)) < 10:
            print(f"[Train] Insufficient human data ({len(raw_files)} sessions). Generating synthetic development sessions...")
            generate_synthetic_dataset(output_dir=synthetic_dir, sessions_per_style=5, seed=seed)

        session_files = [
            os.path.join(synthetic_dir, f)
            for f in os.listdir(synthetic_dir)
            if f.endswith(".jsonl")
        ]
        print(f"[Train] Using {len(session_files)} synthetic development sessions from {synthetic_dir}.")
        print("[Train] NOTICE: Model trained on proxy synthetic data. Real-player validation remains outstanding.")

    # 2. Extract Window Features & Assign Proxy Labels
    all_window_records = []
    for fpath in session_files:
        events = load_session_events(fpath)
        windows = extract_window_features(events, window_size_sec=5.0, stride_sec=2.0)
        for w in windows:
            w["combat_style"] = assign_combat_style(w)
            all_window_records.append(w)

    if len(all_window_records) < 10:
        raise ValueError(f"Insufficient window records ({len(all_window_records)}) to train model.")

    print(f"[Train] Extracted {len(all_window_records)} windows across {len(session_files)} sessions.")

    # 3. Session-Level Train / Test Split
    train_records, test_records = session_train_test_split(all_window_records, test_ratio=0.25, seed=seed)
    train_session_count = len(set(r["session_id"] for r in train_records))
    test_session_count = len(set(r["session_id"] for r in test_records))

    print(f"[Train] Split: {len(train_records)} train windows ({train_session_count} sessions) | "
          f"{len(test_records)} held-out test windows ({test_session_count} sessions).")

    # 4. Fit FeatureScaler on Train Split Only
    X_train_raw = np.array([[float(r[col]) for col in NUMERICAL_FEATURE_NAMES] for r in train_records], dtype=np.float32)
    scaler = FeatureScaler().fit(X_train_raw)
    X_train_norm = scaler.transform(X_train_raw)
    train_labels = [r["combat_style"] for r in train_records]

    # 5. Build Balanced Training Pairs
    x1_train, x2_train, y_train = build_siamese_pairs(X_train_norm, train_labels, pairs_per_sample=4, seed=seed)
    train_dataset = SiamesePairDataset(x1_train, x2_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"[Train] Constructed {len(train_dataset)} balanced training pairs (50% similar, 50% dissimilar).")

    # 6. Initialize Siamese Network & Optimizer
    model = SiameseNet(in_features=len(NUMERICAL_FEATURE_NAMES), embedding_dim=8)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 7. Training Loop
    model.train()
    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batch_count = 0
        for batch_x1, batch_x2, batch_y in train_loader:
            optimizer.zero_grad()
            z1, z2 = model(batch_x1, batch_x2)
            loss = criterion(z1, z2, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        final_loss = total_loss / max(1, batch_count)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"[Train] Epoch {epoch:02d}/{epochs:02d} — Contrastive Loss: {final_loss:.4f}")

    # 8. Compute Class Prototype Centroids in Embedding Space
    model.eval()
    prototypes = {}
    with torch.no_grad():
        for style in ["Aggressive", "Evasive", "Balanced"]:
            style_indices = [i for i, lbl in enumerate(train_labels) if lbl == style]
            if style_indices:
                style_vecs = torch.tensor(X_train_norm[style_indices], dtype=torch.float32)
                style_embeds = model.forward_one(style_vecs)
                centroid = style_embeds.mean(dim=0).cpu().numpy().tolist()
                prototypes[style] = centroid
            else:
                prototypes[style] = [0.0] * 8

    # 9. Save Complete Checkpoint Bundle
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "in_features": len(NUMERICAL_FEATURE_NAMES),
        "embedding_dim": 8,
        "feature_names": NUMERICAL_FEATURE_NAMES,
        "scaler": scaler.to_dict(),
        "class_prototypes": prototypes,
        "margin": margin,
        "provenance": provenance,
        "train_windows": len(train_records),
        "test_windows": len(test_records),
        "train_sessions": train_session_count,
        "test_sessions": test_session_count,
        "final_loss": round(final_loss, 4),
    }

    torch.save(checkpoint, model_save_path)
    print(f"[Train] Model checkpoint and scaler saved to {model_save_path}.")

    return checkpoint


if __name__ == "__main__":
    train_siamese()
