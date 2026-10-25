"""
Offline training script for the Combat Autoencoder (Phase 4).
Trains reconstruction on reference combat patterns, fits preprocessing strictly on the
training split, selects an anomaly threshold without test-set leakage, and saves the
model bundle to saved_models/autoencoder_checkpoint.pt.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ml.features import load_session_events, extract_window_features
from ml.dataset import NUMERICAL_FEATURE_NAMES, FeatureScaler, session_train_test_split
from ml.autoencoder import CombatAutoencoder
from ml.synthetic_generator import generate_synthetic_dataset


def train_autoencoder(
    data_dir: str = "data/raw",
    synthetic_dir: str = "data/synthetic",
    model_save_path: str = "saved_models/autoencoder_checkpoint.pt",
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 0.005,
    seed: int = 42,
) -> dict:
    """Trains the Combat Autoencoder and determines the anomaly threshold on training data."""
    # Reproducible seeds
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
        print(f"[Train-AE] Found {len(session_files)} genuine human sessions in {data_dir}.")
    else:
        # Use synthetic development path
        provenance = "synthetic_development"
        if not os.path.exists(synthetic_dir) or len([f for f in os.listdir(synthetic_dir) if "anomalous" not in f]) < 10:
            print(f"[Train-AE] Insufficient human data ({len(raw_files)} sessions). Generating reference synthetic sessions...")
            generate_synthetic_dataset(output_dir=synthetic_dir, sessions_per_style=5, seed=seed)

        session_files = [
            os.path.join(synthetic_dir, f)
            for f in os.listdir(synthetic_dir)
            if f.endswith(".jsonl") and "anomalous" not in f
        ]
        print(f"[Train-AE] Using {len(session_files)} reference synthetic sessions from {synthetic_dir}.")
        print("[Train-AE] NOTICE: Model trained on proxy synthetic data. Real-player validation remains outstanding.")

    # 2. Extract Window Features
    all_window_records = []
    for fpath in session_files:
        events = load_session_events(fpath)
        windows = extract_window_features(events, window_size_sec=5.0, stride_sec=2.0)
        for w in windows:
            all_window_records.append(w)

    if len(all_window_records) < 10:
        raise ValueError(f"Insufficient window records ({len(all_window_records)}) to train autoencoder.")

    # 3. Session-Level Train / Test Split
    train_records, test_records = session_train_test_split(all_window_records, test_ratio=0.25, seed=seed)
    train_session_count = len(set(r["session_id"] for r in train_records))
    test_session_count = len(set(r["session_id"] for r in test_records))

    print(f"[Train-AE] Split: {len(train_records)} train windows ({train_session_count} sessions) | "
          f"{len(test_records)} held-out reference windows ({test_session_count} sessions).")

    # 4. Fit FeatureScaler on Train Split Only
    X_train_raw = np.array([[float(r[col]) for col in NUMERICAL_FEATURE_NAMES] for r in train_records], dtype=np.float32)
    scaler = FeatureScaler().fit(X_train_raw)
    X_train_norm = scaler.transform(X_train_raw)

    t_train = torch.tensor(X_train_norm, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(t_train), batch_size=batch_size, shuffle=True)

    # 5. Initialize Model, Loss Function, and Optimizer
    model = CombatAutoencoder(in_features=len(NUMERICAL_FEATURE_NAMES), latent_dim=4, hidden_dim=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 6. Training Loop
    model.train()
    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batch_count = 0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            rec_x, _ = model(batch_x)
            loss = criterion(rec_x, batch_x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        final_loss = total_loss / max(1, batch_count)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"[Train-AE] Epoch {epoch:02d}/{epochs:02d} — MSE Reconstruction Loss: {final_loss:.5f}")

    # 7. Threshold Selection on Training Split Only (No Leakage from Held-Out Set)
    model.eval()
    with torch.no_grad():
        train_recon_errors = model.compute_reconstruction_error(t_train).cpu().numpy()

    mean_err = float(np.mean(train_recon_errors))
    std_err = float(np.std(train_recon_errors))
    pct_95_err = float(np.percentile(train_recon_errors, 95))

    # Standard statistical threshold: mean + 2.0 * std (covers ~95-97% of normal reference variance)
    threshold = float(round(mean_err + 2.0 * std_err, 4))
    print(f"[Train-AE] Threshold Selection on Training Data:")
    print(f"           Mean Error: {mean_err:.4f} | Std: {std_err:.4f} | 95th Percentile: {pct_95_err:.4f}")
    print(f"           Validated Anomaly Threshold (Mean + 2*Std): {threshold:.4f}")

    # 8. Save Checkpoint Bundle
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "in_features": len(NUMERICAL_FEATURE_NAMES),
        "latent_dim": 4,
        "hidden_dim": 8,
        "feature_names": NUMERICAL_FEATURE_NAMES,
        "scaler": scaler.to_dict(),
        "threshold": threshold,
        "threshold_method": "mean_plus_2std_train",
        "train_mean_error": round(mean_err, 4),
        "train_std_error": round(std_err, 4),
        "train_95pct_error": round(pct_95_err, 4),
        "provenance": provenance,
        "train_windows": len(train_records),
        "test_windows": len(test_records),
        "train_sessions": train_session_count,
        "test_sessions": test_session_count,
        "final_loss": round(final_loss, 5),
    }

    torch.save(checkpoint, model_save_path)
    print(f"[Train-AE] Autoencoder checkpoint saved to {model_save_path}.")

    return checkpoint


if __name__ == "__main__":
    train_autoencoder()
