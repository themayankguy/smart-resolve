"""
Offline training script for the Action Prediction Transformer (Phase 5).
Trains causal self-attention on chronological action sequences with session-level splitting,
and saves the checkpoint to saved_models/transformer_checkpoint.pt.
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ml.dataset import session_train_test_split, build_action_sequence_dataset
from ml.transformer_model import ActionTransformer, ACTION_VOCAB, ACTION_NAMES
from ml.synthetic_generator import generate_synthetic_dataset


def train_transformer(
    data_dir: str = "data/raw",
    synthetic_dir: str = "data/synthetic",
    model_save_path: str = "saved_models/transformer_checkpoint.pt",
    context_len: int = 16,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_scheme: str = "none",
    seed: int = 42,
) -> dict:
    """Trains ActionTransformer on chronological gameplay sequences with optional class weighting."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Determine Data Source
    raw_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("session_") and f.endswith(".jsonl")
    ] if os.path.exists(data_dir) else []

    if len(raw_files) >= 4:
        session_files = raw_files
        provenance = "human_gameplay"
        print(f"[Train-TF] Found {len(session_files)} genuine human sessions in {data_dir}.")
    else:
        provenance = "synthetic_development"
        if not os.path.exists(synthetic_dir) or len([f for f in os.listdir(synthetic_dir) if "anomalous" not in f]) < 10:
            print(f"[Train-TF] Insufficient human data ({len(raw_files)} sessions). Generating reference synthetic sessions...")
            generate_synthetic_dataset(output_dir=synthetic_dir, sessions_per_style=5, seed=seed)

        session_files = [
            os.path.join(synthetic_dir, f)
            for f in os.listdir(synthetic_dir)
            if f.endswith(".jsonl") and "anomalous" not in f
        ]
        print(f"[Train-TF] Using {len(session_files)} synthetic development sessions from {synthetic_dir}.")
        print("[Train-TF] NOTICE: Model trained on proxy synthetic data. Real-player validation remains outstanding.")

    # 2. Build Chronological Action Sequence Records
    all_records = build_action_sequence_dataset(session_files, context_len=context_len)
    if len(all_records) < 50:
        raise ValueError(f"Insufficient sequence records ({len(all_records)}) to train Transformer.")

    print(f"[Train-TF] Built {len(all_records)} sequential context-target pairs across {len(session_files)} sessions.")

    # 3. Session-Level Train / Held-Out Test Split
    train_records, test_records = session_train_test_split(all_records, test_ratio=0.25, seed=seed)
    train_sessions = sorted(list(set(r["session_id"] for r in train_records)))
    test_sessions = sorted(list(set(r["session_id"] for r in test_records)))

    print(f"[Train-TF] Split: {len(train_records)} train sequences ({len(train_sessions)} sessions) | "
          f"{len(test_records)} held-out test sequences ({len(test_sessions)} sessions).")

    # 4. Prepare PyTorch Tensors & DataLoader
    X_train = torch.tensor([r["input_seq"] for r in train_records], dtype=torch.long)
    y_train = torch.tensor([r["target"] for r in train_records], dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 5. Initialize Model, Loss Function, and Optimizer
    model = ActionTransformer(
        vocab_size=len(ACTION_VOCAB),
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        max_seq_len=context_len,
        dropout=0.1,
    )

    class_weights_tensor = None
    if weight_scheme == "balanced":
        counts = torch.bincount(y_train, minlength=len(ACTION_VOCAB)).float()
        total_samples = float(len(y_train))
        observed = counts > 0
        weights = torch.zeros(len(ACTION_VOCAB))
        weights[observed] = total_samples / (observed.sum().float() * counts[observed])
        weights[ACTION_VOCAB["PAD"]] = 0.0
        class_weights_tensor = weights
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=ACTION_VOCAB["PAD"])
        print(f"[Train-TF] Applied Balanced Inverse-Frequency Class Weights: "
              f"MOVE={weights[ACTION_VOCAB['MOVE']]:.3f}, ATTACK={weights[ACTION_VOCAB['ATTACK']]:.3f}, DODGE={weights[ACTION_VOCAB['DODGE']]:.3f}")
    elif weight_scheme == "sqrt":
        counts = torch.bincount(y_train, minlength=len(ACTION_VOCAB)).float()
        total_samples = float(len(y_train))
        weights = torch.zeros(len(ACTION_VOCAB))
        observed = counts > 0
        weights[observed] = torch.sqrt(total_samples / counts[observed])
        weights[ACTION_VOCAB["PAD"]] = 0.0
        weights = weights * (float(observed.sum()) / weights[observed].sum())
        class_weights_tensor = weights
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=ACTION_VOCAB["PAD"])
        print(f"[Train-TF] Applied Sqrt-Smoothed Class Weights: "
              f"MOVE={weights[ACTION_VOCAB['MOVE']]:.3f}, ATTACK={weights[ACTION_VOCAB['ATTACK']]:.3f}, DODGE={weights[ACTION_VOCAB['DODGE']]:.3f}")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=ACTION_VOCAB["PAD"])
        print("[Train-TF] Using Standard Unweighted Empirical Risk Minimization.")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 6. Training Loop
    model.train()
    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batch_count = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x, causal=True)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        final_loss = total_loss / max(1, batch_count)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"[Train-TF] Epoch {epoch:02d}/{epochs:02d} — Loss ({weight_scheme}): {final_loss:.4f}")

    # 7. Save Model Checkpoint
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": len(ACTION_VOCAB),
            "d_model": 32,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 64,
            "max_seq_len": context_len,
            "dropout": 0.1,
            "weight_scheme": weight_scheme,
        },
        "vocab": ACTION_VOCAB,
        "action_names": ACTION_NAMES,
        "action_to_id": ACTION_VOCAB,
        "id_to_action": {v: k for k, v in ACTION_VOCAB.items()},
        "provenance": provenance,
        "context_len": context_len,
        "weight_scheme": weight_scheme,
        "class_weights": class_weights_tensor.tolist() if class_weights_tensor is not None else None,
        "train_sequences": len(train_records),
        "test_sequences": len(test_records),
        "train_sessions": len(train_sessions),
        "test_sessions": len(test_sessions),
        "final_loss": round(final_loss, 4),
    }

    torch.save(checkpoint, model_save_path)
    print(f"[Train-TF] Transformer checkpoint saved to {model_save_path}.")

    return checkpoint


if __name__ == "__main__":
    train_transformer(weight_scheme="none", model_save_path="saved_models/transformer_checkpoint.pt")
    train_transformer(weight_scheme="balanced", model_save_path="saved_models/transformer_weighted_checkpoint.pt")
