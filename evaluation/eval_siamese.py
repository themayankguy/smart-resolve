"""
Evaluation script for the trained Siamese Network on held-out test sessions.
Calculates pairwise metrics, optimal threshold classification, confusion matrix,
and compares against an unmapped Euclidean distance baseline.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.features import load_session_events, extract_window_features
from ml.dataset import (
    NUMERICAL_FEATURE_NAMES,
    assign_combat_style,
    FeatureScaler,
    session_train_test_split,
    build_siamese_pairs,
)
from ml.siamese import SiameseNet


def evaluate_siamese(
    model_path: str = "saved_models/siamese_checkpoint.pt",
    synthetic_dir: str = "data/synthetic",
    raw_dir: str = "data/raw",
    output_dir: str = "evaluation",
    seed: int = 42,
) -> dict:
    """Evaluates SiameseNet on held-out sessions and compares against raw distance baseline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please run training first.")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    scaler = FeatureScaler.from_dict(checkpoint["scaler"])

    # Load model architecture
    model = SiameseNet(in_features=checkpoint["in_features"], embedding_dim=checkpoint["embedding_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Determine session directory; exclude anomalous sessions created for autoencoder testing
    data_dir = raw_dir if os.path.exists(raw_dir) and len([f for f in os.listdir(raw_dir) if f.startswith("session_")]) >= 4 else synthetic_dir
    session_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".jsonl") and not f.startswith("synthetic_session_anomalous_")
    ]

    # Extract all window records
    all_records = []
    for fpath in session_files:
        events = load_session_events(fpath)
        windows = extract_window_features(events, window_size_sec=5.0, stride_sec=2.0)
        for w in windows:
            w["combat_style"] = assign_combat_style(w)
            all_records.append(w)

    # Perform exact same session-level split to isolate held-out test set
    _, test_records = session_train_test_split(all_records, test_ratio=0.25, seed=seed)
    test_session_ids = sorted(list(set(r["session_id"] for r in test_records)))

    print(f"[Eval] Evaluating on {len(test_records)} held-out windows across {len(test_session_ids)} sessions:")
    for sid in test_session_ids:
        print(f"       - {sid}")

    # Prepare normalized feature vectors for held-out test set
    X_test_raw = np.array([[float(r[col]) for col in NUMERICAL_FEATURE_NAMES] for r in test_records], dtype=np.float32)
    X_test_norm = scaler.transform(X_test_raw)
    test_labels = [r["combat_style"] for r in test_records]

    # Build test pairs (balanced)
    x1_test, x2_test, y_test = build_siamese_pairs(X_test_norm, test_labels, pairs_per_sample=6, seed=seed)
    if len(y_test) == 0:
        raise ValueError("Insufficient test samples to construct evaluation pairs.")

    t_x1 = torch.tensor(x1_test, dtype=torch.float32)
    t_x2 = torch.tensor(x2_test, dtype=torch.float32)

    # 1. Siamese Metric Embeddings & Distances
    with torch.no_grad():
        z1, z2 = model(t_x1, t_x2)
        siamese_distances = F.pairwise_distance(z1, z2).cpu().numpy()

    # 2. Baseline: Raw Normalized Euclidean Distances (without metric learning)
    raw_distances = np.linalg.norm(x1_test - x2_test, axis=1)

    # Calculate distance statistics
    similar_mask = (y_test == 1.0)
    dissimilar_mask = (y_test == 0.0)

    mean_dist_similar = float(np.mean(siamese_distances[similar_mask]))
    mean_dist_dissimilar = float(np.mean(siamese_distances[dissimilar_mask]))

    raw_mean_similar = float(np.mean(raw_distances[similar_mask]))
    raw_mean_dissimilar = float(np.mean(raw_distances[dissimilar_mask]))

    # Sweep thresholds to find optimal threshold for Siamese model vs Baseline
    def compute_best_metrics(distances, targets):
        best_acc = 0.0
        best_thresh = 0.0
        best_cm = {}
        for thresh in np.linspace(np.min(distances), np.max(distances), 100):
            preds = (distances < thresh).astype(float)  # distance < thresh => predict similar (1.0)
            tp = int(np.sum((preds == 1.0) & (targets == 1.0)))
            fp = int(np.sum((preds == 1.0) & (targets == 0.0)))
            tn = int(np.sum((preds == 0.0) & (targets == 0.0)))
            fn = int(np.sum((preds == 0.0) & (targets == 1.0)))

            acc = float((tp + tn) / len(targets))
            if acc > best_acc:
                best_acc = acc
                best_thresh = float(thresh)
                prec = float(tp / max(1, tp + fp))
                rec = float(tp / max(1, tp + fn))
                f1 = float(2 * (prec * rec) / max(1e-6, prec + rec))
                best_cm = {
                    "threshold": float(round(float(thresh), 3)),
                    "accuracy": float(round(acc, 4)),
                    "precision": float(round(prec, 4)),
                    "recall": float(round(rec, 4)),
                    "f1_score": float(round(f1, 4)),
                    "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
                }
        return best_cm

    siamese_metrics = compute_best_metrics(siamese_distances, y_test)
    baseline_metrics = compute_best_metrics(raw_distances, y_test)

    results = {
        "model": "SiameseNet (Phase 3)",
        "provenance": checkpoint.get("provenance", "synthetic_development"),
        "test_windows_count": len(test_records),
        "test_sessions_count": len(test_session_ids),
        "test_pairs_count": len(y_test),
        "siamese_metrics": {
            **siamese_metrics,
            "mean_distance_similar": round(mean_dist_similar, 4),
            "mean_distance_dissimilar": round(mean_dist_dissimilar, 4),
            "margin_separation_ratio": round(mean_dist_dissimilar / max(1e-4, mean_dist_similar), 2),
        },
        "baseline_raw_euclidean": {
            **baseline_metrics,
            "mean_distance_similar": round(raw_mean_similar, 4),
            "mean_distance_dissimilar": round(raw_mean_dissimilar, 4),
            "margin_separation_ratio": round(raw_mean_dissimilar / max(1e-4, raw_mean_similar), 2),
        },
    }

    # Save reports
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "siamese_eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = os.path.join(output_dir, "siamese_eval_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Siamese Network Evaluation Report (Phase 3)\n\n")
        f.write(f"**Data Provenance**: `{results['provenance']}`\n")
        f.write(f"**Test Set**: {results['test_windows_count']} windows across {results['test_sessions_count']} held-out sessions ({results['test_pairs_count']} test pairs)\n\n")
        f.write("### Comparison Against Baseline\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 Score | Similar Dist | Dissimilar Dist |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        s = results["siamese_metrics"]
        b = results["baseline_raw_euclidean"]
        f.write(f"| **SiameseNet (Learned)** | **{s['accuracy']*100:.1f}%** | {s['precision']:.3f} | {s['recall']:.3f} | **{s['f1_score']:.3f}** | {s['mean_distance_similar']:.3f} | {s['mean_distance_dissimilar']:.3f} |\n")
        f.write(f"| Raw Euclidean (Baseline) | {b['accuracy']*100:.1f}% | {b['precision']:.3f} | {b['recall']:.3f} | {b['f1_score']:.3f} | {b['mean_distance_similar']:.3f} | {b['mean_distance_dissimilar']:.3f} |\n\n")
        f.write(f"### Siamese Confusion Matrix\n")
        f.write(f"- **True Positives (TP)**: {s['confusion_matrix']['TP']}\n")
        f.write(f"- **True Negatives (TN)**: {s['confusion_matrix']['TN']}\n")
        f.write(f"- **False Positives (FP)**: {s['confusion_matrix']['FP']}\n")
        f.write(f"- **False Negatives (FN)**: {s['confusion_matrix']['FN']}\n")

    print(f"\n[Eval] Results Summary:")
    print(f"       SiameseNet Test Accuracy: {s['accuracy']*100:.1f}% (F1: {s['f1_score']:.3f})")
    print(f"       Raw Euclidean Baseline:   {b['accuracy']*100:.1f}% (F1: {b['f1_score']:.3f})")
    print(f"       Evaluation reports saved to {json_path} and {md_path}.")

    return results


if __name__ == "__main__":
    evaluate_siamese()
