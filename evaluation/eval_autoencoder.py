"""
Evaluation script for the Combat Autoencoder on held-out test sessions (Phase 4).
Evaluates reconstruction error distribution on held-out normal sessions vs anomalous sessions,
calculates classification metrics at the pre-determined training threshold, and compares against
a normalized feature magnitude baseline.
"""

import os
import sys
import json
import random
import numpy as np
import torch

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.features import load_session_events, extract_window_features
from ml.dataset import NUMERICAL_FEATURE_NAMES, FeatureScaler, session_train_test_split
from ml.autoencoder import CombatAutoencoder
from ml.synthetic_generator import generate_anomalous_dataset


def evaluate_autoencoder(
    model_path: str = "saved_models/autoencoder_checkpoint.pt",
    synthetic_dir: str = "data/synthetic",
    output_dir: str = "evaluation",
    seed: int = 42,
) -> dict:
    """Evaluates CombatAutoencoder on held-out normal vs anomalous sessions."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please run training first.")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    scaler = FeatureScaler.from_dict(checkpoint["scaler"])
    threshold = float(checkpoint["threshold"])
    feature_names = checkpoint.get("feature_names", list(NUMERICAL_FEATURE_NAMES))

    # 1. Load Model
    model = CombatAutoencoder(
        in_features=checkpoint["in_features"],
        latent_dim=checkpoint["latent_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Extract Reference Normal Sessions and Isolate Held-Out Test Split
    normal_files = [
        os.path.join(synthetic_dir, f)
        for f in os.listdir(synthetic_dir)
        if f.endswith(".jsonl") and "anomalous" not in f
    ]

    all_normal_records = []
    for fpath in normal_files:
        events = load_session_events(fpath)
        windows = extract_window_features(events, window_size_sec=5.0, stride_sec=2.0)
        for w in windows:
            all_normal_records.append(w)

    # Re-apply exact session-level split to isolate held-out normal test set
    _, held_out_normal_records = session_train_test_split(all_normal_records, test_ratio=0.25, seed=seed)
    normal_test_sessions = sorted(list(set(r["session_id"] for r in held_out_normal_records)))

    # 3. Extract Anomalous Evaluation Sessions
    anomalous_files = [
        os.path.join(synthetic_dir, f)
        for f in os.listdir(synthetic_dir)
        if f.endswith(".jsonl") and "anomalous" in f
    ]
    if len(anomalous_files) < 4:
        print("[Eval-AE] Generating synthetic anomalous sessions for evaluation...")
        generate_anomalous_dataset(output_dir=synthetic_dir, session_count=5, seed=999)
        anomalous_files = [
            os.path.join(synthetic_dir, f)
            for f in os.listdir(synthetic_dir)
            if f.endswith(".jsonl") and "anomalous" in f
        ]

    anomalous_records = []
    for fpath in anomalous_files:
        events = load_session_events(fpath)
        windows = extract_window_features(events, window_size_sec=5.0, stride_sec=2.0)
        for w in windows:
            anomalous_records.append(w)

    print(f"[Eval-AE] Evaluating on:")
    print(f"          - Held-out normal windows: {len(held_out_normal_records)} across {len(normal_test_sessions)} sessions")
    print(f"          - Anomalous test windows:  {len(anomalous_records)} across {len(anomalous_files)} sessions")

    # 4. Prepare Features
    X_normal_raw = np.array([[float(r[col]) for col in feature_names] for r in held_out_normal_records], dtype=np.float32)
    X_anom_raw = np.array([[float(r[col]) for col in feature_names] for r in anomalous_records], dtype=np.float32)

    X_normal_norm = scaler.transform(X_normal_raw)
    X_anom_norm = scaler.transform(X_anom_raw)

    t_normal = torch.tensor(X_normal_norm, dtype=torch.float32)
    t_anom = torch.tensor(X_anom_norm, dtype=torch.float32)

    # 5. Compute Autoencoder Reconstruction Errors
    with torch.no_grad():
        normal_errors = model.compute_reconstruction_error(t_normal).cpu().numpy()
        anom_errors = model.compute_reconstruction_error(t_anom).cpu().numpy()

    # 6. Simple Baseline: Normalized Feature Mean-Squared Magnitude (Z-score outlier detection)
    baseline_normal_scores = np.mean(X_normal_norm ** 2, axis=1)
    baseline_anom_scores = np.mean(X_anom_norm ** 2, axis=1)
    # Baseline threshold: mean + 2*std on normal baseline scores
    baseline_thresh = float(np.mean(baseline_normal_scores) + 2.0 * np.std(baseline_normal_scores))

    # 7. Classification Metrics at Validated Threshold
    # Positive class = Anomalous (1), Negative class = Normal (0)
    all_ae_scores = np.concatenate([normal_errors, anom_errors])
    all_targets = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anom_errors))])

    all_base_scores = np.concatenate([baseline_normal_scores, baseline_anom_scores])

    def calc_metrics(scores, thresh, targets):
        preds = (scores > thresh).astype(int)
        tp = int(np.sum((preds == 1) & (targets == 1)))
        tn = int(np.sum((preds == 0) & (targets == 0)))
        fp = int(np.sum((preds == 1) & (targets == 0)))
        fn = int(np.sum((preds == 0) & (targets == 1)))

        acc = float((tp + tn) / max(1, len(targets)))
        prec = float(tp / max(1, tp + fp))
        rec = float(tp / max(1, tp + fn))
        f1 = float(2 * (prec * rec) / max(1e-6, prec + rec))

        return {
            "threshold": float(round(thresh, 4)),
            "accuracy": float(round(acc, 4)),
            "precision": float(round(prec, 4)),
            "recall": float(round(rec, 4)),
            "f1_score": float(round(f1, 4)),
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        }

    ae_metrics = calc_metrics(all_ae_scores, threshold, all_targets)
    base_metrics = calc_metrics(all_base_scores, baseline_thresh, all_targets)

    results = {
        "model": "CombatAutoencoder (Phase 4)",
        "provenance": checkpoint.get("provenance", "synthetic_development"),
        "held_out_normal_windows": len(held_out_normal_records),
        "held_out_normal_sessions": len(normal_test_sessions),
        "anomalous_windows": len(anomalous_records),
        "anomalous_sessions": len(anomalous_files),
        "reconstruction_error_distribution": {
            "normal_mean": float(round(float(np.mean(normal_errors)), 4)),
            "normal_std": float(round(float(np.std(normal_errors)), 4)),
            "normal_median": float(round(float(np.median(normal_errors)), 4)),
            "anomalous_mean": float(round(float(np.mean(anom_errors)), 4)),
            "anomalous_std": float(round(float(np.std(anom_errors)), 4)),
            "anomalous_median": float(round(float(np.median(anom_errors)), 4)),
            "separation_ratio": float(round(float(np.mean(anom_errors) / max(1e-4, np.mean(normal_errors))), 2)),
        },
        "autoencoder_metrics": ae_metrics,
        "baseline_zscore_metrics": base_metrics,
    }

    # 8. Save Reports
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "autoencoder_eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = os.path.join(output_dir, "autoencoder_eval_report.md")
    dist = results["reconstruction_error_distribution"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Combat Autoencoder Evaluation Report (Phase 4)\n\n")
        f.write(f"**Data Provenance**: `{results['provenance']}`\n")
        f.write(f"**Test Set**: {results['held_out_normal_windows']} held-out normal windows vs {results['anomalous_windows']} anomalous windows\n")
        f.write(f"**Validated Threshold**: `{threshold:.4f}` (Fitted strictly on training partition)\n\n")
        f.write("### Reconstruction Error Distributions\n\n")
        f.write(f"- **Normal Windows**: Mean = `{dist['normal_mean']:.4f}`, Median = `{dist['normal_median']:.4f}`, Std = `{dist['normal_std']:.4f}`\n")
        f.write(f"- **Anomalous Windows**: Mean = `{dist['anomalous_mean']:.4f}`, Median = `{dist['anomalous_median']:.4f}`, Std = `{dist['anomalous_std']:.4f}`\n")
        f.write(f"- **Separation Ratio**: `{dist['separation_ratio']}x` higher error on anomalous windows\n\n")
        f.write("### Comparison Against Baseline\n\n")
        f.write("| Model | Threshold | Accuracy | Precision | Recall | F1 Score |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        a = results["autoencoder_metrics"]
        b = results["baseline_zscore_metrics"]
        f.write(f"| **CombatAutoencoder (Latent)** | **{a['threshold']:.3f}** | **{a['accuracy']*100:.1f}%** | **{a['precision']:.3f}** | **{a['recall']:.3f}** | **{a['f1_score']:.3f}** |\n")
        f.write(f"| Raw Z-Score Baseline | {b['threshold']:.3f} | {b['accuracy']*100:.1f}% | {b['precision']:.3f} | {b['recall']:.3f} | {b['f1_score']:.3f} |\n\n")
        f.write("### Autoencoder Confusion Matrix\n")
        f.write(f"- **True Positives (TP - Correctly Flagged Anomalies)**: {a['confusion_matrix']['TP']}\n")
        f.write(f"- **True Negatives (TN - Correctly Passed Normal)**: {a['confusion_matrix']['TN']}\n")
        f.write(f"- **False Positives (FP - False Alarms on Normal)**: {a['confusion_matrix']['FP']}\n")
        f.write(f"- **False Negatives (FN - Missed Anomalies)**: {a['confusion_matrix']['FN']}\n")

    print(f"\n[Eval-AE] Results Summary:")
    print(f"          Normal Mean Error:    {dist['normal_mean']:.4f}")
    print(f"          Anomalous Mean Error: {dist['anomalous_mean']:.4f} ({dist['separation_ratio']}x)")
    print(f"          Autoencoder Accuracy: {a['accuracy']*100:.1f}% (F1: {a['f1_score']:.3f})")
    print(f"          Raw Baseline Accuracy:{b['accuracy']*100:.1f}% (F1: {b['f1_score']:.3f})")
    print(f"          Reports saved to {json_path} and {md_path}.")

    return results


if __name__ == "__main__":
    evaluate_autoencoder()
