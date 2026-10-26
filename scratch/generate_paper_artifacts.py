#!/usr/bin/env python3
"""
SmartResolve Research Paper Empirical Artifact Generator
Executes real evaluation protocols on data/seed_complaints.csv, computes baseline comparisons,
benchmarks latencies, runs few-shot ablation, exports JSON reports, and generates publication figures.
"""

import os
import sys
import time
import json
import random
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    f1_score
)
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

# Set global seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR = '/Users/mayanksmac/Desktop/atml2'
DATA_PATH = os.path.join(BASE_DIR, 'data/seed_complaints.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'research paper/results')
FIGURES_DIR = os.path.join(BASE_DIR, 'research paper/figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

def main():
    print("=== Step 1: Loading Dataset & Establishing Stratified Splits ===")
    df = pd.read_csv(DATA_PATH)
    texts = df['complaint_text'].tolist()
    categories = df['category'].values
    is_normals = df['is_normal'].values
    N_total = len(df)
    print(f"Total samples: {N_total}")

    # 70% / 15% / 15% Stratified Split
    indices = np.arange(N_total)
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=categories, random_state=RANDOM_SEED)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1765, stratify=categories[train_val_idx], random_state=RANDOM_SEED)

    print(f"Train samples: {len(train_idx)} ({len(train_idx)/N_total*100:.1f}%)")
    print(f"Val samples:   {len(val_idx)} ({len(val_idx)/N_total*100:.1f}%)")
    print(f"Test samples:  {len(test_idx)} ({len(test_idx)/N_total*100:.1f}%)")

    train_texts = [texts[i] for i in train_idx]
    train_y = categories[train_idx]
    train_is_normal = is_normals[train_idx]

    val_texts = [texts[i] for i in val_idx]
    val_y = categories[val_idx]
    val_is_normal = is_normals[val_idx]

    test_texts = [texts[i] for i in test_idx]
    test_y = categories[test_idx]
    test_is_normal = is_normals[test_idx]

    # --- Step 2: Baseline Model (TF-IDF + Logistic Regression) ---
    print("\n=== Step 2: Evaluating Baseline 1 (TF-IDF + Logistic Regression) ===")
    tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_test_tfidf = tfidf.transform(test_texts)

    baseline_clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=RANDOM_SEED)
    baseline_clf.fit(X_train_tfidf, train_y)
    baseline_preds = baseline_clf.predict(X_test_tfidf)

    base_acc = float(accuracy_score(test_y, baseline_preds))
    base_p, base_r, base_f1, _ = precision_recall_fscore_support(test_y, baseline_preds, average='macro', zero_division=0)
    print(f"Baseline TF-IDF -> Acc: {base_acc*100:.2f}%, Macro F1: {base_f1*100:.2f}%")

    # --- Step 3: Proposed Model (Sentence-BERT + Logistic Regression) ---
    print("\n=== Step 3: Evaluating Proposed Model (SBERT + Logistic Regression) ===")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    t0 = time.perf_counter()
    X_all_sbert = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    emb_time = time.perf_counter() - t0
    print(f"Computed 900 SBERT embeddings in {emb_time:.2f}s")

    X_train_sbert = X_all_sbert[train_idx]
    X_val_sbert = X_all_sbert[val_idx]
    X_test_sbert = X_all_sbert[test_idx]

    clf = LogisticRegression(C=2.0, max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
    clf.fit(X_train_sbert, train_y)
    test_preds = clf.predict(X_test_sbert)
    test_probs = clf.predict_proba(X_test_sbert)

    sbert_acc = float(accuracy_score(test_y, test_preds))
    sbert_p, sbert_r, sbert_f1, _ = precision_recall_fscore_support(test_y, test_preds, average='macro', zero_division=0)
    _, _, sbert_f1_weighted, _ = precision_recall_fscore_support(test_y, test_preds, average='weighted', zero_division=0)
    print(f"Proposed SBERT -> Acc: {sbert_acc*100:.2f}%, Macro F1: {sbert_f1*100:.2f}%, Weighted F1: {sbert_f1_weighted*100:.2f}%")

    classes_list = sorted(list(np.unique(categories)))
    cm = confusion_matrix(test_y, test_preds, labels=classes_list)
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(test_y, test_preds, labels=classes_list, zero_division=0)

    per_class_dict = {}
    for c, p, r, f, s in zip(classes_list, p_per, r_per, f1_per, sup_per):
        per_class_dict[str(c)] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s)
        }

    # --- Step 4: PyTorch Deep Autoencoder Anomaly Detection ---
    print("\n=== Step 4: Training & Evaluating PyTorch Deep Autoencoder ===")
    class EmbeddingAutoencoder(nn.Module):
        def __init__(self, input_dim=384, hidden_dim=128, latent_dim=32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.LeakyReLU(0.1)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, input_dim)
            )

        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            return nn.functional.normalize(out, p=2, dim=-1)

    # Train strictly on normal IT training tickets
    normal_train_X = X_train_sbert[train_is_normal == 1]
    ae = EmbeddingAutoencoder()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(normal_train_X, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_tensor), batch_size=16, shuffle=True
    )

    ae.train()
    for epoch in range(40):
        for (bx,) in train_loader:
            optimizer.zero_grad()
            rec = ae(bx)
            loss = criterion(rec, bx)
            loss.backward()
            optimizer.step()

    ae.eval()
    with torch.no_grad():
        # Calibrate threshold on normal validation split
        val_tensor = torch.tensor(X_val_sbert, dtype=torch.float32)
        val_recon = ae(val_tensor)
        val_mse = torch.mean((val_tensor - val_recon) ** 2, dim=1).numpy()
        val_normal_mse = val_mse[val_is_normal == 1]
        calibrated_threshold = float(np.percentile(val_normal_mse, 90) + 0.0001)

        # Test evaluation
        test_tensor = torch.tensor(X_test_sbert, dtype=torch.float32)
        test_recon = ae(test_tensor)
        test_mse = torch.mean((test_tensor - test_recon) ** 2, dim=1).numpy()

    # Ground truth: 1 for anomaly (is_normal == 0)
    y_test_anomaly_true = (test_is_normal == 0).astype(int)
    ae_roc_auc = float(roc_auc_score(y_test_anomaly_true, test_mse))
    ae_pr_auc = float(average_precision_score(y_test_anomaly_true, test_mse))
    
    pred_anomalies = (test_mse > calibrated_threshold).astype(int)
    ae_prec, ae_rec, ae_f1, _ = precision_recall_fscore_support(y_test_anomaly_true, pred_anomalies, average='binary', zero_division=0)
    fp_rate = float(np.sum((pred_anomalies == 1) & (y_test_anomaly_true == 0)) / max(1, np.sum(y_test_anomaly_true == 0)))

    print(f"Autoencoder -> Calibrated Threshold: {calibrated_threshold:.4f}")
    print(f"Autoencoder -> Test ROC-AUC: {ae_roc_auc*100:.2f}%, PR-AUC: {ae_pr_auc*100:.2f}%")
    print(f"Autoencoder -> Precision: {ae_prec*100:.2f}%, Recall: {ae_rec*100:.2f}%, F1: {ae_f1*100:.2f}%, FPR: {fp_rate*100:.2f}%")

    # --- Step 5: Prototypical Few-Shot Classification Experiment ---
    print("\n=== Step 5: Prototypical Few-Shot Learning Evaluation across k={1, 3, 5} ===")
    few_shot_results = []
    
    # Pre-group training texts by category
    category_examples_map = {}
    for cat in classes_list:
        category_examples_map[cat] = [
            train_texts[i] for i in range(len(train_texts))
            if train_y[i] == cat and train_is_normal[i] == 1
        ]

    for k_shots in [1, 3, 5]:
        trial_accs = []
        trial_f1s = []
        for trial in range(10):
            # Sample k examples per class to form prototypes
            prototypes = {}
            for cat in classes_list:
                pool = category_examples_map[cat]
                chosen = random.sample(pool, min(k_shots, len(pool)))
                embs = embedder.encode(chosen, normalize_embeddings=True, show_progress_bar=False)
                centroid = np.mean(embs, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-6:
                    centroid = centroid / norm
                prototypes[cat] = centroid

            # Evaluate on held-out test split
            preds_fs = []
            for test_emb in X_test_sbert:
                sims = {c: float(np.dot(test_emb, p)) for c, p in prototypes.items()}
                best_c = max(sims, key=sims.get)
                preds_fs.append(best_c)

            t_acc = accuracy_score(test_y, preds_fs)
            t_f1 = f1_score(test_y, preds_fs, average='macro', zero_division=0)
            trial_accs.append(t_acc)
            trial_f1s.append(t_f1)

        mean_acc = float(np.mean(trial_accs))
        std_acc = float(np.std(trial_accs))
        mean_f1 = float(np.mean(trial_f1s))
        print(f"Few-Shot k={k_shots} -> Mean Acc: {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%), Macro F1: {mean_f1*100:.2f}%")
        few_shot_results.append({
            "k_shots": k_shots,
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "macro_f1": round(mean_f1, 4),
            "test_samples": len(test_idx)
        })

    # --- Step 6: Operational Latency Benchmarking ---
    print("\n=== Step 6: Measuring Operational Inference Latencies (50 iterations) ===")
    sample_text = "The laser printer on floor 3 is offline with a paper jam in tray 2."
    latencies_emb = []
    latencies_clf = []
    latencies_ae = []
    latencies_total = []

    for _ in range(50):
        t_start = time.perf_counter()
        
        # Step A: Embedding
        t_e0 = time.perf_counter()
        vec = embedder.encode([sample_text], normalize_embeddings=True, show_progress_bar=False)[0]
        lat_emb = (time.perf_counter() - t_e0) * 1000
        latencies_emb.append(lat_emb)

        # Step B: Classification
        t_c0 = time.perf_counter()
        prob = clf.predict_proba([vec])[0]
        pred_label = clf.classes_[np.argmax(prob)]
        lat_clf = (time.perf_counter() - t_c0) * 1000
        latencies_clf.append(lat_clf)

        # Step C: Autoencoder Anomaly Scoring
        t_a0 = time.perf_counter()
        with torch.no_grad():
            tx_single = torch.tensor([vec], dtype=torch.float32)
            rec_single = ae(tx_single)
            err_single = torch.mean((tx_single - rec_single) ** 2).item()
        lat_ae = (time.perf_counter() - t_a0) * 1000
        latencies_ae.append(lat_ae)

        # End-to-end total
        lat_tot = (time.perf_counter() - t_start) * 1000
        latencies_total.append(lat_tot)

    latency_metrics = {
        "embedding_latency_ms": {
            "mean": round(float(np.mean(latencies_emb)), 2),
            "median": round(float(np.median(latencies_emb)), 2),
            "p95": round(float(np.percentile(latencies_emb, 95)), 2)
        },
        "classification_latency_ms": {
            "mean": round(float(np.mean(latencies_clf)), 2),
            "median": round(float(np.median(latencies_clf)), 2),
            "p95": round(float(np.percentile(latencies_clf, 95)), 2)
        },
        "autoencoder_latency_ms": {
            "mean": round(float(np.mean(latencies_ae)), 2),
            "median": round(float(np.median(latencies_ae)), 2),
            "p95": round(float(np.percentile(latencies_ae, 95)), 2)
        },
        "end_to_end_inference_latency_ms": {
            "mean": round(float(np.mean(latencies_total)), 2),
            "median": round(float(np.median(latencies_total)), 2),
            "p95": round(float(np.percentile(latencies_total, 95)), 2)
        }
    }
    print(f"End-to-End Latency: Mean = {latency_metrics['end_to_end_inference_latency_ms']['mean']} ms, p95 = {latency_metrics['end_to_end_inference_latency_ms']['p95']} ms")

    # --- Step 7: Export JSON Results ---
    print("\n=== Step 7: Exporting Results JSON Files ===")
    
    # 1. classification_report.json
    class_report_data = {
        "dataset_name": "Customer IT Support Dataset (Tobi-Bueck/customer-support-tickets)",
        "split_protocol": "70% train / 15% val / 15% test stratified",
        "random_seed": RANDOM_SEED,
        "test_samples": len(test_idx),
        "overall_metrics": {
            "accuracy": round(sbert_acc, 4),
            "macro_precision": round(sbert_p, 4),
            "macro_recall": round(sbert_r, 4),
            "macro_f1": round(sbert_f1, 4),
            "weighted_f1": round(sbert_f1_weighted, 4)
        },
        "per_class": per_class_dict,
        "confusion_matrix": {
            "labels": classes_list,
            "matrix": cm.tolist()
        }
    }
    with open(os.path.join(RESULTS_DIR, 'classification_report.json'), 'w') as f:
        json.dump(class_report_data, f, indent=2)

    # 2. evaluation_summary.json
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_total_samples": N_total,
        "dataset_classes_count": len(classes_list),
        "dataset_splits": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx)
        },
        "baseline_comparison": {
            "tfidf_logistic_regression": {
                "accuracy": round(base_acc, 4),
                "macro_f1": round(base_f1, 4)
            },
            "sbert_logistic_regression": {
                "accuracy": round(sbert_acc, 4),
                "macro_f1": round(sbert_f1, 4),
                "improvement_over_baseline_pct": round((sbert_acc - base_acc) * 100, 2)
            }
        },
        "autoencoder_anomaly_detection": {
            "latent_dimension": 32,
            "calibrated_threshold": round(calibrated_threshold, 4),
            "roc_auc": round(ae_roc_auc, 4),
            "pr_auc": round(ae_pr_auc, 4),
            "precision": round(ae_prec, 4),
            "recall": round(ae_rec, 4),
            "f1_score": round(ae_f1, 4),
            "false_positive_rate": round(fp_rate, 4),
            "normal_test_count": int(np.sum(test_is_normal == 1)),
            "anomaly_test_count": int(np.sum(test_is_normal == 0))
        },
        "few_shot_prototypical_evaluation": few_shot_results,
        "latency_benchmarks": latency_metrics
    }
    with open(os.path.join(RESULTS_DIR, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    # 3. experiment_metadata.json
    meta_data = {
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "transformers_embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": 384
        },
        "hyperparameters": {
            "classifier_solver": "lbfgs",
            "classifier_C": 2.0,
            "classifier_class_weight": "balanced",
            "autoencoder_epochs": 40,
            "autoencoder_batch_size": 16,
            "autoencoder_lr": 0.001,
            "autoencoder_weight_decay": 0.0001,
            "autoencoder_threshold_percentile": 90
        },
        "random_seed": RANDOM_SEED
    }
    with open(os.path.join(RESULTS_DIR, 'experiment_metadata.json'), 'w') as f:
        json.dump(meta_data, f, indent=2)

    print("Exported JSON result files successfully!")

    # --- Step 8: Generate Publication Figures (PNG) ---
    print("\n=== Step 8: Generating High-Resolution Publication Figures ===")
    
    # 1. Dataset Distribution
    plt.figure(figsize=(9, 4.5), dpi=300)
    cat_counts = df['category'].value_counts()
    colors = sns.color_palette('Blues_r', len(cat_counts))
    bars = plt.barh(cat_counts.index, cat_counts.values, color=colors, edgecolor='#1e293b', linewidth=0.8)
    plt.xlabel('Number of Complaints (N)')
    plt.title('Figure 4: Ground-Truth Category Distribution in SmartResolve Benchmark (N=900)', pad=12, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 2, bar.get_y() + bar.get_height()/2, f'{int(w)}', va='center', ha='left', fontsize=9, color='#0f172a', fontweight='semibold')
    plt.xlim(0, max(cat_counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'dataset_distribution.png'))
    plt.close()

    # 2. Confusion Matrix
    plt.figure(figsize=(8.5, 7), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_list, yticklabels=classes_list, cbar=True)
    plt.ylabel('Ground Truth Category', fontweight='bold', labelpad=8)
    plt.xlabel('Predicted Category', fontweight='bold', labelpad=8)
    plt.title('Figure 5: Normalized Confusion Matrix on Held-Out Test Set (N=135)', pad=12, fontweight='bold')
    plt.xticks(rotation=35, ha='right', fontsize=8.5)
    plt.yticks(rotation=0, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'))
    plt.close()

    # 3. Per-Class Classification Metrics (P, R, F1)
    plt.figure(figsize=(10, 5), dpi=300)
    x = np.arange(len(classes_list))
    width = 0.26
    plt.bar(x - width, p_per, width, label='Precision', color='#3b82f6', edgecolor='#1e3a8a')
    plt.bar(x, r_per, width, label='Recall', color='#10b981', edgecolor='#065f46')
    plt.bar(x + width, f1_per, width, label='F1-Score', color='#8b5cf6', edgecolor='#4c1d95')
    plt.ylabel('Score (0.0 to 1.0)', fontweight='bold')
    plt.title('Figure 6: Per-Class Precision, Recall, and F1-Score Breakdown (Held-Out Test Set)', pad=12, fontweight='bold')
    plt.xticks(x, classes_list, rotation=30, ha='right', fontsize=8.5)
    plt.ylim(0, 1.1)
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'classification_metrics.png'))
    plt.close()

    # 4. Anomaly Detection Results (Reconstruction Error Distribution + ROC Curve)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    
    normal_errs = test_mse[y_test_anomaly_true == 0]
    ano_errs = test_mse[y_test_anomaly_true == 1]
    
    ax1.hist(normal_errs, bins=15, alpha=0.6, label='Normal IT (in-domain)', color='#2563eb', density=True)
    ax1.hist(ano_errs, bins=10, alpha=0.6, label='OOD Anomalies', color='#ef4444', density=True)
    ax1.axvline(calibrated_threshold, color='#0f172a', linestyle='--', linewidth=1.5, label=f'Threshold $\\tau={calibrated_threshold:.4f}$')
    ax1.set_xlabel('Reconstruction MSE Error $e(x)$')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('(a) Reconstruction Error Distribution', fontweight='bold')
    ax1.legend(fontsize=8.5)

    fpr_vals, tpr_vals, _ = roc_curve(y_test_anomaly_true, test_mse)
    ax2.plot(fpr_vals, tpr_vals, color='#2563eb', lw=2, label=f'Autoencoder (ROC-AUC = {ae_roc_auc*100:.1f}%)')
    ax2.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1, label='Random Chance (50%)')
    ax2.set_xlabel('False Positive Rate (FPR)')
    ax2.set_ylabel('True Positive Rate (TPR)')
    ax2.set_title('(b) Anomaly Detection ROC Curve', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8.5)

    plt.suptitle('Figure 7: Autoencoder Anomaly Detection Performance & Error Separation', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'anomaly_detection_results.png'), bbox_inches='tight')
    plt.close()

    # 5. Few-Shot Evaluation
    plt.figure(figsize=(7, 4.5), dpi=300)
    k_vals = [r['k_shots'] for r in few_shot_results]
    acc_vals = [r['mean_accuracy'] * 100 for r in few_shot_results]
    std_vals = [r['std_accuracy'] * 100 for r in few_shot_results]
    f1_vals = [r['macro_f1'] * 100 for r in few_shot_results]

    plt.errorbar(k_vals, acc_vals, yerr=std_vals, fmt='-o', color='#2563eb', lw=2, capsize=4, label='Top-1 Accuracy (%)')
    plt.plot(k_vals, f1_vals, '-s', color='#10b981', lw=2, label='Macro F1-Score (%)')
    plt.axhline(12.5, color='#94a3b8', linestyle=':', label='Random Baseline (12.5%)')
    plt.xlabel('Support Examples per Category ($k$-shots)', fontweight='bold')
    plt.ylabel('Performance Metric (%)', fontweight='bold')
    plt.title('Figure 8: Prototypical Few-Shot Performance vs. Support Set Size (k)', pad=12, fontweight='bold')
    plt.xticks(k_vals)
    plt.ylim(0, 85)
    plt.legend(frameon=True, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'few_shot_evaluation.png'))
    plt.close()

    # 6. Diagram 1: System Architecture
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('off')
    arch_text = """
    +---------------------------------------------------------------------------------------------------------+
    |                                SMARTRESOLVE SYSTEM ARCHITECTURE                                         |
    +---------------------------------------------------------------------------------------------------------+
    |  PRESENTATION LAYER (React + Vite + Recharts + Lucide-React)                                            |
    |  [Employee Portal]             [IT Resolver Console]                [ML Admin Console & Few-Shot Studio]|
    |  - Natural language ticket     - Queue manager, SLA priority badges - Empirical metrics dashboard       |
    |  - Real-time tracking status   - Resolution notes & audit workflow  - Category prototype manager        |
    +----------------------------------------------------+----------------------------------------------------+
                                                         | REST / JSON (Reverse Proxy)
    +----------------------------------------------------v----------------------------------------------------+
    |  APPLICATION & SERVICE LAYER (FastAPI - Python 3.13)                                                    |
    |  - POST /api/tickets: Ingestion, Validation, Triage, Routing                                             |
    |  - GET  /api/dashboard: Real-time operational metrics, volume, department distribution                  |
    |  - POST /api/models/train: Re-training, validation-calibration & evaluation execution                   |
    |  - Rule-Based Priority & Department Router: Matrix of Urgency x Impact Scope                            |
    +----------------------------------------------------+----------------------------------------------------+
                                                         |
    +----------------------------------------------------v----------------------------------------------------+
    |  MACHINE LEARNING & INTELLIGENCE ENGINE (ml_engine.py)                                                  |
    |  [Sentence-BERT: all-MiniLM-L6-v2] -> 384-dimensional dense semantic embeddings (unit-sphere)            |
    |  [Module A: Supervised Classifier] -> Multiclass Logistic Regression (8 ITIL categories, C=2.0)          |
    |  [Module B: Semantic Autoencoder] -> PyTorch 32D Bottleneck Anomaly Detector (Threshold tau=0.0021)     |
    |  [Module C: Prototypical Engine]  -> Few-shot classification via dynamic support set centroids          |
    +----------------------------------------------------+----------------------------------------------------+
                                                         | SQLite WAL Driver
    +----------------------------------------------------v----------------------------------------------------+
    |  DATA & PERSISTENCE LAYER (SQLite with Write-Ahead Logging - backend/smartresolve.db)                   |
    |  - `tickets`: Complete ticket records, status lifecycle, resolver notes, priority & classification      |
    |  - `categories`: Master ITIL category list, department routing mapping, description                     |
    |  - `support_examples`: Labeled reference examples powering few-shot prototypical centroids              |
    |  - `complaint_dataset`: 900 benchmark tickets (800 IT tickets + 100 OOD anomalies)                      |
    |  - `model_runs`: Audit history of accuracy, macro F1, autoencoder threshold, and ROC-AUC                |
    +---------------------------------------------------------------------------------------------------------+
    """
    ax.text(0.5, 0.5, arch_text, fontsize=7.5, family='monospace', va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8fafc', edgecolor='#cbd5e1', lw=1.5))
    plt.title('Figure 1: SmartResolve End-to-End System Architecture (Layered Model)', pad=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'system_architecture.png'))
    plt.close()

    # 7. Diagram 2: Complaint Processing Workflow
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('off')
    flow_text = """
    USER COMPLAINT SUBMISSION: "Our floor printer has a paper jam and error code E-04"
                      |
                      v
    [Step 1: Input Validation & Sanitization] -> Verify text length (>5 chars), sanitize HTML/special chars
                      |
                      v
    [Step 2: Dense Semantic Encoding] --------> Pretrained Sentence-BERT (all-MiniLM-L6-v2) generates z in R^384
                      |
                      +---------------------------------------+
                      |                                       |
                      v                                       v
    [Step 3: Supervised Classification]     [Step 4: Autoencoder Reconstruction]
    Predict category & softmax confidence   Compute MSE error e(x) = ||z - z_hat||^2
    Class: "Printer & Peripheral" (p=0.88)  Error: 0.0011 (Threshold tau = 0.0021)
                      |                                       |
                      +-------------------+-------------------+
                                          |
                                          v
    [Step 5: Triage & Anomaly Assessment Decision Logic]
    IF e(x) > tau OR max(p) < 0.60:
         Flag: [HUMAN REVIEW REQUIRED] -> Escalate to General Service Desk Triage
    ELSE:
         Flag: [AUTOMATED TRIAGE CONFIRMED]
                                          |
                                          v
    [Step 6: Priority & Routing Recommendation]
    Impact: Individual | Urgency: Medium  ==>  Calculated Priority: "Medium" (SLA: 24h)
    Category: "Printer & Peripheral"      ==>  Assigned Group: "Field Support & Peripherals"
                                          |
                                          v
    [Step 7: Ticket Generation & Persistence] -> Generate ID "SR-A9F21B", write to SQLite, notify Resolver
    """
    ax.text(0.5, 0.5, flow_text, fontsize=8.0, family='monospace', va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8fafc', edgecolor='#cbd5e1', lw=1.5))
    plt.title('Figure 2: Complaint Ingestion, Semantic Triage, and Dispatch Workflow', pad=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'complaint_processing_workflow.png'))
    plt.close()

    # 8. Diagram 3: ML Pipeline
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('off')
    ml_pipe_text = """
    RAW COMPLAINT TEXT x_i: "Unable to log in to corporate portal after password reset"
                                   |
                                   v
    +----------------------------------------------------------------------------------------------------+
    | FEATURE EXTRACTION: Sentence-BERT Pretrained Transformer (Frozen, all-MiniLM-L6-v2)               |
    | - Mean Pooling of Subword Token Representations -> Unit-Sphere Projection z_i in S^{383}          |
    +----------------------------------------------+-----------------------------------------------------+
                                                   |
                   +-------------------------------+-------------------------------+
                   |                               |                               |
                   v                               v                               v
    +-----------------------------+ +-----------------------------+ +-----------------------------+
    | MODULE A: SUPERVISED        | | MODULE B: PYTORCH DEEP      | | MODULE C: PROTOCOL FEW-SHOT |
    | CLASSIFIER                  | | AUTOENCODER                 | | PROTOTYPICAL CLASSIFIER     |
    | Algorithm: Logistic Regres. | | Architecture: 384->128->32  | | Centroid: p_c = (1/|S_c|)   |
    | Loss: Cross-Entropy (C=2.0) | | Latent Bottleneck: 32 dim   | |           * sum(z_i)        |
    | Output: P(y=c|z) across     | | Reconstruction: z_hat       | | Metric: Cosine Similarity   |
    |         8 ITIL Categories   | | Anomaly Score: ||z-z_hat||^2| |         sim(z, p_c)         |
    +--------------+--------------+ +--------------+--------------+ +--------------+--------------+
                   |                               |                               |
                   v                               v                               v
            Predicted Class               Out-of-Distribution              Similarity Score for
        "Account & Authentication"               Score                      Emerging / Zero-Shot
               (p = 0.91)                  (0.0009 < 0.0021)                      Categories
    """
    ax.text(0.5, 0.5, ml_pipe_text, fontsize=8.0, family='monospace', va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8fafc', edgecolor='#cbd5e1', lw=1.5))
    plt.title('Figure 3: Tri-Component ATML Machine Learning Pipeline Architecture', pad=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'ml_pipeline.png'))
    plt.close()

    print("Generated all 8 research figures successfully in research paper/figures/!")

if __name__ == '__main__':
    main()
