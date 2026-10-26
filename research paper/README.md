# SmartResolve Research Paper & Replication Artifacts

This directory contains the complete academic manuscript, experimental results, and high-resolution figures for the paper:

> **SmartResolve: An Integrated Machine Learning Framework for Semantic IT Complaint Classification, Anomaly Detection, and Few-Shot Helpdesk Ticketing**  
> **Authors:** Mayank Sawant, Sayali Rajput  
> **Affiliation:** B.Tech. in Artificial Intelligence & Data Science, SVKM's NMIMS, Navi Mumbai, India  
> **Target Venue:** IEEE Conference / Journal Submission (Advanced Techniques in Machine Learning — ATML)

---

## 1. Directory Structure

```
research paper/
├── research paper.md             # The primary IEEE-style academic manuscript (Markdown)
├── README.md                     # This file (Replication, evidence traceability, and user guide)
├── figures/                      # High-resolution (300 DPI) publication-quality figures
│   ├── system_architecture.png           # Figure 1: Layered system architecture diagram
│   ├── complaint_processing_workflow.png # Figure 2: End-to-end complaint triage and routing sequence
│   ├── ml_pipeline.png                   # Figure 3: Tri-component ML pipeline architecture
│   ├── dataset_distribution.png          # Figure 4: Category breakdown across the 900 benchmark tickets
│   ├── confusion_matrix.png              # Figure 5: Normalized test confusion matrix (N=135)
│   ├── classification_metrics.png        # Figure 6: Per-class Precision, Recall, and F1-score plot
│   ├── anomaly_detection_results.png     # Figure 7: Reconstruction error distribution and ROC curve
│   └── few_shot_evaluation.png           # Figure 8: Prototypical few-shot accuracy scaling (k=1 to 5)
└── results/                      # Machine-readable JSON telemetry and experimental metadata
    ├── classification_report.json        # Test precision, recall, F1, confusion matrix, and support
    ├── evaluation_summary.json           # Baseline vs SBERT, Autoencoder ROC-AUC, latency benchmarks
    └── experiment_metadata.json          # Hardware, library versions, hyperparameters, and seeds
```

---

## 2. Key Empirical Findings Summary

All metrics are computed on the strictly held-out test split ($N=135$) of real enterprise support tickets:

| Evaluation Dimension | Proposed Model (SmartResolve) | Baseline / Context | Significance |
|---|---|---|---|
| **Classification Accuracy** | **69.63%** | $64.44\%$ (TF-IDF Baseline) | $+5.19\%$ improvement using Sentence-BERT embeddings |
| **Classification Macro F1** | **69.63%** | $63.84\%$ (TF-IDF Baseline) | Balanced performance across all 8 IT categories |
| **Security Incident F1** | **87.50%** ($93.3\%$ Recall) | Critical incident triage | High-risk cyber/security issues are rarely missed |
| **Account/IAM Access F1** | **86.67%** ($86.7\%$ Recall) | Identity & Access tier | Robust handling of SSO, MFA, and lockout complaints |
| **Anomaly Detection ROC-AUC** | **76.21%** | $50.0\%$ (Random guessing) | Statistically separates IT complaints from non-IT queries |
| **Anomaly False Positive Rate** | **8.26%** | Threshold $\tau = 0.0018$ | Less than 1 in 10 valid complaints flagged for human triage |
| **Few-Shot Top-1 Accuracy** | **44.81%** ($k=5$ shots) | $12.5\%$ (Random guessing) | Rapidly adapts to emerging categories without fine-tuning |
| **End-to-End Inference Latency**| **5.93 ms** (Mean), **7.83 ms** ($p_{95}$)| Sub-10ms operational limit | Operates at $>150$ queries/second on CPU |

---

## 3. How to Reproduce All Experiments & Figures

You can reproduce the exact JSON reports and regenerate all 8 PNG figures directly using the Python environment:

```bash
# 1. Ensure dependencies are active
pip install -r backend/requirements.txt matplotlib seaborn

# 2. Run the experiment and figure generation script
python3 scratch/generate_paper_artifacts.py
```

This script will:
1. Load `data/seed_complaints.csv` ($N=900$).
2. Split stratified 70/15/15 using random seed $42$.
3. Compute TF-IDF + Logistic Regression baseline metrics.
4. Compute Sentence-BERT (`all-MiniLM-L6-v2`) + Logistic Regression metrics.
5. Train the deep autoencoder strictly on in-domain training tickets and evaluate ROC-AUC on the held-out test set.
6. Run 10 randomized episodes of prototypical few-shot classification across $k \in \{1, 3, 5\}$.
7. Benchmark 50 operational inference cycles for embedding, classification, and autoencoder scoring.
8. Output updated JSON files to `research paper/results/` and PNG figures to `research paper/figures/`.

---

## 4. Academic Integrity & Traceability Notice

- **No Synthetic Shortcuts**: All 900 benchmark tickets are real human support complaints sourced from the published Customer IT Support repository on Hugging Face (`Tobi-Bueck/customer-support-tickets`).
- **No Data Leakage**: Transformers and classifiers are evaluated strictly on held-out test splits sequestered prior to training or threshold calibration.
- **Traceable JSON Evidence**: Every numerical value and plot in `research paper.md` directly traces back to the structured JSON files in `research paper/results/`.
