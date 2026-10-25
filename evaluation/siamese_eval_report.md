# Siamese Network Evaluation Report (Phase 3)

**Data Provenance**: `synthetic_development`
**Test Set**: 32 windows across 3 held-out sessions (384 test pairs)

### Comparison Against Baseline

| Model | Accuracy | Precision | Recall | F1 Score | Similar Dist | Dissimilar Dist |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SiameseNet (Learned)** | **85.4%** | 0.821 | 0.906 | **0.861** | 0.220 | 0.959 |
| Raw Euclidean (Baseline) | 66.7% | 0.676 | 0.641 | 0.658 | 2.328 | 3.028 |

### Siamese Confusion Matrix
- **True Positives (TP)**: 174
- **True Negatives (TN)**: 154
- **False Positives (FP)**: 38
- **False Negatives (FN)**: 18
