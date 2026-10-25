# Combat Autoencoder Evaluation Report (Phase 4)

**Data Provenance**: `synthetic_development`
**Test Set**: 32 held-out normal windows vs 50 anomalous windows
**Validated Threshold**: `0.0966` (Fitted strictly on training partition)

### Reconstruction Error Distributions

- **Normal Windows**: Mean = `0.0555`, Median = `0.0389`, Std = `0.0514`
- **Anomalous Windows**: Mean = `2129.9875`, Median = `2088.8442`, Std = `903.2073`
- **Separation Ratio**: `38370.15x` higher error on anomalous windows

### Comparison Against Baseline

| Model | Threshold | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CombatAutoencoder (Latent)** | **0.097** | **93.9%** | **0.909** | **1.000** | **0.952** |
| Raw Z-Score Baseline | 1.550 | 97.6% | 0.962 | 1.000 | 0.980 |

### Autoencoder Confusion Matrix
- **True Positives (TP - Correctly Flagged Anomalies)**: 50
- **True Negatives (TN - Correctly Passed Normal)**: 27
- **False Positives (FP - False Alarms on Normal)**: 5
- **False Negatives (FN - Missed Anomalies)**: 0
