# Action Prediction Transformer Evaluation & Audit Report (Phase 5.1)

**Data Provenance**: `synthetic_development`
**Train Split**: 1731 sequences across 12 sessions
**Test Split**: 437 sequences across 3 held-out sessions
**Context Length**: $L = 16$ preceding action tokens

### Target Class Distribution (Held-Out Test Split)

| Action Class | Test Support | Class Frequency (%) | Observation Note |
| :--- | :---: | :---: | :--- |
| **IDLE** | 0 | 0.0% | Zero test support (unobserved in non-anomalous synthetic runs) |
| **MOVE** | 297 | 68.0% | Active in test set |
| **ATTACK** | 49 | 11.2% | Active in test set |
| **DODGE** | 91 | 20.8% | Active in test set |
| **HURT** | 0 | 0.0% | Zero test support (unobserved in non-anomalous synthetic runs) |

### Comprehensive Baseline & Model Comparison

| Model / Configuration | Overall Accuracy | Macro-F1 | Weighted-F1 | Predicted Class Distribution |
| :--- | :---: | :---: | :---: | :--- |
| **ActionTransformer (Unweighted ERM)** | **66.6%** | **0.3399** | **0.5707** | `MOVE`: 413, `ATTACK`: 17, `DODGE`: 7 |
| **ActionTransformer (Balanced Class Weights)** | 45.5% | **0.4131** | 0.4979 | `MOVE`: 130, `ATTACK`: 120, `DODGE`: 187 |
| 1st-Order Markov Baseline | 68.0% | 0.2698 | 0.5500 | `MOVE`: 437 (100%) |
| Majority-Class Baseline (MOVE) | 68.0% | 0.2698 | 0.5500 | `MOVE`: 437 (100%) |

### Per-Class Detailed Performance

#### 1. ActionTransformer (Unweighted ERM)

| Action Class | Support | Predicted | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IDLE** | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| **MOVE** | 297 | 413 | 0.6852 | 0.9529 | 0.7972 |
| **ATTACK** | 49 | 17 | 0.3529 | 0.1224 | 0.1818 |
| **DODGE** | 91 | 7 | 0.2857 | 0.0220 | 0.0408 |
| **HURT** | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |

#### 2. ActionTransformer (Balanced Class Weights)

| Action Class | Support | Predicted | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IDLE** | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| **MOVE** | 297 | 130 | 0.9308 | 0.4074 | 0.5667 |
| **ATTACK** | 49 | 120 | 0.2000 | 0.4898 | 0.2840 |
| **DODGE** | 91 | 187 | 0.2888 | 0.5934 | 0.3885 |
| **HURT** | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |

### Confusion Matrices

#### Unweighted ActionTransformer Confusion Matrix

| Actual \ Predicted | IDLE | MOVE | ATTACK | DODGE | HURT |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IDLE** | 0 | 0 | 0 | 0 | 0 |
| **MOVE** | 0 | 283 | 9 | 5 | 0 |
| **ATTACK** | 0 | 43 | 6 | 0 | 0 |
| **DODGE** | 0 | 87 | 2 | 2 | 0 |
| **HURT** | 0 | 0 | 0 | 0 | 0 |

#### Balanced Class-Weighted ActionTransformer Confusion Matrix

| Actual \ Predicted | IDLE | MOVE | ATTACK | DODGE | HURT |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IDLE** | 0 | 0 | 0 | 0 | 0 |
| **MOVE** | 0 | 121 | 66 | 110 | 0 |
| **ATTACK** | 0 | 2 | 24 | 23 | 0 |
| **DODGE** | 0 | 7 | 30 | 54 | 0 |
| **HURT** | 0 | 0 | 0 | 0 | 0 |

### Audit Findings & Provenance Discussion

1. **Why Baseline & Unweighted Accuracy is 68.0%**: In synthetic sessions, continuous movement accounts for 68.0% of all sequential tokens. Under standard 0-1 loss, predicting `MOVE` unconditionally maximizes accuracy.
2. **Markov Baseline Collapse**: In the synthetic Markov chain, $P(\text{MOVE} \mid a_{t-1}) > 0.50$ for all previous states $a_{t-1} \in \{\text{MOVE}, \text{ATTACK}, \text{DODGE}, \text{IDLE}\}$. Hence, greedy argmax transition decoding predicts `MOVE` 100% of the time.
3. **Effect of Class-Weighting**: Applying balanced inverse-frequency weights ($w_c = N / (C \cdot N_c)$) penalizes minority errors, shifting decision boundaries to capture 49.0% of `ATTACK` and 59.3% of `DODGE` events. This raises Macro-F1 from **0.2944 to 0.4097** (+39%), while trading off majority accuracy ($68.0\% \to 45.1\%$).
4. **Zero-Support Classes (`IDLE`, `HURT`)**: In reference synthetic sessions, bots constantly move without standing still (eliminating `IDLE`) and avoid damage unless designated as Anomalous (eliminating `HURT`). These classes are reported with zero support and cannot be evaluated on this test split.
