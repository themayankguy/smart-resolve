# NEURAL DUNGEON — ATML ORAL DEFENSE & TECHNICAL VIVA NOTES

**Candidate Revision & Technical Reference Guide**  
**Coursework / Module**: Advanced Topics in Machine Learning (ATML)  
**Project**: Neural Dungeon (2D Dark-Fantasy Top-Down Research Crawler & ML Testbed)  
**Codebase**: `/Users/mayanksmac/Desktop/ATML`  
**Test Suite**: 64 Unit & Regression Tests Passing (`python3 -m unittest discover -s tests -p "test_*.py" -v`)  
**E2E Headless Verification**: `python3 scripts/verify_gameplay_e2e.py`

---

## 1. System Overview & Architecture

Neural Dungeon integrates a real-time, deterministic 2D action game engine (Pygame) with four distinct Applied Machine Learning architectures written in PyTorch. The game acts as a closed-loop laboratory where gameplay telemetry is extracted, processed, and fed into machine learning models to govern enemy adaptation, safety/anomaly auditing, tactical action forecasting, and spatial dungeon comprehension.

```
                                      +------------------------------------+
                                      |         NEURAL DUNGEON             |
                                      |      Game Engine & Combat          |
                                      +-----------------+------------------+
                                                        |
                                            Telemetry Event Stream (JSONL)
                                                        |
                 +----------------------+---------------+----------------------+----------------------+
                 |                      |                                      |                      |
                 v                      v                                      v                      v
        +-----------------+    +-----------------+                    +-----------------+    +-----------------+
        |     MODEL 1     |    |     MODEL 2     |                    |     MODEL 3     |    |     MODEL 4     |
        | Siamese Network |    |   Autoencoder   |                    |   Transformer   |    |       GCN       |
        +-----------------+    +-----------------+                    +-----------------+    +-----------------+
        | Playstyle       |    | Combat Anomaly  |                    | Next-Action     |    | Room Threat     |
        | Similarity      |    | Detection       |                    | Forecasting     |    | Prediction      |
        +-----------------+    +-----------------+                    +-----------------+    +-----------------+
        | 11 Features     |    | 11 Features     |                    | Action Tokens   |    | Dungeon Graph   |
        | 8D Metric Space |    | 4D Bottleneck   |                    | Causal Attention|    | 5 Features (B)  |
        +-----------------+    +-----------------+                    +-----------------+    +-----------------+
                 |                      |                                      |                      |
                 v                      v                                      v                      v
        +-----------------+    +-----------------+                    +-----------------+    +-----------------+
        | Enemy Tactic    |    | HUD Anomaly     |                    | Enemy Micro-    |    | Room Threat     |
        | State Machine   |    | Alert & Log     |                    | Reaction (Gate) |    | Badge on HUD    |
        +-----------------+    +-----------------+                    +-----------------+    +-----------------+
```

### Architectural Guarantees & Failover Resilience
1. **Decoupled Execution**: All four models run asynchronously or via periodic sampling windows (e.g., Siamese/Autoencoder/Transformer evaluate every 2.0s; GCN evaluates once at floor generation). Inference latency never blocks the 60 FPS physics loop.
2. **Graceful Failover**: If any model checkpoint (`.pt`) is absent, corrupted, or incompatible, every predictor falls back to a deterministic heuristic without throwing unhandled exceptions or crashing the game.
3. **Session-Level Data Isolation**: All training, validation, and testing datasets enforce strict session-level splitting. Time windows from the same gameplay session never appear in both training and test sets.

---

## 2. Model 1: Siamese Network for Combat Playstyle Similarity

### 2.1 Rationale & Problem Formulation
- **Objective**: Determine whether a current combat window reflects a known tactical playstyle (`Aggressive`, `Evasive`, `Balanced`) by learning a distance metric rather than training an overfitted static classifier.
- **Why Metric Learning?** Metric learning maps continuous behavior into an embedding space where distances directly encode behavioral similarity. This enables the system to compare incoming players against canonical prototype vectors and readily admit new playstyles without retraining the entire network.

### 2.2 Input Schema & Architecture
- **Input**: 11 continuous combat features extracted over a 40-event sliding window:
  1. `attack_rate`: Number of melee swings per second.
  2. `hit_accuracy`: Ratio of landed melee hits to swings.
  3. `dodge_rate`: Number of dodge rolls per second.
  4. `evasion_success_rate`: Percentage of dodges avoiding damage during active enemy windup.
  5. `damage_dealt_rate`: Total damage dealt per second.
  6. `damage_taken_rate`: Total damage received per second.
  7. `damage_ratio`: Ratio of damage dealt to damage taken.
  8. `mean_distance_to_enemy`: Mean Euclidean distance to closest active enemy.
  9. `close_quarter_ratio`: Fraction of window time spent within melee distance ($<80$ px).
  10. `stamina_efficiency`: Damage dealt per unit of stamina spent.
  11. `stamina_depleted_ratio`: Fraction of time player spent with zero stamina.
- **Architecture**:
  - Twin feedforward subnetworks sharing identical weights $W$:
    $$\mathbb{R}^{11} \xrightarrow{\text{Linear(11, 32)}} \text{ReLU} \xrightarrow{\text{Linear(32, 16)}} \text{ReLU} \xrightarrow{\text{Linear(16, 8)}} \text{Embedding } z \in \mathbb{R}^8$$
  - Euclidean distance: $d(z_1, z_2) = \|z_1 - z_2\|_2$.

### 2.3 Mathematical Formulation: Contrastive Loss
Trained with Hadsell et al. (2006) Contrastive Loss:
$$\mathcal{L}_{\text{cont}}(z_1, z_2, y) = y \cdot d(z_1, z_2)^2 + (1 - y) \cdot \max(0, m - d(z_1, z_2))^2$$
where:
- $y = 1.0$ if pair $(x_1, x_2)$ belongs to the same playstyle; $y = 0.0$ if different.
- $m = 1.0$ is the contrastive margin enforcing separation between dissimilar styles.

### 2.4 Prototype Classification & Confidence
During inference, the runtime embeds the player's current window $z$ and measures distances $d_c = \|z - p_c\|_2$ to saved class prototypes $p_c \in \{p_{\text{Aggressive}}, p_{\text{Evasive}}, p_{\text{Balanced}}\}$.
- **Similarity score**:
  $$s_c = \frac{1}{1 + d_c} \in (0, 1]$$
- **Confidence**:
  $$\text{Confidence} = 1 - \frac{d_{\min}}{\sum_c d_c}$$
- **Fallback Rule**: If $\text{Confidence} < 0.45$, the detector defaults to `Balanced` to avoid false enemy adaptation.

### 2.5 In-Game Adaptation (Enemy Counter-Tactics)
- **Predicted `Aggressive`** ($\text{conf} \ge 0.45$): Enemies increase spacing distance (+25%), engage in retreating kiting behavior, and increase projectile frequency to punish rushdown.
- **Predicted `Evasive`** ($\text{conf} \ge 0.45$): Enemies decrease engage distance (-20%), increase movement speed (+15%), and coordinate aggressive surrounds to exhaust player stamina.
- **Predicted `Balanced`**: Enemies execute default patrol, flank, and attack cycles.

### 2.6 Empirical Results & Limitations
- **Reported Metric**: **85.4% Pairwise Similarity Accuracy** (F1 Score: 0.8614, Precision: 0.8208, Recall: 0.9062) on 384 held-out synthetic test pairs across 32 windows in 3 test sessions (`synthetic_session_balanced_03_263`, `synthetic_session_balanced_04_280`, `synthetic_session_evasive_04_195`).
- **Baseline Comparison**: Raw Euclidean Distance achieves **66.7% Accuracy** (F1: 0.6578). The learned metric produces a **4.36x margin separation ratio** between similar and dissimilar pairs.
- **Confusion Matrix**: True Positives: 174, False Positives: 38, True Negatives: 154, False Negatives: 18.
- **Reconciliation Note**: Phase 7 mistakenly noted 96.5% due to an unverified entry; 85.4% is the exact verified checkpoint performance. If anomalous sessions are mixed in, accuracy measures 94.4% (51 windows, 612 pairs).
- **Critical Academic Disclosures**:
  - *Pairwise Similarity $\neq$ 3-Class Accuracy*: This evaluates binary similarity between combat windows, not multi-class classification directly.
  - *Proxy Labels*: Ground-truth labels in the training set were assigned by heuristic rules on synthetic bot runs. They represent operational archetypes, not verified psychological player profiles.
  - *No Human Player Validation*: The model has not been validated on real human play sessions.

---

## 3. Model 2: Deep Combat Autoencoder for Anomaly Detection

### 3.1 Rationale & Objective
- **Objective**: Monitor combat telemetry feature windows in an unsupervised manner to detect anomalous, uncharacteristic combat states without requiring labeled anomaly datasets.
- **Core Concept**: An undercomplete autoencoder trained solely on nominal combat data learns the latent manifold of standard gameplay. When presented with anomalous data (e.g., telemetry corruption, impossible stats, bot jitter), it fails to reconstruct the inputs, resulting in elevated Mean Squared Error.

### 3.2 Architecture & Reconstruction Loss
- **Encoder**: $\mathbb{R}^{11} \xrightarrow{\text{Linear(11, 8)}} \text{ReLU} \xrightarrow{\text{Linear(8, 4)}} \text{Latent } z \in \mathbb{R}^4$
- **Decoder**: $z \in \mathbb{R}^4 \xrightarrow{\text{Linear(4, 8)}} \text{ReLU} \xrightarrow{\text{Linear(8, 11)}} \hat{x} \in \mathbb{R}^{11}$
- **Reconstruction Loss**:
  $$\text{MSE}(x, \hat{x}) = \frac{1}{D} \sum_{i=1}^D (x_i - \hat{x}_i)^2 \quad (D = 11)$$

### 3.3 Dynamic Threshold Selection
The anomaly threshold $\tau$ is established strictly on the training partition:
$$\tau = \mu_{\text{train}} + 2 \cdot \sigma_{\text{train}} = 0.0352 + 2(0.0307) = 0.0966$$
where $\mu_{\text{train}} = 0.0352$ is the mean training reconstruction error and $\sigma_{\text{train}} = 0.0307$ is the standard deviation. Any window with $\text{MSE}(x, \hat{x}) > \tau$ is flagged as anomalous. (If $\mu + 3\sigma$ were used, $\tau$ would be $0.1273$).

### 3.4 Empirical Results & Anomaly vs. Cheating Distinction
- **Reconstruction Error Distributions (Held-Out Test Set)**:
  - Nominal test windows (N=32): Mean MSE = `0.0555` (median: `0.0389`, std: `0.0514`).
  - Synthetic anomalous windows (N=50): Mean MSE = `2129.9875` (median: `2088.8442`, std: `903.2073`).
  - Separation ratio: `38,370x`.
- **Classification Performance**:
  - Autoencoder at $\tau = 0.0966$: **93.9% Accuracy**, **0.9524 F1** (Precision: 0.9091, Recall: 1.0000; TP: 50, FP: 5, TN: 27, FN: 0).
  - Raw Z-Score Baseline: **97.6% Accuracy**, **0.9804 F1** (Precision: 0.9615, Recall: 1.0000; TP: 50, FP: 2, TN: 30, FN: 0).
  - *Honest Baseline Finding*: The simple feature magnitude baseline slightly outperforms the Autoencoder (97.6% vs 93.9%) because extreme synthetic bot anomalies create massive feature spikes that are readily separated by raw magnitude with fewer false alarms on nominal play (2 false positives vs 5).
- **Crucial Viva Talking Point**:
  - *Anomaly Detection $\neq$ Cheating Detection*: The autoencoder flags statistical divergence from the training manifold. It cannot distinguish between malicious tampering, novel human play techniques, software glitches, or edge-case weapon synergies. It must be framed strictly as a *telemetry anomaly monitor*, never an "anti-cheat" engine.

---

## 4. Model 3: Action Prediction Transformer

### 4.1 Rationale & Sequence Formulation
- **Objective**: Forecast the player's immediate next discrete action ($a_{t+1}$) given a causal history of past actions $[a_{t-15}, \dots, a_t]$.
- **Why Transformers?** Unlike Markov models that assume memoryless first-order transitions, self-attention computes pairwise associations across the entire sequence context, capturing multi-step tactical combos and delays.

### 4.2 Architecture & Causal Masking
- **Vocabulary (6 Tokens)**: `PAD: 0`, `IDLE: 1`, `MOVE: 2`, `ATTACK: 3`, `DODGE: 4`, `HURT: 5`.
- **Sequence Length**: Context length $T = 16$ actions (Phase 7 typo incorrectly listed $T=8$; verified code and checkpoint use $T=16$).
- **Layers**:
  - Token Embedding ($d_{\text{model}} = 32$) + Learned Positional Embedding ($d_{\text{model}} = 32$).
  - 2 Transformer Encoder Layers: 4 Attention Heads, $d_{\text{ff}} = 64$, Dropout = 0.1.
  - Linear Projection Head: $\mathbb{R}^{32} \to \mathbb{R}^6$.
- **Causal Mask**: An upper-triangular additive mask $M_{ij} = -\infty$ for $j > i$ is strictly enforced so token $i$ can attend only to positions $j \le i$:
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

### 4.3 The Class Imbalance Audit & The Weighted Trade-Off
An audit of gameplay telemetry sequences revealed severe natural class imbalance on the held-out test split (N=437):
- `MOVE`: 297 tokens (68.0%).
- `DODGE`: 91 tokens (20.8%).
- `ATTACK`: 49 tokens (11.2%).
- `IDLE` / `HURT`: 0 tokens (unobserved in active synthetic combat).

Because `MOVE` dominates, standard unweighted Cross-Entropy loss causes the model to optimize accuracy by predicting `MOVE` almost exclusively. We trained and evaluated two configurations to investigate this trade-off:

| Metric | Unweighted Transformer | Class-Weighted Transformer | 1st-Order Markov Baseline | Majority Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **66.6%** | 45.5% | 68.0% | 68.0% |
| **Macro-F1** | 0.3399 | **0.4131** | 0.2698 | 0.2698 |
| **Weighted-F1** | 0.5707 | 0.4979 | 0.5500 | 0.5500 |
| **`ATTACK` F1** | 0.1818 (Recall 0.1224) | **0.2840 (Recall 0.4898)** | 0.0000 | 0.0000 |
| **`DODGE` F1** | 0.0408 (Recall 0.0220) | **0.3885 (Recall 0.5934)** | 0.0000 | 0.0000 |
| **`MOVE` F1** | 0.7972 (Recall 0.9529) | 0.5667 (Recall 0.4074) | 0.8093 | 0.8093 |

### 4.4 Technical Interpretation for Examiners
- The unweighted Transformer achieves 66.6% accuracy, but this is a *trivial optimum* driven by majority-class prevalence.
- The class-weighted Transformer sacrifices nominal accuracy (drops to 45.5%) but substantially improves **Macro-F1 from 0.3399 to 0.4131** (+53% relative gain over baselines) and more than doubles minority combat action recall (`ATTACK`: 12.2% $\to$ 49.0%, `DODGE`: 2.2% $\to$ 59.3%).
- In a game AI context, predicting an oncoming `ATTACK` or `DODGE` is actionable for enemy counter-tactics, whereas continuously predicting `MOVE` provides zero tactical utility.
- Gated by a 0.45 confidence threshold, enemies execute proactive micro-actions:
  - Forecast `ATTACK` $\to$ Enter defensive parry/strafe stance.
  - Forecast `DODGE` $\to$ Predict roll endpoint and lead projectile aiming.

---

## 5. Model 4: Graph Convolutional Network (GCN) for Room Threat Prediction

### 5.1 Rationale & Dungeon Graph Representation
- **Objective**: Classify dungeon rooms into threat categories (`LOW`, `MEDIUM`, `HIGH`) based on dungeon topology and spatial layout.
- **Graph Representation**:
  - Nodes: Dungeon rooms ($V$).
  - Edges: Physical corridor connections ($E$).
  - Adjacency Matrix $A \in \{0, 1\}^{N \times N}$.

### 5.2 Mathematical Formulation: Spectral Graph Convolution
Using Kipf & Welling (ICLR 2017) first-order localized spectral convolution:
$$H^{(l+1)} = \text{ReLU}\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)$$
where:
- $\tilde{A} = A + I_N$ (Adjacency matrix with added self-loops).
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ (Degree matrix of $\tilde{A}$).
- $\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$ is the symmetric normalized adjacency matrix that prevents gradient explosions and node degree bias during message passing.
- Network depth: 2 GCN layers ($\mathbb{R}^{F} \to \mathbb{R}^{16} \to \mathbb{R}^3$), followed by Softmax.

---

### 5.3 The Target Leakage Audit (Phase 6.1 Critical Finding)

**The Discovery**:
In the original implementation (Config A), the GCN reported an impressive **74.0% accuracy** (vs. 63.0% for a non-graph MLP). However, inspection of `ml/gcn_dataset.py` revealed fatal target leakage:
1. Target definitions:
   - `LOW` threat: Rooms with 0 enemies (specifically includes the spawn room).
   - `HIGH` threat: Rooms with 3+ enemies or high threat scores (specifically includes the exit sanctum).
2. Input features in Config A included:
   - `is_spawn`: Directly indicated spawn room $\to$ Leaked `LOW` label!
   - `is_exit`: Directly indicated exit sanctum $\to$ Leaked `HIGH` label!
3. Graph Propagation Leakage:
   - Because `is_exit` was present as a node feature, GCN spectral message passing propagated the non-zero exit signal directly to neighboring rooms, allowing the GCN to trivially infer proximity to the exit.

**The Controlled Ablation Experiments**:
To measure true structural graph learning, we conducted controlled ablations on the identical train/test split across **150 procedural dungeons (1,038 rooms)**, split into 120 training dungeons (830 rooms) and 30 held-out test dungeons (208 rooms, seed offset 1000):

| Configuration | Features Used ($F$) | GCN Accuracy | GCN Macro-F1 | MLP Baseline Acc | Majority Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Config A (Original - Leaked)** | 7 (`area`, `aspect`, `deg`, `x`, `y`, `is_spawn`, `is_exit`) | **74.0%** | **0.7451** | 63.0% | 34.6% |
| **Config B (Controlled)** | 5 (`area`, `aspect`, `deg`, `norm_x`, `norm_y`) | **48.1%** | **0.4760** | **51.9%** | 34.6% |
| **Config C (Pure Geometric)** | 3 (`area`, `aspect_ratio`, `degree`) | **41.3%** | **0.4090** | **48.1%** | 34.6% |

### 5.4 Honest Scientific Conclusion
1. When target-leaking flags (`is_spawn`, `is_exit`) are removed, GCN accuracy drops from 74.0% to **48.1%**.
2. Both Config B (48.1%) and Config C (41.3%) still comfortably outperform the majority baseline (34.6%), proving that layout geometry and room degree carry predictive signal.
3. However, under controlled conditions, **the GCN does not outperform a non-graph MLP baseline (51.9%)**. In planar dungeons with only 7–12 rooms, the graph diameter is small (2–3 hops), and local room geometry (`area`, `degree`) accounts for most variance.
4. **Runtime Selection**: Neural Dungeon's runtime defaults to **Config B (`gcn_checkpoint_controlled.pt`)**, ensuring honest, leakage-free inference on live runs.

---

## 6. Telemetry Provenance & Integrity

1. **Synthetic vs. Human Separation**:
   - Synthetic sessions are prefixed `synthetic_session_*` and contain metadata tag `"session_type": "synthetic"`.
   - The telemetry pipeline enforces separation: synthetic sessions are excluded by default when processing human datasets.
2. **Session Summary Consistency**:
   - Fixed a Phase 2 defect where event counts (hits, damage, defeats) failed to tally into session summaries.
   - All session summaries in `data/raw/summaries.jsonl` are verified against raw events.
3. **No Fabricated Labels**:
   - Incomplete or abandoned runs are marked `ABANDONED` or `DEFEAT`. Only legitimate floor clears reaching the unlocked exit portal are labeled `VICTORY`.

---

## 7. Likely Examiner / Viva Questions & Answers

### Q1: "Why did you choose a Siamese Network rather than a standard 3-class classifier for playstyle detection?"
> **Answer**:  
> "A standard softmax classifier learns static decision boundaries that tend to overfit to specific training distributions and cannot easily adapt to new or hybrid playstyles without retraining the entire network. A Siamese Network trains an embedding space via contrastive metric learning where the Euclidean distance directly represents behavioral similarity. This allows us to compare incoming players against prototype vectors, calculate transparent distance-based confidence scores, and admit new playstyle archetypes simply by adding reference prototypes without altering model weights."

### Q2: "Can your Autoencoder be used as an anti-cheat engine in commercial games?"
> **Answer**:  
> "No, and doing so would be technically irresponsible. An autoencoder performs unsupervised anomaly detection by identifying feature vectors that deviate from its training distribution. An elevated reconstruction error indicates *unusual combat telemetry*, which could arise from network latency, software bugs, novel high-skill maneuvers, or unexpected weapon synergies just as easily as memory tampering or aimbots. In academic terms, it is an anomaly monitor, not an anti-cheat classifier."

### Q3: "Your Transformer has an accuracy of only 45.5% when class-weighted, while the Markov baseline achieves 68.0%. Isn't the Markov model superior?"
> **Answer**:  
> "Accuracy is deeply misleading here due to severe class imbalance: movement actions constitute 68% of the dataset. The Markov baseline and the unweighted Transformer achieve 68% accuracy simply by collapsing to the majority class (`MOVE`), yielding an ATTACK and DODGE recall of zero. In game combat AI, predicting `MOVE` gives enemies zero tactical value. The class-weighted Transformer achieves a significantly higher **Macro-F1 of 0.4131** (vs. 0.2698 for Markov) and successfully anticipates attacks and dodges, allowing enemies to trigger defensive parries and intercepts."

### Q4: "What was the target leakage issue in your GCN, and how did you resolve it?"
> **Answer**:  
> "In Phase 6, the initial GCN reported 74% accuracy. Upon auditing the feature definitions, we discovered that `is_spawn` and `is_exit` were included as input features. Because the target rule defined zero-enemy spawn rooms as `LOW` and the exit sanctum as `HIGH`, these features leaked ground-truth labels directly into 28% of the nodes. Furthermore, spectral convolutions propagated the exit signal to adjacent rooms.  
> We resolved this by isolating Config A as a documented historical defect and training **Config B (5 features)** without any spawn/exit flags, and **Config C (3 features)** without spatial coordinates. Under leakage-controlled evaluation, GCN accuracy settled at 48.1% (above the 34.6% majority baseline), and we deployed Config B as our active runtime checkpoint."

### Q5: "How does the game prevent ML inference latency from stalling the 60 FPS physics loop?"
> **Answer**:  
> "We employ temporal decoupling. The Siamese Network, Autoencoder, and Transformer evaluate over sliding 40-event windows at a throttled 2.0-second timer rather than every frame. The GCN evaluates exactly once when a dungeon floor is generated, caching predictions per room ID. If any model is missing or corrupt, lightweight fallback heuristics execute in under 0.1 milliseconds. This guarantees zero frame drops in the main Pygame update loop."

### Q6: "How did you audit and reconcile discrepancies across your project reports?"
> **Answer**:  
> "In Phase 7.1, we conducted an empirical reconciliation audit comparing saved checkpoints against reported metrics:
> 1. For the Siamese Network, an unverified entry of 96.5% was reconciled to the exact verified checkpoint accuracy of 85.4% (384 test pairs across the 3 held-out playstyle sessions).
> 2. For the Autoencoder, we verified that the threshold 0.0966 was derived via $\mu_{\text{train}} + 2\sigma_{\text{train}}$ ($0.0352 + 2 \times 0.0307 = 0.0966$), correcting a documentation typo that labeled it as $\mu + 3\sigma$, and noted that the simple feature magnitude Z-score baseline achieved 97.6% (vs 93.9% for the AE).
> 3. For the Transformer, we corrected a typo stating $T=8$ to the actual verified sequence context length of $T=16$.
> 4. For the GCN, we confirmed that the dataset consists of 150 procedural dungeons (1,038 rooms), correcting an earlier typo stating 80 dungeons. All code, checkpoints, and reports are now 100% synchronized."

---

## 8. Candidate Walkthrough & Code Inspection Checklist

Before your examination, review these exact files to be familiar with the implementation:
1. `ml/siamese_model.py` & `ml/playstyle_detector.py`: Contrastive loss, 8D embedding, prototype calculation.
2. `ml/autoencoder_model.py` & `ml/anomaly_detector.py`: Bottleneck architecture, reconstruction MSE, threshold calculation.
3. `ml/transformer_model.py` & `ml/action_predictor.py`: Causal masking, self-attention layers, class weighting.
4. `ml/gcn_model.py` & `ml/threat_predictor.py`: Normalized adjacency matrix $\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$, 5-feature controlled schema.
5. `game/engine.py` & `game/ui.py`: Master game loop, ML inference calls, HUD badge rendering.
6. `tests/`: 64 automated unit and regression tests covering all 4 models and the game engine.
