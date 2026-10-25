# Room Threat GCN Evaluation & Target-Leakage Audit Report (Phase 6.1)

**Data Provenance**: `synthetic_procedural_dungeons`  
**Train Split**: 120 dungeon graphs (830 rooms)  
**Test Split**: 30 held-out dungeon graphs (208 rooms)  
**Evaluation Mode**: Complete graph-level seed separation (zero dungeon overlap)  

---

### 1. Target Class Distribution (Held-Out Test Split, N = 208)

| Threat Category | Test Support (Rooms) | Class Share (%) | Target Rule Summary |
| :--- | :---: | :---: | :--- |
| **LOW** | 68 | 32.7% | Spawn room (0 enemies) or open low-risk rooms ($S \\le 1.0$) |
| **MEDIUM** | 68 | 32.7% | Spacious combat rooms ($S = 2.0$) |
| **HIGH** | 72 | 34.6% | Exit sanctum or cramped/deep bottleneck rooms ($S \\ge 3.0$) |

---

### 2. Controlled Feature Ablation Comparison (Held-Out Test Split)

| Configuration | Features Used | Model | Overall Accuracy | Macro-F1 | Weighted-F1 | Leakage Risk / Mechanism |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Config A** (Original) | 7: `area, aspect, degree, is_spawn, is_exit, norm_cx, norm_cy` | **RoomThreatGCN** | **74.0%** | **0.7451** | **0.7460** | High: `is_spawn` and `is_exit` leak targets directly |
| | | NonGraphMLP | 63.0% | 0.6241 | 0.6264 | Leaked flags memorized without graph |
| **Config B** (Controlled) | 5: `area, aspect, degree, norm_cx, norm_cy` | **RoomThreatGCN** | **48.1%** | **0.4760** | **0.4761** | None: Flags excluded; spatial coordinates retained |
| | | NonGraphMLP | 51.9% | 0.4855 | 0.4878 | Spatial distance heuristic |
| **Config C** (Pure Geom) | 3: `area, aspect, degree` | **RoomThreatGCN** | **41.3%** | **0.4090** | **0.4080** | None: Geometry and corridor degree only |
| | | NonGraphMLP | 48.1% | 0.4555 | 0.4572 | Isolated room geometry |
| **Baseline** | None (Static Guess) | Majority (`HIGH`) | 34.6% | 0.1714 | 0.1780 | Zero learning baseline |

---

### 3. Target Leakage Investigation & Root Cause Analysis

1. **Deterministic Target Leakage in Config A**:
   - In `ml/gcn_dataset.py`, the target labeling rule explicitly checks:
     `if room.room_type == "spawn": return "LOW"`
     `if room.room_type == "exit" or is_exit: return "HIGH"`
   - In Config A, the input feature vector directly contained `is_spawn` and `is_exit`.
   - Because exactly 1 room per dungeon is spawn and 1 room is exit, **2 out of 7 rooms (28.6% of all nodes)** had their exact ground-truth class given directly in the input vector.
   - Furthermore, because GCN aggregates messages across corridors ($\hat{A} H$), the GCN propagated the `is_exit` flag to adjacent rooms, artificially enabling the GCN to detect the `exit_adjacent` bonus ($+1.0$) in the labeling rule!
2. **Controlled Performance in Config B (Without Flags)**:
   - When `is_spawn` and `is_exit` are removed, the artificial shortcut disappears.
   - GCN accuracy adjusts to **48.1%** and MLP accuracy to **51.9%** (both substantially beating the Majority baseline of 34.6%).
   - This demonstrates that without leaked anchor flags, neither model receives free identity cues for spawn and exit rooms.
3. **Pure Geometric Analysis in Config C (No Coordinates)**:
   - When coordinates (`norm_center_x`, `norm_center_y`) are also removed, the models must classify room threat strictly from local room area, aspect ratio, and degree.
   - GCN accuracy is **41.3%** and MLP accuracy is **48.1%**.

---

### 4. Per-Class Metrics Comparison

#### Config A: Original 7-Feature Model
| Class | Support | GCN Pred | GCN Prec | GCN Rec | GCN F1 | MLP Pred | MLP Prec | MLP Rec | MLP F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOW** | 68 | 61 | 0.8361 | 0.7500 | 0.7907 | 79 | 0.6709 | 0.7794 | 0.7211 |
| **MEDIUM** | 68 | 85 | 0.5882 | 0.7353 | 0.6536 | 63 | 0.4286 | 0.3971 | 0.4122 |
| **HIGH** | 72 | 62 | 0.8548 | 0.7361 | 0.7910 | 66 | 0.7727 | 0.7083 | 0.7391 |

#### Config B: Leakage-Controlled 5-Feature Model (No Spawn/Exit Flags)
| Class | Support | GCN Pred | GCN Prec | GCN Rec | GCN F1 | MLP Pred | MLP Prec | MLP Rec | MLP F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOW** | 68 | 90 | 0.4778 | 0.6324 | 0.5443 | 82 | 0.5854 | 0.7059 | 0.6400 |
| **MEDIUM** | 68 | 61 | 0.4262 | 0.3824 | 0.4031 | 36 | 0.3056 | 0.1618 | 0.2115 |
| **HIGH** | 72 | 57 | 0.5439 | 0.4306 | 0.4806 | 90 | 0.5444 | 0.6806 | 0.6049 |

#### Config C: Pure Structural 3-Feature Model (No Flags, No Coordinates)
| Class | Support | GCN Pred | GCN Prec | GCN Rec | GCN F1 | MLP Pred | MLP Prec | MLP Rec | MLP F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOW** | 68 | 84 | 0.4524 | 0.5588 | 0.5000 | 86 | 0.5581 | 0.7059 | 0.6234 |
| **MEDIUM** | 68 | 67 | 0.3731 | 0.3676 | 0.3704 | 44 | 0.2500 | 0.1618 | 0.1964 |
| **HIGH** | 72 | 57 | 0.4035 | 0.3194 | 0.3566 | 78 | 0.5256 | 0.5694 | 0.5467 |

---

### 5. Confusion Matrices

#### Config A: Original Schema Confusion Matrices
**GCN (7 Features)**:
```
Actual \ Predicted | LOW | MEDIUM | HIGH
LOW (68)           |  51 |     16 |    1
MEDIUM (68)        |  10 |     50 |    8
HIGH (72)          |   0 |     19 |   53
```

**NonGraphMLP (7 Features)**:
```
Actual \ Predicted | LOW | MEDIUM | HIGH
LOW (68)           |  53 |     15 |    0
MEDIUM (68)        |  26 |     27 |   15
HIGH (72)          |   0 |     21 |   51
```

#### Config B: Leakage-Controlled Schema Confusion Matrices (No Spawn/Exit Flags)
**GCN (5 Features)**:
```
Actual \ Predicted | LOW | MEDIUM | HIGH
LOW (68)           |  43 |     17 |    8
MEDIUM (68)        |  24 |     26 |   18
HIGH (72)          |  23 |     18 |   31
```

**NonGraphMLP (5 Features)**:
```
Actual \ Predicted | LOW | MEDIUM | HIGH
LOW (68)           |  48 |      9 |   11
MEDIUM (68)        |  27 |     11 |   30
HIGH (72)          |   7 |     16 |   49
```

#### Config C: Pure Geometric Schema Confusion Matrices
**GCN (3 Features)**:
```
Actual \ Predicted | LOW | MEDIUM | HIGH
LOW (68)           |  38 |     15 |   15
MEDIUM (68)        |  24 |     25 |   19
HIGH (72)          |  22 |     27 |   23
```

**NonGraphMLP (3 Features)**:
```
Actual \ Predicted | LOW | MEDIUM | HIGH
LOW (68)           |  48 |     11 |    9
MEDIUM (68)        |  29 |     11 |   28
HIGH (72)          |   9 |     22 |   41
```
