# SmartResolve: 100% Feature-Complete Multi-Profile Video Demonstration Plan
## End-to-End Walkthrough Covering Every Platform Feature Across All 3 Roles

**Document:** `coursera/video_demonstration_plan.md`  
**Target Duration:** ~3 minutes  
**Resolution:** 1920 × 1080 (Full HD, 16:9)  
**Live Application URL:** `http://127.0.0.1:5173/`

---

## 1. Complete Feature Matrix & Role Access Mapping

Every single feature built into SmartResolve is mapped to its primary persona and demonstration scene:

| # | Feature Name | Core Functionality | Demonstrated Profile | Scene |
|---|---|---|---|---|
| **F-01** | **Role Switcher (RBAC)** | Dynamic UI role toggle between User, Resolver, and Admin | All Profiles | Header |
| **F-02** | **Natural Language Intake** | Freeform text complaint intake with auto-expansion | 👤 Employee / User | Act I |
| **F-03** | **Pre-Submission AI Preview** | `Test AI Preview` button testing `/classify` without saving | 👤 Employee / User | Act I |
| **F-04** | **Impact & Urgency Selectors** | Organization / Dept / Individual scope + P1–P4 urgency | 👤 Employee / User | Act I |
| **F-05** | **AI Triage Modal** | Real-time SBERT category, confidence meter, SLA target | 👤 Employee / User | Act I |
| **F-06** | **Deep Anomaly Detector** | PyTorch Autoencoder reconstruction loss vs. threshold $\tau$ | 👤 Employee / User | Act I |
| **F-07** | **My Tickets History** | User-scoped ticket log with live status & SLA countdown | 👤 Employee / User | Act I |
| **F-08** | **Executive KPI Dashboard** | Total, Open, In Progress, Resolved, Critical, Anomalies | 🛡️ IT Resolver | Act II |
| **F-09** | **Domain Distribution Charts** | Recharts Category Volume Bar Chart & Lifecycle Donut | 🛡️ IT Resolver | Act II |
| **F-10** | **Recent Anomaly Alert Feed** | Real-time security and out-of-distribution alert box | 🛡️ IT Resolver | Act II |
| **F-11** | **Full-Text Ticket Search** | Live keyword search across complaint text and IDs | 🛡️ IT Resolver | Act II |
| **F-12** | **Multi-Facet Queue Filters** | Filter by Category, Priority, Status, and Anomaly status | 🛡️ IT Resolver | Act II |
| **F-13** | **Ticket Status Transition** | Lifecycle workflow toggle (`Open` $\to$ `In Progress` $\to$ `Resolved`) | 🛡️ IT Resolver | Act II |
| **F-14** | **Priority Override Engine** | Manual technician priority override with audit log | 🛡️ IT Resolver | Act II |
| **F-15** | **Technician Resolution Notes** | Diagnostic updates and closure verification notes | 🛡️ IT Resolver | Act II |
| **F-16** | **Module A: Classifier Health** | SBERT + Logistic Regression accuracy and Macro F1 | ⚙️ ML Admin | Act III |
| **F-17** | **Module B: Autoencoder ROC** | Deep Autoencoder ROC-AUC, threshold $\tau = 0.0018$ | ⚙️ ML Admin | Act III |
| **F-18** | **Module C: Few-Shot Engine** | Prototypical few-shot accuracy ($k=1, 3, 5$) | ⚙️ ML Admin | Act III |
| **F-19** | **Confusion & Class Metrics** | Precision, recall, F1, and test support per category | ⚙️ ML Admin | Act III |
| **F-20** | **Live Retrain Trigger** | `Retrain All Models` button on fresh splits | ⚙️ ML Admin | Act III |
| **F-21** | **Dataset CSV Importer** | Batch external dataset file upload and validation | ⚙️ ML Admin | Act III |
| **F-22** | **Dynamic Category Creator** | Register ad-hoc IT categories (Name, Dept, Scope) | ⚙️ ML Admin | Act III |
| **F-23** | **Prototype Support Exemplar** | Add labeled examples to update cosine centroids live | ⚙️ ML Admin | Act III |
| **F-24** | **Few-Shot Testing Sandbox** | Real-time cosine similarity distance evaluation | ⚙️ ML Admin | Act III |
| **F-25** | **Sub-10ms Hardware Latency** | Median and $p_{95}$ inference latency diagnostics | ⚙️ ML Admin | Act III |

---

## 2. Minute-by-Minute Script & Action Storyboard

### **ACT I: 👤 Profile 1 — Employee / User (Self-Service Portal)** `[0:00 – 1:00]`
* **Role Switcher:** Click top-right role dropdown $\to$ Select **`👤 Employee / User`**.
* **Portal Title:** *Employee Self-Service Portal* (`Report Issue`, `My Tickets`).

#### Step 1: Pre-Submission AI Preview (Feature F-03)
* In the textarea, type:
  > *"My Outlook desktop client cannot connect to Exchange after changing my password, error 0x80040115."*
* Click **Test AI Preview** (Feature F-03):
  - Notice the instant non-blocking preview badge showing:
    - *Predicted: Account and Authentication (Confidence: 89%)*
    - *Anomaly Check: PASS (Normal technical issue)*

#### Step 2: Impact Scope & Submission (Features F-02, F-04, F-05, F-06)
* Select **Reported Impact Scope:** `Individual User Workstation`.
* Select **User Urgency:** `High - Escalated Turnaround`.
* Click **Submit & Generate Ticket**.
* **Deep Explainability Modal Pops Up (4s pause):**
  - **Ticket ID:** `SR-XXXXXX`
  - **Category:** `Account and Authentication`
  - **Reconstruction Error:** `0.0028` vs. threshold `0.0047` (`NORMAL PATTERN`)
  - **Assigned Dept:** `Identity & Access Management (IAM)`
  - **SLA Resolution Target:** `4 Hours`
* Click **Close Window**.

#### Step 3: Out-of-Distribution Anomaly Demonstration (Feature F-06)
* Type a non-technical / OOD query:
  > *"Where can I buy a fresh sandwich and claim meal reimbursement vouchers?"*
* Click **Test AI Preview**:
  - The badge highlights in red: **`⚠️ ANOMALOUS PATTERN DETECTED`** (Reconstruction loss exceeds threshold $\tau$).
* Voiceover explains how the PyTorch Autoencoder gates non-technical noise before wasting IT bandwidth.

#### Step 4: Tracking in "My Tickets" (Feature F-07)
* Click **My Tickets** tab.
* Show the employee's personal ticket queue with real-time status `Open` and priority badges.

---

### **ACT II: 🛡️ Profile 2 — IT Resolver / Support (Incident Console)** `[1:00 – 2:00]`
* **Role Switcher:** Click top-right role dropdown $\to$ Select **`🛡️ IT Resolver / Support`**.
* **Console Display:** *IT Incident Resolver Console* (`Dashboard`, `Ticket Queue`, `Log Ticket`).

#### Step 1: Operations Dashboard Overview (Features F-08, F-09, F-10)
* Showcase the KPI metrics: Total Tickets, Open Tickets, In Progress, Resolved, High & Critical, Anomalies Detected, Training Samples.
* Hover over the **Category Volume Bar Chart** and **Ticket Lifecycle Donut Chart**.
* Highlight the **Recent Anomaly Alert Feed** showing automated security/OOD detections.

#### Step 2: Interactive Filtering & Search (Features F-11, F-12)
* Switch to the **Ticket Queue** tab.
* In the search input, type `"Outlook"`. The table filters in real-time.
* Toggle the quick status filters (`Open`, `In Progress`, `High Priority`).

#### Step 3: Ticket Inspection, Priority Override & Resolution (Features F-13, F-14, F-15)
* Click on the Outlook ticket row to open the full **Diagnostic Modal**:
  - Review the AI Prioritization Rationale.
  - Change **Status** from `Open` to **`In Progress`**.
  - Change **Override Priority** to **`High`**.
  - In **Resolution / Administrator Notes**, type:
    > *"Reset Kerberos authentication ticket cache. Triggered remote Active Directory profile synchronization."*
  - Click **Close Window**. The table reflects the update immediately.

---

### **ACT III: ⚙️ Profile 3 — ML Admin / Data Scientist (AI Core Console)** `[2:00 – 2:50]`
* **Role Switcher:** Click top-right role dropdown $\to$ Select **`⚙️ ML Admin / Data Scientist`**.
* **Console Display:** Unlocks the specialized **`Model Evaluation`** and **`Few-Shot Studio`** tabs.

#### Step 1: Model Health & Empirical Benchmarks (Features F-16, F-17, F-18, F-19, F-25)
* Click **Model Evaluation** tab:
  - **Module A Card:** Sentence-BERT + Logistic Regression (Accuracy: 69.63%, Macro F1: 69.63%).
  - **Module B Card:** PyTorch Symmetric Autoencoder (ROC-AUC: 76.21%, Threshold $\tau = 0.0018$).
  - **Module C Card:** Prototypical Few-Shot Metric Engine ($k=5$: 44.81% vs. 12.5% baseline).
  - **Hardware Latency Percentiles:** Mean 5.93 ms, Median 5.32 ms, $p_{95}$ 7.83 ms.
* Scroll through the per-category metrics table (Security Incident: 87.5% F1).
* Point out the **Retrain All Models** button (F-20) and **Import Complaint Dataset (CSV)** component (F-21).

#### Step 2: Few-Shot Category Studio & Prototype Matching (Features F-22, F-23, F-24)
* Click **Few-Shot Studio** tab:
  - Show the **Prototypical Embedding Space** interface.
  - In **Create New IT Category**, enter:
    * Category Name: `Cloud & Kubernetes`
    * Resolver Department: `Cloud Operations / SRE`
    * Scope: `Container crashes, node evictions, and cloud quota limits`
  - Click **Register Category** (updates prototypes instantly in memory).
* In the **Few-Shot Classification Testing Sandbox**:
  - In the test input, type:
    > *"Docker container exited with exit code 137 OOMKilled in Kubernetes pod."*
  - Click **Evaluate Similarities**.
  - Observe the real-time cosine similarity ranking against category prototype centroids!

---

### **ACT IV: Summary & Deliverables Wrap-Up** `[2:50 – 3:00]`
* Return to the central **Dashboard**.
* Voiceover concludes with delivery highlights: 100% test pass rate, strict train/test isolation, sub-10ms latency, and open-source GitHub release.
