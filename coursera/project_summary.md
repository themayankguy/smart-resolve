# SmartResolve: AI-Based Smart Helpdesk Ticketing System
## Project Summary & Agile Engineering Report

**Course Context:** Coursera Guided Project Submission  
**Authors:** Mayank Sawant & Sayali Rajput  
**Institution:** Dept. of Artificial Intelligence & Data Science, MPSTME, SVKM's NMIMS, Navi Mumbai, India  
**Software/Tool Used:** **Asana** (Agile Sprint Tracking & Kanban Management)  
**Repository:** [https://github.com/themayankguy/smart-resolve.git](https://github.com/themayankguy/smart-resolve.git)  
**Deliverable Document:** `coursera/Project_Summary_SmartResolve.pdf` (Strict 5-page PDF)

---

## 1. Title of Project
**SmartResolve: AI-Based Smart Helpdesk Ticketing System (with Agile Project Delivery in Asana)**

---

## 2. Software / Tool Used: Asana
**Asana** was selected and used as the central collaborative project management and Agile sprint tracking software throughout the entire lifecycle of the **SmartResolve** project. Asana enabled the development team to decompose an end-to-end artificial intelligence engineering system—spanning dataset curation, Sentence-BERT semantic vectorization, PyTorch deep autoencoders, asynchronous FastAPI REST microservices, and reactive React/Vite interfaces—into structured user stories, tasks, acceptance tests, and sprint milestones.

### Agile Workflow Structure in Asana:
- **Backlog:** Conceptual ideas, raw requirements, and future capability roadmaps.
- **Ready:** Refined, peer-reviewed engineering tasks scheduled for immediate pickup.
- **In Progress:** Active feature implementation, unit testing, and model fine-tuning.
- **Review & Validation:** Code review, metric reproducibility verification, data leakage audits, and automated test execution.
- **Done:** Fully merged, production-verified deliverables meeting all acceptance criteria.

### Screenshot of Tool Mentioned (Asana):
![Figure 1: Asana Project Management Board](images/asana_tool_screenshot.png)
*Figure 1: Sprint management board for SmartResolve in Asana, illustrating live task distribution across Backlog, Done, Review & Validation, In Progress, and Ready columns.*

**Key Agile Metrics Tracked in Asana:**
- **8 Core Engineering Milestones Completed:** SQLite schema & CRUD operations, FastAPI routes & Pydantic schemas, Sentence-BERT embedding pipeline, PyTorch Autoencoder training, React/Vite management dashboard, dataset ingestion pipeline, research artifacts, and backend test suites.
- **Audits & Quality Checks:** Dedicated cards for data leakage prevention, ML edge case validation, and full-stack smoke tests.
- **Pass Rate:** 100% test pass rate across automated pytest suites.

---

## 3. Objectives and Requirements

### 3.1 Primary Objectives
Modern enterprise IT departments face significant operational friction from ticket backlogs, non-standard user phrasing, manual triage fatigue, and SLA breaches. SmartResolve automates the complaint ingestion, classification, risk scoring, and routing pipeline:
1. **Automated Natural Language Ticket Intake:** Ingest heterogeneous user complaints and encode them into 384-dimensional continuous representations using Sentence-BERT (`all-MiniLM-L6-v2`).
2. **Multi-Class IT Domain Classification:** Route tickets across 8 enterprise IT service categories: *Hardware*, *Software*, *Network/VPN*, *Account/IAM Access*, *Security Incident*, *Database/Storage*, *Printer and Peripheral*, and *Other IT Services*.
3. **Unsupervised Out-of-Distribution (OOD) Anomaly Detection:** Detect spam, non-technical queries, zero-day failures, or novel attack vectors using a deep symmetric PyTorch Autoencoder reconstruction threshold ($\tau = 0.0018$).
4. **Prototypical Few-Shot Metric Learning:** Dynamically adapt to emerging IT categories from minimal support examples ($k = 1, 3, 5$) without retraining or catastrophic forgetting.
5. **Dynamic SLA & Escalation Prediction:** Deterministically compute SLA target turnaround times (2h to 48h) and escalation risk from ticket priority and organizational impact scope.
6. **Ultra-Low Latency Inference:** Guarantee sub-10ms per-ticket inference latency on commodity CPU hardware.

### 3.2 Requirements Specification

#### Functional Requirements (FRs):
| Req ID | Requirement Category | Description & Engineering Implementation |
|---|---|---|
| **FR-1** | Natural Language Intake | Accept unstructured complaint text via REST API or interactive web interface; normalize whitespace and encode via Sentence-BERT into 384-dim dense vectors. |
| **FR-2** | Multi-Class Routing | Generate posterior probability distributions across 8 enterprise IT domains using regularized multinomial logistic regression. |
| **FR-3** | Anomaly Detection | Pass embeddings through a 5-layer PyTorch Autoencoder ($384 \to 128 \to 64 \to 128 \to 384$); flag reconstruction MSE $> 0.0018$ as anomalous. |
| **FR-4** | Few-Shot Adaptation | Support rapid creation of ad-hoc support classes by calculating class prototype centroids in cosine metric space from $k$ sample tickets. |
| **FR-5** | SLA & Priority Engine | Synthesize ticket priority (P1–P4) and organizational impact (Organization, Department, Individual) into deterministic SLA targets (2h–48h). |
| **FR-6** | Full-Stack Dashboard | Provide IT agents with real-time analytics, queue management, model evaluation visualizations, and ticket status resolution toggles. |

#### Non-Functional Requirements (NFRs):
- **NFR-1 (Latency):** Average end-to-end processing latency under 10 ms per ticket on commodity multi-core CPU architecture.
- **NFR-2 (Data Isolation):** Strict 85/15 train-test split isolation with zero data leakage between training and evaluation partitions.
- **NFR-3 (Lightweight Footprint):** Zero microservice sprawl; single-process asynchronous FastAPI backend with ACID SQLite persistence.
- **NFR-4 (Verification & Reliability):** 100% automated test pass rate for backend API endpoints, database operations, and ML routines.

---

## 4. Technical Summary & Experimental Findings

### 4.1 System Architecture
SmartResolve implements a dual-inference machine learning pipeline:
1. **Embedding Layer:** Sentence-BERT (`all-MiniLM-L6-v2`) mapping text to a 384-dimensional dense semantic space.
2. **Dual-Head Inference:**
   - **Supervised Classifier:** $\ell_2$-regularized multinomial logistic regression producing calibrated class posteriors.
   - **Unsupervised Anomaly Detector:** Deep symmetric PyTorch Autoencoder trained exclusively on normal in-distribution tickets using MSE loss.
   - **Few-Shot Engine:** Prototypical cosine metric nearest-centroid matching.
3. **Backend & Persistence:** Python FastAPI asynchronous REST service backed by SQLite with write-ahead logging (WAL).
4. **User Interface:** React 19 + Vite dashboard featuring live analytics, queue management, and model diagnostics.

### 4.2 Quantitative Experimental Results
Evaluated on a real-world enterprise IT support benchmark comprising **900 tickets** (800 in-distribution across 8 balanced categories + 100 out-of-distribution anomaly inquiries) using a strictly isolated held-out test split ($N = 135$):

| Evaluation Metric | TF-IDF Baseline | SmartResolve (Ours) | Performance Gain / Status |
|---|---|---|---|
| **Overall Test Accuracy** | 64.44% | **69.63%** | **+5.19% absolute gain** (5.5× random baseline) |
| **Macro F1-Score** | 63.12% | **69.63%** | Balanced across all 8 IT categories |
| **Security Incident F1** | 74.20% | **87.50%** (93.3% recall) | High-precision routing for critical security events |
| **Account / IAM Access F1** | 78.10% | **86.67%** (86.7% recall) | High fidelity on credential and authorization requests |
| **Anomaly Detection ROC-AUC** | 52.30% (Random) | **76.21%** (PR-AUC: 50.22%) | **8.26% False Positive Rate** at $\tau = 0.0018$ |
| **Few-Shot Accuracy (k=5)** | 12.50% (Random) | **44.81%** (3.6× baseline) | Rapid adaptation to emerging edge cases |
| **Mean Inference Latency** | 14.20 ms | **5.93 ms** ($p_{95}$: 7.83 ms) | **2.4× faster**; exceeds sub-10ms real-time target |

---

## 5. Screenshots of Project

### Figure 2: Live Operations Dashboard
![Figure 2: Operations Dashboard](images/dashboard_overview.png)
*Visualizing real-time ticket volume metrics (Total, Open, Escalated, Normal/Anomaly count), category distribution bar charts, priority breakdown, and the dynamic ticket queue table with live status filtering.*

### Figure 3: Interactive Ticket Ingestion Form
![Figure 3: Ticket Ingestion Form](images/submit_ticket.png)
*End-user complaint intake portal featuring natural language input, suggested category chips, priority selection, and organizational impact scope tagging (Organization, Department, Individual) for automated SLA scheduling.*

### Figure 4: Automated AI Diagnostic Record & Triage Modal
![Figure 4: Diagnostic Modal](images/ticket_diagnostic_modal.png)
*Deep inspection card showing predicted category (Network and Wi-Fi, 77.5% confidence), reconstruction error (0.003 vs threshold 0.0047), assigned support group, and prioritization rationale.*

### Figure 5: Live Model Evaluation & Benchmark Studio
![Figure 5: Model Evaluation Studio](images/model_evaluation.png)
*Real-time performance dashboard displaying confusion matrix, per-category precision/recall/F1 metrics, autoencoder ROC curve, and hardware latency percentiles (5.93 ms mean, 5.32 ms median).*

### Figure 6: Prototypical Few-Shot Category Studio
![Figure 6: Few-Shot Studio](images/few_shot_studio.png)
*Interactive sandbox allowing IT administrators to dynamically register new IT incident types with 1 to 5 labeled examples and evaluate cosine similarity distance in embedding space.*

---

## 6. Conclusion & Deliverables Summary
SmartResolve successfully achieved all technical milestones defined in Asana. The project delivers a production-ready, sub-10ms AI ticketing system with an empirically verified 69.63% classification accuracy, 76.21% Autoencoder ROC-AUC, 100% test pass rate, and full reproducible source code published at [https://github.com/themayankguy/smart-resolve.git](https://github.com/themayankguy/smart-resolve.git).
