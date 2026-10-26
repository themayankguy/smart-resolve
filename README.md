# SmartResolve — AI-Based Smart Helpdesk Ticketing System

> **SIH Problem Statement SIH25195:** Smart Helpdesk Ticketing Solution for IT Services  
> **Academic Subject:** Advanced Techniques in Machine Learning (ATML)

---

## 1. Project Overview

**SmartResolve** is an end-to-end AI-powered IT service desk platform engineered to eliminate informal, unmonitored IT complaints and replace them with centralized, intelligent tracking.

The system automatically:
1. **Understands Complaint Semantics:** Encodes unstructured complaint text using Sentence-BERT (`sentence-transformers/all-MiniLM-L6-v2`) and classifies issues into IT domain categories with confidence scoring.
2. **Detects Anomaly Patterns:** Reconstructs complaint embeddings via a PyTorch deep autoencoder trained on normal operational IT complaints, flagging out-of-distribution or corrupted submissions based on validation-calibrated reconstruction error thresholds.
3. **Supports Rapid Few-Shot Domain Expansion:** Implements an embedding-based prototypical few-shot classifier allowing administrators to introduce brand new IT categories with 1 to 5 labelled support examples.
4. **Recommends Transparent Priorities:** Evaluates reported impact, user urgency, security threat signatures, and service outage indicators with an explainable rationale list.
5. **Generates Persistent Tickets & Admin Controls:** Manages tickets in SQLite with status tracking (`Open`, `In Progress`, `Resolved`, `Closed`), department reassignment, and technician resolution notes.

---

## 2. Compact Architecture (Minimal Backend)

Adhering strictly to project constraints, the backend consists of **three Python source files only**:

```text
smartresolve/
├── backend/
│   ├── main.py              # FastAPI app, Pydantic schemas, routes, and orchestration
│   ├── ml_engine.py         # All ML models (Transformers, Autoencoder, Few-Shot, Evaluation)
│   ├── database.py          # SQLite connection, schema, indexes, and CRUD operations
│   ├── requirements.txt     # Backend dependencies
│   └── saved_models/        # Persisted model checkpoints and evaluation metadata
├── frontend/
│   ├── package.json
│   ├── vite.config.js       # Vite config with /api reverse proxy
│   ├── index.html
│   └── src/
│       ├── App.jsx          # SaaS navigation, ticket form, filterable table, modal, studio
│       ├── main.jsx         # React root
│       ├── index.css        # Enterprise design tokens, glassmorphism, responsive grid
│       └── components/
│           └── Dashboard.jsx# Recharts visualizations, KPI cards, and recent activity
├── data/
│   ├── README.md            # Dataset specification and CSV format instructions
│   └── seed_complaints.csv  # 100+ realistic IT complaints and anomaly cases for SIH25195
├── tests/
│   └── test_backend.py      # Automated pytest suite for all APIs and ML modules
├── README.md
└── .gitignore
```

---

## 3. Core AI/ML Pipeline Modules

### Module A: Transformer-Based Complaint Classification
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` producing 384-dimensional normalized dense vectors.
- **Classifier:** Multi-class Logistic Regression with calibrated probability distributions (`predict_proba`).
- **Initial Categories:**
  1. `Network and Wi-Fi` (Routes to *Network Infrastructure*)
  2. `Hardware` (Routes to *Hardware & Asset Support*)
  3. `Software and Application` (Routes to *Application Support Group*)
  4. `Account and Authentication` (Routes to *Identity & Access Management - IAM*)
  5. `Email and Communication` (Routes to *Messaging & Collaboration*)
  6. `Printer and Peripheral` (Routes to *Field Support & Peripherals*)
  7. `Security Incident` (Routes to *Cybersecurity & Incident Response Team - CERT*)
  8. `Other / Unknown` (Routes to *General IT Service Desk*)
- **Confidence & Fallback:** Displays model confidence. Low-confidence predictions (< 0.40) trigger human review flags and prototype fallback.

### Module B: Autoencoder-Based Anomaly Detection
- **Architecture:** PyTorch `EmbeddingAutoencoder` ($384 \to 64 \to 16 \to 64 \to 384$) with LeakyReLU activations, Batch Normalization, and unit-sphere reconstruction normalization.
- **Training Strategy:** Trained strictly on normal operational complaints (`is_normal == 1`).
- **Error Metric:** Mean Squared Error (MSE) between input embedding $\mathbf{x}$ and reconstructed embedding $\hat{\mathbf{x}}$.
- **Threshold Calibration:** Calibrated on a held-out validation split (90th percentile of normal validation errors).
- **Explanation:** Produces human-readable explanations explaining whether the reconstruction error exceeds the calibrated baseline.

### Module C: Prototypical Few-Shot Learning
- **Algorithm:** Embedding centroid averaging over support examples $\mathbf{c}_k = \frac{1}{|S_k|}\sum_{x \in S_k} f(x)$.
- **Dynamic Updates:** When an administrator registers a new category or adds support examples, the prototype vector is updated in memory and persisted.
- **Classification:** Uses cosine similarity $\text{sim}(\mathbf{x}, \mathbf{c}_k)$ to assign the nearest prototype.

### Module D: Transparent Priority Recommender
- **Heuristic Engine:** Evaluates:
  - Security threat keywords (`ransomware`, `phishing`, `unauthorized`, `stolen`) $\implies$ High/Critical with mandatory CERT review.
  - Outage indicators (`down`, `outage`, `entire floor`, `offline`) $\implies$ High/Critical.
  - Impact scope (`Individual`, `Department`, `Enterprise`).
  - Urgency level (`Low`, `Normal`, `Urgent`).
  - Autoencoder anomaly status.
- **Explainability:** Returns a list of concrete reasons explaining why the priority was recommended.

---

## 4. Setup and Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 1. Backend Setup & Startup
```bash
# In project root
python3 -m pip install -r backend/requirements.txt

# Start FastAPI server on port 8000
python3 -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```
*On first startup, SQLite database `smartresolve.db` is initialized and automatically seeded with `data/seed_complaints.csv`, and models are trained on train/val/test splits.*

### 2. Frontend Setup & Startup
```bash
# In frontend directory
cd frontend
npm install
npm run dev
```
Open **`http://127.0.0.1:5173/`** in your browser.

### 3. Run Automated Tests
```bash
# In project root
python3 -m pytest tests/test_backend.py -v
```

---

## 5. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service status and ML engine readiness |
| `GET` | `/api/dashboard` | SQLite-calculated statistics for charts and cards |
| `POST` | `/api/tickets` | Classifies complaint, detects anomalies, recommends priority, creates ticket |
| `GET` | `/api/tickets` | Lists tickets with category, priority, status, and search filters |
| `GET` | `/api/tickets/{id}` | Retrieves specific ticket details |
| `PATCH` | `/api/tickets/{id}` | Updates ticket status, priority, department, or admin notes |
| `POST` | `/api/classify` | Tests AI pipeline inference without persisting a ticket |
| `GET` | `/api/models/evaluation` | Returns test-split evaluation metrics and confusion matrix |
| `POST` | `/api/models/train` | Retrains classifier and autoencoder on dataset |
| `GET` | `/api/categories` | Lists all active categories and support counts |
| `POST` | `/api/categories` | Adds a new category |
| `POST` | `/api/categories/{name}/examples` | Adds a support example and recalculates prototype vector |
| `POST` | `/api/dataset/upload` | Validates and imports training CSV |

---

## 6. Evaluation Methodology & Academic Integrity

- **No Data Leakage:** Raw records are split into 70% Train, 15% Validation, and 15% Test *before* training.
- **Evaluation on Test Split Only:** Accuracy, Precision, Recall, Macro F1, and Confusion Matrices are evaluated on the test partition.
- **Validation Calibration:** Autoencoder anomaly thresholds are tuned solely on normal validation samples.
- **No Generative Hallucinations:** Predictions are generated by deterministic classifiers and embedding projections, never by unverified generative LLM tokens.
