# SmartResolve Dataset & Research Benchmark

This directory contains the dataset, ingestion pipeline, and empirical research benchmark for **SmartResolve** (SIH25195: Smart Helpdesk Ticketing Solution for IT Services).

---

## 1. Primary Dataset Source & Citation

The training and evaluation benchmark is extracted from the open-source **Customer IT Support Dataset** on Hugging Face:
- **Hugging Face Repository**: [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
- **Direct Dataset File**: [`aa_dataset-tickets-multi-lang-5-2-50-version.csv`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main/aa_dataset-tickets-multi-lang-5-2-50-version.csv)
- **Data Ingestion Script**: [`data/ingest_real_dataset.py`](file:///Users/mayanksmac/Desktop/atml2/data/ingest_real_dataset.py)

---

## 2. Benchmark Composition & Class Distribution

The benchmark dataset ([`seed_complaints.csv`](file:///Users/mayanksmac/Desktop/atml2/data/seed_complaints.csv)) comprises **900 real tickets** curated with balanced class representation:

| Class / Category | Samples ($N$) | Type | Description |
|---|---|---|---|
| **Network and Wi-Fi** | 100 | In-Distribution IT | VPN drops, Wi-Fi connectivity, DNS errors, subnet routing |
| **Hardware** | 100 | In-Distribution IT | Laptop battery failure, motherboard overheating, RAM issues |
| **Software and Application** | 100 | In-Distribution IT | App crashes, patch installation failures, runtime exceptions |
| **Account and Authentication** | 100 | In-Distribution IT | SSO login failure, MFA tokens, locked accounts, password resets |
| **Email and Communication** | 100 | In-Distribution IT | Outlook sync, Teams outages, mailbox quotas, SMTP failures |
| **Printer and Peripheral** | 100 | In-Distribution IT | Network printers, paper jams, spooler queue, scanner drivers, webcams |
| **Security Incident** | 100 | In-Distribution IT | Phishing attempts, ransomware warnings, unauthorized access |
| **Other / Unknown** | 100 | In-Distribution IT | General IT inquiries, system maintenance notifications |
| **Out-of-Distribution Anomalies** | 100 | Out-of-Distribution (`is_normal=0`) | Real non-IT tickets (billing, retail returns, sales queries) for Autoencoder validation |
| **Total** | **900** | — | **Balanced Real Benchmark** |

---

## 3. Empirical Evaluation Metrics (Held-out Test Split, $N=135$)

Evaluated on a strictly stratified 70/15/15 train-validation-test split:
- **Classifier Accuracy**: **68.15%** (8-way real multi-class)
- **Classifier Macro F1**: **67.77%**
- **Autoencoder ROC-AUC**: **60.09%**
- **Calibrated Anomaly Threshold**: **0.0021** (90th percentile of normal validation errors)
- **Prototypical Few-Shot Accuracy**: **60.74%**
