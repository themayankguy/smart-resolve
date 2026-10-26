#!/usr/bin/env python3
"""
Ingest real-world IT helpdesk support tickets from Kaggle/HuggingFace dataset (Tobi-Bueck/customer-support-tickets).
Maps real tickets into SmartResolve's 8 ITIL categories + OOD anomalies.
Replaces database complaint_dataset table and retrains ML models.
"""

import os
import re
import random
import sqlite3
import pandas as pd
from datetime import datetime

# URLs and filepaths
HF_DATASET_URL = 'https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main/aa_dataset-tickets-multi-lang-5-2-50-version.csv'
CSV_OUTPUT_PATH = '/Users/mayanksmac/Desktop/atml2/data/seed_complaints.csv'
DB_PATH = '/Users/mayanksmac/Desktop/atml2/backend/smartresolve.db'

PERIPHERAL_KEYWORDS = [
    'printer', 'scanner', 'toner', 'cartridge', 'paper jam', 'spooler', 'plotter',
    'webcam', 'headset', 'mouse', 'keyboard', 'projector', 'external monitor',
    'display screen', 'docking station', 'usb', 'peripheral', 'hdmi'
]

def clean_ticket_text(subject, body):
    """Combine subject and body into clean complaint text, stripping standard greetings/closings."""
    subj = str(subject).strip() if pd.notna(subject) else ''
    b = str(body).strip() if pd.notna(body) else ''
    
    # Clean greeting
    b = re.sub(r'^(dear\s+support\s+team|dear\s+customer\s+support|hello\s+support|hi\s+team|dear\s+it\s+support)[,\.\s\n]*', '', b, flags=re.IGNORECASE)
    # Clean closing
    b = re.sub(r'(best\s+regards|sincerely|thanks\s+and\s+regards|thank\s+you|regards)[,\.\s\n]*.*$', '', b, flags=re.IGNORECASE)
    
    text = f"{subj}. {b}".strip()
    text = re.sub(r'\s+', ' ', text)
    return text[:600]

def categorize_ticket(row):
    text = (str(row['subject']) + ' ' + str(row['body'])).lower()
    tags = ' '.join([str(row[f'tag_{i}']) for i in range(1, 9) if pd.notna(row[f'tag_{i}'])]).lower()
    queue = str(row['queue']).lower()
    
    # Non-IT OOD anomaly queues
    if any(q in queue for q in ['billing', 'returns', 'sales', 'human resources']):
        return 'Other / Unknown', 0
    
    # 1. Printer and Peripheral
    if any(k in text for k in PERIPHERAL_KEYWORDS):
        return 'Printer and Peripheral', 1
    
    # 2. Security Incident
    if 'security' in tags or any(k in text for k in ['breach', 'vulnerability', 'phishing', 'malware', 'cyberattack', 'ransomware', 'unauthorized access', 'compromised']):
        return 'Security Incident', 1
        
    # 3. Account and Authentication
    if 'account' in tags or any(k in text for k in ['login', 'password', '2fa', 'mfa', 'authentication', 'credentials', 'reset password', 'sso', 'locked account', 'portal access', 'permission denied']):
        return 'Account and Authentication', 1
        
    # 4. Email and Communication
    if any(k in text for k in ['email', 'outlook', 'teams', 'exchange', 'inbox', 'mailbox', 'smtp', 'imap', 'calendar', 'zoom', 'chat']):
        return 'Email and Communication', 1
        
    # 5. Network and Wi-Fi
    if 'network' in tags or 'connectivity' in tags or any(k in text for k in ['wi-fi', 'wifi', 'vpn', 'router', 'dns', 'ethernet', 'bandwidth', 'latency', 'gateway', 'subnet', 'internet connection']):
        return 'Network and Wi-Fi', 1
        
    # 6. Hardware
    if 'hardware' in tags or any(k in text for k in ['laptop', 'desktop', 'motherboard', 'ram', 'cpu', 'battery', 'power supply', 'hard drive', 'ssd', 'fan noise', 'overheating', 'chassis']):
        return 'Hardware', 1
        
    # 7. Software and Application
    if any(k in tags for k in ['software', 'bug', 'crash']) or any(k in text for k in ['application', 'install', 'software crash', 'patch', 'update failed', 'runtime error', 'version upgrade', 'freeze', 'operating system', 'windows update']):
        return 'Software and Application', 1
        
    return 'Other / Unknown', 1

def map_priority(priority_raw):
    p = str(priority_raw).strip().lower()
    if 'critical' in p or 'urgent' in p:
        return 'Critical'
    if 'high' in p:
        return 'High'
    if 'low' in p:
        return 'Low'
    return 'Medium'

def infer_impact_scope(text):
    t = text.lower()
    if any(k in t for k in ['entire office', 'organization', 'all users', 'company-wide', 'whole company', 'infrastructure outage', 'multiple locations']):
        return 'Organization-wide'
    if any(k in t for k in ['department', 'finance', 'engineering', 'operations', 'sales team', 'floor', 'entire team']):
        return 'Department'
    if any(k in t for k in ['our team', 'project group', 'colleagues', 'several people']):
        return 'Team'
    return 'Individual'

def main():
    if os.path.exists(CSV_OUTPUT_PATH):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found existing {CSV_OUTPUT_PATH}, loading...")
        final_df = pd.read_csv(CSV_OUTPUT_PATH)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching real tickets from HuggingFace/Kaggle...")
        df = pd.read_csv(HF_DATASET_URL, nrows=35000)
        df_en = df[df['language'] == 'en'].copy()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(df_en)} English tickets.")
        
        results = [categorize_ticket(r) for _, r in df_en.iterrows()]
        df_en['mapped_category'] = [r[0] for r in results]
        df_en['is_normal'] = [r[1] for r in results]
        
        categories = [
            'Network and Wi-Fi',
            'Hardware',
            'Software and Application',
            'Account and Authentication',
            'Email and Communication',
            'Printer and Peripheral',
            'Security Incident',
            'Other / Unknown'
        ]
        
        sampled_rows = []
        TARGET_PER_CAT = 100
        
        random.seed(42)
        
        for cat in categories:
            sub = df_en[(df_en['is_normal'] == 1) & (df_en['mapped_category'] == cat)]
            n_take = min(len(sub), TARGET_PER_CAT)
            print(f"  Category '{cat}': found {len(sub)}, taking {n_take}")
            sampled = sub.sample(n=n_take, random_state=42)
            for _, row in sampled.iterrows():
                text = clean_ticket_text(row['subject'], row['body'])
                urgency = map_priority(row['priority'])
                impact = infer_impact_scope(text)
                sampled_rows.append({
                    'complaint_text': text,
                    'category': cat,
                    'urgency': urgency,
                    'impact_scope': impact,
                    'is_normal': 1
                })
                
        ood_sub = df_en[df_en['is_normal'] == 0]
        n_ood = min(len(ood_sub), TARGET_PER_CAT)
        print(f"  OOD Anomalies (non-IT): found {len(ood_sub)}, taking {n_ood}")
        sampled_ood = ood_sub.sample(n=n_ood, random_state=42)
        for _, row in sampled_ood.iterrows():
            text = clean_ticket_text(row['subject'], row['body'])
            urgency = map_priority(row['priority'])
            impact = infer_impact_scope(text)
            sampled_rows.append({
                'complaint_text': text,
                'category': 'Other / Unknown',
                'urgency': urgency,
                'impact_scope': impact,
                'is_normal': 0
            })
            
        final_df = pd.DataFrame(sampled_rows)
        final_df.drop_duplicates(subset=['complaint_text'], inplace=True)
        final_df.to_csv(CSV_OUTPUT_PATH, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {len(final_df)} real benchmark samples to {CSV_OUTPUT_PATH}")
    
    # Update SQLite database
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM complaint_dataset;")
    now_iso = datetime.now().isoformat()
    for _, r in final_df.iterrows():
        cur.execute("""
            INSERT INTO complaint_dataset (complaint_text, category, is_normal, priority, impact, source, created_at)
            VALUES (?, ?, ?, ?, ?, 'real_kaggle_hf', ?)
        """, (r['complaint_text'], r['category'], int(r['is_normal']), r['urgency'], r['impact_scope'], now_iso))
    conn.commit()
    conn.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully re-seeded SQLite complaint_dataset table ({len(final_df)} rows).")

    # Retrain ML Engine
    import sys
    sys.path.insert(0, '/Users/mayanksmac/Desktop/atml2')
    from backend.ml_engine import engine
    from backend.database import get_all_dataset_records, save_model_run
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Retraining ML Engine on real data...")
    records = get_all_dataset_records()
    metrics = engine.train_and_evaluate(records)
    run_id = save_model_run({
        "classifier_accuracy": metrics["classifier"]["accuracy"],
        "classifier_macro_f1": metrics["classifier"]["macro_f1"],
        "classifier_metrics_json": metrics["classifier"],
        "autoencoder_threshold": metrics["autoencoder"]["calibrated_threshold"],
        "autoencoder_metrics_json": metrics["autoencoder"],
        "few_shot_metrics_json": metrics["few_shot"],
        "train_samples": metrics["dataset_splits"]["train_samples"],
        "val_samples": metrics["dataset_splits"]["val_samples"],
        "test_samples": metrics["dataset_splits"]["test_samples"],
        "status": "SUCCESS",
        "notes": "Kaggle/HuggingFace real IT ticket benchmark ingestion"
    })
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Retraining Complete! Saved Model Run #{run_id}")
    print("Final Empirical Metrics on Held-out Real Test Set:")
    print(f"  Classifier Accuracy: {metrics['classifier']['accuracy'] * 100:.2f}%")
    print(f"  Classifier Macro F1: {metrics['classifier']['macro_f1'] * 100:.2f}%")
    print(f"  Autoencoder ROC-AUC: {metrics['autoencoder']['roc_auc'] * 100:.2f}%")
    print(f"  Calibrated Anomaly Threshold: {metrics['autoencoder']['calibrated_threshold']:.4f}")
    print(f"  Few-Shot Accuracy: {metrics['few_shot']['accuracy'] * 100:.2f}%")
    print(f"  Test Samples Evaluated: {metrics['dataset_splits']['test_samples']}")

if __name__ == '__main__':
    main()
