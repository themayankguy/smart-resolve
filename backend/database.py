import sqlite3
import os
import json
import csv
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smartresolve.db")

DEFAULT_CATEGORIES = [
    ("Network and Wi-Fi", "Network Infrastructure", "Issues with internet, Wi-Fi, switches, routers, and connectivity"),
    ("Hardware", "Hardware & Asset Support", "Desktops, laptops, monitors, power supplies, physical components"),
    ("Software and Application", "Application Support Group", "Enterprise software, ERP, office suites, licensing, OS bugs"),
    ("Account and Authentication", "Identity & Access Management (IAM)", "Logins, password resets, Active Directory, MFA, permissions"),
    ("Email and Communication", "Messaging & Collaboration", "Outlook, Teams, email delivery, spam filtering, distribution lists"),
    ("Printer and Peripheral", "Field Support & Peripherals", "Printers, scanners, barcode readers, plotters, projectors"),
    ("Security Incident", "Cybersecurity & Incident Response Team (CERT)", "Phishing, malware, suspicious logins, ransomware, data leaks"),
    ("Other / Unknown", "General IT Service Desk", "General inquiries, miscellaneous facility or unclassified requests")
]

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db(db_path: str = DEFAULT_DB_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            department TEXT NOT NULL,
            description TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS support_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL,
            example_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (category_name) REFERENCES categories (name) ON DELETE CASCADE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            complaint_text TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            department TEXT NOT NULL,
            priority TEXT NOT NULL,
            urgency TEXT NOT NULL DEFAULT 'Normal',
            impact TEXT NOT NULL DEFAULT 'Individual',
            is_anomaly INTEGER NOT NULL DEFAULT 0,
            reconstruction_error REAL DEFAULT 0.0,
            anomaly_threshold REAL DEFAULT 0.0,
            anomaly_explanation TEXT,
            few_shot_category TEXT,
            few_shot_similarity REAL DEFAULT 0.0,
            priority_reasons TEXT,
            status TEXT NOT NULL DEFAULT 'Open',
            assigned_to TEXT,
            admin_notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaint_dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_text TEXT NOT NULL,
            category TEXT NOT NULL,
            is_normal INTEGER NOT NULL DEFAULT 1,
            priority TEXT DEFAULT 'Medium',
            impact TEXT DEFAULT 'Individual',
            source TEXT NOT NULL DEFAULT 'seed',
            created_at TEXT NOT NULL
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT NOT NULL,
            classifier_accuracy REAL,
            classifier_macro_f1 REAL,
            classifier_metrics_json TEXT,
            autoencoder_threshold REAL,
            autoencoder_metrics_json TEXT,
            few_shot_metrics_json TEXT,
            train_samples INTEGER,
            val_samples INTEGER,
            test_samples INTEGER,
            status TEXT NOT NULL,
            notes TEXT
        );
        """)

        # Indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets(priority);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets(created_at);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset_category ON complaint_dataset(category);")
        
        # Populate default categories if empty
        cursor.execute("SELECT COUNT(*) FROM categories")
        if cursor.fetchone()[0] == 0:
            now = datetime.now(timezone.utc).isoformat()
            cursor.executemany(
                "INSERT INTO categories (name, department, description, is_active, created_at) VALUES (?, ?, ?, 1, ?)",
                [(name, dept, desc, now) for name, dept, desc in DEFAULT_CATEGORIES]
            )

        conn.commit()

def seed_initial_dataset(db_path: str = DEFAULT_DB_PATH, csv_path: Optional[str] = None):
    init_db(db_path)
    if not csv_path:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "data", "seed_complaints.csv")
    
    if not os.path.exists(csv_path):
        return 0

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM complaint_dataset")
        if cursor.fetchone()[0] > 0:
            return 0  # Already seeded
        
        now = datetime.now(timezone.utc).isoformat()
        records = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                complaint_text = row.get("complaint_text", "").strip()
                category = row.get("category", "Other / Unknown").strip()
                is_normal = int(row.get("is_normal", 1))
                priority = row.get("priority", "Medium").strip()
                impact = row.get("impact", "Individual").strip()
                if complaint_text:
                    records.append((complaint_text, category, is_normal, priority, impact, "seed", now))
        
        cursor.executemany(
            """INSERT INTO complaint_dataset 
            (complaint_text, category, is_normal, priority, impact, source, created_at) 
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            records
        )

        # Seed initial support examples for few-shot learning
        for text, cat, is_n, pri, imp, src, dt in records[:30]:
            if is_n == 1:
                cursor.execute(
                    "INSERT INTO support_examples (category_name, example_text, created_at) VALUES (?, ?, ?)",
                    (cat, text, dt)
                )

        conn.commit()
        return len(records)

# --- Tickets CRUD ---

def create_ticket(ticket_data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        
        reasons_str = ticket_data.get("priority_reasons", [])
        if isinstance(reasons_str, list):
            reasons_str = json.dumps(reasons_str)

        cursor.execute("""
            INSERT INTO tickets (
                ticket_id, complaint_text, category, confidence, department,
                priority, urgency, impact, is_anomaly, reconstruction_error,
                anomaly_threshold, anomaly_explanation, few_shot_category,
                few_shot_similarity, priority_reasons, status, assigned_to,
                admin_notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket_data["ticket_id"],
            ticket_data["complaint_text"],
            ticket_data["category"],
            float(ticket_data.get("confidence", 0.0)),
            ticket_data.get("department", "General IT Service Desk"),
            ticket_data.get("priority", "Medium"),
            ticket_data.get("urgency", "Normal"),
            ticket_data.get("impact", "Individual"),
            1 if ticket_data.get("is_anomaly") else 0,
            float(ticket_data.get("reconstruction_error", 0.0)),
            float(ticket_data.get("anomaly_threshold", 0.0)),
            ticket_data.get("anomaly_explanation", ""),
            ticket_data.get("few_shot_category", ""),
            float(ticket_data.get("few_shot_similarity", 0.0)),
            reasons_str,
            ticket_data.get("status", "Open"),
            ticket_data.get("assigned_to", ""),
            ticket_data.get("admin_notes", ""),
            ticket_data.get("created_at", now),
            now
        ))
        conn.commit()
    return get_ticket_by_id(ticket_data["ticket_id"], db_path)

def get_ticket_by_id(ticket_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("priority_reasons"):
            try:
                d["priority_reasons"] = json.loads(d["priority_reasons"])
            except Exception:
                d["priority_reasons"] = [d["priority_reasons"]]
        else:
            d["priority_reasons"] = []
        d["is_anomaly"] = bool(d["is_anomaly"])
        return d

def list_tickets(
    category: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    is_anomaly: Optional[bool] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[List[Dict[str, Any]], int]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM tickets WHERE 1=1"
        count_query = "SELECT COUNT(*) FROM tickets WHERE 1=1"
        params: List[Any] = []

        if category and category != "All":
            query += " AND category = ?"
            count_query += " AND category = ?"
            params.append(category)
        if priority and priority != "All":
            query += " AND priority = ?"
            count_query += " AND priority = ?"
            params.append(priority)
        if status and status != "All":
            query += " AND status = ?"
            count_query += " AND status = ?"
            params.append(status)
        if is_anomaly is not None:
            query += " AND is_anomaly = ?"
            count_query += " AND is_anomaly = ?"
            params.append(1 if is_anomaly else 0)
        if search:
            query += " AND (ticket_id LIKE ? OR complaint_text LIKE ? OR department LIKE ?)"
            count_query += " AND (ticket_id LIKE ? OR complaint_text LIKE ? OR department LIKE ?)"
            term = f"%{search}%"
            params.extend([term, term, term])

        # Get total count
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for r in rows:
            d = dict(r)
            if d.get("priority_reasons"):
                try:
                    d["priority_reasons"] = json.loads(d["priority_reasons"])
                except Exception:
                    d["priority_reasons"] = [d["priority_reasons"]]
            else:
                d["priority_reasons"] = []
            d["is_anomaly"] = bool(d["is_anomaly"])
            results.append(d)
        return results, total

def update_ticket(ticket_id: str, updates: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    allowed = ["status", "priority", "department", "assigned_to", "admin_notes"]
    set_clauses = []
    params = []
    
    for key in allowed:
        if key in updates:
            set_clauses.append(f"{key} = ?")
            params.append(updates[key])
            
    if not set_clauses:
        return get_ticket_by_id(ticket_id, db_path)
        
    set_clauses.append("updated_at = ?")
    now = datetime.now(timezone.utc).isoformat()
    params.append(now)
    params.append(ticket_id)
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE tickets SET {', '.join(set_clauses)} WHERE ticket_id = ?", params)
        conn.commit()
    return get_ticket_by_id(ticket_id, db_path)

# --- Dashboard Metrics ---

def get_dashboard_metrics(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM tickets")
        total_tickets = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tickets WHERE status = 'Open'")
        open_tickets = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tickets WHERE status = 'In Progress'")
        in_progress_tickets = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tickets WHERE status = 'Resolved'")
        resolved_tickets = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tickets WHERE status = 'Closed'")
        closed_tickets = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tickets WHERE priority IN ('High', 'Critical')")
        high_critical_tickets = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tickets WHERE is_anomaly = 1")
        anomalous_tickets = cursor.fetchone()[0]

        cursor.execute("SELECT category, COUNT(*) as count FROM tickets GROUP BY category")
        category_distribution = [{"category": row["category"], "count": row["count"]} for row in cursor.fetchall()]

        cursor.execute("SELECT priority, COUNT(*) as count FROM tickets GROUP BY priority")
        priority_distribution = [{"priority": row["priority"], "count": row["count"]} for row in cursor.fetchall()]

        cursor.execute("SELECT status, COUNT(*) as count FROM tickets GROUP BY status")
        status_distribution = [{"status": row["status"], "count": row["count"]} for row in cursor.fetchall()]

        # Recent tickets
        cursor.execute("SELECT * FROM tickets ORDER BY created_at DESC LIMIT 6")
        recent = []
        for r in cursor.fetchall():
            d = dict(r)
            d["is_anomaly"] = bool(d["is_anomaly"])
            if d.get("priority_reasons"):
                try:
                    d["priority_reasons"] = json.loads(d["priority_reasons"])
                except Exception:
                    d["priority_reasons"] = [d["priority_reasons"]]
            else:
                d["priority_reasons"] = []
            recent.append(d)

        # Dataset count
        cursor.execute("SELECT COUNT(*) FROM complaint_dataset")
        dataset_count = cursor.fetchone()[0]

        return {
            "total_tickets": total_tickets,
            "open_tickets": open_tickets,
            "in_progress_tickets": in_progress_tickets,
            "resolved_tickets": resolved_tickets,
            "closed_tickets": closed_tickets,
            "high_critical_tickets": high_critical_tickets,
            "anomalous_tickets": anomalous_tickets,
            "category_distribution": category_distribution,
            "priority_distribution": priority_distribution,
            "status_distribution": status_distribution,
            "recent_tickets": recent,
            "dataset_count": dataset_count
        }

# --- Categories & Support Examples ---

def get_categories(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM categories ORDER BY name ASC")
        return [dict(r) for r in cursor.fetchall()]

def add_category(name: str, department: str, description: str = "", db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            """INSERT INTO categories (name, department, description, is_active, created_at)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(name) DO UPDATE SET department=excluded.department, description=excluded.description""",
            (name, department, description, now)
        )
        conn.commit()
        cursor.execute("SELECT * FROM categories WHERE name = ?", (name,))
        return dict(cursor.fetchone())

def get_support_examples(category_name: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if category_name:
            cursor.execute("SELECT * FROM support_examples WHERE category_name = ? ORDER BY id DESC", (category_name,))
        else:
            cursor.execute("SELECT * FROM support_examples ORDER BY id DESC")
        return [dict(r) for r in cursor.fetchall()]

def add_support_example(category_name: str, example_text: str, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "INSERT INTO support_examples (category_name, example_text, created_at) VALUES (?, ?, ?)",
            (category_name, example_text, now)
        )
        conn.commit()
        return {"id": cursor.lastrowid, "category_name": category_name, "example_text": example_text, "created_at": now}

# --- Dataset & Model Runs ---

def get_all_dataset_records(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM complaint_dataset ORDER BY id ASC")
        return [dict(r) for r in cursor.fetchall()]

def insert_dataset_records(records: List[Dict[str, Any]], source: str = "upload", db_path: str = DEFAULT_DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                r["complaint_text"],
                r["category"],
                int(r.get("is_normal", 1)),
                r.get("priority", "Medium"),
                r.get("impact", "Individual"),
                source,
                now
            )
            for r in records
        ]
        cursor.executemany(
            """INSERT INTO complaint_dataset 
            (complaint_text, category, is_normal, priority, impact, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows
        )
        conn.commit()
        return len(rows)

def save_model_run(run_data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> int:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_runs (
                run_timestamp, classifier_accuracy, classifier_macro_f1,
                classifier_metrics_json, autoencoder_threshold, autoencoder_metrics_json,
                few_shot_metrics_json, train_samples, val_samples, test_samples,
                status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_data.get("run_timestamp", datetime.now(timezone.utc).isoformat()),
            run_data.get("classifier_accuracy"),
            run_data.get("classifier_macro_f1"),
            json.dumps(run_data.get("classifier_metrics_json", {})),
            run_data.get("autoencoder_threshold"),
            json.dumps(run_data.get("autoencoder_metrics_json", {})),
            json.dumps(run_data.get("few_shot_metrics_json", {})),
            run_data.get("train_samples", 0),
            run_data.get("val_samples", 0),
            run_data.get("test_samples", 0),
            run_data.get("status", "SUCCESS"),
            run_data.get("notes", "")
        ))
        conn.commit()
        return cursor.lastrowid

def get_latest_model_run(db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_runs ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        for json_col in ["classifier_metrics_json", "autoencoder_metrics_json", "few_shot_metrics_json"]:
            if d.get(json_col):
                try:
                    d[json_col] = json.loads(d[json_col])
                except Exception:
                    pass
        return d
