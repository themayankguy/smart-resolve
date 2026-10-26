import os
import uuid
import csv
import io
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import (
    init_db, seed_initial_dataset, create_ticket, get_ticket_by_id,
    list_tickets, update_ticket, get_dashboard_metrics,
    get_categories, add_category, get_support_examples, add_support_example,
    get_all_dataset_records, insert_dataset_records, save_model_run, get_latest_model_run
)
from ml_engine import engine

# --- Pydantic Models ---

class TicketCreateRequest(BaseModel):
    complaint_text: str = Field(..., min_length=5, description="User complaint description")
    impact: str = Field("Individual", description="Impact scope: Individual, Department, or Enterprise")
    urgency: str = Field("Normal", description="Urgency level: Low, Normal, or Urgent")

class TicketUpdateRequest(BaseModel):
    status: Optional[str] = Field(None, description="Open, In Progress, Resolved, Closed")
    priority: Optional[str] = Field(None, description="Low, Medium, High, Critical")
    department: Optional[str] = None
    assigned_to: Optional[str] = None
    admin_notes: Optional[str] = None

class CategoryCreateRequest(BaseModel):
    name: str = Field(..., min_length=2)
    department: str = Field(..., min_length=2)
    description: Optional[str] = ""

class SupportExampleCreateRequest(BaseModel):
    example_text: str = Field(..., min_length=5)

class ClassifyRequest(BaseModel):
    complaint_text: str = Field(..., min_length=5)
    impact: str = "Individual"
    urgency: str = "Normal"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes SQLite, seeds demo data, and trains/calibrates ML models if not yet ready."""
    init_db()
    seed_initial_dataset()
    
    # Load all support examples for few-shot prototypes
    support_records = get_support_examples()
    support_map: Dict[str, List[str]] = {}
    for r in support_records:
        support_map.setdefault(r["category_name"], []).append(r["example_text"])
    
    if support_map:
        engine.update_prototypes(support_map)

    # If models are untrained, train on initial dataset
    if not engine.classifier or not engine.autoencoder:
        records = get_all_dataset_records()
        if len(records) >= 15:
            try:
                eval_metrics = engine.train_and_evaluate(records)
                save_model_run({
                    "classifier_accuracy": eval_metrics["classifier"]["accuracy"],
                    "classifier_macro_f1": eval_metrics["classifier"]["macro_f1"],
                    "classifier_metrics_json": eval_metrics["classifier"],
                    "autoencoder_threshold": eval_metrics["autoencoder"]["calibrated_threshold"],
                    "autoencoder_metrics_json": eval_metrics["autoencoder"],
                    "few_shot_metrics_json": eval_metrics["few_shot"],
                    "train_samples": eval_metrics["dataset_splits"]["train_samples"],
                    "val_samples": eval_metrics["dataset_splits"]["val_samples"],
                    "test_samples": eval_metrics["dataset_splits"]["test_samples"],
                    "status": "INITIAL_SUCCESS",
                    "notes": "Automatic initialization training on seed dataset"
                })
            except Exception as e:
                print(f"Warning: Auto-training on startup failed: {e}")
    yield

app = FastAPI(
    title="SmartResolve AI Helpdesk API",
    description="SIH25195: AI-Based Smart Helpdesk Ticketing Solution for IT Services",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.get("/api/health")
def get_health():
    return {
        "status": "healthy",
        "service": "SmartResolve AI Helpdesk API",
        "ml_engine": engine.get_status()
    }

@app.get("/api/dashboard")
def get_dashboard():
    return get_dashboard_metrics()

@app.post("/api/tickets", status_code=status.HTTP_201_CREATED)
def create_ticket_endpoint(req: TicketCreateRequest):
    if not req.complaint_text.strip():
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")

    # Run AI inference pipeline
    analysis = engine.analyze_complaint(
        complaint_text=req.complaint_text,
        impact=req.impact,
        urgency=req.urgency
    )

    ticket_id = f"SR-{uuid.uuid4().hex[:8].upper()}"
    ticket_payload = {
        "ticket_id": ticket_id,
        "complaint_text": req.complaint_text.strip(),
        "category": analysis["category"],
        "confidence": analysis["confidence"],
        "department": analysis["department"],
        "priority": analysis["priority"],
        "urgency": req.urgency,
        "impact": req.impact,
        "is_anomaly": analysis["is_anomaly"],
        "reconstruction_error": analysis["reconstruction_error"],
        "anomaly_threshold": analysis["anomaly_threshold"],
        "anomaly_explanation": analysis["anomaly_explanation"],
        "few_shot_category": analysis["few_shot_category"],
        "few_shot_similarity": analysis["few_shot_similarity"],
        "priority_reasons": analysis["priority_reasons"],
        "status": "Open",
        "assigned_to": analysis["department"],
        "admin_notes": ""
    }

    created = create_ticket(ticket_payload)
    return {
        "ticket": created,
        "ai_analysis": analysis
    }

@app.get("/api/tickets")
def list_tickets_endpoint(
    category: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    is_anomaly: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    tickets, total = list_tickets(
        category=category,
        priority=priority,
        status=status,
        is_anomaly=is_anomaly,
        search=search,
        limit=limit,
        offset=offset
    )
    return {
        "tickets": tickets,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/api/tickets/{ticket_id}")
def get_ticket_endpoint(ticket_id: str):
    ticket = get_ticket_by_id(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    return ticket

@app.patch("/api/tickets/{ticket_id}")
def update_ticket_endpoint(ticket_id: str, updates: TicketUpdateRequest):
    existing = get_ticket_by_id(ticket_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    
    update_dict = {k: v for k, v in updates.model_dump().items() if v is not None}
    updated = update_ticket(ticket_id, update_dict)
    return updated

@app.post("/api/classify")
def test_classify_endpoint(req: ClassifyRequest):
    if not req.complaint_text.strip():
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")
    
    return engine.analyze_complaint(
        complaint_text=req.complaint_text,
        impact=req.impact,
        urgency=req.urgency
    )

@app.get("/api/categories")
def get_categories_endpoint():
    cats = get_categories()
    examples = get_support_examples()
    
    # Augment with prototype support count
    counts = {}
    for ex in examples:
        counts[ex["category_name"]] = counts.get(ex["category_name"], 0) + 1
        
    for c in cats:
        c["support_count"] = counts.get(c["name"], 0)
        c["has_prototype"] = c["name"] in engine.prototypes

    return cats

@app.post("/api/categories", status_code=status.HTTP_201_CREATED)
def create_category_endpoint(req: CategoryCreateRequest):
    try:
        new_cat = add_category(req.name.strip(), req.department.strip(), req.description or "")
        engine.category_dept_map[req.name.strip()] = req.department.strip()
        return new_cat
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create category: {str(e)}")

@app.get("/api/categories/{category_name}/examples")
def list_support_examples_endpoint(category_name: str):
    return get_support_examples(category_name)

@app.post("/api/categories/{category_name}/examples", status_code=status.HTTP_201_CREATED)
def add_support_example_endpoint(category_name: str, req: SupportExampleCreateRequest):
    cats = [c["name"] for c in get_categories()]
    if category_name not in cats:
        raise HTTPException(status_code=404, detail=f"Category '{category_name}' does not exist.")
    
    record = add_support_example(category_name, req.example_text.strip())
    
    # Recalculate prototypes
    support_records = get_support_examples()
    support_map: Dict[str, List[str]] = {}
    for r in support_records:
        support_map.setdefault(r["category_name"], []).append(r["example_text"])
    engine.update_prototypes(support_map)

    return {
        "example": record,
        "prototype_updated": True,
        "support_count": len(support_map.get(category_name, []))
    }

@app.post("/api/dataset/upload")
async def upload_dataset_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    try:
        text = contents.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    if not rows:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    # Validate headers
    first_row = rows[0]
    if "complaint_text" not in first_row or "category" not in first_row:
        raise HTTPException(
            status_code=400,
            detail="Missing required columns. CSV must contain at least 'complaint_text' and 'category'."
        )

    valid_records = []
    errors = []
    for idx, r in enumerate(rows, start=1):
        ct = r.get("complaint_text", "").strip()
        cat = r.get("category", "").strip()
        if not ct:
            errors.append(f"Row {idx}: 'complaint_text' is empty.")
            continue
        if not cat:
            errors.append(f"Row {idx}: 'category' is empty.")
            continue
        
        is_normal = 1
        if "is_normal" in r and r["is_normal"].strip():
            try:
                is_normal = int(r["is_normal"].strip())
            except ValueError:
                is_normal = 1

        valid_records.append({
            "complaint_text": ct,
            "category": cat,
            "is_normal": is_normal,
            "priority": r.get("priority", "Medium").strip() or "Medium",
            "impact": r.get("impact", "Individual").strip() or "Individual"
        })

    if not valid_records:
        raise HTTPException(
            status_code=400,
            detail={"message": "No valid records found.", "errors": errors[:10]}
        )

    inserted = insert_dataset_records(valid_records, source=file.filename)
    return {
        "message": f"Successfully imported {inserted} records.",
        "imported_count": inserted,
        "skipped_errors": errors[:10]
    }

@app.post("/api/models/train")
def train_models_endpoint():
    records = get_all_dataset_records()
    if len(records) < 15:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient training examples ({len(records)} found). At least 15 labelled complaints required."
        )

    try:
        metrics = engine.train_and_evaluate(records)
        save_model_run({
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
            "notes": "Admin triggered retraining"
        })
        return {
            "status": "SUCCESS",
            "message": "Models successfully trained, calibrated, and evaluated.",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.get("/api/models/evaluation")
def get_evaluation_endpoint():
    latest_run = get_latest_model_run()
    if engine.latest_eval_metrics:
        return {
            "source": "live_engine",
            "metrics": engine.latest_eval_metrics,
            "latest_run": latest_run
        }
    elif latest_run:
        return {
            "source": "database_history",
            "metrics": {
                "classifier": latest_run.get("classifier_metrics_json", {}),
                "autoencoder": latest_run.get("autoencoder_metrics_json", {}),
                "few_shot": latest_run.get("few_shot_metrics_json", {}),
                "dataset_splits": {
                    "train_samples": latest_run.get("train_samples", 0),
                    "val_samples": latest_run.get("val_samples", 0),
                    "test_samples": latest_run.get("test_samples", 0)
                }
            },
            "latest_run": latest_run
        }
    else:
        return {
            "status": "NO_EVALUATION_YET",
            "message": "Models have not been trained or evaluated yet. Use POST /api/models/train."
        }

@app.get("/api/models/status")
def get_model_status_endpoint():
    return engine.get_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
