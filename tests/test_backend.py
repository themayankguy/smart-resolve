import os
import sys
import tempfile
import pytest
from fastapi.testclient import TestClient

# Insert backend directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

import database
from main import app
from ml_engine import engine

@pytest.fixture(scope="module")
def test_client():
    # Use temporary database
    temp_dir = tempfile.mkdtemp()
    temp_db = os.path.join(temp_dir, "test_smartresolve.db")
    database.DEFAULT_DB_PATH = temp_db
    database.init_db(temp_db)
    database.seed_initial_dataset(temp_db)

    # Initialize client
    with TestClient(app) as client:
        yield client

def test_api_health(test_client):
    res = test_client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "ml_engine" in data

def test_ticket_creation_and_persistence(test_client):
    payload = {
        "complaint_text": "Cisco core switch in server room is offline and entire floor is disconnected",
        "impact": "Enterprise",
        "urgency": "Urgent"
    }
    res = test_client.post("/api/tickets", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert "ticket" in data
    assert "ai_analysis" in data
    
    ticket = data["ticket"]
    assert ticket["ticket_id"].startswith("SR-")
    assert ticket["category"] == "Network and Wi-Fi"
    assert ticket["department"] == "Network Infrastructure"
    assert ticket["priority"] in ["High", "Critical"]
    assert len(ticket["priority_reasons"]) > 0

    # Verify retrieval
    get_res = test_client.get(f"/api/tickets/{ticket['ticket_id']}")
    assert get_res.status_code == 200
    assert get_res.json()["ticket_id"] == ticket["ticket_id"]

def test_ticket_listing_and_filtering(test_client):
    res = test_client.get("/api/tickets?category=Network and Wi-Fi")
    assert res.status_code == 200
    data = res.json()
    assert "tickets" in data
    assert data["total"] >= 1
    assert all(t["category"] == "Network and Wi-Fi" for t in data["tickets"])

def test_ticket_status_update(test_client):
    # First create
    create_res = test_client.post("/api/tickets", json={
        "complaint_text": "Printer in HR room is jamming continuously",
        "impact": "Department",
        "urgency": "Normal"
    })
    ticket_id = create_res.json()["ticket"]["ticket_id"]

    # Patch status
    patch_res = test_client.patch(f"/api/tickets/{ticket_id}", json={
        "status": "In Progress",
        "admin_notes": "Printer technician dispatched."
    })
    assert patch_res.status_code == 200
    updated = patch_res.json()
    assert updated["status"] == "In Progress"
    assert updated["admin_notes"] == "Printer technician dispatched."

def test_invalid_input_handling(test_client):
    # Short text
    res = test_client.post("/api/tickets", json={"complaint_text": "hi"})
    assert res.status_code == 422 # Pydantic min_length error

    # Non-existent ticket
    res_404 = test_client.get("/api/tickets/SR-NONEXISTENT")
    assert res_404.status_code == 404

def test_few_shot_new_category_and_examples(test_client):
    # Create new category
    cat_payload = {
        "name": "Cloud Storage & Backup",
        "department": "Cloud Operations",
        "description": "AWS S3, Google Cloud, and enterprise backup storage"
    }
    cat_res = test_client.post("/api/categories", json=cat_payload)
    assert cat_res.status_code == 201

    # Add support examples
    ex_res = test_client.post("/api/categories/Cloud Storage & Backup/examples", json={
        "example_text": "AWS S3 bucket synchronization failed with access denied 403 error"
    })
    assert ex_res.status_code == 201
    assert ex_res.json()["prototype_updated"] is True

    # Test classification using few-shot
    cls_res = test_client.post("/api/classify", json={
        "complaint_text": "Amazon AWS bucket backup synchronization failed overnight"
    })
    assert cls_res.status_code == 200
    data = cls_res.json()
    assert data["few_shot_category"] == "Cloud Storage & Backup"

def test_model_training_and_evaluation(test_client):
    train_res = test_client.post("/api/models/train")
    assert train_res.status_code == 200
    data = train_res.json()
    assert data["status"] == "SUCCESS"
    assert "metrics" in data
    assert "classifier" in data["metrics"]
    assert "autoencoder" in data["metrics"]
    assert "few_shot" in data["metrics"]

    eval_res = test_client.get("/api/models/evaluation")
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert "metrics" in eval_data
