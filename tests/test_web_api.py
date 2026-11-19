"""
tests/test_web_api.py
Unit and integration tests for the Flask REST API and web endpoints.
"""

import os
import json
import pytest
from web.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    rv = client.get("/")
    assert rv.status_code == 200
    assert b"Mild Steel (MS) Weld Property" in rv.data

def test_api_predict_success(client):
    payload = {
        "Base_UTS": 938.0,
        "Base_YS": 685.0,
        "Base_Elongation": 29.5,
        "Gauge_Length_L0": 175.0,
        "Specimen_Width": 30.0,
        "Specimen_Thickness": 3.0,
        "Weld_Current_A": 150.0,
        "Weld_Voltage_V": 22.0,
        "Heat_Input_kJ_mm": 1.2,
        "HAZ_Width_mm": 4.5,
        "Pass_Count": 1,
        "Joint_Type_DoubleV": 1,
        "Weld_Process_TIG": 1,
        "Failure_Location_Weld": 0
    }
    rv = client.post("/api/predict", json=payload)
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    assert "predictions" in data
    assert "uncertainty" in data
    assert "margin_of_safety" in data
    assert "Welded_UTS" in data["predictions"]
    assert data["predictions"]["Welded_UTS"] > 800.0

def test_api_predict_empty_payload(client):
    rv = client.post("/api/predict", json={})
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True

def test_api_curve_success(client):
    payload = {"Welded_UTS": 925.0, "Welded_YS": 705.0, "Welded_Elongation": 17.5}
    rv = client.post("/api/curve", json=payload)
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    assert len(data["curve"]["strain"]) == 200
    assert len(data["curve"]["stress"]) == 200

def test_api_curve_references(client):
    rv = client.post("/api/curve", json={})
    data = rv.get_json()
    assert "references" in data
    assert len(data["references"]) > 0

def test_api_leaderboard_success(client):
    rv = client.get("/api/leaderboard")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    assert "leaderboard" in data
    assert "XGBoost" in data["leaderboard"] or "Stacking Ensemble" in data["leaderboard"]

def test_api_metrics_success(client):
    rv = client.get("/api/metrics?model=XGBoost")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    assert "metrics" in data

def test_api_data_success(client):
    rv = client.get("/api/data")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    assert "summary" in data
    assert len(data["summary"]) > 0

def test_api_shap_success(client):
    rv = client.get("/api/shap")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["success"] is True
    assert len(data["shap"]) > 0

def test_api_export_pdf_success(client):
    payload = {
        "specimen_id": "TEST-SPEC-API",
        "Welded_UTS": 925.0,
        "Welded_YS": 700.0,
        "Welded_Elongation": 18.0
    }
    rv = client.post("/api/export-pdf", json=payload)
    assert rv.status_code == 200
    assert rv.headers["Content-Type"] == "application/pdf"
    assert len(rv.data) > 2000
