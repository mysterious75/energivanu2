"""Tests for the FastAPI REST API endpoints."""
from fastapi.testclient import TestClient

from energivanu.api import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_predict_endpoint():
    resp = client.post("/predict", json={"power_trace": [100, 120, 115, 130]})
    assert resp.status_code == 200
    data = resp.json()
    assert "power_forecast" in data
    assert "signal" in data
    assert "signal_probabilities" in data
    assert len(data["power_forecast"]) == 10
    assert data["signal"] in ("hold", "discharge", "charge")


def test_predict_with_nan_rejected():
    resp = client.post("/predict", json={"power_trace": [100, None, 200]})
    assert resp.status_code == 422


def test_predict_large_values_accepted():
    resp = client.post("/predict", json={"power_trace": [100, 9999, 200]})
    assert resp.status_code == 200


def test_predict_empty_trace_rejected():
    resp = client.post("/predict", json={"power_trace": []})
    assert resp.status_code == 422


def test_optimize_battery_endpoint():
    resp = client.post("/optimize/battery", json={
        "current_power_mw": 180.0,
        "target_power_mw": 200.0,
        "soc": 0.5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "battery_action_mw" in data
    assert "grid_power_mw" in data
    assert "soc" in data
    assert isinstance(data["battery_action_mw"], float)


def test_optimize_peak_shave():
    resp = client.post("/optimize/peak-shave", json={
        "hourly_power": [200 + 50 * (i % 12) for i in range(24)]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "peak_before_mw" in data
    assert "peak_after_mw" in data
    assert "peak_reduction_pct" in data
    assert "monthly_savings_usd" in data
    assert data["peak_reduction_pct"] >= 0


def test_predict_large_trace():
    trace = [140.0 + i * 0.1 for i in range(1000)]
    resp = client.post("/predict", json={"power_trace": trace})
    assert resp.status_code == 200


def test_optimize_battery_edge_soc():
    for soc in [0.0, 0.05, 0.95, 1.0]:
        resp = client.post("/optimize/battery", json={
            "current_power_mw": 180.0,
            "target_power_mw": 200.0,
            "soc": soc,
        })
        assert resp.status_code == 200
