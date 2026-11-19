"""
tests/test_uncertainty.py
Unit tests for Monte Carlo Dropout epistemic uncertainty quantification
and Aerospace Margin of Safety calculation.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from core.uncertainty import MonteCarloDropout, compute_margin_of_safety, enable_dropout_eval

class DummyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(14, 6)
        self.drop = nn.Dropout(p=0.5)
    def forward(self, x):
        return self.drop(self.fc(x))

def test_enable_dropout_eval():
    model = DummyTorchModel()
    model.eval()
    enable_dropout_eval(model)
    assert model.drop.training is True

def test_mc_dropout_passes_count():
    model = DummyTorchModel()
    uq = MonteCarloDropout(model, n_passes=50, is_torch=True)
    X = torch.randn(4, 14)
    res = uq.predict_with_uncertainty(X)
    assert res["all_passes"].shape[0] == 50

def test_mc_dropout_shape():
    model = DummyTorchModel()
    uq = MonteCarloDropout(model, n_passes=20, is_torch=True)
    X = torch.randn(5, 14)
    res = uq.predict_with_uncertainty(X)
    assert res["mean"].shape == (5, 6)
    assert res["sigma"].shape == (5, 6)
    assert res["min_guaranteed"].shape == (5, 6)

def test_mc_dropout_min_guaranteed_below_mean():
    model = DummyTorchModel()
    uq = MonteCarloDropout(model, n_passes=30, is_torch=True)
    X = torch.randn(4, 14)
    res = uq.predict_with_uncertainty(X)
    assert np.all(res["min_guaranteed"] <= res["mean"])

def test_mc_dropout_upper_bound_above_mean():
    model = DummyTorchModel()
    uq = MonteCarloDropout(model, n_passes=30, is_torch=True)
    X = torch.randn(4, 14)
    res = uq.predict_with_uncertainty(X)
    assert np.all(res["upper_bound"] >= res["mean"])

def test_mc_dropout_sigma_positive():
    model = DummyTorchModel()
    uq = MonteCarloDropout(model, n_passes=30, is_torch=True)
    X = torch.randn(4, 14)
    res = uq.predict_with_uncertainty(X)
    assert np.all(res["sigma"] >= 0.0)

def test_mc_dropout_numpy_model():
    class DummyNumpyModel:
        def predict(self, X):
            return np.ones((X.shape[0], 6)) * 1448.0
    uq = MonteCarloDropout(DummyNumpyModel(), n_passes=50, is_torch=False)
    X = np.random.randn(3, 14)
    res = uq.predict_with_uncertainty(X)
    assert res["mean"].shape == (3, 6)
    assert np.all(res["min_guaranteed"] <= res["mean"])

def test_margin_of_safety_approved():
    res = compute_margin_of_safety(min_guaranteed_uts=1420.0, applied_hoop_stress=1350.0)
    assert res["margin_of_safety"] > 0.0
    assert res["flight_clearance_granted"] is True
    assert "APPROVED" in res["status"]

def test_margin_of_safety_rejected():
    res = compute_margin_of_safety(min_guaranteed_uts=1300.0, applied_hoop_stress=1350.0)
    assert res["margin_of_safety"] < 0.0
    assert res["flight_clearance_granted"] is False
    assert "REJECTED" in res["status"]

def test_margin_of_safety_formula():
    allowable = 1485.0
    applied = 1350.0
    expected_ms = (allowable / applied) - 1.0
    res = compute_margin_of_safety(allowable, applied)
    assert abs(res["margin_of_safety"] - expected_ms) < 1e-5
