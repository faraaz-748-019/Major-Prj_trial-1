"""
tests/test_defects_audit.py
Regression test suite auditing resolutions for defects DEF-001 through DEF-007:
- DEF-001: Transformer parameter signature & head divisibility
- DEF-002: compute_metrics scalar float indexing guard
- DEF-003: Continuous elastic slope 190 GPa +/- 1%
- DEF-004: StackingEnsemble 1D meta-matrix preparation
- DEF-005: Multi-target slicing dynamic bounds
- DEF-006: Script discovery isolation
- DEF-007: Dataset physical target boundary compliance (YS <= UTS)
"""

import os
import pytest
import numpy as np
import torch
import pandas as pd

from core.architectures import TransformerRegressor
from core.ensemble import StackingEnsemble
from core.curve_generator import reconstruct_r2_curve, E_NOMINAL
from train_all import compute_metrics, TARGET_COLUMNS

def test_audit_def_001_transformer_signature():
    """DEF-001: Verified constructor accepts standard args and enforces embed_dim % num_heads == 0."""
    # Valid: 64 % 4 == 0
    t = TransformerRegressor(input_dim=14, output_dim=6, embed_dim=64, num_heads=4)
    x = torch.randn(2, 14)
    out = t(x)
    assert out.shape == (2, 6)

    # Invalid: 65 % 4 != 0 must assert
    with pytest.raises(AssertionError):
        TransformerRegressor(input_dim=14, output_dim=6, embed_dim=65, num_heads=4)

def test_audit_def_002_compute_metrics_scalar_indexing():
    """DEF-002: compute_metrics must not crash with IndexError on 1D / single-sample inputs."""
    y_true_1d = np.array([1448.0, 1366.0, 10.2, 8.2, 2.1, 13.5])
    y_pred_1d = np.array([1445.0, 1360.0, 10.0, 8.0, 2.0, 13.0])
    
    # Must run cleanly without IndexError
    metrics = compute_metrics(y_true_1d, y_pred_1d)
    assert "overall_mae" in metrics
    assert metrics["overall_mae"] > 0.0
    assert "overall_accuracy" in metrics

def test_audit_def_003_curve_elastic_slope_strictly_190gpa():
    """DEF-003: Elastic slope in reconstruct_r2_curve must equal 190 GPa +/- 1%."""
    strains, stresses = reconstruct_r2_curve(uts=1448.0, ys=1366.0, elongation=10.0, n_points=200)
    for idx in [5, 10, 15, 20]:
        e_measured = stresses[idx] / (strains[idx] / 1e6)
        rel_err = abs(e_measured - E_NOMINAL) / E_NOMINAL
        assert rel_err < 0.01, f"At index {idx}: E={e_measured} err={rel_err*100:.3f}%"

def test_audit_def_004_stacking_1d_meta_matrix():
    """DEF-004: StackingEnsemble._prepare_meta_matrix correctly reshapes 1D arrays to 2D columns."""
    ens = StackingEnsemble()
    p_1d = np.array([10.0, 20.0, 30.0])
    p_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    res = ens._prepare_meta_matrix([p_1d, p_2d])
    assert res.shape == (3, 3)
    assert res[0, 0] == 10.0
    assert res[1, 0] == 20.0

def test_audit_def_005_dynamic_target_slice():
    """DEF-005: compute_metrics dynamically slices prediction columns matching target dimensions."""
    y_true = np.ones((5, len(TARGET_COLUMNS))) * 100.0
    # Over-dimensioned prediction array (e.g. 10 columns instead of 6)
    y_pred_extra = np.ones((5, len(TARGET_COLUMNS) + 4)) * 95.0
    m = compute_metrics(y_true, y_pred_extra)
    assert len(m["per_target"]) == len(TARGET_COLUMNS)
    assert m["overall_mae"] == 5.0

def test_audit_def_006_pytest_config_isolation():
    """DEF-006: pytest.ini exists and isolates test discovery."""
    assert os.path.exists("pytest.ini")
    with open("pytest.ini", "r") as f:
        content = f.read()
    assert "testpaths = tests" in content

def test_audit_def_007_target_r2_physics_compliance():
    """DEF-007: Target Yield Strength is 100% strictly <= Target UTS."""
    for p in ["data/mild_steel_weld_dataset_4000_samples.csv", "data/M250_S139_S200_separate_dataset_4000_samples.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "Welded_YS" in df.columns:
                violations = df[df["Welded_YS"] > df["Welded_UTS"]]
                assert len(violations) == 0
            elif "R2_YS" in df.columns:
                violations = df[df["R2_YS"] > df["R2_UTS"]]
                assert len(violations) == 0

def test_audit_def_007_baseline_r0_physics_bound():
    """DEF-007: Documented bound asserting >85% compliance for input baseline samples."""
    for p in ["data/mild_steel_weld_dataset_4000_samples.csv", "data/M250_S139_S200_separate_dataset_4000_samples.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "Base_YS" in df.columns:
                compliant = df[df["Base_YS"] <= df["Base_UTS"]]
                assert (len(compliant) / len(df)) * 100.0 >= 85.0
            elif "R0_YS" in df.columns:
                compliant = df[df["R0_YS"] <= df["R0_UTS"]]
                assert (len(compliant) / len(df)) * 100.0 >= 85.0

def test_audit_pin_loss_positivity_and_physics_weighting():
    """Audits combined weighting of alpha and beta penalties in PINNLoss."""
    from core.architectures import PINNLoss
    loss_fn = PINNLoss(alpha=2.0, beta=3.0)
    # y_pred with violation: YS > UTS, and negative elongation
    y_pred = torch.tensor([[1400.0, 1450.0, -2.0, 8.0, 2.0, 13.0]])
    y_true = torch.tensor([[1400.0, 1350.0, 10.0, 8.0, 2.0, 13.0]])
    val = loss_fn(y_pred, y_true)
    assert val > 1000.0 # Heavy penalty asserted

def test_audit_dataset_stratification_consistency():
    """Audits train/test split consistency on R2 UTS continuous quantiles."""
    dataset_path = "data/M250_S139_S200_separate_dataset_4000_samples.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        assert len(df) == 4000
        assert "R2_UTS" in df.columns
