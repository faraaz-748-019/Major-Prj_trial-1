"""
tests/test_curve.py
Unit tests for continuous stress-strain trajectory synthesis, Pchip interpolation,
and Hooke's Law elastic modulus enforcement (E = 190 GPa +/- 1%).
"""

import pytest
import numpy as np
import torch
from core.curve_generator import (
    generate_curve_landmarks, reconstruct_r2_curve, generate_reference_curves, E_NOMINAL
)
from core.curve_lstm import CurveLSTMGenerator

def test_generate_curve_landmarks_count():
    strains, stresses = generate_curve_landmarks(uts=925.0, ys=700.0, elongation=20.0)
    assert len(strains) == 7
    assert len(stresses) == 7

def test_generate_curve_landmarks_monotonic_strains():
    strains, _ = generate_curve_landmarks(uts=925.0, ys=700.0, elongation=20.0)
    assert np.all(np.diff(strains) > 0)

def test_generate_curve_landmarks_yield_below_uts():
    _, stresses = generate_curve_landmarks(uts=925.0, ys=700.0, elongation=20.0)
    ys_val = stresses[2]
    uts_val = stresses[5]
    assert ys_val < uts_val

def test_reconstruct_r2_curve_length():
    strains, stresses = reconstruct_r2_curve(uts=925.0, ys=700.0, elongation=20.0, n_points=200)
    assert len(strains) == 200
    assert len(stresses) == 200

def test_reconstruct_r2_curve_elastic_slope_190gpa():
    # DEF-003 regression test: strictly enforces Hooke's Law E = 200 GPa +/- 1%
    strains, stresses = reconstruct_r2_curve(uts=925.0, ys=700.0, elongation=20.0, n_points=200)
    # Point index 10 is well within the linear elastic proportional zone
    # Strain is in microstrain (1e-6), so slope = delta_stress / (delta_strain * 1e-6)
    delta_stress = stresses[10] - stresses[0]
    delta_strain = (strains[10] - strains[0]) / 1e6
    measured_e = delta_stress / delta_strain
    relative_err = abs(measured_e - E_NOMINAL) / E_NOMINAL
    assert relative_err < 0.01, f"Elastic modulus error: {relative_err*100:.3f}% (measured {measured_e:.1f} vs nominal {E_NOMINAL})"

def test_reconstruct_r2_curve_starts_at_zero():
    strains, stresses = reconstruct_r2_curve(uts=925.0, ys=700.0, elongation=20.0)
    assert strains[0] == 0.0
    assert stresses[0] == 0.0

def test_reconstruct_r2_curve_peak_matches_uts():
    uts_target = 925.0
    strains, stresses = reconstruct_r2_curve(uts=uts_target, ys=700.0, elongation=20.0)
    peak_stress = np.max(stresses)
    assert abs(peak_stress - uts_target) < 1.0

def test_reconstruct_r2_curve_pchip_no_nan():
    strains, stresses = reconstruct_r2_curve(uts=925.0, ys=700.0, elongation=20.0)
    assert not np.isnan(strains).any()
    assert not np.isnan(stresses).any()

def test_generate_reference_curves_keys():
    refs = generate_reference_curves()
    assert len(refs) > 0
    first_key = list(refs.keys())[0]
    assert "strain" in refs[first_key]
    assert "stress" in refs[first_key]
    assert len(refs[first_key]["strain"]) == 200

def test_curve_lstm_generator_shape():
    cl = CurveLSTMGenerator(input_dim=14, hidden_dim=64, num_layers=2, output_dim=6)
    x = torch.randn(4, 14)
    out = cl(x)
    assert out.shape == (4, 6)
