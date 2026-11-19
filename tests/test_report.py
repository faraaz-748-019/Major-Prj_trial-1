"""
tests/test_report.py
Unit tests for ReportLab automated PDF flight qualification certificate generation.
"""

import os
import pytest
from core.report_generator import build_pdf_certificate, generate_curve_image
from core.uncertainty import compute_margin_of_safety

def test_generate_curve_image_buffer():
    import numpy as np
    strains = np.linspace(0, 18000, 200)
    stresses = np.linspace(0, 1450, 200)
    buf = generate_curve_image(strains, stresses, min_guaranteed_uts=1411.0)
    assert buf is not None
    assert buf.getbuffer().nbytes > 1000

def test_build_pdf_certificate_creates_file(tmp_path):
    out_pdf = str(tmp_path / "test_certificate.pdf")
    specimen_data = {"specimen_id": "TEST-001", "stage": "S200", "location": "Root", "depth": 2.1}
    predictions = {"uts": 1448.0, "ys": 1366.0, "elongation": 10.2, "haz_width": 8.2, "depth": 2.35, "austenite": 13.8}
    uncertainty_data = {"min_guaranteed_uts": 1411.0, "sigma_uts": 18.5}
    margin_data = compute_margin_of_safety(1411.0, 1350.0)

    res = build_pdf_certificate(out_pdf, specimen_data, predictions, uncertainty_data, margin_data)
    assert os.path.exists(out_pdf)
    assert os.path.getsize(out_pdf) > 5000

def test_build_pdf_certificate_custom_curve(tmp_path):
    out_pdf = str(tmp_path / "test_custom_curve.pdf")
    specimen_data = {"specimen_id": "TEST-002", "stage": "S139", "location": "Center", "depth": 1.8}
    predictions = {"uts": 1440.0, "ys": 1350.0, "elongation": 11.0, "haz_width": 7.8, "depth": 2.0, "austenite": 12.5}
    uncertainty_data = {"min_guaranteed_uts": 1400.0, "sigma_uts": 20.0}
    margin_data = compute_margin_of_safety(1400.0, 1350.0)
    curve_data = {"strain": [0, 5000, 15000], "stress": [0, 950, 1440]}

    build_pdf_certificate(out_pdf, specimen_data, predictions, uncertainty_data, margin_data, curve_data=curve_data)
    assert os.path.exists(out_pdf)

def test_build_pdf_certificate_rejected_status(tmp_path):
    out_pdf = str(tmp_path / "test_rejected.pdf")
    specimen_data = {"specimen_id": "TEST-REJECTED", "stage": "S139", "location": "Root", "depth": 3.5}
    predictions = {"uts": 1320.0, "ys": 1250.0, "elongation": 6.5, "haz_width": 11.0, "depth": 3.7, "austenite": 20.0}
    uncertainty_data = {"min_guaranteed_uts": 1280.0, "sigma_uts": 20.0}
    margin_data = compute_margin_of_safety(1280.0, 1350.0) # Negative MS

    build_pdf_certificate(out_pdf, specimen_data, predictions, uncertainty_data, margin_data)
    assert os.path.exists(out_pdf)

def test_build_pdf_certificate_default_values(tmp_path):
    out_pdf = str(tmp_path / "test_defaults.pdf")
    build_pdf_certificate(out_pdf, {}, {}, {}, {})
    assert os.path.exists(out_pdf)

def test_build_pdf_certificate_non_empty():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        temp_name = tf.name
    try:
        build_pdf_certificate(temp_name, {}, {}, {}, {})
        assert os.path.getsize(temp_name) > 0
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)

def test_generate_curve_image_dimensions():
    import numpy as np
    buf = generate_curve_image(np.linspace(0, 1000, 50), np.linspace(0, 500, 50), 450.0)
    from PIL import Image
    im = Image.open(buf)
    assert im.size[0] > 400
    assert im.size[1] > 200

def test_margin_of_safety_positive_threshold():
    m = compute_margin_of_safety(min_guaranteed_uts=1411.2, applied_hoop_stress=1350.0)
    assert m["margin_of_safety"] > 0.04
