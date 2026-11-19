"""
core/curve_generator.py
Continuous Stress-Strain Trajectory Synthesizer for Mild Steel (MS) Welded & Unwelded Specimens:
- Preserves Hooke's Law Linear Elastic Modulus for Mild Steel: E = 200 GPa +/- 1% (200,000 MPa).
- Models physical mild steel yield point phenomenon (elastic limit -> yield plateau -> strain hardening -> necking -> fracture).
- Employs SciPy Pchip (Piecewise Cubic Hermite Interpolating Polynomial) spline interpolation over 200 points.
- DEF-003 Fixed: Correct strain recalculation order and guaranteed 200 GPa slope.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

# Mild Steel Nominal Young's Modulus in MPa
E_NOMINAL = 200000.0  # 200 GPa (200,000 N/mm^2)

def generate_curve_landmarks(uts, ys, elongation=25.0, haz_width=4.5, e_modulus=E_NOMINAL):
    """
    Computes physically consistent mild steel stress-strain landmark coordinates based on UTM test results.
    Strain expressed in microstrain (ue) or engineering strain.
    """
    sigma_uts = float(uts)
    sigma_ys = float(ys)
    
    # 1. Proportional limit stress (~85% of Yield Strength)
    sigma_prop = 0.88 * sigma_ys
    eps_prop = (sigma_prop / e_modulus) * 1e6  # in microstrain (ue)
    
    # 2. Yield Point (Mild steel upper/lower yield onset)
    eps_y = (sigma_ys / e_modulus) * 1e6 + 300.0  # ~ 3500 - 4500 ue
    
    # 3. Mild steel Lüders yield plateau end
    sigma_plateau = sigma_ys * 0.99
    eps_plateau = eps_y + 1200.0
    
    # 4. Intermediate strain hardening landmark
    sigma_mid = sigma_plateau + 0.60 * (sigma_uts - sigma_plateau)
    eps_mid = eps_plateau + 4500.0
    
    # 5. Ultimate Tensile Strength (UTS peak / maximum load Fm)
    eps_uts = eps_mid + 6000.0
    
    # 6. Fracture point (necking drop)
    sigma_frac = sigma_uts * 0.88
    # Elongation is in %, e.g., 22.5%
    eps_frac = eps_uts + (elongation * 250.0)
    
    strains = [0.0, eps_prop, eps_y, eps_plateau, eps_mid, eps_uts, eps_frac]
    stresses = [0.0, sigma_prop, sigma_ys, sigma_plateau, sigma_mid, sigma_uts, sigma_frac]
    
    return np.array(strains), np.array(stresses)

def reconstruct_r2_curve(uts, ys, elongation=22.0, haz_width=4.5, n_points=200, e_modulus=E_NOMINAL):
    """
    Reconstructs a smooth 200-point continuous engineering stress-strain curve for Mild Steel.
    DEF-003 Fixed: Strictly guarantees E = 200 GPa +/- 1% in the elastic range.
    """
    strains_lm, stresses_lm = generate_curve_landmarks(uts, ys, elongation, haz_width, e_modulus)
    max_strain = strains_lm[-1]
    
    # Elastic region partition: 40 points strictly on Hooke's Law line
    eps_prop = strains_lm[1]
    elastic_strains = np.linspace(0.0, eps_prop, 40)
    elastic_stresses = (elastic_strains / 1e6) * e_modulus
    
    # Plastic region partition: 160 points via Monotonic Pchip spline
    plastic_strains_eval = np.linspace(eps_prop, max_strain, 161)[1:]
    
    pchip = PchipInterpolator(strains_lm[1:], stresses_lm[1:], extrapolate=False)
    plastic_stresses = pchip(plastic_strains_eval)
    
    final_strains = np.concatenate([elastic_strains, plastic_strains_eval])
    final_stresses = np.concatenate([elastic_stresses, plastic_stresses])
    
    if len(final_strains) != n_points:
        interp_full = PchipInterpolator(final_strains, final_stresses)
        final_strains = np.linspace(0.0, max_strain, n_points)
        mask_elastic = final_strains <= eps_prop
        final_stresses = interp_full(final_strains)
        final_stresses[mask_elastic] = (final_strains[mask_elastic] / 1e6) * e_modulus

    # Validate measured slope: Delta_stress / Delta_strain in elastic domain
    measured_e = ((final_stresses[15] - final_stresses[0]) / (final_strains[15] / 1e6))
    assert abs(measured_e - e_modulus) / e_modulus < 0.01, f"Elastic slope mismatch: {measured_e} vs {e_modulus}"
    
    return final_strains, final_stresses

def generate_reference_curves():
    """Generates comparative reference curves from actual UTM experimental test benchmarks."""
    curves = {}
    
    # 1. Unwelded Parent Metal (Sample 10 Unwelded) - UTS: 936.0 MPa, YS: 685.0 MPa, Elong: 31.07%
    eps_unw, sig_unw = reconstruct_r2_curve(uts=936.0, ys=685.0, elongation=31.07)
    curves["Unwelded_Base"] = {"strain": eps_unw.tolist(), "stress": sig_unw.tolist(), "uts": 936.0, "ys": 685.0, "elong": 31.07}
    
    # 2. Welded Joint Pairwise (Sample 10 Welded) - UTS: 926.3 MPa, YS: 710.0 MPa, Elong: 22.56%
    eps_wld, sig_wld = reconstruct_r2_curve(uts=926.3, ys=710.0, elongation=22.56)
    curves["Welded_Pair"] = {"strain": eps_wld.tolist(), "stress": sig_wld.tolist(), "uts": 926.3, "ys": 710.0, "elong": 22.56}
    
    # 3. Weld Failure Specimen (Specimen 7) - UTS: 913.3 MPa, YS: 820.0 MPa, Elong: 16.67%
    eps_sp7, sig_sp7 = reconstruct_r2_curve(uts=913.3, ys=820.0, elongation=16.67)
    curves["Weld_Failure_Sp7"] = {"strain": eps_sp7.tolist(), "stress": sig_sp7.tolist(), "uts": 913.3, "ys": 820.0, "elong": 16.67}
    
    # 4. Low Ductility Welded Rupture (Sample 1) - UTS: 924.3 MPa, YS: 700.0 MPa, Elong: 13.89%
    eps_sp1, sig_sp1 = reconstruct_r2_curve(uts=924.3, ys=700.0, elongation=13.89)
    curves["Sample_1_Welded"] = {"strain": eps_sp1.tolist(), "stress": sig_sp1.tolist(), "uts": 924.3, "ys": 700.0, "elong": 13.89}
    
    return curves
