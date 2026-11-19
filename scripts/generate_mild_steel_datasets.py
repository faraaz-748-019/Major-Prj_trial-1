"""
scripts/generate_mild_steel_datasets.py
Generates calibrated datasets for Mild Steel (MS) Welded vs Unwelded Mechanical Property Prediction:
1. data/experimental_utm_mild_steel_records.csv (11 verified laboratory UTM test records from Vasavi College of Eng.)
2. data/mild_steel_weld_dataset_4000_samples.csv (4,000 specimen calibrated dataset)

Ground truth values directly derived from UTM tests:
- Specimen geometry: 30 mm width, 3 mm thickness, 90.0 mm^2 area.
- Unwelded base metal UTS: ~934 - 956 MPa, Elongation: ~27.9 - 31.1%
- Welded joint UTS: ~913 - 926 MPa, Elongation: ~13.9 - 22.6%
- Max Force: Fm = UTS * 90.0 N (82,200 - 86,070 N)
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_experimental_records_csv():
    records = [
        {"Record_No": 1, "Date": "30-04-2026", "Specimen_ID": "Specimen-7", "Condition": "Welded", "Gauge_Length_L0_mm": 175.0, "Final_Length_Lf_mm": 204.2, "Width_mm": 30.0, "Thickness_mm": 3.0, "Area_mm2": 90.0, "Max_Force_Fm_N": 82200.0, "Disp_at_Fm_mm": 8.9, "Max_Disp_mm": 11.9, "UTS_MPa": 913.333, "Elongation_Pct": 16.667, "Yield_Strength_Re_MPa": 913.0, "Failure_Location": "Weld Region", "Remarks": "Specimen failed at weld region, Re/Rm=1.000"},
        {"Record_No": 2, "Date": "08-05-2026", "Specimen_ID": "Sample-1", "Condition": "Welded", "Gauge_Length_L0_mm": 180.0, "Final_Length_Lf_mm": 205.0, "Width_mm": 30.0, "Thickness_mm": 3.0, "Area_mm2": 90.0, "Max_Force_Fm_N": 83190.0, "Disp_at_Fm_mm": 16.3, "Max_Disp_mm": 16.9, "UTS_MPa": 924.333, "Elongation_Pct": 13.889, "Yield_Strength_Re_MPa": 700.0, "Failure_Location": "HAZ / Weld", "Remarks": "Lowest elongation (13.89%), brittle rupture at 16.9 mm"},
        {"Record_No": 3, "Date": "08-05-2026", "Specimen_ID": "Sample-2", "Condition": "Unwelded", "Gauge_Length_L0_mm": 170.0, "Final_Length_Lf_mm": 220.0, "Width_mm": 30.0, "Thickness_mm": 3.0, "Area_mm2": 90.0, "Max_Force_Fm_N": 86070.0, "Disp_at_Fm_mm": 23.7, "Max_Disp_mm": 29.4, "UTS_MPa": 956.333, "Elongation_Pct": 29.412, "Yield_Strength_Re_MPa": 690.0, "Failure_Location": "Base Metal", "Remarks": "Peak tensile strength (956.3 MPa), ductile cup-and-cone"},
        {"Record_No": 4, "Date": "08-05-2026", "Specimen_ID": "Sample-3", "Condition": "Unwelded", "Gauge_Length_L0_mm": 175.0, "Final_Length_Lf_mm": 227.0, "Width_mm": 30.0, "Thickness_mm": 3.0, "Area_mm2": 90.0, "Max_Force_Fm_N": 84060.0, "Disp_at_Fm_mm": 24.2, "Max_Disp_mm": 27.6, "UTS_MPa": 934.000, "Elongation_Pct": 29.714, "Yield_Strength_Re_MPa": 680.0, "Failure_Location": "Base Metal", "Remarks": "Standard ductile failure"},
        {"Record_No": 5, "Date": "08-05-2026", "Specimen_ID": "Sample-4", "Condition": "Unwelded", "Gauge_Length_L0_mm": 172.0, "Final_Length_Lf_mm": 220.0, "Width_mm": 30.0, "Thickness_mm": 3.0, "Area_mm2": 90.0, "Max_Force_Fm_N": 83280.0, "Disp_at_Fm_mm": 20.6, "Max_Disp_mm": 28.3, "UTS_MPa": 925.333, "Elongation_Pct": 27.907, "Yield_Strength_Re_MPa": 675.0, "Failure_Location": "Base Metal", "Remarks": "Uniform yield plateau"},
        {"Record_No": 6, "Date": "08-05-2026", "Specimen_ID": "Sample-5", "Condition": "Unwelded", "Gauge_Length_L0_mm": 173.0, "Final_Length_Lf_mm": 213.0, "Width_mm": 30.0, "Thickness_mm": 3.0, "Area_mm2": 90.0, "Max_Force_Fm_N": 84030.0, "Disp_at_Fm_mm": 20.9, "Max_Disp_mm": 23.0, "UTS_MPa": 933.667, "Elongation_Pct": 23.121, "Yield_Strength_Re_MPa": 685.0, "Failure_Location": "Base Metal", "Remarks": "Moderate base ductility"},
        {"Record_No": 7, "Date": "08-05-2026", "Specimen_ID": "Sample-6", "Condition": "Unwelded", "Gauge_Length_L0_mm": 172.0, "Final_Length_Lf_mm": 225.0, "Width_mm": 30.0, "Thickness_mm": 3.0, "Area_mm2": 90.0, "Max_Force_Fm_N": 85110.0, "Disp_at_Fm_mm": 25.8, "Max_Disp_mm": 30.8, "UTS_MPa": 945.667, "Elongation_Pct": 30.814, "Yield_Strength_Re_MPa": 695.0, "Failure_Location": "Base Metal", "Remarks": "High elongation and strength"},
        {"Record_No": 8, "Date": "08-05-2026", "Specimen_ID": "Sample-8", "Condition": "Unwelded", "Gauge_Length_L0_mm": 175.0, "Final_Length_Lf_mm": 225.0, "Width_mm": 30.0, "Thickness_mm": 3.0, "Area_mm2": 90.0, "Max_Force_Fm_N": 84510.0, "Disp_at_Fm_mm": 23.8, "Max_Disp_mm": 29.5, "UTS_MPa": 939.000, "Elongation_Pct": 28.571, "Yield_Strength_Re_MPa": 688.0, "Failure_Location": "Base Metal", "Remarks": "Consistent parent metal"},
        {"Record_No": 9, "Date": "08-05-2026", "Specimen_ID": "Sample-9", "Condition": "Unwelded", "Gauge_Length_L0_mm": 176.0, "Final_Length_Lf_mm": 230.0, "Width_mm": 30.0, "Thickness_mm": 3.0, "Area_mm2": 90.0, "Max_Force_Fm_N": 82770.0, "Disp_at_Fm_mm": 26.2, "Max_Disp_mm": 34.7, "UTS_MPa": 919.667, "Elongation_Pct": 30.682, "Yield_Strength_Re_MPa": 670.0, "Failure_Location": "Base Metal", "Remarks": "Maximum displacement (34.7 mm)"},
        {"Record_No": 10, "Date": "08-05-2026", "Specimen_ID": "Sample-10-Unwelded", "Condition": "Unwelded", "Gauge_Length_L0_mm": 177.0, "Final_Length_Lf_mm": 232.0, "Width_mm": 30.0, "Thickness_mm": 3.0, "Area_mm2": 90.0, "Max_Force_Fm_N": 84240.0, "Disp_at_Fm_mm": 25.6, "Max_Disp_mm": 34.2, "UTS_MPa": 936.000, "Elongation_Pct": 31.073, "Yield_Strength_Re_MPa": 685.0, "Failure_Location": "Base Metal", "Remarks": "Highest elongation (31.07%), pairwise baseline"},
        {"Record_No": 11, "Date": "08-05-2026", "Specimen_ID": "Sample-10-Welded", "Condition": "Welded", "Gauge_Length_L0_mm": 195.0, "Final_Length_Lf_mm": 239.0, "Width_mm": 30.0, "Thickness_mm": 3.0, "Area_mm2": 90.0, "Max_Force_Fm_N": 83370.0, "Disp_at_Fm_mm": 25.7, "Max_Disp_mm": 30.8, "UTS_MPa": 926.333, "Elongation_Pct": 22.564, "Yield_Strength_Re_MPa": 710.0, "Failure_Location": "Weld Toe / HAZ", "Remarks": "Direct welded comparison: -27.4% elongation drop"}
    ]
    df = pd.DataFrame(records)
    out_path = os.path.join("data", "experimental_utm_mild_steel_records.csv")
    df.to_csv(out_path, index=False)
    print(f"Experimental UTM records saved to {out_path} ({len(df)} records)")
    return df

def generate_calibrated_mild_steel_dataset(n_samples=4000):
    specimen_ids = [f"MS-COUPON-{i+1:04d}" for i in range(n_samples)]
    
    # Base Metal Properties (calibrated to unwelded UTM tests: mean UTS ~936 MPa, std ~11 MPa)
    base_uts = np.random.normal(938.0, 10.5, size=n_samples)
    base_ys = base_uts * np.random.uniform(0.71, 0.76, size=n_samples)
    base_elongation = np.random.normal(29.2, 1.6, size=n_samples)
    
    # Specimen Geometry
    gauge_length = np.random.choice([170.0, 172.0, 173.0, 175.0, 176.0, 177.0, 180.0, 195.0], size=n_samples)
    width = np.random.normal(30.0, 0.08, size=n_samples)
    thickness = np.random.normal(3.0, 0.04, size=n_samples)
    area = width * thickness  # ~90.0 mm^2
    
    # Welding Process Parameters (TIG / SMAW / GMAW)
    process = np.random.choice(["TIG", "SMAW", "GMAW"], size=n_samples, p=[0.60, 0.25, 0.15])
    current = np.where(process == "TIG", np.random.normal(130.0, 10.0, n_samples), np.random.normal(145.0, 12.0, n_samples))
    voltage = np.where(process == "TIG", np.random.normal(20.0, 1.2, n_samples), np.random.normal(23.0, 1.5, n_samples))
    heat_input = (current * voltage * 0.06) / np.random.uniform(120.0, 160.0, n_samples) # kJ/mm (~0.75 - 1.20)
    
    haz_width = 2.5 + (heat_input * 2.2) + np.random.normal(0.0, 0.2, n_samples) # mm (~4.2 - 5.5 mm)
    pass_count = np.random.choice([1, 2], size=n_samples, p=[0.7, 0.3])
    
    joint_types = np.random.choice(["Square_Butt", "Double_V", "Single_V"], size=n_samples, p=[0.4, 0.35, 0.25])
    
    # Welded Properties Calibration (Welded UTS ~ 913 - 930 MPa, Elongation ~ 13.8 - 22.6%)
    # Joint strength retention factor ~ 0.97 - 0.99
    uts_retention = 0.988 - (heat_input * 0.009) - (pass_count * 0.004) + np.random.normal(0.0, 0.003, n_samples)
    welded_uts = base_uts * uts_retention
    
    # Yield strength in welded region is slightly elevated due to thermal grain refinement / residual stress (700 - 740 MPa)
    welded_ys = np.minimum(welded_uts * np.random.uniform(0.75, 0.81, n_samples), welded_uts - 12.0)
    
    # Ductility drop: Welded elongation collapses by 25% to 55% as proven by UTM tests (Sample 1: 13.89%, Sample 10: 22.56%)
    elong_retention = 0.72 - (heat_input * 0.12) - np.where(process == "TIG", 0.0, 0.06) + np.random.normal(0.0, 0.04, n_samples)
    welded_elong = np.clip(base_elongation * elong_retention, 11.5, 24.5)
    
    # Max Force Fm (N) = Welded UTS * Cross-Sectional Area
    max_force_fm = welded_uts * area
    
    # Displacement at Fm (mm): Welded specimens exhibit lower peak displacement (8.9 - 25.7 mm)
    disp_fm = (welded_elong / 100.0) * gauge_length * np.random.uniform(0.65, 0.85, n_samples)
    
    # Joint Efficiency (%)
    joint_eff = (welded_uts / base_uts) * 100.0
    
    df = pd.DataFrame({
        "Specimen_ID": specimen_ids,
        "Base_UTS": np.round(base_uts, 2),
        "Base_YS": np.round(base_ys, 2),
        "Base_Elongation": np.round(base_elongation, 2),
        "Gauge_Length_L0": np.round(gauge_length, 1),
        "Specimen_Width": np.round(width, 2),
        "Specimen_Thickness": np.round(thickness, 2),
        "Weld_Current_A": np.round(current, 1),
        "Weld_Voltage_V": np.round(voltage, 1),
        "Heat_Input_kJ_mm": np.round(heat_input, 3),
        "HAZ_Width_mm": np.round(haz_width, 2),
        "Pass_Count": pass_count,
        "Joint_Type_DoubleV": (joint_types == "Double_V").astype(int),
        "Weld_Process_TIG": (process == "TIG").astype(int),
        "Failure_Location_Weld": np.where(joint_eff < 97.5, 1, 0),
        
        # 6 Target properties
        "Welded_UTS": np.round(welded_uts, 2),
        "Welded_YS": np.round(welded_ys, 2),
        "Welded_Elongation": np.round(welded_elong, 2),
        "Max_Force_Fm": np.round(max_force_fm, 1),
        "Disp_at_Fm": np.round(disp_fm, 2),
        "Joint_Efficiency_Pct": np.round(joint_eff, 2)
    })
    
    out_path = os.path.join("data", "mild_steel_weld_dataset_4000_samples.csv")
    df.to_csv(out_path, index=False)
    print(f"Mild Steel 4,000 dataset saved to {out_path} ({df.shape})")
    return df

if __name__ == "__main__":
    generate_experimental_records_csv()
    generate_calibrated_mild_steel_dataset(4000)
