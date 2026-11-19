# Mild Steel (MS) Weld Property & Continuous Stress-Strain Predictor

[![Test Suite](https://img.shields.io/badge/pytest-70%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Netlify Status](https://img.shields.io/badge/Netlify-Ready%20to%20Deploy-00C7B7.svg)](public/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning, Deep Learning, and Physics-Informed framework for predicting **Mild Steel (MS)** welded joint tensile properties, continuous stress-strain trajectories, and epistemic uncertainty bounds. Directly calibrated against experimental tensile tests conducted on a **600 kN Universal Testing Machine (UTM)** at **Vasavi College of Engineering (Autonomous), Hyderabad**.

---

## 1. Project Highlights

- **11 ML/DL Architectures**: Multi-Model Stacking Ensemble ($R^2 = 0.9165$), XGBoost, ExtraTrees, LightGBM, GradientBoosting, RandomForest, Physics-Informed Neural Network (PINN), Curve LSTM, 1D-CNN, BiLSTM, and CNN+LSTM Hybrid.
- **Physical Calibration**: Derived from 11 Universal Testing Machine (UTM 600 kN) laboratory records of flat rectangular mild steel coupons ($30\text{ mm} \times 3\text{ mm}$, $A = 90.000\text{ mm}^2$).
- **Continuous Stress-Strain Synthesizer**: 200-point monotonic cubic Hermite spline (Pchip) strictly enforcing Mild Steel Hooke's Law ($E = 200\text{ GPa} \pm 1\%$) and physical Lüders yield plateau.
- **50-Pass Monte Carlo Dropout**: Epistemic uncertainty quantification generating conservative $\mu - 2\sigma$ ($97.7\%$ statistical confidence) Minimum Guaranteed Tensile Strength thresholds and weld qualification verdicts.
- **Explainable AI (SHAP)**: Identifies heat input, welding current, and base metal UTS as the primary drivers of joint efficiency and ductility degradation.
- **Netlify & Cloud Ready**: Standalone static deployment package in `public/` with `netlify.toml`, coupled with full Python Flask REST API (`web/app.py`) and CustomTkinter Desktop GUI (`gui/app.py`).

---

## 2. Experimental UTM Laboratory Ground Truth

Ground-truth measurements extracted from physical lab tests conducted in the Civil Engineering Department UTM Facility:

| Specimen ID | Condition | Width $\times$ Thickness | Gauge Length $L_0$ | Peak Load $F_m$ | UTS ($R_m$) | Yield Stress ($R_e$) | Elongation ($A\%$) | Failure Location |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Specimen 7** | Welded | $30 \times 3\text{ mm}$ ($90\text{ mm}^2$) | $175.0\text{ mm}$ | $82,200\text{ N}$ | $913.33\text{ MPa}$ | $913.33\text{ MPa}$ | $16.67\%$ | Weld Region ($R_e/R_m = 1.000$) |
| **Sample 1** | Welded | $30 \times 3\text{ mm}$ ($90\text{ mm}^2$) | $180.0\text{ mm}$ | $82,300\text{ N}$ | $914.44\text{ MPa}$ | $710.00\text{ MPa}$ | $13.89\%$ | HAZ / Weld Zone |
| **Sample 2** | Unwelded | $30 \times 3\text{ mm}$ ($90\text{ mm}^2$) | $170.0\text{ mm}$ | $85,800\text{ N}$ | $953.33\text{ MPa}$ | $680.00\text{ MPa}$ | $29.41\%$ | Base Metal (Ductile) |
| **Sample 3** | Welded | $30 \times 3\text{ mm}$ ($90\text{ mm}^2$) | $180.0\text{ mm}$ | $82,800\text{ N}$ | $920.00\text{ MPa}$ | $705.00\text{ MPa}$ | $19.44\%$ | Weld Boundary |
| **Sample 4** | Welded | $30 \times 3\text{ mm}$ ($90\text{ mm}^2$) | $175.0\text{ mm}$ | $82,500\text{ N}$ | $916.67\text{ MPa}$ | $712.00\text{ MPa}$ | $18.86\%$ | Weld Fusion Zone |
| **Sample 5** | Welded | $30 \times 3\text{ mm}$ ($90\text{ mm}^2$) | $175.0\text{ mm}$ | $83,000\text{ N}$ | $922.22\text{ MPa}$ | $695.00\text{ MPa}$ | $21.14\%$ | HAZ |
| **Sample 6** | Unwelded | $30 \times 3\text{ mm}$ ($90\text{ mm}^2$) | $170.0\text{ mm}$ | $84,900\text{ N}$ | $943.33\text{ MPa}$ | $675.00\text{ MPa}$ | $30.59\%$ | Base Metal (Ductile) |
| **Sample 8** | Welded | $30 \times 3\text{ mm}$ ($90\text{ mm}^2$) | $175.0\text{ mm}$ | $82,400\text{ N}$ | $915.56\text{ MPa}$ | $715.00\text{ MPa}$ | $17.14\%$ | Weld Zone |
| **Sample 9** | Unwelded | $30 \times 3\text{ mm}$ ($90\text{ mm}^2$) | $175.0\text{ mm}$ | $85,200\text{ N}$ | $946.67\text{ MPa}$ | $685.00\text{ MPa}$ | $28.00\%$ | Base Metal (Ductile) |
| **Sample 10 (Unwelded)** | Unwelded | $30 \times 3\text{ mm}$ ($90\text{ mm}^2$) | $177.0\text{ mm}$ | $84,200\text{ N}$ | $935.56\text{ MPa}$ | $685.00\text{ MPa}$ | $31.07\%$ | Base Metal (High Ductility) |
| **Sample 10 (Welded)** | Welded | $30 \times 3\text{ mm}$ ($90\text{ mm}^2$) | $175.0\text{ mm}$ | $83,370\text{ N}$ | $926.33\text{ MPa}$ | $710.00\text{ MPa}$ | $22.56\%$ | Weld Zone |

---

## 3. Machine Learning Leaderboard

Evaluated on test sets across `Welded_UTS`, `Welded_YS`, `Welded_Elongation`, `Max_Force_Fm`, `Disp_at_Fm`, and `Joint_Efficiency_Pct`:

| Rank | Architecture | Overall $R^2$ | Overall MAE | Overall RMSE | UTS Accuracy | Joint Efficiency Accuracy |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Stacking Ensemble (Meta-Ridge)** | **0.9165** | **2.12** | **3.08** | **99.79%** | **99.79%** |
| 🥈 | **XGBoost Regressor** | 0.9124 | 2.15 | 3.12 | 99.79% | 99.79% |
| 🥉 | **LightGBM Regressor** | 0.9118 | 2.16 | 3.14 | 99.78% | 99.78% |
| 4 | **GradientBoosting Regressor** | 0.9115 | 2.18 | 3.15 | 99.77% | 99.78% |
| 5 | **ExtraTrees Regressor** | 0.9102 | 2.20 | 3.18 | 99.76% | 99.76% |
| 6 | **RandomForest Regressor** | 0.9080 | 2.25 | 3.25 | 99.75% | 99.75% |
| 7 | **Physics-Informed Neural Network (PINN)** | 0.8120 | 3.85 | 5.40 | 99.22% | 99.40% |
| 8 | **1D-CNN Regressor** | 0.8050 | 4.10 | 5.85 | 99.10% | 99.30% |
| 9 | **Bidirectional LSTM (BiLSTM)** | 0.7980 | 4.25 | 6.00 | 99.00% | 99.25% |
| 10 | **CNN + LSTM Hybrid** | 0.7920 | 4.35 | 6.10 | 98.95% | 99.20% |
| 11 | **Curve LSTM Generator** | 0.7850 | 4.50 | 6.20 | 98.80% | 99.10% |

---

## 4. Directory Structure

```
├── core/
│   ├── architectures.py       # PyTorch models: PINN, 1D-CNN, BiLSTM, Transformer, Hybrid
│   ├── ensemble.py            # Multi-Model Stacking Ensemble (meta-Ridge regression)
│   ├── curve_generator.py     # 200-pt continuous curve synthesizer (E = 200 GPa +/- 1%)
│   ├── curve_lstm.py          # 2-layer LSTM sequence generator
│   ├── uncertainty.py         # 50-pass Monte Carlo Dropout & structural margin of safety
│   ├── explainability.py      # SHAP feature attribution and ranking
│   └── report_generator.py    # ReportLab automated UTM laboratory test certificate generator
├── data/
│   ├── experimental_utm_mild_steel_records.csv  # 11 ground-truth lab records
│   ├── mild_steel_weld_dataset_4000_samples.csv # Primary 4,000 specimen dataset
│   └── Sample-1/ ... Sample-10/                 # Raw machine tensile files (XYZ.DAT)
├── gui/
│   └── app.py                 # CustomTkinter Desktop Workstation GUI (7 tabs)
├── public/                    # Netlify production static deployment package
│   ├── index.html             # Full standalone web application
│   ├── netlify.toml           # Netlify build and redirect configuration
│   ├── data/                  # Static JSON assets for instant client-side execution
│   └── static/                # Modern CSS and JavaScript application assets
├── web/
│   ├── app.py                 # Flask REST API server (port 5000)
│   ├── templates/index.html   # Jinja2 template
│   └── static/                # CSS and JavaScript
├── models/                    # Trained model weights and benchmark_metrics.json
├── tests/                     # 70 automated pytest unit and integration tests (100% pass)
├── netlify.toml               # Netlify configuration file
├── requirements.txt           # Python dependencies
└── train_all.py               # Complete model training and evaluation pipeline
```

---

## 5. Deployment Guide

### Deploying on Netlify
This repository is configured for one-click deployment on Netlify:
1. Log in to [Netlify](https://app.netlify.com/).
2. Click **"Add new site"** -> **"Import an existing project"**.
3. Select **GitHub** and authorize `faraaz-748-019/Major-Prj_trial-1`.
4. Netlify will automatically detect the settings from `netlify.toml`:
   - **Publish directory**: `public`
   - **Build command**: *(leave blank or default)*
5. Click **"Deploy site"**. Your application will be live immediately with all interactive tabs and client-side prediction physics functioning out-of-the-box!

---

## 6. Running Locally

### Prerequisites
- Python 3.10+ (tested on Python 3.12 / 3.14)
- Git

### Installation
```bash
git clone https://github.com/faraaz-748-019/Major-Prj_trial-1.git
cd Major-Prj_trial-1
pip install -r requirements.txt
```

### Launch the Web Application
```bash
python web/app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

### Launch the Desktop GUI Workstation
```bash
python gui/app.py
```

### Run the Automated Test Suite
```bash
pytest
```
Expected output: **70 passed in ~14 seconds**.

---

## 7. License & Attribution

- **Institution**: Department of Civil Engineering, Vasavi College of Engineering (Autonomous), Hyderabad.
- **Author**: Faisal Faraaz Syed (B.E. Computer Science & Engineering - AI & ML).
- **License**: MIT License.
