"""
train_all.py
Comprehensive End-to-End Training and Benchmarking Pipeline for Mild Steel (MS) Welded Property Prediction:
- Loads data/mild_steel_weld_dataset_4000_samples.csv (calibrated against UTM test records).
- Performs continuous quantile stratification on Welded_UTS (80% train = 3,200 specimens, 20% test = 800 specimens, seed=42).
- Fits leakage-free StandardScaler on training split.
- Trains and benchmarks 11 predictive architectures (5 tree models, Stacking Ensemble, and 5 deep neural networks).
- DEF-002 Fixed: compute_metrics includes scalar dimensionality guards.
- DEF-005 Fixed: Dynamic multi-target slicing.
- Saves trained models and benchmark_metrics.json to models/ directory.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from core.architectures import (
    PINNRegressor, PINNLoss, CNNRegressor, BiLSTMRegressor,
    TransformerRegressor, CNNLSTMHybrid
)
from core.ensemble import StackingEnsemble
from core.curve_lstm import CurveLSTMGenerator

# Input features (14 features) and Target properties (6 targets)
INPUT_FEATURES = [
    "Base_UTS", "Base_YS", "Base_Elongation", "Gauge_Length_L0",
    "Specimen_Width", "Specimen_Thickness", "Weld_Current_A", "Weld_Voltage_V",
    "Heat_Input_kJ_mm", "HAZ_Width_mm", "Pass_Count", "Joint_Type_DoubleV",
    "Weld_Process_TIG", "Failure_Location_Weld"
]

TARGET_COLUMNS = [
    "Welded_UTS", "Welded_YS", "Welded_Elongation",
    "Max_Force_Fm", "Disp_at_Fm", "Joint_Efficiency_Pct"
]

def compute_metrics(y_true, y_pred):
    """
    Computes audited multi-target metrics.
    DEF-002 Fixed: Guards against scalar indexing errors when input is a 1-sample 1D array.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)

    n_samples, n_targets = y_true.shape
    
    # DEF-005 Fixed: Ensure exact target slice matching
    if y_pred.shape[1] > n_targets:
        y_pred = y_pred[:, :n_targets]

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0
    accuracy = float(np.clip(100.0 - mape, 0.0, 100.0))

    per_target = {}
    for i, target_name in enumerate(TARGET_COLUMNS[:n_targets]):
        t_true = y_true[:, i]
        t_pred = y_pred[:, i]
        t_mae = float(mean_absolute_error(t_true, t_pred))
        t_rmse = float(root_mean_squared_error(t_true, t_pred))
        t_r2 = float(r2_score(t_true, t_pred))
        t_mape = np.mean(np.abs((t_true - t_pred) / np.clip(np.abs(t_true), 1e-6, None))) * 100.0
        t_acc = float(np.clip(100.0 - t_mape, 0.0, 100.0))
        per_target[target_name] = {
            "mae": t_mae,
            "rmse": t_rmse,
            "r2": t_r2,
            "accuracy": t_acc
        }

    return {
        "overall_mae": mae,
        "overall_rmse": rmse,
        "overall_r2": r2,
        "overall_accuracy": accuracy,
        "uts_rmse": per_target.get("Welded_UTS", {}).get("rmse", rmse),
        "ys_rmse": per_target.get("Welded_YS", {}).get("rmse", rmse),
        "uts_acc": per_target.get("Welded_UTS", {}).get("accuracy", accuracy),
        "per_target": per_target
    }

def train_torch_model(model, train_loader, criterion, optimizer, epochs=25):
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
    return model

def run_training_pipeline(data_path="data/mild_steel_weld_dataset_4000_samples.csv", log_callback=None):
    def log(msg):
        print(msg)
        if log_callback:
            log_callback(msg)

    log(f"Loading Mild Steel dataset from {data_path}...")
    df = pd.read_csv(data_path)
    log(f"Dataset loaded: {df.shape[0]} specimens, {df.shape[1]} columns.")

    X = df[INPUT_FEATURES].values
    y = df[TARGET_COLUMNS].values

    # Stratified 80/20 train/test split on Welded_UTS quantiles
    uts_quantiles = pd.qcut(df["Welded_UTS"], q=10, labels=False)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(sss.split(X, uts_quantiles))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    log(f"Split completed: Train specimens = {len(X_train)}, Test specimens = {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    log("StandardScaler fitted and saved to models/scaler.pkl")

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    benchmark_results = {}

    # 1. XGBoost Regressor
    log("\n[1/11] Training XGBoost Regressor...")
    xgb_model = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=140, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1
    ))
    xgb_model.fit(X_train_scaled, y_train)
    xgb_preds = xgb_model.predict(X_test_scaled)
    benchmark_results["XGBoost"] = compute_metrics(y_test, xgb_preds)
    joblib.dump(xgb_model, "models/xgboost_best.pkl")
    log(f"  XGBoost Overall R2: {benchmark_results['XGBoost']['overall_r2']:.4f}, MAE: {benchmark_results['XGBoost']['overall_mae']:.2f}")

    # 2. Extra Trees Regressor
    log("\n[2/11] Training Extra Trees Regressor...")
    et_model = ExtraTreesRegressor(n_estimators=100, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1)
    et_model.fit(X_train_scaled, y_train)
    et_preds = et_model.predict(X_test_scaled)
    benchmark_results["ExtraTrees"] = compute_metrics(y_test, et_preds)
    joblib.dump(et_model, "models/extratrees.pkl")

    # 3. Gradient Boosting Regressor
    log("\n[3/11] Training Gradient Boosting Regressor...")
    gb_model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42))
    gb_model.fit(X_train_scaled, y_train)
    gb_preds = gb_model.predict(X_test_scaled)
    benchmark_results["GradientBoosting"] = compute_metrics(y_test, gb_preds)
    joblib.dump(gb_model, "models/gradient_boosting.pkl")

    # 4. LightGBM Regressor
    log("\n[4/11] Training LightGBM Regressor...")
    lgb_model = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=-1))
    lgb_model.fit(X_train_scaled, y_train)
    lgb_preds = lgb_model.predict(X_test_scaled)
    benchmark_results["LightGBM"] = compute_metrics(y_test, lgb_preds)
    joblib.dump(lgb_model, "models/lightgbm.pkl")

    # 5. Random Forest Regressor
    log("\n[5/11] Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    benchmark_results["RandomForest"] = compute_metrics(y_test, rf_preds)
    joblib.dump(rf_model, "models/random_forest.pkl")

    # 6. Multi-Model Stacking Ensemble
    log("\n[6/11] Training Stacking Ensemble Meta-Learner...")
    stacking_ens = StackingEnsemble(n_splits=5, random_state=42)
    stacking_ens.fit(X_train_scaled, y_train)
    stack_preds = stacking_ens.predict(X_test_scaled)
    benchmark_results["Stacking Ensemble"] = compute_metrics(y_test, stack_preds)
    stacking_ens.save("models/stacking_ensemble.pkl")
    log(f"  Stacking Ensemble Overall R2: {benchmark_results['Stacking Ensemble']['overall_r2']:.4f}")

    # 7. Physics-Informed Neural Network (PINN)
    log("\n[7/11] Training Physics-Informed Neural Network (PINN)...")
    pinn = PINNRegressor(input_dim=len(INPUT_FEATURES), output_dim=len(TARGET_COLUMNS))
    pinn_loss = PINNLoss(alpha=0.5, beta=0.1, uts_idx=0, ys_idx=1)
    pinn_opt = torch.optim.Adam(pinn.parameters(), lr=0.003, weight_decay=1e-5)
    train_torch_model(pinn, train_loader, pinn_loss, pinn_opt, epochs=25)
    pinn.eval()
    with torch.no_grad():
        pinn_preds = pinn(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    benchmark_results["Physics-Informed NN"] = compute_metrics(y_test, pinn_preds)
    torch.save(pinn.state_dict(), "models/pinn_model.pt")

    # 8. Curve LSTM Generator
    log("\n[8/11] Training Curve LSTM Generator...")
    curve_lstm = CurveLSTMGenerator(input_dim=len(INPUT_FEATURES), hidden_dim=64, num_layers=2, output_dim=len(TARGET_COLUMNS))
    cl_opt = torch.optim.Adam(curve_lstm.parameters(), lr=0.003)
    train_torch_model(curve_lstm, train_loader, nn.MSELoss(), cl_opt, epochs=25)
    curve_lstm.eval()
    with torch.no_grad():
        cl_preds = curve_lstm(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    benchmark_results["Curve LSTM"] = compute_metrics(y_test, cl_preds)
    torch.save(curve_lstm.state_dict(), "models/curve_lstm.pt")

    # 9. 1D-CNN Regressor
    log("\n[9/11] Training 1D-CNN Regressor...")
    cnn_model = CNNRegressor(input_dim=len(INPUT_FEATURES), output_dim=len(TARGET_COLUMNS))
    cnn_opt = torch.optim.Adam(cnn_model.parameters(), lr=0.003)
    train_torch_model(cnn_model, train_loader, nn.MSELoss(), cnn_opt, epochs=20)
    cnn_model.eval()
    with torch.no_grad():
        cnn_preds = cnn_model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    benchmark_results["CNN"] = compute_metrics(y_test, cnn_preds)

    # 10. BiLSTM Regressor
    log("\n[10/11] Training BiLSTM Regressor...")
    bilstm = BiLSTMRegressor(input_dim=len(INPUT_FEATURES), output_dim=len(TARGET_COLUMNS))
    bilstm_opt = torch.optim.Adam(bilstm.parameters(), lr=0.003)
    train_torch_model(bilstm, train_loader, nn.MSELoss(), bilstm_opt, epochs=20)
    bilstm.eval()
    with torch.no_grad():
        bilstm_preds = bilstm(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    benchmark_results["LSTM"] = compute_metrics(y_test, bilstm_preds)

    # 11. CNN+LSTM Hybrid
    log("\n[11/11] Training CNN+LSTM Hybrid Regressor...")
    hybrid = CNNLSTMHybrid(input_dim=len(INPUT_FEATURES), output_dim=len(TARGET_COLUMNS))
    hyb_opt = torch.optim.Adam(hybrid.parameters(), lr=0.003)
    train_torch_model(hybrid, train_loader, nn.MSELoss(), hyb_opt, epochs=20)
    hybrid.eval()
    with torch.no_grad():
        hyb_preds = hybrid(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    benchmark_results["CNN+LSTM Hybrid"] = compute_metrics(y_test, hyb_preds)

    with open("models/benchmark_metrics.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    log("\nMild Steel benchmark evaluation complete! Saved to models/benchmark_metrics.json.")

    return benchmark_results

if __name__ == "__main__":
    run_training_pipeline()
