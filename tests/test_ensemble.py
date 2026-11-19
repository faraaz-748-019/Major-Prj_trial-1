"""
tests/test_ensemble.py
Unit tests for Stacking Ensemble and Tree Base Regressors.
"""

import os
import pytest
import numpy as np
from core.ensemble import StackingEnsemble, get_base_models

def test_get_base_models():
    models = get_base_models()
    assert "ExtraTrees" in models
    assert "RandomForest" in models
    assert "GradientBoosting" in models
    assert "XGBoost" in models
    assert "LightGBM" in models

def test_prepare_meta_matrix_1d_and_2d():
    # DEF-004 regression test
    ens = StackingEnsemble()
    p1 = np.array([1.0, 2.0, 3.0])                # 1D array
    p2 = np.array([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]) # 2D array
    meta = ens._prepare_meta_matrix([p1, p2])
    assert meta.shape == (3, 3)
    np.testing.assert_array_equal(meta[:, 0], p1)
    np.testing.assert_array_equal(meta[:, 1:], p2)

def test_stacking_ensemble_fit_predict_shape():
    X = np.random.randn(40, 14)
    y = np.random.randn(40, 6)
    
    # Fast lightweight base models for unit testing
    from sklearn.linear_model import Ridge, LinearRegression
    mini_base = {
        "m1": Ridge(alpha=1.0),
        "m2": LinearRegression()
    }
    ens = StackingEnsemble(base_models=mini_base, n_splits=3)
    ens.fit(X, y)
    preds = ens.predict(X[:5])
    assert preds.shape == (5, 6)

def test_stacking_ensemble_multi_target_output():
    X = np.random.randn(30, 14)
    y = np.random.randn(30, 2)
    from sklearn.linear_model import Ridge
    mini_base = {"m1": Ridge()}
    ens = StackingEnsemble(base_models=mini_base, n_splits=2)
    ens.fit(X, y)
    preds = ens.predict(X)
    assert preds.shape == (30, 2)

def test_stacking_ensemble_save_load(tmp_path):
    X = np.random.randn(30, 14)
    y = np.random.randn(30, 6)
    from sklearn.linear_model import Ridge
    ens = StackingEnsemble(base_models={"m1": Ridge()}, n_splits=2)
    ens.fit(X, y)
    save_file = str(tmp_path / "test_stack.pkl")
    ens.save(save_file)
    assert os.path.exists(save_file)

    loaded = StackingEnsemble.load(save_file)
    preds_orig = ens.predict(X[:4])
    preds_loaded = loaded.predict(X[:4])
    np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)

def test_tree_model_xgboost_fit():
    from sklearn.multioutput import MultiOutputRegressor
    import xgboost as xgb
    m = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=42))
    X = np.random.randn(20, 14)
    y = np.random.randn(20, 6)
    m.fit(X, y)
    p = m.predict(X[:3])
    assert p.shape == (3, 6)

def test_tree_model_extratrees_fit():
    from sklearn.ensemble import ExtraTreesRegressor
    m = ExtraTreesRegressor(n_estimators=5, max_depth=3, random_state=42)
    X = np.random.randn(20, 14)
    y = np.random.randn(20, 6)
    m.fit(X, y)
    p = m.predict(X[:3])
    assert p.shape == (3, 6)

def test_tree_model_gradient_boosting_fit():
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    m = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5, max_depth=3, random_state=42))
    X = np.random.randn(20, 14)
    y = np.random.randn(20, 6)
    m.fit(X, y)
    p = m.predict(X[:3])
    assert p.shape == (3, 6)

def test_tree_model_lightgbm_fit():
    from sklearn.multioutput import MultiOutputRegressor
    import lightgbm as lgb
    m = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=5, max_depth=3, random_state=42, verbose=-1))
    X = np.random.randn(20, 14)
    y = np.random.randn(20, 6)
    m.fit(X, y)
    p = m.predict(X[:3])
    assert p.shape == (3, 6)

def test_tree_model_random_forest_fit():
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    X = np.random.randn(20, 14)
    y = np.random.randn(20, 6)
    m.fit(X, y)
    p = m.predict(X[:3])
    assert p.shape == (3, 6)
