"""
core/ensemble.py
Multi-Model Stacking Ensemble Architecture for Aerospace Post-Weld Repair Property Prediction:
- Aggregates 5 classical tree models:
  1. Extra Trees Regressor
  2. Random Forest Regressor
  3. Gradient Boosting Regressor
  4. XGBoost Regressor
  5. LightGBM Regressor
- Employs 5-fold out-of-fold cross-validation to construct meta-features.
- Meta-Learner: Regularized Ridge Regression.
- DEF-004 Fixed: Includes _prepare_meta_matrix helper handling 1D and 2D arrays.
"""

import os
import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

def get_base_models(random_state=42):
    """Initializes the standard suite of tuned tree ensemble models."""
    return {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=100, max_depth=15, min_samples_split=4, random_state=random_state, n_jobs=-1
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_split=4, random_state=random_state, n_jobs=-1
        ),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=random_state)
        ),
        "XGBoost": MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=random_state, n_jobs=-1)
        ),
        "LightGBM": MultiOutputRegressor(
            lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=random_state, verbose=-1, n_jobs=-1)
        )
    }

class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Multi-Target Out-Of-Fold Stacking Ensemble.
    Stacks predictions from multiple base regressors and trains a Ridge meta-model.
    """
    def __init__(self, base_models=None, meta_model=None, n_splits=5, random_state=42):
        self.base_models = base_models if base_models is not None else get_base_models(random_state)
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.n_splits = n_splits
        self.random_state = random_state
        self.fitted_base_models = {}
        self.fitted_meta_model = None

    def _prepare_meta_matrix(self, preds_list):
        """
        DEF-004 Fix: Safely prepares the meta-feature matrix handling both 1D and 2D model outputs.
        Ensures each prediction array is 2D (N, n_targets) before horizontal concatenation.
        """
        standardized_preds = []
        for p in preds_list:
            arr = np.asarray(p)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            standardized_preds.append(arr)
        return np.hstack(standardized_preds)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        n_targets = y.shape[1] if y.ndim > 1 else 1

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Out-of-fold prediction dictionary
        oof_predictions = {name: np.zeros((n_samples, n_targets)) for name in self.base_models}

        # Generate out-of-fold predictions
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            for name, model in self.base_models.items():
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_train, y_train)
                val_pred = fold_model.predict(X_val)
                if val_pred.ndim == 1 and n_targets > 1:
                    val_pred = val_pred.reshape(-1, 1)
                oof_predictions[name][val_idx] = val_pred

        # Construct out-of-fold meta-matrix
        oof_list = [oof_predictions[name] for name in self.base_models]
        meta_X = self._prepare_meta_matrix(oof_list)

        # Train meta-model on out-of-fold predictions
        if n_targets > 1 and not hasattr(self.meta_model, "n_targets"):
            self.fitted_meta_model = MultiOutputRegressor(self.meta_model)
        else:
            self.fitted_meta_model = self.meta_model
        self.fitted_meta_model.fit(meta_X, y)

        # Retrain full base models on entire dataset
        for name, model in self.base_models.items():
            from sklearn.base import clone
            full_model = clone(model)
            full_model.fit(X, y)
            self.fitted_base_models[name] = full_model

        return self

    def predict(self, X):
        X = np.asarray(X)
        preds_list = []
        for name in self.base_models:
            model = self.fitted_base_models[name]
            p = model.predict(X)
            preds_list.append(p)
        
        meta_X = self._prepare_meta_matrix(preds_list)
        return self.fitted_meta_model.predict(meta_X)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
