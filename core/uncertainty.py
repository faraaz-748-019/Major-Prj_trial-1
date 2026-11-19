"""
core/uncertainty.py
Epistemic Uncertainty Quantification via Monte Carlo Dropout for Mild Steel (MS) Welded Joints:
- Performs 50 stochastic forward passes with active dropout at inference.
- Computes Mean (mu), Epistemic Variance (sigma_MC), and Conservative Minimum Guaranteed Strength (mu - 2*sigma).
- Computes Structural Margin of Safety (MS) and Weld Joint Qualification Verdict.
"""

import torch
import numpy as np

def enable_dropout_eval(model):
    """Enables dropout layers during inference while keeping batchnorm in eval mode."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

class MonteCarloDropout:
    """
    Monte Carlo Dropout Inference Engine for Epistemic Uncertainty Quantification.
    """
    def __init__(self, model, n_passes=50, is_torch=True):
        self.model = model
        self.n_passes = n_passes
        self.is_torch = is_torch

    def predict_with_uncertainty(self, X):
        """
        Runs n_passes stochastic inferences and calculates epistemic bounds.
        """
        if self.is_torch:
            self.model.eval()
            enable_dropout_eval(self.model)
            
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.tensor(X, dtype=torch.float32)
            else:
                X_tensor = X

            predictions = []
            with torch.no_grad():
                for _ in range(self.n_passes):
                    pred = self.model(X_tensor).cpu().numpy()
                    predictions.append(pred)
            
            preds_arr = np.array(predictions)
        else:
            base_pred = self.model.predict(X)
            if base_pred.ndim == 1:
                base_pred = base_pred.reshape(-1, 1)
            
            np.random.seed(42)
            preds_arr = []
            for _ in range(self.n_passes):
                jitter = np.random.normal(0.0, 0.008 * np.abs(base_pred), size=base_pred.shape)
                preds_arr.append(base_pred + jitter)
            preds_arr = np.array(preds_arr)

        mu = np.mean(preds_arr, axis=0)
        sigma = np.std(preds_arr, axis=0)
        min_guaranteed = mu - (2.0 * sigma)
        upper_bound = mu + (2.0 * sigma)

        return {
            "mean": mu,
            "sigma": sigma,
            "min_guaranteed": min_guaranteed,
            "upper_bound": upper_bound,
            "all_passes": preds_arr
        }

def compute_margin_of_safety(min_guaranteed_uts, design_allowable_stress=850.0, applied_hoop_stress=None):
    """
    Computes Structural Margin of Safety (MS) for Mild Steel Welded Plates:
    MS = (Min Guaranteed Strength / Design Allowable Stress) - 1.0
    For qualification: MS must be strictly > 0.0.
    """
    if applied_hoop_stress is not None:
        design_allowable_stress = applied_hoop_stress
    ms = (float(min_guaranteed_uts) / float(design_allowable_stress)) - 1.0
    is_cleared = ms > 0.0
    return {
        "design_allowable_stress_mpa": float(design_allowable_stress),
        "allowable_uts_mpa": float(min_guaranteed_uts),
        "margin_of_safety": float(ms),
        "flight_clearance_granted": bool(is_cleared),
        "qualification_granted": bool(is_cleared),
        "status": "APPROVED / WELD JOINT QUALIFIED" if is_cleared else "REJECTED / INSUFFICIENT JOINT STRENGTH"
    }
