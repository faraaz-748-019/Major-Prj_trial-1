"""
core/curve_lstm.py
2-Layer LSTM Parameter Generator for Continuous Stress-Strain Trajectory Synthesis.
Predicts 6 key R2 stress-strain parameters from baseline input features.
Coupled with SciPy Pchip monotonic spline interpolation to reconstruct 200-point curves.
"""

import torch
import torch.nn as nn
import numpy as np
from core.curve_generator import reconstruct_r2_curve

class CurveLSTMGenerator(nn.Module):
    """
    2-Layer LSTM Network for predicting continuous stress-strain curve parameters.
    """
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=2, output_dim=6, dropout=0.1):
        super(CurveLSTMGenerator, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim) -> (batch_size, input_dim, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        # Take the final sequence step hidden representation
        last_hidden = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_hidden))
        out = self.fc2(out)
        return out

    def generate_curve(self, input_features, uts_pred, ys_pred, elongation=10.0, haz_width=8.0, n_points=200):
        """
        Synthesizes a 200-point continuous engineering stress-strain curve.
        Enforces Hooke's Law E = 190 GPa +/- 1% on the initial slope.
        """
        strains, stresses = reconstruct_r2_curve(
            uts=uts_pred,
            ys=ys_pred,
            elongation=elongation,
            haz_width=haz_width,
            n_points=n_points
        )
        return strains, stresses
