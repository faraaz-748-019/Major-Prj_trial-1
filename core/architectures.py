"""
core/architectures.py
PyTorch Deep Neural Network Architectures for M250 Maraging Steel Mechanical Property Prediction:
1. CNNRegressor (1D-CNN)
2. BiLSTMRegressor (Bidirectional LSTM)
3. TransformerRegressor (Self-Attention Transformer Encoder) - DEF-001 Fixed
4. CNNLSTMHybrid (Convolutional + Recurrent Hybrid)
5. PINNRegressor (Physics-Informed Neural Network) with PINNLoss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNLoss(nn.Module):
    """
    Physics-Informed Loss Objective enforcing metallurgical laws:
    1. Yield Strength (YS) <= Ultimate Tensile Strength (UTS)
    2. Non-negativity across physical properties (UTS, YS, Elongation, HAZ Width, Depth, Austenite)
    """
    def __init__(self, alpha=0.5, beta=0.1, uts_idx=0, ys_idx=1):
        super(PINNLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha      # Physics violation penalty weight
        self.beta = beta        # Positivity constraint penalty weight
        self.uts_idx = uts_idx
        self.ys_idx = ys_idx

    def forward(self, y_pred, y_true):
        # Base data fidelity loss
        loss_mse = self.mse(y_pred, y_true)
        
        # Physical constraint: YS <= UTS
        # If YS > UTS, penalty = (YS - UTS)^2
        uts_pred = y_pred[:, self.uts_idx]
        ys_pred = y_pred[:, self.ys_idx]
        physics_violation = F.relu(ys_pred - uts_pred)
        loss_physics = torch.mean(physics_violation ** 2)
        
        # Positivity constraint: All mechanical targets must be positive
        positivity_violation = F.relu(-y_pred)
        loss_positivity = torch.mean(positivity_violation ** 2)
        
        total_loss = loss_mse + (self.alpha * loss_physics) + (self.beta * loss_positivity)
        return total_loss

class PINNRegressor(nn.Module):
    """
    Physics-Informed Neural Network with Dense architecture and residual skip connections.
    """
    def __init__(self, input_dim=14, output_dim=6, hidden_dims=[128, 128, 64, 64], dropout=0.1):
        super(PINNRegressor, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out

class CNNRegressor(nn.Module):
    """
    1D Convolutional Neural Network for localized feature pattern extraction.
    """
    def __init__(self, input_dim=14, output_dim=6, num_filters=[64, 128], kernel_size=3, dropout=0.1):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters[0], kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        self.conv2 = nn.Conv1d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_filters[1])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters[1], 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out).squeeze(-1)
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)
        return out

class BiLSTMRegressor(nn.Module):
    """
    Bidirectional LSTM network for cross-feature sequential state dependencies.
    """
    def __init__(self, input_dim=14, output_dim=6, hidden_dim=64, num_layers=2, dropout=0.1):
        super(BiLSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim) -> (batch_size, input_dim, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        # Take the mean over the sequence dimension
        pooled = torch.mean(lstm_out, dim=1)
        out = self.dropout(F.relu(self.fc1(pooled)))
        out = self.fc2(out)
        return out

class TransformerRegressor(nn.Module):
    """
    Transformer Multi-Head Attention architecture for tabular feature interactions.
    DEF-001 Fixed: Consistent parameter signature and verified embed_dim % num_heads == 0.
    """
    def __init__(self, input_dim=14, output_dim=6, embed_dim=64, num_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(1, embed_dim)
        
        # Positional encoding for sequence order of features
        self.pos_encoder = nn.Parameter(torch.randn(1, input_dim, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim) -> (batch_size, input_dim, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x_proj = self.input_proj(x) + self.pos_encoder[:, :x.size(1), :]
        trans_out = self.transformer_encoder(x_proj)
        pooled = torch.mean(trans_out, dim=1)
        out = self.dropout(F.gelu(self.fc1(pooled)))
        out = self.fc2(out)
        return out

class CNNLSTMHybrid(nn.Module):
    """
    Hybrid architecture combining 1D-CNN local feature extraction with LSTM state aggregation.
    """
    def __init__(self, input_dim=14, output_dim=6, conv_filters=32, lstm_hidden=64, dropout=0.1):
        super(CNNLSTMHybrid, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=conv_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(conv_filters)
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        conv_out = F.relu(self.bn(self.conv(x)))
        # conv_out: (batch_size, conv_filters, input_dim) -> (batch_size, input_dim, conv_filters)
        lstm_in = conv_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        pooled = lstm_out[:, -1, :]  # Last hidden state
        out = self.dropout(F.relu(self.fc1(pooled)))
        out = self.fc2(out)
        return out
