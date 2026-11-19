"""
tests/test_architectures.py
Unit tests for PyTorch architectures and Physics-Informed loss constraints.
"""

import pytest
import torch
from core.architectures import (
    PINNLoss, PINNRegressor, CNNRegressor, BiLSTMRegressor,
    TransformerRegressor, CNNLSTMHybrid
)

def test_pinn_loss_basic():
    criterion = PINNLoss(alpha=0.5, beta=0.1)
    y_pred = torch.tensor([[1450.0, 1360.0, 10.0, 8.0, 2.0, 13.0]])
    y_true = torch.tensor([[1448.0, 1366.0, 10.2, 8.2, 2.1, 13.5]])
    loss = criterion(y_pred, y_true)
    assert loss.item() > 0.0

def test_pinn_loss_physics_penalty_when_ys_greater_than_uts():
    criterion = PINNLoss(alpha=1.0, beta=0.0)
    # Physical violation: YS (1500) > UTS (1400)
    y_pred_bad = torch.tensor([[1400.0, 1500.0, 10.0, 8.0, 2.0, 13.0]])
    y_true = torch.tensor([[1400.0, 1350.0, 10.0, 8.0, 2.0, 13.0]])
    loss_bad = criterion(y_pred_bad, y_true)

    # Valid: YS (1350) < UTS (1400)
    y_pred_good = torch.tensor([[1400.0, 1350.0, 10.0, 8.0, 2.0, 13.0]])
    loss_good = criterion(y_pred_good, y_true)

    assert loss_bad.item() > loss_good.item()

def test_pinn_loss_zero_physics_penalty_when_ys_less_than_uts():
    criterion = PINNLoss(alpha=10.0, beta=0.0)
    # When YS <= UTS, physics penalty is 0, so total loss == MSE loss
    y_pred = torch.tensor([[1450.0, 1360.0, 10.0, 8.0, 2.0, 13.0]])
    y_true = torch.tensor([[1440.0, 1350.0, 10.0, 8.0, 2.0, 13.0]])
    mse = torch.nn.MSELoss()(y_pred, y_true)
    total_loss = criterion(y_pred, y_true)
    assert torch.isclose(mse, total_loss)

def test_pinn_loss_positivity_penalty_on_negative_values():
    criterion = PINNLoss(alpha=0.0, beta=5.0)
    y_pred_neg = torch.tensor([[1450.0, 1360.0, -5.0, 8.0, 2.0, 13.0]])
    y_true = torch.tensor([[1450.0, 1360.0, 5.0, 8.0, 2.0, 13.0]])
    loss = criterion(y_pred_neg, y_true)
    assert loss.item() > 0.0

def test_pinn_regressor_forward_shape():
    model = PINNRegressor(input_dim=14, output_dim=6)
    x = torch.randn(8, 14)
    out = model(x)
    assert out.shape == (8, 6)

def test_pinn_regressor_gradient_flow():
    model = PINNRegressor(input_dim=14, output_dim=6)
    x = torch.randn(4, 14)
    y = torch.randn(4, 6)
    loss = torch.nn.MSELoss()(model(x), y)
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None

def test_cnn_regressor_forward_shape():
    model = CNNRegressor(input_dim=14, output_dim=6)
    x = torch.randn(8, 14)
    out = model(x)
    assert out.shape == (8, 6)

def test_cnn_regressor_dim_handling():
    model = CNNRegressor(input_dim=14, output_dim=6)
    x_3d = torch.randn(8, 1, 14)
    out = model(x_3d)
    assert out.shape == (8, 6)

def test_bilstm_regressor_forward_shape():
    model = BiLSTMRegressor(input_dim=14, output_dim=6)
    x = torch.randn(8, 14)
    out = model(x)
    assert out.shape == (8, 6)

def test_transformer_regressor_forward_shape():
    model = TransformerRegressor(input_dim=14, output_dim=6, embed_dim=64, num_heads=4)
    x = torch.randn(8, 14)
    out = model(x)
    assert out.shape == (8, 6)

def test_transformer_regressor_assertion_divisible_heads():
    # DEF-001 regression assertion
    with pytest.raises(AssertionError):
        TransformerRegressor(input_dim=14, output_dim=6, embed_dim=65, num_heads=4)

def test_cnn_lstm_hybrid_forward_shape():
    model = CNNLSTMHybrid(input_dim=14, output_dim=6)
    x = torch.randn(8, 14)
    out = model(x)
    assert out.shape == (8, 6)
