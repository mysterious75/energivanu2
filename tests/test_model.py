import torch

from energivanu.model import EnergivanuPEB


def test_model_forward_shapes():
    model = EnergivanuPEB(n_features=15, seq_len=30, pred_horizon=10)
    model.eval()
    x = torch.randn(4, 30, 15)
    power, signal = model(x)
    assert power.shape == (4, 10)
    assert signal.shape == (4, 3)


def test_model_single_sample():
    model = EnergivanuPEB(n_features=15, seq_len=30, pred_horizon=10)
    model.eval()
    x = torch.randn(1, 30, 15)
    power, signal = model(x)
    assert power.shape == (1, 10)
    assert signal.shape == (1, 3)


def test_model_deterministic():
    model = EnergivanuPEB(n_features=15, seq_len=30, pred_horizon=10)
    model.eval()
    x = torch.randn(5, 30, 15)
    p1, s1 = model(x)
    p2, s2 = model(x)
    assert torch.allclose(p1, p2)
    assert torch.allclose(s1, s2)


def test_model_count_parameters():
    model = EnergivanuPEB(n_features=15, seq_len=30, pred_horizon=10)
    n = model.count_parameters()
    assert n > 0
    assert isinstance(n, int)


def test_model_different_batch_sizes():
    model = EnergivanuPEB(n_features=15, seq_len=30, pred_horizon=10)
    model.eval()
    for bs in [1, 8, 32]:
        x = torch.randn(bs, 30, 15)
        power, signal = model(x)
        assert power.shape == (bs, 10)
        assert signal.shape == (bs, 3)
