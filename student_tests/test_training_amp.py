"""Exercise actual GradScaler overflow recovery on CPU."""

import pytest

torch = pytest.importorskip("torch")

from student.training import _try_optimizer_step


def test_overflow_reduces_scale_without_updating_weights_then_recovers():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    scaler.scale(parameter.sum()).backward()
    parameter.grad.fill_(float("inf"))
    assert _try_optimizer_step(torch, optimizer, scaler, [parameter], 1.0) is None
    assert scaler.get_scale() == 512.0
    assert parameter.item() == 1.0
    assert parameter.grad is None

    scaler.scale(parameter.sum()).backward()
    grad_norm, learning_rate = _try_optimizer_step(torch, optimizer, scaler, [parameter], 1.0)
    assert grad_norm == pytest.approx(1.0)
    assert learning_rate == pytest.approx(0.1)
    assert parameter.item() == pytest.approx(0.9)
    assert parameter.grad is None
