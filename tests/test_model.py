"""Unit tests for the model and the training loop (no dataset download needed)."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import CLASSES, SimpleCNN  # noqa: E402
from train import evaluate, train_one_epoch  # noqa: E402


def test_output_shape():
    model = SimpleCNN()
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, len(CLASSES))


def test_training_step_reduces_loss():
    """The model should be able to overfit a tiny random batch."""
    torch.manual_seed(0)
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, len(CLASSES), (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    loss_before, _ = evaluate(model, loader, criterion, device)
    for _ in range(30):
        train_one_epoch(model, loader, criterion, optimizer, device)
    loss_after, acc_after = evaluate(model, loader, criterion, device)

    assert loss_after < loss_before
    assert acc_after > 0.5
