"""Shared test fixtures for EmbdGuard tests."""
import sys
import os
import pytest
import torch

# Add repo root and dlattack_research to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dlattack_research"))

from src.model import build_ebc, TwoTower, TwoTowerTrainTask, make_kjt


@pytest.fixture
def small_ebc():
    return build_ebc(50, 100, 16, device=torch.device("cpu"))


@pytest.fixture
def small_two_tower(small_ebc):
    return TwoTower(small_ebc, layer_sizes=[32, 16], device=torch.device("cpu"))


@pytest.fixture
def small_model(small_two_tower):
    return TwoTowerTrainTask(small_two_tower)


@pytest.fixture
def sample_kjt():
    """A small KJT with 3 user-item pairs."""
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([10, 20, 30])
    return make_kjt(users, items)


@pytest.fixture
def sample_labels():
    return torch.tensor([1.0, 0.0, 1.0])
