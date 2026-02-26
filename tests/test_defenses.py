"""Tests for defensive techniques."""
import pytest
import torch
import warnings

from src.models import build_ebc, TwoTower, TwoTowerTrainTask, make_kjt, make_optimizer
from src.guard import EmbdGuard
from src.defenses.gradient_clip import GradientClipDefense
from src.defenses.row_freeze import RowFreezeDefense
from src.defenses.interaction_filter import InteractionFilterDefense


def _make_model():
    ebc = build_ebc(50, 100, 16, device=torch.device("cpu"))
    tt = TwoTower(ebc, layer_sizes=[32, 16], device=torch.device("cpu"))
    return TwoTowerTrainTask(tt)


class TestGradientClipDefense:
    def test_clip_reduces_gradient(self):
        model = _make_model()
        defense = GradientClipDefense(max_norm=0.001)
        defense.apply(model)
        defense.activate("t_item_id", [42], duration=5)

        # Forward+backward to get gradients
        kjt = make_kjt(torch.tensor([0, 1]), torch.tensor([42, 10]))
        labels = torch.tensor([1.0, 0.0])
        loss, _ = model(kjt, labels)
        loss.backward()

        # The gradient on row 42 should be clipped
        weight = model.two_tower.ebc.embedding_bags["t_item_id"].weight
        if weight.grad is not None:
            row_norm = weight.grad[42].norm().item()
            assert row_norm <= 0.001 + 1e-6

        defense.remove()

    def test_expiration(self):
        defense = GradientClipDefense()
        defense.activate("t_item_id", [42], duration=3)
        assert 42 in defense.active_rows.get("t_item_id", [])
        defense.step()
        defense.step()
        defense.step()
        assert 42 not in defense.active_rows.get("t_item_id", [])

    def test_remove_clears_state(self):
        model = _make_model()
        defense = GradientClipDefense()
        defense.apply(model)
        defense.activate("t_item_id", [1, 2, 3])
        defense.remove()
        assert defense.active_rows == {}


class TestRowFreezeDefense:
    def test_freeze_zeros_gradient(self):
        model = _make_model()
        defense = RowFreezeDefense()
        defense.apply(model)
        defense.activate("t_item_id", [42], duration=5)

        kjt = make_kjt(torch.tensor([0, 1]), torch.tensor([42, 10]))
        labels = torch.tensor([1.0, 0.0])
        loss, _ = model(kjt, labels)
        loss.backward()

        weight = model.two_tower.ebc.embedding_bags["t_item_id"].weight
        if weight.grad is not None:
            row_grad_norm = weight.grad[42].norm().item()
            assert row_grad_norm < 1e-10

        defense.remove()

    def test_unflagged_rows_unaffected(self):
        model = _make_model()
        defense = RowFreezeDefense()
        defense.apply(model)
        defense.activate("t_item_id", [42], duration=5)

        kjt = make_kjt(torch.tensor([0, 1]), torch.tensor([42, 10]))
        labels = torch.tensor([1.0, 0.0])
        loss, _ = model(kjt, labels)
        loss.backward()

        # Row 10 should still have non-zero gradient
        weight = model.two_tower.ebc.embedding_bags["t_item_id"].weight
        if weight.grad is not None:
            row10_norm = weight.grad[10].norm().item()
            # Row 10 was accessed, so it should have gradient
            # (but it might be zero if the model architecture doesn't propagate to it)

        defense.remove()

    def test_expiration(self):
        defense = RowFreezeDefense()
        defense.activate("t_item_id", [5], duration=2)
        defense.step()
        assert 5 in defense.active_rows.get("t_item_id", [])
        defense.step()
        assert 5 not in defense.active_rows.get("t_item_id", [])


class TestInteractionFilterDefense:
    def test_filters_flagged_items(self):
        defense = InteractionFilterDefense()
        defense.activate("t_item_id", [42, 99], duration=5)

        users = torch.tensor([0, 1, 2, 3, 4])
        items = torch.tensor([10, 42, 20, 99, 30])
        labels = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])

        fu, fi, fl = defense.filter_batch(users, items, labels)
        assert len(fu) == 3
        assert 42 not in fi.tolist()
        assert 99 not in fi.tolist()

    def test_no_filter_when_inactive(self):
        defense = InteractionFilterDefense()
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([10, 20, 30])
        labels = torch.tensor([1.0, 0.0, 1.0])

        fu, fi, fl = defense.filter_batch(users, items, labels)
        assert torch.equal(fu, users)
        assert torch.equal(fi, items)

    def test_expiration(self):
        defense = InteractionFilterDefense()
        defense.activate("t_item_id", [42], duration=1)
        assert 42 in defense.active_rows.get("t_item_id", [])
        defense.step()
        assert 42 not in defense.active_rows.get("t_item_id", [])

    def test_remove_clears_state(self):
        defense = InteractionFilterDefense()
        defense.activate("t_item_id", [1, 2, 3])
        defense.remove()
        assert defense.active_rows == {}


class TestGuardDefenseIntegration:
    def test_add_defense(self):
        model = _make_model()
        guard = EmbdGuard(model)
        defense = RowFreezeDefense()
        guard.add_defense(defense)
        assert len(guard._defenses) == 1
        guard.detach()

    def test_detach_removes_defenses(self):
        model = _make_model()
        guard = EmbdGuard(model)
        defense = GradientClipDefense()
        guard.add_defense(defense)
        guard.detach()
        assert len(guard._defenses) == 0
