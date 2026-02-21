"""Tests for EBCHooks — hook attachment and stat collection."""
import torch
from embdguard.hooks import EBCHooks


def test_attach_and_detach(small_ebc):
    hooks = EBCHooks(small_ebc)
    assert not hooks.attached

    hooks.attach()
    assert hooks.attached
    assert len(hooks._handles) == 4  # 2 tables × (fwd + bwd)

    hooks.detach()
    assert not hooks.attached
    assert len(hooks._handles) == 0


def test_attach_idempotent(small_ebc):
    hooks = EBCHooks(small_ebc)
    hooks.attach()
    hooks.attach()  # second call should be no-op
    assert len(hooks._handles) == 4
    hooks.detach()


def test_table_names_default(small_ebc):
    hooks = EBCHooks(small_ebc)
    assert set(hooks._table_names) == {"t_user_id", "t_item_id"}


def test_table_names_filter(small_ebc):
    hooks = EBCHooks(small_ebc, table_names=["t_item_id"])
    assert hooks._table_names == ["t_item_id"]
    hooks.attach()
    assert len(hooks._handles) == 2  # 1 table × (fwd + bwd)
    hooks.detach()


def test_forward_captures_indices(small_model, sample_kjt, sample_labels):
    ebc = small_model.two_tower.ebc
    hooks = EBCHooks(ebc)
    hooks.attach()

    # Forward pass
    loss, _ = small_model(sample_kjt, sample_labels)

    stats = hooks.collect()
    # Should have stats for at least one table with accessed indices
    has_indices = False
    for tname, tstat in stats.items():
        if "n_accessed" in tstat:
            assert tstat["n_accessed"] > 0
            assert isinstance(tstat["accessed_ids"], list)
            has_indices = True
    assert has_indices

    hooks.detach()


def test_backward_captures_gradients(small_model, sample_kjt, sample_labels):
    ebc = small_model.two_tower.ebc
    hooks = EBCHooks(ebc)
    hooks.attach()

    loss, _ = small_model(sample_kjt, sample_labels)
    loss.backward()

    stats = hooks.collect()
    has_grads = False
    for tname, tstat in stats.items():
        if "grad_norm" in tstat:
            assert tstat["grad_norm"] > 0
            assert tstat["grad_max"] > 0
            has_grads = True
    assert has_grads

    hooks.detach()


def test_collect_resets_buffers(small_model, sample_kjt, sample_labels):
    ebc = small_model.two_tower.ebc
    hooks = EBCHooks(ebc)
    hooks.attach()

    loss, _ = small_model(sample_kjt, sample_labels)
    loss.backward()

    stats1 = hooks.collect()
    stats2 = hooks.collect()

    # First collect should have stats
    assert any("grad_norm" in s for s in stats1.values())
    # Second collect should be empty (buffers were reset)
    assert all(len(s) == 0 for s in stats2.values())

    hooks.detach()


def test_collect_no_forward():
    """Collect without a forward pass should return empty stats."""
    import torch.nn as nn
    eb = nn.EmbeddingBag(10, 4, mode="mean")

    class FakeEBC(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding_bags = nn.ModuleDict({"test": eb})

    ebc = FakeEBC()
    hooks = EBCHooks(ebc)
    hooks.attach()
    stats = hooks.collect()
    assert stats["test"] == {}
    hooks.detach()
