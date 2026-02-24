"""
Shared model utilities: EBC/KJT imports (with pure-PyTorch fallbacks),
build_ebc(), make_kjt(), _build_mlp().
"""
import torch
import torch.nn as nn
from dataclasses import dataclass, field

try:
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchrec.modules.embedding_modules import EmbeddingBagCollection
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    _TORCHREC_AVAILABLE = True
except (ImportError, OSError, AttributeError):
    _TORCHREC_AVAILABLE = False

if not _TORCHREC_AVAILABLE:
    import logging
    logging.getLogger(__name__).warning(
        "TorchRec unavailable — using pure-PyTorch fallback for EmbeddingBagCollection and KeyedJaggedTensor"
    )


# ── Pure-PyTorch fallbacks (used when torchrec is broken) ───────────────

if not _TORCHREC_AVAILABLE:

    @dataclass
    class EmbeddingBagConfig:
        name: str = ""
        num_embeddings: int = 0
        embedding_dim: int = 0
        feature_names: list = field(default_factory=list)

    class _KeyedTensorResult:
        """Dict-like wrapper returned by EmbeddingBagCollection.forward()."""
        def __init__(self, data: dict):
            self._data = data
        def __getitem__(self, key):
            return self._data[key]

    class EmbeddingBagCollection(nn.Module):
        """Pure-PyTorch drop-in for torchrec.EmbeddingBagCollection.

        Wraps nn.EmbeddingBag modules in a ModuleDict with the same API
        surface used by the rest of the codebase.
        """
        def __init__(self, tables, device=None):
            super().__init__()
            if device is None:
                device = torch.device("cpu")
            self._configs = list(tables)
            self.embedding_bags = nn.ModuleDict()
            self._feature_to_table = {}
            for config in self._configs:
                bag = nn.EmbeddingBag(
                    config.num_embeddings, config.embedding_dim,
                    mode="mean", include_last_offset=True,
                ).to(device)
                self.embedding_bags[config.name] = bag
                for feat in config.feature_names:
                    self._feature_to_table[feat] = config.name

        @property
        def embedding_bag_configs(self):
            return self._configs

        def forward(self, kjt):
            result = {}
            n_keys = len(kjt._keys)
            total_bags = len(kjt._offsets) - 1
            per_key = total_bags // n_keys
            for i, key in enumerate(kjt._keys):
                table_name = self._feature_to_table[key]
                bag = self.embedding_bags[table_name]
                start_bag = i * per_key
                end_bag = start_bag + per_key
                val_start = kjt._offsets[start_bag].item()
                val_end = kjt._offsets[end_bag].item()
                sub_values = kjt._values[val_start:val_end]
                sub_offsets = kjt._offsets[start_bag:end_bag + 1] - kjt._offsets[start_bag]
                result[key] = bag(sub_values, sub_offsets)
            return _KeyedTensorResult(result)

    class KeyedJaggedTensor:
        """Pure-PyTorch drop-in for torchrec.KeyedJaggedTensor."""
        def __init__(self, keys, values, offsets=None, lengths=None):
            self._keys = keys
            self._values = values
            self._offsets = offsets
            self._lengths = lengths

        def keys(self):
            return self._keys

        def values(self):
            return self._values

        def offsets(self):
            return self._offsets


# ── Shared builders ─────────────────────────────────────────────────────

def build_ebc(
    n_users: int,
    n_items: int,
    embedding_dim: int,
    device: torch.device = None,
) -> EmbeddingBagCollection:
    """Construct a single EBC with user and item embedding tables."""
    if device is None:
        device = torch.device("cpu")
    return EmbeddingBagCollection(
        tables=[
            EmbeddingBagConfig(
                name="t_user_id",
                num_embeddings=n_users,
                embedding_dim=embedding_dim,
                feature_names=["user_id"],
            ),
            EmbeddingBagConfig(
                name="t_item_id",
                num_embeddings=n_items,
                embedding_dim=embedding_dim,
                feature_names=["item_id"],
            ),
        ],
        device=device,
    )


def make_kjt(user_ids: torch.Tensor, item_ids: torch.Tensor) -> KeyedJaggedTensor:
    """Build a KeyedJaggedTensor from user and item ID tensors.

    Each ID is a single-element bag (length=1), so lengths is all ones
    and offsets is [0, 1, 2, ...]. We provide offsets directly to avoid
    the fbgemm cumsum kernel (not available on macOS).
    """
    n = len(user_ids)
    values = torch.cat([user_ids, item_ids])
    offsets = torch.arange(2 * n + 1, dtype=torch.long, device=user_ids.device)
    return KeyedJaggedTensor(
        keys=["user_id", "item_id"],
        values=values,
        offsets=offsets,
    )


def _build_mlp(in_dim: int, layer_sizes: list) -> nn.Sequential:
    layers = []
    cur = in_dim
    for h in layer_sizes:
        layers += [nn.Linear(cur, h), nn.ReLU()]
        cur = h
    return nn.Sequential(*layers)
