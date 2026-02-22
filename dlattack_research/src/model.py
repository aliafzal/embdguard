"""
TorchRec-native Two-Tower recommender model.

Architecture:
  EBC -> TwoTower -> TwoTowerTrainTask + make_optimizer()

  Single EBC with two tables (user, item)
  User tower: EBC["user_id"] -> MLP -> L2-normalized vector
  Item tower: EBC["item_id"] -> MLP -> L2-normalized vector
  Score:      dot product of normalized towers

Optimizer: Adam with parameter groups — embedding_lr=0.1 for EBC params,
  lr=0.001 for MLP params. Embeddings are sparse (each row only gets
  gradients when that user/item appears in the batch), so they need a
  much higher LR than the dense MLP layers.

When torchrec is unavailable or broken (e.g. fbgemm_gpu version mismatch),
pure-PyTorch fallbacks are used automatically. The API is identical.
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


# ── Model code (identical regardless of backend) ───────────────────────

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


class TwoTower(nn.Module):
    """Two-Tower model that takes an EBC and projects through MLP towers."""

    def __init__(
        self,
        ebc: EmbeddingBagCollection,
        layer_sizes: list = None,
        device: torch.device = None,
    ):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [128, 64]
        if device is None:
            device = torch.device("cpu")

        self.ebc = ebc
        configs = ebc.embedding_bag_configs
        if callable(configs):
            configs = configs()
        embedding_dim = configs[0].embedding_dim
        self.user_proj = _build_mlp(embedding_dim, layer_sizes).to(device)
        self.item_proj = _build_mlp(embedding_dim, layer_sizes).to(device)

    def forward(self, kjt):
        """Returns (user_emb, item_emb) both L2-normalized."""
        pooled = self.ebc(kjt)
        user_emb = self.user_proj(pooled["user_id"])
        item_emb = self.item_proj(pooled["item_id"])
        user_emb = nn.functional.normalize(user_emb, dim=-1)
        item_emb = nn.functional.normalize(item_emb, dim=-1)
        return user_emb, item_emb

    def get_item_embeddings(self) -> torch.Tensor:
        """Return the raw item embedding matrix (n_items, embed_dim)."""
        return self.ebc.embedding_bags["t_item_id"].weight.data

    def resize_user_embedding(self, new_n_users: int):
        """Expand user embedding table to accommodate fake users, preserving weights."""
        old_table = self.ebc.embedding_bags["t_user_id"]
        old_n_users = old_table.weight.shape[0]
        if new_n_users <= old_n_users:
            return
        old_weight = old_table.weight.data.clone()
        device = old_weight.device
        embedding_dim = old_weight.shape[1]
        n_items = self.ebc.embedding_bags["t_item_id"].weight.shape[0]
        old_item_weight = self.ebc.embedding_bags["t_item_id"].weight.data.clone()

        self.ebc = build_ebc(new_n_users, n_items, embedding_dim, device)
        with torch.no_grad():
            self.ebc.embedding_bags["t_user_id"].weight[:old_n_users] = old_weight
            self.ebc.embedding_bags["t_item_id"].weight.copy_(old_item_weight)


def make_optimizer(model, lr=0.001, embedding_lr=0.1):
    """Create Adam with separate LRs for embeddings (high) and MLPs (normal).

    Embeddings are sparse — each row only gets gradients when that user/item
    appears in the batch — so they need a much higher LR than the dense MLP
    layers to learn effectively.
    """
    from torch.optim import Adam
    two_tower = model.two_tower if hasattr(model, "two_tower") else model
    ebc_params = list(two_tower.ebc.parameters())
    mlp_params = (list(two_tower.user_proj.parameters()) +
                  list(two_tower.item_proj.parameters()))
    return Adam([
        {"params": ebc_params, "lr": embedding_lr},
        {"params": mlp_params, "lr": lr},
    ])


class TwoTowerTrainTask(nn.Module):
    """Wraps TwoTower with BCEWithLogitsLoss for training.

    Args:
        logit_temp: Temperature scaling for dot-product logits. Since embeddings
            are L2-normalized, raw dot products lie in [-1, 1], which constrains
            sigmoid to [0.27, 0.73] — too narrow for BCE to produce strong
            gradients. Multiplying by temperature stretches logits to
            [-temp, temp], giving the model full dynamic range.
    """

    def __init__(self, two_tower: TwoTower, logit_temp: float = 10.0):
        super().__init__()
        self.two_tower = two_tower
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.logit_temp = logit_temp

    def forward(self, kjt, labels: torch.Tensor):
        user_emb, item_emb = self.two_tower(kjt)
        logits = (user_emb * item_emb).sum(dim=1) * self.logit_temp
        loss = self.loss_fn(logits, labels)
        return loss, (loss.detach(), logits.detach(), labels.detach())
