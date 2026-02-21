"""
TorchRec-native Two-Tower recommender model.

Follows the official TorchRec two-tower example pattern:
  EBC (meta device) -> TwoTower -> TwoTowerTrainTask
                                          |
                      apply_optimizer_in_backward (RowWiseAdagrad)
                                          |
                                DistributedModelParallel (DMP)
                                          +
                                KeyedOptimizerWrapper (Adam for dense)

Architecture:
  Single EBC with two tables (user, item)
  User tower: EBC["user_id"] -> MLP -> L2-normalized vector
  Item tower: EBC["item_id"] -> MLP -> L2-normalized vector
  Score:      dot product of normalized towers
"""
import torch
import torch.nn as nn
import torchrec
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


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
            torchrec.EmbeddingBagConfig(
                name="t_user_id",
                num_embeddings=n_users,
                embedding_dim=embedding_dim,
                feature_names=["user_id"],
            ),
            torchrec.EmbeddingBagConfig(
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

    Each ID is a single-element bag (length=1), so lengths is all ones.
    """
    return KeyedJaggedTensor(
        keys=["user_id", "item_id"],
        values=torch.cat([user_ids, item_ids]),
        lengths=torch.ones(len(user_ids) * 2, dtype=torch.long, device=user_ids.device),
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
        embedding_dim = ebc.embedding_bag_configs()[0].embedding_dim
        self.user_proj = _build_mlp(embedding_dim, layer_sizes).to(device)
        self.item_proj = _build_mlp(embedding_dim, layer_sizes).to(device)

    def forward(self, kjt: KeyedJaggedTensor):
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

    def forward(self, kjt: KeyedJaggedTensor, labels: torch.Tensor):
        user_emb, item_emb = self.two_tower(kjt)
        logits = (user_emb * item_emb).sum(dim=1) * self.logit_temp
        loss = self.loss_fn(logits, labels)
        return loss, (loss.detach(), logits.detach(), labels.detach())
