"""
Two-Tower recommender model.

Architecture:
  EBC -> TwoTower -> TwoTowerTrainTask + make_optimizer()

  Single EBC with two tables (user, item)
  User tower: EBC["user_id"] -> MLP -> L2-normalized vector
  Item tower: EBC["item_id"] -> MLP -> L2-normalized vector
  Score:      dot product of normalized towers
"""
import torch
import torch.nn as nn

from .common import build_ebc, _build_mlp


class TwoTower(nn.Module):
    """Two-Tower model that takes an EBC and projects through MLP towers."""

    def __init__(
        self,
        ebc,
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

    def score(self, kjt) -> torch.Tensor:
        """Full forward pass returning dot-product logits (B,)."""
        user_emb, item_emb = self.forward(kjt)
        return (user_emb * item_emb).sum(dim=1)

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
