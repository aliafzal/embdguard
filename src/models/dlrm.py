"""
DLRM (Deep Learning Recommendation Model) for MovieLens-1M.

Architecture (sparse-only: user_id + item_id, no dense features):
  EBC lookup -> user_emb (B, D), item_emb (B, D)
  -> interaction: user_emb * item_emb element-wise (B, D)
  -> concat: [user_emb, item_emb, interaction] (B, 3D)
  -> top MLP: 3D -> 128 -> 64 -> 1
  -> logit (B,)

Key differences from Two-Tower:
  - Two-Tower: separate MLPs -> L2-normalize -> dot product (late interaction)
  - DLRM: raw embeddings -> element-wise interaction -> shared top MLP (early interaction)
  - No logit_temp needed: top MLP outputs unbounded logits
"""
import torch
import torch.nn as nn

from .common import build_ebc, _build_mlp


class DLRM(nn.Module):
    """DLRM model with element-wise interaction and top MLP."""

    def __init__(
        self,
        ebc,
        top_layer_sizes: list = None,
        device: torch.device = None,
    ):
        super().__init__()
        if top_layer_sizes is None:
            top_layer_sizes = [128, 64]
        if device is None:
            device = torch.device("cpu")

        self.ebc = ebc
        configs = ebc.embedding_bag_configs
        if callable(configs):
            configs = configs()
        embedding_dim = configs[0].embedding_dim

        # Top MLP: input is [user_emb, item_emb, interaction] = 3 * embedding_dim
        top_layers = []
        cur = 3 * embedding_dim
        for h in top_layer_sizes:
            top_layers += [nn.Linear(cur, h), nn.ReLU()]
            cur = h
        top_layers.append(nn.Linear(cur, 1))
        self.top_mlp = nn.Sequential(*top_layers).to(device)

    def forward(self, kjt):
        """Returns (user_emb, item_emb) — raw EBC embeddings (not normalized)."""
        pooled = self.ebc(kjt)
        user_emb = pooled["user_id"]
        item_emb = pooled["item_id"]
        return user_emb, item_emb

    def score(self, kjt) -> torch.Tensor:
        """Full DLRM forward: embeddings -> interaction -> top MLP -> logits (B,)."""
        user_emb, item_emb = self.forward(kjt)
        interaction = user_emb * item_emb
        x = torch.cat([user_emb, item_emb, interaction], dim=1)
        return self.top_mlp(x).squeeze(1)

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


def make_dlrm_optimizer(model, lr=0.001, embedding_lr=0.1):
    """Create Adam with separate LRs for embeddings (high) and top MLP (normal)."""
    from torch.optim import Adam
    dlrm = model.dlrm if hasattr(model, "dlrm") else model
    ebc_params = list(dlrm.ebc.parameters())
    mlp_params = list(dlrm.top_mlp.parameters())
    return Adam([
        {"params": ebc_params, "lr": embedding_lr},
        {"params": mlp_params, "lr": lr},
    ])


class DLRMTrainTask(nn.Module):
    """Wraps DLRM with BCEWithLogitsLoss for training.

    Unlike TwoTowerTrainTask, no temperature scaling is needed because
    the top MLP outputs unbounded logits directly.
    """

    def __init__(self, dlrm: DLRM):
        super().__init__()
        self.dlrm = dlrm
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, kjt, labels: torch.Tensor):
        logits = self.dlrm.score(kjt)
        loss = self.loss_fn(logits, labels)
        return loss, (loss.detach(), logits.detach(), labels.detach())
