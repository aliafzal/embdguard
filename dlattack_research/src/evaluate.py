"""
Evaluation metrics for Two-Tower recommender.

Accepts either a plain TwoTower or a TwoTowerTrainTask, unwrapping as needed.
Uses KJT-based forward passes for scoring.
"""
import torch
import numpy as np

from src.model import make_kjt


def _unwrap(model):
    """Unwrap TrainTask wrapper to get the inner model, or return as-is."""
    if hasattr(model, "two_tower"):
        return model.two_tower
    if hasattr(model, "dlrm"):
        return model.dlrm
    return model


def _score_pairs(model, user_ids: torch.Tensor,
                 item_ids: torch.Tensor, batch_size: int = 65536) -> np.ndarray:
    """Score (user, item) pairs using model.score(), return numpy array."""
    all_scores = []
    for start in range(0, len(user_ids), batch_size):
        end = min(start + batch_size, len(user_ids))
        kjt = make_kjt(user_ids[start:end], item_ids[start:end])
        scores = model.score(kjt)
        all_scores.append(scores.cpu())
    return torch.cat(all_scores).numpy()


@torch.no_grad()
def evaluate(model, test_df, train_df, n_items: int,
             n_neg: int = 99, k: int = 10, device="cpu") -> dict:
    """
    For each user in test_df, score their held-out item against n_neg negatives.
    Batched on GPU for speed.
    """
    two_tower = _unwrap(model)
    two_tower.to(device)
    two_tower.eval()

    pos_items = dict(zip(test_df["user_id"], test_df["item_id"]))
    user_items = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    rng = np.random.default_rng(42)
    all_items = np.arange(n_items)

    n_eval = n_neg + 1
    user_list = []
    item_list = []

    for uid, pos_item in pos_items.items():
        seen = user_items.get(uid, set()) | {pos_item}
        cands = np.setdiff1d(all_items, list(seen))
        neg_items = rng.choice(cands, size=min(n_neg, len(cands)), replace=False)
        eval_items = np.concatenate([[pos_item], neg_items])
        user_list.append(np.full(len(eval_items), uid, dtype=np.int64))
        item_list.append(eval_items.astype(np.int64))

    all_users = torch.from_numpy(np.concatenate(user_list)).to(device)
    all_items_t = torch.from_numpy(np.concatenate(item_list)).to(device)

    all_scores = _score_pairs(two_tower, all_users, all_items_t)

    n_users_eval = len(pos_items)
    scores_2d = all_scores[:n_users_eval * n_eval].reshape(n_users_eval, n_eval)
    pos_scores = scores_2d[:, 0:1]
    neg_scores = scores_2d[:, 1:]
    ranks = (neg_scores >= pos_scores).sum(axis=1) + 1

    hr = (ranks <= k).mean()
    ndcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0).mean()
    return {"HR@K": float(hr), "NDCG@K": float(ndcg), "K": k}


@torch.no_grad()
def target_item_hit_ratio(model, target_item_id: int,
                           test_df, train_df, n_items: int,
                           n_neg: int = 99, k: int = 10,
                           device: str = "cpu") -> float:
    """
    Measures how often target_item appears in the top-K recommendation list.
    """
    two_tower = _unwrap(model)
    two_tower.to(device)
    two_tower.eval()

    user_items = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    rng = np.random.default_rng(42)
    all_items = np.arange(n_items)
    users = test_df["user_id"].unique()

    n_eval = n_neg + 1
    user_list = []
    item_list = []

    for uid in users:
        seen = user_items.get(uid, set())
        cands = np.setdiff1d(all_items, list(seen | {target_item_id}))
        negs = rng.choice(cands, size=min(n_neg, len(cands)), replace=False)
        eval_items = np.concatenate([[target_item_id], negs])
        user_list.append(np.full(len(eval_items), uid, dtype=np.int64))
        item_list.append(eval_items.astype(np.int64))

    all_users = torch.from_numpy(np.concatenate(user_list)).to(device)
    all_items_t = torch.from_numpy(np.concatenate(item_list)).to(device)

    all_scores = _score_pairs(two_tower, all_users, all_items_t)

    scores_2d = all_scores[:len(users) * n_eval].reshape(len(users), n_eval)
    pos_scores = scores_2d[:, 0:1]
    neg_scores = scores_2d[:, 1:]
    ranks = (neg_scores >= pos_scores).sum(axis=1) + 1

    return float((ranks <= k).mean())
