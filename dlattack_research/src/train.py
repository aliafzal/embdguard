"""
Training loop for Two-Tower recommender.

Uses KJT-based forward passes. All parameters (embeddings + MLP) are
optimized with a single Adam optimizer.
"""
import torch
import pandas as pd

from src.model import make_kjt


def _negative_sample_tensors(user_t, item_t, n_items, n_neg, device):
    """Fast GPU-native negative sampling. All ops on device, no Python loops."""
    n_pos = len(user_t)

    neg_users = user_t.repeat_interleave(n_neg)
    neg_items = torch.randint(0, n_items, (n_pos * n_neg,), device=device)
    neg_labels = torch.zeros(n_pos * n_neg, device=device)

    pos_labels = torch.ones(n_pos, device=device)

    all_users = torch.cat([user_t, neg_users])
    all_items = torch.cat([item_t, neg_items])
    all_labels = torch.cat([pos_labels, neg_labels])

    perm = torch.randperm(len(all_users), device=device)
    return all_users[perm], all_items[perm], all_labels[perm]


def train(model, optimizer, train_df, n_items, epochs=20,
          batch_size=1024, n_neg=4, device="cpu",
          save_path=None, eval_fn=None) -> list:
    """
    Training loop for TwoTowerTrainTask with Adam optimizer.

    Args:
        model: TwoTowerTrainTask
        optimizer: Adam optimizer for all parameters
        train_df: DataFrame with user_id, item_id columns
        n_items: total number of items
        epochs: number of training epochs
        batch_size: mini-batch size
        n_neg: negative samples per positive
        device: device string
        save_path: path to save model checkpoint
        eval_fn: optional evaluation callback
    """
    history = []
    print(f"  Training: {len(train_df)} interactions, epochs={epochs}, "
          f"batch={batch_size}, n_neg={n_neg}")

    # Move positive interactions to device once
    pos_users = torch.tensor(train_df["user_id"].values, dtype=torch.long, device=device)
    pos_items = torch.tensor(train_df["item_id"].values, dtype=torch.long, device=device)

    for epoch in range(1, epochs + 1):
        model.train()

        users, items, labels = _negative_sample_tensors(
            pos_users, pos_items, n_items, n_neg, device
        )

        n_samples = len(users)
        effective_batch = min(batch_size, n_samples)
        total_loss = 0.0

        for start in range(0, n_samples, effective_batch):
            end = min(start + effective_batch, n_samples)
            kjt = make_kjt(users[start:end], items[start:end])
            batch_labels = labels[start:end]

            loss, (_loss_detached, _logits, _labels) = model(kjt, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (end - start)

        train_loss = total_loss / n_samples

        metrics = {}
        if eval_fn is not None and (epoch % 5 == 0 or epoch == epochs):
            metrics = eval_fn(model.two_tower)
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
                  f"HR@{metrics['K']}={metrics['HR@K']:.4f} | "
                  f"NDCG@{metrics['K']}={metrics['NDCG@K']:.4f}")
        else:
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f}")

        history.append((epoch, train_loss, metrics))

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"  Model saved -> {save_path}")
    return history


# Keep for backward compat
class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]
