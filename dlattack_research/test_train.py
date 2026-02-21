# test_train.py
import torch
import os
from src.dataset import download_ml1m, load_ratings, split_data
from src.model import build_ebc, TwoTower, TwoTowerTrainTask, make_optimizer
from src.train import train
from src.evaluate import evaluate

os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
download_ml1m()
df, n_users, n_items, _, _ = load_ratings()
train_df, test_df = split_data(df)

# Use only first 500 users for fast smoke test
small_users = list(range(500))
train_small = train_df[train_df["user_id"].isin(small_users)]
test_small = test_df[test_df["user_id"].isin(small_users)]

# Build model
ebc = build_ebc(n_users, n_items, embedding_dim=64, device=device)
two_tower = TwoTower(ebc, layer_sizes=[128, 64], device=device)
model = TwoTowerTrainTask(two_tower)
optimizer = make_optimizer(model)

# Wrap eval function
eval_fn = lambda m: evaluate(m, test_small, train_small, n_items, n_neg=99, k=10, device=str(device))

# Train 2 epochs
history = train(model, optimizer, train_small, n_items,
                epochs=2, batch_size=512, device=str(device),
                eval_fn=eval_fn, save_path="checkpoints/smoke_test.pt")

assert len(history) == 2, "Expected 2 epochs of history"
assert history[0][1] > 0, "Train loss should be positive"
print("\ntraining smoke test OK")
