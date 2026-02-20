# test_train.py
import torch
import os
from src.dataset import download_ml1m, load_ratings, split_data
from src.model import build_ebc, TwoTower, TwoTowerTrainTask
from src.train import train
from src.evaluate import evaluate
from src.distributed import init_process_group, wrap_with_dmp, unwrap_model

os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_process_group(device)

# Load data
download_ml1m()
df, n_users, n_items, _, _ = load_ratings()
train_df, test_df = split_data(df)

# Use only first 500 users for fast smoke test
small_users = list(range(500))
train_small = train_df[train_df["user_id"].isin(small_users)]
test_small = test_df[test_df["user_id"].isin(small_users)]

# Build model with meta device for DMP path
ebc = build_ebc(n_users, n_items, embedding_dim=64, device=torch.device("meta"))
two_tower = TwoTower(ebc, layer_sizes=[128, 64], device=device)
train_task = TwoTowerTrainTask(two_tower)

# Wrap with DMP
dmp_model, dense_optimizer = wrap_with_dmp(train_task, device, lr=0.001)

# Wrap eval function
eval_fn = lambda m: evaluate(m, test_small, train_small, n_items, n_neg=99, k=10, device=str(device))

# Train 2 epochs
history = train(dmp_model, dense_optimizer, train_small, n_items,
                epochs=2, batch_size=512, device=str(device),
                eval_fn=eval_fn, save_path="checkpoints/smoke_test.pt")

assert len(history) == 2, "Expected 2 epochs of history"
assert history[0][1] > 0, "Train loss should be positive"
print("\ntraining smoke test OK")
