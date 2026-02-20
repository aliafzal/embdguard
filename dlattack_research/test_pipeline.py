"""
Runs the entire pipeline end-to-end on a tiny subset (300 users, 2 attack rounds)
to confirm no crashes before running the full experiment.
Uses full DMP pipeline.
"""
import torch, pandas as pd, os
from src.dataset import download_ml1m, load_ratings, split_data
from src.model import build_ebc, TwoTower, TwoTowerTrainTask
from src.train import train
from src.attack import run_dlattack
from src.evaluate import evaluate, target_item_hit_ratio
from src.detect import detect_fake_users
from src.distributed import init_process_group, wrap_with_dmp, unwrap_model

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_process_group(device)

download_ml1m()
df, n_users, n_items, _, _ = load_ratings()
train_df, test_df = split_data(df)

# Use only 300 users for speed
small_uids = list(range(300))
train_small = train_df[train_df["user_id"].isin(small_uids)]
test_small = test_df[test_df["user_id"].isin(small_uids)]

# Build DMP-wrapped model
ebc = build_ebc(n_users, n_items, embedding_dim=32, device=torch.device("meta"))
two_tower = TwoTower(ebc, layer_sizes=[128, 64], device=device)
train_task = TwoTowerTrainTask(two_tower)
dmp_model, dense_optimizer = wrap_with_dmp(train_task, device, lr=0.001)

eval_fn = lambda m: evaluate(m, test_small, train_small, n_items, n_neg=50, k=10, device=str(device))

# Quick train
train(dmp_model, dense_optimizer, train_small, n_items,
      epochs=3, batch_size=512, device=str(device), eval_fn=eval_fn,
      save_path="checkpoints/pipeline_test.pt")

# Quick attack (2 rounds, 2 fake users, 50 optimization steps)
target = int(train_small["item_id"].value_counts().index[5])
results, poisoned, dmp_model, dense_optimizer = run_dlattack(
    dmp_model, dense_optimizer, train_small, test_small, n_users, n_items,
    target_item_id=target, rounds=2, m=2,
    n_filler=10, n_optim_steps=50, retrain_epochs=3,
    lr=0.001, device=str(device),
    eval_fn=eval_fn,
)

assert "round_0_clean" in results, "Missing pre-attack metrics"
assert "round_1" in results, "Missing round 1 metrics"
assert "round_2" in results, "Missing round 2 metrics"

# Detection
two_tower = unwrap_model(dmp_model)
flagged = detect_fake_users(poisoned, two_tower, target, n_items)
fake_ids = set(poisoned[poisoned["user_id"] >= n_users]["user_id"].unique())
print(f"\nFake users: {fake_ids}")
print(f"Flagged:    {flagged}")
print(f"\nFull pipeline smoke test OK -- ready for full run")
