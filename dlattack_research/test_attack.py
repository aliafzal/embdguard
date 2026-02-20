"""
Smoke test for the attack module using a tiny synthetic dataset.
Uses a plain TwoTower (no DMP needed) for attack smoke test.
"""
import torch, numpy as np, pandas as pd
from src.model import build_ebc, TwoTower
from src.attack import optimize_fake_user, generate_fake_users

N_USERS, N_ITEMS = 50, 100
TARGET_ITEM = 42
M_FAKE = 3

# Build a randomly-initialized plain TwoTower (no DMP needed for attack)
ebc = build_ebc(N_USERS, N_ITEMS, embedding_dim=32, device=torch.device("cpu"))
model = TwoTower(ebc, layer_sizes=[64, 32], device=torch.device("cpu"))

# --- Test 1: optimize_fake_user returns valid binary vector ---
vec = optimize_fake_user(model, TARGET_ITEM, N_ITEMS, n_filler=10, n_optim_steps=50)
assert vec.shape == (N_ITEMS,), f"Expected ({N_ITEMS},), got {vec.shape}"
assert vec[TARGET_ITEM] == 1.0, "Target item must be in fake user interactions"
assert vec.sum() >= 2, "Fake user should have target + at least 1 filler"
assert set(np.unique(vec)) <= {0.0, 1.0}, "Vector should be binary"
print(f"  optimize_fake_user: {int(vec.sum())} interactions, target included")

# --- Test 2: generate_fake_users returns proper DataFrame ---
fake_df = generate_fake_users(model, TARGET_ITEM, N_USERS, N_ITEMS,
                              m=M_FAKE, n_filler=10, n_optim_steps=50)
assert "user_id" in fake_df.columns and "item_id" in fake_df.columns
assert fake_df["user_id"].nunique() == M_FAKE, \
    f"Expected {M_FAKE} fake users, got {fake_df['user_id'].nunique()}"
assert TARGET_ITEM in fake_df["item_id"].values, \
    "Target item should appear in fake user interactions"
print(f"  generate_fake_users: {M_FAKE} fake users, {len(fake_df)} rows")

# --- Test 3: fake user IDs do not collide with real user IDs ---
assert fake_df["user_id"].min() >= N_USERS, \
    f"Fake user IDs should start at {N_USERS}"
print(f"  Fake user IDs: {sorted(fake_df['user_id'].unique())} (no collision)")

print("\nattack module smoke test OK")
