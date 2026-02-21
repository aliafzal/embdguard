# test_detect.py
import torch, pandas as pd, numpy as np
from src.model import build_ebc, TwoTower
from src.detect import detect_fake_users, compute_user_anomaly_scores

N_USERS, N_ITEMS, TARGET = 100, 200, 42

# Build a plain TwoTower for testing
ebc = build_ebc(N_USERS + 10, N_ITEMS, embedding_dim=32, device=torch.device("cpu"))
model = TwoTower(ebc, layer_sizes=[64, 32], device=torch.device("cpu"))

# Set up item embeddings so items 40-50 are close to TARGET (42) in embedding space.
with torch.no_grad():
    base_emb = model.ebc.embedding_bags["t_item_id"].weight[TARGET].clone()
    for i in range(40, 51):
        model.ebc.embedding_bags["t_item_id"].weight[i] = base_emb + torch.randn(32) * 0.01

# Synthetic clean users -- interact with items spread across the full range
rows = []
for u in range(N_USERS):
    for i in range(5):
        item = (u * 7 + i * 37) % N_ITEMS
        rows.append({"user_id": u, "item_id": item, "label": 1})

# Synthetic fake users -- interact with target item + items similar to target
for fu in range(N_USERS, N_USERS + 10):
    rows.append({"user_id": fu, "item_id": TARGET, "label": 1})
    rows.append({"user_id": fu, "item_id": TARGET + 1, "label": 1})
    rows.append({"user_id": fu, "item_id": TARGET + 2, "label": 1})
    rows.append({"user_id": fu, "item_id": TARGET - 1, "label": 1})
    rows.append({"user_id": fu, "item_id": TARGET - 2, "label": 1})

df = pd.DataFrame(rows)

flagged = detect_fake_users(df, model, TARGET, N_ITEMS, threshold_percentile=90)

# Fake users (ids 100-109) should be flagged more than clean users
fake_ids = set(range(N_USERS, N_USERS + 10))
clean_ids = set(range(N_USERS))
fake_flagged = len(flagged & fake_ids)
clean_flagged = len(flagged & clean_ids)

print(f"\n  Fake users flagged:  {fake_flagged}/10")
print(f"  Clean users flagged: {clean_flagged}/100")
assert fake_flagged > clean_flagged, "Detector should flag more fake than clean users"
print("detection module OK")
