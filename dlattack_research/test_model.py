# test_model.py
import torch
from src.model import build_ebc, TwoTower, TwoTowerTrainTask, make_kjt

# Build model components
ebc = build_ebc(n_users=100, n_items=200, embedding_dim=32, device=torch.device("cpu"))
model = TwoTower(ebc, layer_sizes=[64, 32], device=torch.device("cpu"))

# Test forward pass with KJT
user_ids = torch.tensor([0, 1, 2, 5, 7])
item_ids = torch.tensor([10, 20, 30, 40, 50])
kjt = make_kjt(user_ids, item_ids)
user_emb, item_emb = model(kjt)

assert user_emb.shape == (5, 32), f"Expected user_emb shape (5, 32), got {user_emb.shape}"
assert item_emb.shape == (5, 32), f"Expected item_emb shape (5, 32), got {item_emb.shape}"

# Scores via dot product
scores = (user_emb * item_emb).sum(dim=1)
assert scores.shape == (5,), f"Expected shape (5,), got {scores.shape}"
assert not scores.isnan().any(), "NaN in scores"
assert scores.min() >= -1.01, "Scores out of [-1,1] (normalized dot product)"
assert scores.max() <= 1.01, "Scores out of [-1,1] (normalized dot product)"

# Test item embeddings access
item_embs = model.get_item_embeddings()
assert item_embs.shape == (200, 32), f"Expected (200, 32), got {item_embs.shape}"

# Test TwoTowerTrainTask
train_task = TwoTowerTrainTask(model)
labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
loss, (loss_d, logits, labels_d) = train_task(kjt, labels)
assert loss.shape == (), f"Expected scalar loss, got {loss.shape}"
assert loss.item() > 0, "Loss should be positive"

# Test resize_user_embedding
model.resize_user_embedding(150)
new_item_embs = model.get_item_embeddings()
assert new_item_embs.shape == (200, 32), "Item embeddings should be unchanged after user resize"

print(f"Scores: {scores.detach().tolist()}")
print(f"Item embedding table shape: {item_embs.shape}")
print(f"Train task loss: {loss.item():.4f}")
print("model forward pass OK")
