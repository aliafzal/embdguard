"""Train a clean baseline Two-Tower model."""
import torch, os, json
from torch.optim import Adam
from src.dataset import download_ml1m, load_ratings, split_data
from src.model import build_ebc, TwoTower, TwoTowerTrainTask
from src.train import train
from src.evaluate import evaluate

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

download_ml1m()
df, n_users, n_items, _, _ = load_ratings()
train_df, test_df = split_data(df)

# Build model
ebc = build_ebc(n_users, n_items, embedding_dim=64, device=device)
two_tower = TwoTower(ebc, layer_sizes=[128, 64], device=device)
model = TwoTowerTrainTask(two_tower)
optimizer = Adam(model.parameters(), lr=0.001)

eval_fn = lambda m: evaluate(m, test_df, train_df, n_items, n_neg=99, k=10, device=str(device))

print(f"\n=== Baseline Training | Users={n_users} Items={n_items} ===")
history = train(model, optimizer, train_df, n_items,
                epochs=50, batch_size=4096, n_neg=4,
                device=str(device), eval_fn=eval_fn,
                save_path="checkpoints/baseline.pt")

final_metrics = history[-1][2]
print(f"\nFinal Baseline -> HR@10={final_metrics['HR@K']:.4f}, NDCG@10={final_metrics['NDCG@K']:.4f}")

with open("results/baseline_metrics.json", "w") as f:
    json.dump({"n_users": n_users, "n_items": n_items,
               "HR@10": final_metrics.get("HR@K"),
               "NDCG@10": final_metrics.get("NDCG@K")}, f, indent=2)
