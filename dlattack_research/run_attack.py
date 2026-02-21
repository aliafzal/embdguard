"""Execute full DLAttack."""
import torch, json, os
from torch.optim import Adam
from src.dataset import load_ratings, split_data
from src.model import build_ebc, TwoTower, TwoTowerTrainTask
from src.attack import run_dlattack
from src.evaluate import evaluate

os.makedirs("results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df, n_users, n_items, _, _ = load_ratings()
train_df, test_df = split_data(df)

# Build model on real device, load baseline weights
ebc = build_ebc(n_users, n_items, embedding_dim=64, device=device)
two_tower = TwoTower(ebc, layer_sizes=[128, 64], device=device)
model = TwoTowerTrainTask(two_tower)

state = torch.load("checkpoints/baseline.pt", map_location=device, weights_only=False)
model.load_state_dict(state, strict=False)
optimizer = Adam(model.parameters(), lr=0.001)
print("Loaded baseline model.")

# Choose target item: use a mid-popularity item for realistic attack
item_counts = train_df["item_id"].value_counts()
mid_items = item_counts[(item_counts > 20) & (item_counts < 100)].index.tolist()
target_item = int(mid_items[0])
print(f"Target item ID: {target_item} "
      f"(appears {item_counts[target_item]} times in training data)")

eval_fn = lambda m: evaluate(m, test_df, train_df, n_items, n_neg=99, k=10, device=str(device))

results, poisoned_train, model, optimizer = run_dlattack(
    model=model,
    optimizer=optimizer,
    train_df=train_df,
    test_df=test_df,
    n_users=n_users,
    n_items=n_items,
    target_item_id=target_item,
    embedding_dim=64,
    layer_sizes=[128, 64],
    rounds=5,
    m=5,
    n_filler=30,
    n_optim_steps=200,
    retrain_epochs=10,
    lr=0.001,
    device=str(device),
    eval_fn=eval_fn,
)

# Save poisoned model and results
torch.save(model.state_dict(), "checkpoints/attacked_model.pt")
poisoned_train.to_csv("results/poisoned_training_data.csv", index=False)

with open("results/attack_results.json", "w") as f:
    json.dump({"target_item": target_item, "rounds": results}, f, indent=2)

print("\n=== Attack Summary ===")
for key, metrics in results.items():
    hr = metrics.get('HR@K')
    ndcg = metrics.get('NDCG@K')
    print(f"  {key}: HR@10={hr:.4f}, NDCG@10={ndcg:.4f}")
