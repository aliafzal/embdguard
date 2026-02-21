"""Compare clean vs attacked models using plain TwoTower for evaluation."""
import torch, json
from src.dataset import load_ratings, split_data
from src.model import build_ebc, TwoTower
from src.evaluate import evaluate, target_item_hit_ratio

df, n_users, n_items, _, _ = load_ratings()
train_df, test_df = split_data(df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("results/attack_results.json") as f:
    attack_meta = json.load(f)
target_item = attack_meta["target_item"]


def _load_two_tower(ckpt_path):
    """Load a plain TwoTower from a TwoTowerTrainTask checkpoint.
    Infers user/item counts from checkpoint shapes."""
    from src.distributed import deshard_state_dict
    state = deshard_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)
    )
    tt_state = {}
    for k, v in state.items():
        if k.startswith("two_tower."):
            tt_state[k[len("two_tower."):]] = v
    if tt_state:
        state = tt_state
    ckpt_n_users = state["ebc.embedding_bags.t_user_id.weight"].shape[0]
    ckpt_n_items = state["ebc.embedding_bags.t_item_id.weight"].shape[0]
    ebc = build_ebc(ckpt_n_users, ckpt_n_items, embedding_dim=64, device=device)
    tt = TwoTower(ebc, layer_sizes=[128, 64], device=device)
    tt.load_state_dict(state, strict=False)
    return tt


clean_model = _load_two_tower("checkpoints/baseline.pt")
attacked_model = _load_two_tower("checkpoints/attacked_model.pt")

clean_thr = target_item_hit_ratio(clean_model, target_item, test_df, train_df, n_items, device=str(device))
attacked_thr = target_item_hit_ratio(attacked_model, target_item, test_df, train_df, n_items, device=str(device))

clean_overall = evaluate(clean_model, test_df, train_df, n_items, device=str(device))
attacked_overall = evaluate(attacked_model, test_df, train_df, n_items, device=str(device))

print("\n=== Attack Evaluation ===")
print(f"  Target item ID: {target_item}")
print(f"  Target item HR@10 (clean):   {clean_thr:.4f}")
print(f"  Target item HR@10 (attacked):{attacked_thr:.4f}  (up = attack worked)")
print(f"  Overall HR@10   (clean):     {clean_overall['HR@K']:.4f}")
print(f"  Overall HR@10   (attacked):  {attacked_overall['HR@K']:.4f}  (should stay ~stable)")
