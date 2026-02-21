"""
Full pipeline: download -> train baseline -> attack -> evaluate -> detect
All phases use DMP-wrapped models for fused TBE kernels.

Usage:
  python main.py --phase all          # run everything
  python main.py --phase baseline     # only train clean model
  python main.py --phase attack       # only run attack (requires baseline)
  python main.py --phase evaluate     # only evaluate (requires both checkpoints)
"""
import argparse, json, os, torch
from src.dataset import download_ml1m, load_ratings, split_data
from src.model import build_ebc, TwoTower, TwoTowerTrainTask
from src.train import train
from src.attack import run_dlattack
from src.evaluate import evaluate, target_item_hit_ratio
from src.detect import detect_fake_users
from src.distributed import (
    init_process_group, wrap_with_dmp, unwrap_model, extract_state_dict,
)


def get_target_item(train_df, min_count=20, max_count=100):
    counts = train_df["item_id"].value_counts()
    mid = counts[(counts > min_count) & (counts < max_count)].index.tolist()
    return int(mid[0]) if mid else int(train_df["item_id"].mode()[0])


def _build_dmp_model(n_users, n_items, embed_dim, layer_sizes, device, lr):
    """Build a fresh DMP-wrapped model."""
    ebc = build_ebc(n_users, n_items, embed_dim, device=torch.device("meta"))
    two_tower = TwoTower(ebc, layer_sizes=layer_sizes, device=device)
    train_task = TwoTowerTrainTask(two_tower)
    return wrap_with_dmp(train_task, device, lr=lr)


def _load_plain_two_tower(n_users, n_items, embed_dim, layer_sizes, device, ckpt_path):
    """Load a plain TwoTower from checkpoint (for evaluation without DMP)."""
    ebc = build_ebc(n_users, n_items, embed_dim, device=device)
    two_tower = TwoTower(ebc, layer_sizes=layer_sizes, device=device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # The checkpoint is from TwoTowerTrainTask, so filter for two_tower keys
    tt_state = {}
    for k, v in state.items():
        if k.startswith("two_tower."):
            tt_state[k[len("two_tower."):]] = v
    if tt_state:
        two_tower.load_state_dict(tt_state, strict=False)
    else:
        two_tower.load_state_dict(state, strict=False)
    return two_tower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["all", "baseline", "attack", "evaluate"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_process_group(device)

    download_ml1m()
    df, n_users, n_items, _, _ = load_ratings()
    train_df, test_df = split_data(df)
    layer_sizes = [128, 64]

    if args.phase in ("all", "baseline"):
        dmp_model, dense_optimizer = _build_dmp_model(
            n_users, n_items, args.embed_dim, layer_sizes, device, args.lr
        )
        eval_fn = lambda m: evaluate(m, test_df, train_df, n_items, n_neg=99, k=10, device=str(device))
        train(dmp_model, dense_optimizer, train_df, n_items,
              epochs=args.epochs, device=str(device), eval_fn=eval_fn,
              save_path="checkpoints/baseline.pt")

    if args.phase in ("all", "attack"):
        dmp_model, dense_optimizer = _build_dmp_model(
            n_users, n_items, args.embed_dim, layer_sizes, device, args.lr
        )
        # Load baseline weights
        state = torch.load("checkpoints/baseline.pt", map_location=device, weights_only=False)
        dmp_model.module.load_state_dict(state)

        target = get_target_item(train_df)
        eval_fn = lambda m: evaluate(m, test_df, train_df, n_items, n_neg=99, k=10, device=str(device))

        results, poisoned, dmp_model, dense_optimizer = run_dlattack(
            dmp_model, dense_optimizer, train_df, test_df, n_users, n_items,
            target_item_id=target, rounds=args.rounds, m=args.m,
            lr=args.lr, device=str(device), eval_fn=eval_fn,
        )
        torch.save(extract_state_dict(dmp_model), "checkpoints/attacked_model.pt")
        poisoned.to_csv("results/poisoned_train.csv", index=False)
        with open("results/attack_results.json", "w") as f:
            json.dump({"target_item": target, "rounds": results}, f, indent=2)
        print(f"\nAttack complete. Results saved to results/")

    if args.phase in ("all", "evaluate"):
        with open("results/attack_results.json") as f:
            meta = json.load(f)
        target = meta["target_item"]
        poisoned = __import__("pandas").read_csv("results/poisoned_train.csv")

        clean_tt = _load_plain_two_tower(
            n_users, n_items, args.embed_dim, layer_sizes, device, "checkpoints/baseline.pt"
        )
        attacked_tt = _load_plain_two_tower(
            n_users, n_items, args.embed_dim, layer_sizes, device, "checkpoints/attacked_model.pt"
        )

        print("\n=== Final Evaluation ===")
        for label, tt in [("Clean", clean_tt), ("Attacked", attacked_tt)]:
            overall = evaluate(tt, test_df, train_df, n_items, device=str(device))
            target_hr = target_item_hit_ratio(tt, target, test_df, train_df, n_items, device=str(device))
            print(f"  [{label}] Overall HR@10={overall['HR@K']:.4f} | "
                  f"Target item HR@10={target_hr:.4f}")

        # Detection
        print("\n=== Detection (TIA) ===")
        flagged = detect_fake_users(poisoned, attacked_tt, target, n_items)
        fake_ids = set(poisoned[poisoned["user_id"] >= n_users]["user_id"].unique())
        if fake_ids:
            precision = len(flagged & fake_ids) / max(len(flagged), 1)
            recall = len(flagged & fake_ids) / max(len(fake_ids), 1)
            print(f"  Detection precision: {precision:.3f}  recall: {recall:.3f}")


if __name__ == "__main__":
    main()
