"""
Full pipeline for DLAttack on Amazon Reviews 2023 (Video_Games, 5-core).

Phases: baseline, attack, evaluate, all.

Usage:
  python amazon_main.py --phase all          # run everything
  python amazon_main.py --phase baseline     # only train clean model
  python amazon_main.py --phase attack       # only run attack (requires baseline)
  python amazon_main.py --phase evaluate     # only evaluate (requires both checkpoints)

Smoke test (local CPU):
  python amazon_main.py --phase all --epochs 3 --rounds 1 --m 2
"""
import argparse, json, os, torch
import pandas as pd

from src.amazon_dataset import load_amazon_reviews
from src.dataset import split_data
from src.model import build_ebc, TwoTower, TwoTowerTrainTask, make_optimizer
from src.train import train
from src.attack import run_dlattack
from src.evaluate import evaluate, target_item_hit_ratio
from src.detect import (
    detect_fake_users,
    sweep_detection_thresholds,
    compute_detection_auc,
)

CKPT_DIR = "checkpoints/amazon"
RESULTS_DIR = "results/amazon"


def get_target_item(train_df, min_count=20, max_count=200):
    counts = train_df["item_id"].value_counts()
    mid = counts[(counts > min_count) & (counts < max_count)].index.tolist()
    return int(mid[0]) if mid else int(train_df["item_id"].mode()[0])


def _build_model(n_users, n_items, embed_dim, layer_sizes, device, lr):
    ebc = build_ebc(n_users, n_items, embed_dim, device=device)
    two_tower = TwoTower(ebc, layer_sizes=layer_sizes, device=device)
    model = TwoTowerTrainTask(two_tower)
    optimizer = make_optimizer(model, lr=lr)
    return model, optimizer


def _load_plain_two_tower(embed_dim, layer_sizes, device, ckpt_path):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    tt_state = {}
    for k, v in state.items():
        if k.startswith("two_tower."):
            tt_state[k[len("two_tower."):]] = v
    if tt_state:
        state = tt_state
    ckpt_n_users = state["ebc.embedding_bags.t_user_id.weight"].shape[0]
    ckpt_n_items = state["ebc.embedding_bags.t_item_id.weight"].shape[0]
    ebc = build_ebc(ckpt_n_users, ckpt_n_items, embed_dim, device=device)
    two_tower = TwoTower(ebc, layer_sizes=layer_sizes, device=device)
    two_tower.load_state_dict(state, strict=False)
    return two_tower


def _print_detection_table(sweep_results, auc):
    print("\n  Detection (TIA):")
    print(f"  {'Threshold':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for label, m in sweep_results.items():
        print(f"  {label:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    print(f"  {'AUC-ROC':<12} {auc:>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="DLAttack pipeline on Amazon Reviews 2023"
    )
    parser.add_argument("--phase", default="all",
                        choices=["all", "baseline", "attack", "evaluate"])
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--n_filler", type=int, default=50)
    parser.add_argument("--n_optim_steps", type=int, default=300)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df, n_users, n_items, _, _ = load_amazon_reviews(category=args.category)
    train_df, test_df = split_data(df)
    layer_sizes = [128, 64]

    baseline_path = os.path.join(CKPT_DIR, "baseline.pt")
    attacked_path = os.path.join(CKPT_DIR, "attacked_model.pt")
    results_path = os.path.join(RESULTS_DIR, "attack_results.json")
    poisoned_path = os.path.join(RESULTS_DIR, "poisoned_train.csv")

    # ── Baseline ──────────────────────────────────────────────────────
    if args.phase in ("all", "baseline"):
        print(f"\n{'='*60}")
        print(f"  BASELINE TRAINING  ({args.category})")
        print(f"{'='*60}")
        model, optimizer = _build_model(
            n_users, n_items, args.embed_dim, layer_sizes, device, args.lr
        )
        eval_fn = lambda m: evaluate(
            m, test_df, train_df, n_items, n_neg=99, k=10, device=str(device)
        )
        train(
            model, optimizer, train_df, n_items,
            epochs=args.epochs, batch_size=args.batch_size,
            device=str(device), eval_fn=eval_fn,
            save_path=baseline_path,
        )

    # ── Attack ────────────────────────────────────────────────────────
    if args.phase in ("all", "attack"):
        print(f"\n{'='*60}")
        print(f"  DLATTACK  ({args.category})")
        print(f"{'='*60}")
        ebc = build_ebc(n_users, n_items, args.embed_dim, device=device)
        two_tower = TwoTower(ebc, layer_sizes=layer_sizes, device=device)
        model = TwoTowerTrainTask(two_tower)
        state = torch.load(baseline_path, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
        optimizer = make_optimizer(model, lr=args.lr)

        target = get_target_item(train_df, max_count=200)
        print(f"  Target item: {target}")
        eval_fn = lambda m: evaluate(
            m, test_df, train_df, n_items, n_neg=99, k=10, device=str(device)
        )

        results, poisoned, model, optimizer = run_dlattack(
            model, optimizer, train_df, test_df, n_users, n_items,
            target_item_id=target, embedding_dim=args.embed_dim,
            layer_sizes=layer_sizes, rounds=args.rounds, m=args.m,
            n_filler=args.n_filler, n_optim_steps=args.n_optim_steps,
            lr=args.lr, device=str(device), eval_fn=eval_fn,
        )
        torch.save(model.state_dict(), attacked_path)
        poisoned.to_csv(poisoned_path, index=False)
        with open(results_path, "w") as f:
            json.dump({
                "target_item": target,
                "n_users": n_users,
                "n_items": n_items,
                "category": args.category,
                "rounds": results,
            }, f, indent=2)
        print(f"\nAttack complete. Results saved to {RESULTS_DIR}/")

    # ── Evaluate ──────────────────────────────────────────────────────
    if args.phase in ("all", "evaluate"):
        print(f"\n{'='*60}")
        print(f"  FINAL EVALUATION  ({args.category})")
        print(f"{'='*60}")

        with open(results_path) as f:
            meta = json.load(f)
        target = meta["target_item"]
        n_real_users = meta.get("n_users", n_users)
        poisoned = pd.read_csv(poisoned_path)

        clean_tt = _load_plain_two_tower(
            args.embed_dim, layer_sizes, device, baseline_path
        )
        attacked_tt = _load_plain_two_tower(
            args.embed_dim, layer_sizes, device, attacked_path
        )

        print("\n  === Overall Metrics ===")
        for label, tt in [("Clean", clean_tt), ("Attacked", attacked_tt)]:
            overall = evaluate(tt, test_df, train_df, n_items, device=str(device))
            target_hr = target_item_hit_ratio(
                tt, target, test_df, train_df, n_items, device=str(device)
            )
            print(f"  [{label}] HR@10={overall['HR@K']:.4f} | "
                  f"NDCG@10={overall['NDCG@K']:.4f} | "
                  f"Target HR@10={target_hr:.4f}")

        # Detection
        print("\n  === Detection (TIA) ===")
        flagged = detect_fake_users(poisoned, attacked_tt, target, n_items)
        fake_ids = set(
            poisoned[poisoned["user_id"] >= n_real_users]["user_id"].unique()
        )
        if fake_ids:
            precision = len(flagged & fake_ids) / max(len(flagged), 1)
            recall = len(flagged & fake_ids) / max(len(fake_ids), 1)
            print(f"  Detection precision: {precision:.3f}  recall: {recall:.3f}")

        # Detailed sweep
        sweep = sweep_detection_thresholds(
            poisoned, attacked_tt, target, n_items, n_real_users
        )
        auc = compute_detection_auc(
            poisoned, attacked_tt, target, n_items, n_real_users
        )
        _print_detection_table(sweep, auc)

        # Save detection results
        detection_results = {
            "threshold_sweep": {
                k: {sk: sv for sk, sv in v.items()}
                for k, v in sweep.items()
            },
            "auc_roc": auc,
        }
        det_path = os.path.join(RESULTS_DIR, "detection_results.json")
        with open(det_path, "w") as f:
            json.dump(detection_results, f, indent=2)
        print(f"\n  Detection results saved to {det_path}")


if __name__ == "__main__":
    main()
