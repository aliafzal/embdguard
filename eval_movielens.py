"""Evaluate EmbdGuard detectors on the full MovieLens-1M DLAttack pipeline.

Loads a trained baseline, runs a multi-round DLAttack with EmbdGuard
monitoring the poisoned retraining, and reports which detectors fire and when.

Usage:
    python eval_movielens.py
    python eval_movielens.py --rounds 3 --m 10
"""
import sys, os, argparse, warnings
import importlib

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DLATTACK_DIR = os.path.join(REPO_ROOT, "dlattack_research")

# Import EmbdGuard modules first (from REPO_ROOT/src/)
sys.path.insert(0, REPO_ROOT)

from src.guard import EmbdGuard
from src.detectors.gradient_anomaly import GradientAnomalyDetector
from src.detectors.access_frequency import AccessFrequencyDetector
from src.detectors.embedding_drift import EmbeddingDriftDetector
from src.detectors.gradient_distribution import GradientDistributionDetector
from src.detectors.temporal_access import TemporalAccessDetector

# Save EmbdGuard's src module reference
import src as embdguard_src

# Now import dlattack modules by temporarily swapping sys.path
# We need dlattack_research/ on path so its internal `from src.model import ...` works
sys.path.insert(0, DLATTACK_DIR)
# Remove the cached `src` module so Python re-discovers from dlattack dir
del sys.modules["src"]
for key in list(sys.modules.keys()):
    if key.startswith("src."):
        del sys.modules[key]

import src.dataset as dl_dataset
import src.model as dl_model
import src.train as dl_train
import src.attack as dl_attack
import src.evaluate as dl_evaluate

# Restore EmbdGuard's src module
sys.modules["src"] = embdguard_src
sys.path.remove(DLATTACK_DIR)

import torch
import numpy as np
import pandas as pd


def get_target_item(train_df, min_count=20, max_count=100):
    counts = train_df["item_id"].value_counts()
    mid = counts[(counts > min_count) & (counts < max_count)].index.tolist()
    return int(mid[0]) if mid else int(train_df["item_id"].mode()[0])


def main():
    parser = argparse.ArgumentParser(description="EmbdGuard on MovieLens-1M DLAttack")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--retrain-epochs", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cpu")
    layer_sizes = [128, 64]

    # dataset.py uses relative paths from dlattack_research/
    os.chdir(DLATTACK_DIR)

    # ── Load data ──
    print("=" * 60)
    print("  Loading MovieLens-1M")
    print("=" * 60)
    dl_dataset.download_ml1m()
    df, n_users, n_items, _, _ = dl_dataset.load_ratings()
    train_df, test_df = dl_dataset.split_data(df)
    target = get_target_item(train_df)
    print(f"  Target item: {target}")
    print(f"  Train: {len(train_df)} interactions, Test: {len(test_df)}")

    # ── Load baseline ──
    print(f"\n{'=' * 60}")
    print("  Loading baseline model")
    print("=" * 60)
    ebc = dl_model.build_ebc(n_users, n_items, args.embed_dim, device=device)
    two_tower = dl_model.TwoTower(ebc, layer_sizes=layer_sizes, device=device)
    model = dl_model.TwoTowerTrainTask(two_tower)
    state = torch.load("checkpoints/baseline.pt", map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    optimizer = dl_model.make_optimizer(model, lr=args.lr)

    eval_fn = lambda m: dl_evaluate.evaluate(
        m, test_df, train_df, n_items, n_neg=99, k=10, device=str(device)
    )
    baseline_metrics = eval_fn(model.two_tower)
    print(f"  Baseline HR@10={baseline_metrics['HR@K']:.4f} | "
          f"NDCG@10={baseline_metrics['NDCG@K']:.4f}")

    # ── Run DLAttack with EmbdGuard monitoring ──
    poisoned_train = train_df.copy()
    fake_user_id_start = n_users
    all_round_alerts = {}
    total_alerts_by_detector = {}

    for r in range(1, args.rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"  ATTACK ROUND {r}/{args.rounds} — target_item={target}")
        print("=" * 60)

        # Step 1: Build surrogate, retrain
        max_user_id = int(poisoned_train["user_id"].max()) + 1
        surrogate = dl_attack._build_surrogate(
            model, max_user_id, n_items, args.embed_dim, layer_sizes, str(device)
        )
        print(f"  Retraining surrogate...")
        surr_task = dl_model.TwoTowerTrainTask(surrogate)
        surr_optimizer = dl_model.make_optimizer(surr_task, lr=args.lr)
        dl_train.train(surr_task, surr_optimizer, poisoned_train, n_items,
                       epochs=args.retrain_epochs, batch_size=2048,
                       device=str(device), eval_fn=None, save_path=None)
        surrogate = surr_task.two_tower

        # Step 2: Generate fake users
        print(f"  Generating {args.m} fake users...")
        n_current_fake = (r - 1) * args.m
        fake_df = dl_attack.generate_fake_users(
            surrogate, target, n_users=n_users + n_current_fake,
            n_items=n_items, m=args.m, n_filler=30, n_optim_steps=200,
            fake_user_id_start=fake_user_id_start, device=str(device),
        )
        fake_user_id_start += args.m

        # Step 3: Inject
        poisoned_train = pd.concat([poisoned_train, fake_df]).reset_index(drop=True)
        print(f"  Poisoned set: {len(poisoned_train)} interactions")

        # Step 4: Build expanded model, copy weights
        max_user_id = int(poisoned_train["user_id"].max()) + 1
        new_two_tower = dl_attack._build_plain_two_tower(
            max_user_id, n_items, args.embed_dim, layer_sizes, str(device)
        )
        dl_attack._copy_weights_to_plain(model, new_two_tower)
        model = dl_model.TwoTowerTrainTask(new_two_tower)
        optimizer = dl_model.make_optimizer(model, lr=args.lr)

        # Step 5: Retrain with EmbdGuard monitoring
        # Thresholds tuned for MovieLens-1M scale (3706 items, ~5M samples/epoch)
        print(f"  Retraining main model with EmbdGuard attached...")
        guard = EmbdGuard(model)
        guard.add_detector(GradientAnomalyDetector(threshold_z=5.0, min_steps=50))
        guard.add_detector(AccessFrequencyDetector(concentration_threshold=3.0, min_steps=30))
        guard.add_detector(EmbeddingDriftDetector(
            drift_threshold_z=8.0, min_steps=50, snapshot_interval=100))
        guard.add_detector(GradientDistributionDetector(
            kurtosis_z=5.0, concentration_threshold=50.0, min_steps=50))
        guard.add_detector(TemporalAccessDetector(
            burst_window=10, burst_threshold=1.0, top_k=3, min_steps=50))

        pos_users = torch.tensor(
            poisoned_train["user_id"].values, dtype=torch.long, device=device)
        pos_items = torch.tensor(
            poisoned_train["item_id"].values, dtype=torch.long, device=device)

        round_alerts = []
        step = 0
        for epoch in range(1, args.retrain_epochs + 1):
            model.train()
            users, items, labels = dl_train._negative_sample_tensors(
                pos_users, pos_items, n_items, 4, device
            )
            n_samples = len(users)
            batch_size = 2048
            total_loss = 0.0
            epoch_alerts = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                kjt = dl_model.make_kjt(users[start:end], items[start:end])
                batch_labels = labels[start:end]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    loss, _ = model(kjt, batch_labels)
                    optimizer.zero_grad()
                    loss.backward()
                optimizer.step()

                step += 1
                alerts = guard.step()
                round_alerts.extend(alerts)
                epoch_alerts += len(alerts)
                total_loss += loss.item() * (end - start)

            train_loss = total_loss / n_samples
            print(f"    Epoch {epoch:2d} | loss={train_loss:.4f} | alerts={epoch_alerts}")

        guard.detach()

        # Summarize round alerts
        round_by_detector = {}
        for a in round_alerts:
            round_by_detector.setdefault(a.detector, []).append(a)
            total_alerts_by_detector.setdefault(a.detector, []).append(a)

        print(f"\n  Round {r} alert summary:")
        if not round_by_detector:
            print(f"    No alerts fired")
        for det_name, det_alerts in sorted(round_by_detector.items()):
            print(f"    {det_name}: {len(det_alerts)} alerts")
            for a in det_alerts[:3]:
                print(f"      Step {a.step}: {a.message}")
            if len(det_alerts) > 3:
                print(f"      ... and {len(det_alerts) - 3} more")

        all_round_alerts[r] = round_alerts

        # Evaluate
        metrics = eval_fn(model.two_tower)
        target_hr = dl_evaluate.target_item_hit_ratio(
            model.two_tower, target, test_df, train_df, n_items,
            n_neg=99, k=10, device=str(device)
        )
        print(f"\n  After Round {r}: HR@10={metrics['HR@K']:.4f} | "
              f"NDCG@10={metrics['NDCG@K']:.4f} | Target HR@10={target_hr:.4f}")

    # ── Final summary ──
    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print("=" * 60)
    total = sum(len(alerts) for alerts in all_round_alerts.values())
    print(f"  Total alerts across {args.rounds} rounds: {total}")

    if total_alerts_by_detector:
        print(f"\n  Alerts by detector:")
        for det_name, det_alerts in sorted(total_alerts_by_detector.items()):
            first_step = min(a.step for a in det_alerts)
            print(f"    {det_name}: {len(det_alerts)} alerts (first at step {first_step})")

        for det_name in ["access_frequency", "embedding_drift"]:
            if det_name in total_alerts_by_detector:
                det_alerts = total_alerts_by_detector[det_name]
                flagged_rows = set()
                for a in det_alerts:
                    row = a.details.get("hottest_row") or a.details.get("row_id")
                    if row is not None:
                        flagged_rows.add(row)
                print(f"\n  {det_name} flagged rows: {sorted(flagged_rows)[:20]}")
                if target in flagged_rows:
                    print(f"    Target item {target} was flagged!")
                else:
                    print(f"    Target item {target} was NOT flagged")
    else:
        print("\n  No alerts fired across any rounds.")
        print("  This is expected — with MovieLens-1M's 3706 items and batch_size=2048,")
        print("  DLAttack's 5 fake users per round create very subtle poisoning signals")
        print("  that blend into the dense access patterns.")


if __name__ == "__main__":
    main()
