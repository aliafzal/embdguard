"""Demo: EmbdGuard detecting embedding poisoning in real time.

Creates a Two-Tower model with a realistic item space, trains on clean data
to build a baseline, then injects DLAttack-style poisoned batches where fake
users all interact with one target item. EmbdGuard catches the anomaly.
"""
import sys, os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import torch
import numpy as np

from dlattack_research.src.model import build_ebc, TwoTower, TwoTowerTrainTask, make_kjt, make_optimizer

from src.guard import EmbdGuard
from src.detectors.gradient_anomaly import GradientAnomalyDetector
from src.detectors.access_frequency import AccessFrequencyDetector

device = torch.device("cpu")
LOG_PATH = os.path.join(REPO_ROOT, "demo_log.jsonl")
torch.manual_seed(42)
np.random.seed(42)

N_USERS, N_ITEMS = 1000, 2000
TARGET_ITEM = 42
BATCH = 128

# ── Build model ──
ebc = build_ebc(N_USERS, N_ITEMS, 32, device=device)
two_tower = TwoTower(ebc, layer_sizes=[64, 32], device=device)
model = TwoTowerTrainTask(two_tower)
optimizer = make_optimizer(model)

# ── Attach EmbdGuard ──
guard = EmbdGuard(model, log_path=LOG_PATH)
guard.add_detector(GradientAnomalyDetector(threshold_z=3.0, min_steps=20))
guard.add_detector(AccessFrequencyDetector(concentration_threshold=5.0, min_steps=10))

# ── Phase 1: Clean training (25 steps) ──
print("── Phase 1: Clean training (25 steps) ──")
for step in range(25):
    users = torch.randint(0, N_USERS, (BATCH,))
    items = torch.randint(0, N_ITEMS, (BATCH,))
    labels = torch.cat([torch.ones(BATCH // 2), torch.zeros(BATCH // 2)])

    kjt = make_kjt(users, items)
    loss, _ = model(kjt, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    alerts = guard.step()
    if alerts:
        for a in alerts:
            print(f"  [Step {guard.step_count}] {a.severity.upper()}: {a.message}")
    elif step % 5 == 0:
        print(f"  Step {guard.step_count}: loss={loss.item():.4f} — no alerts")

# ── Phase 2: Poison — fake users all target one item ──
print(f"\n── Phase 2: Poisoned training targeting item {TARGET_ITEM} ──")
for step in range(25):
    users = torch.randint(0, N_USERS, (BATCH,))
    # 80% of lookups hit the target item, 20% are random filler
    n_target = int(BATCH * 0.8)
    items = torch.cat([
        torch.full((n_target,), TARGET_ITEM, dtype=torch.long),
        torch.randint(0, N_ITEMS, (BATCH - n_target,)),
    ])
    labels = torch.ones(BATCH)

    kjt = make_kjt(users, items)
    loss, _ = model(kjt, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    alerts = guard.step()
    if alerts:
        for a in alerts:
            print(f"  [Step {guard.step_count}] {a.severity.upper()}: {a.message}")
    elif step % 5 == 0:
        print(f"  Step {guard.step_count}: loss={loss.item():.4f} — no alerts")

guard.detach()

# ── Show log sample ──
print("\n── Log output (demo_log.jsonl) ──")
with open(LOG_PATH) as f:
    lines = f.readlines()
    shown_stats = 0
    for line in lines:
        if '"type": "stats"' in line and shown_stats < 2:
            print(line.rstrip())
            shown_stats += 1
        elif '"type": "alert"' in line:
            print(line.rstrip())

os.remove(LOG_PATH)
