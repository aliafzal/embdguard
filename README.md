# EmbdGuard

Real-time embedding-level poisoning detection for TorchRec recommender systems.

EmbdGuard attaches lightweight PyTorch hooks to TorchRec `EmbeddingBagCollection` modules, captures per-step embedding statistics during training, and runs pluggable anomaly detectors to catch data poisoning attacks (e.g., DLAttack) as they happen — not after training is done.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- TorchRec
- NumPy
- pandas (for TIA detector and DLAttack pipeline)
- scikit-learn (for DLAttack data splits)

## Setup

```bash
git clone https://github.com/aliafzal/embdguard.git
cd embdguard
pip install torch torchrec numpy pandas scikit-learn pytest
```

## Quick Start

```python
from guard import EmbdGuard
from detectors.gradient_anomaly import GradientAnomalyDetector
from detectors.access_frequency import AccessFrequencyDetector

# Wrap any model that has a TorchRec EBC
guard = EmbdGuard(model)
guard.add_detector(GradientAnomalyDetector())
guard.add_detector(AccessFrequencyDetector())

# Add one line to your training loop
for batch in dataloader:
    loss = model(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    alerts = guard.step()  # <-- this is it
    for alert in alerts:
        print(alert)

guard.detach()
```

## Running the DLAttack Pipeline

The `dlattack_research/` directory contains a full replication of the DLAttack data poisoning attack (Huang et al., NDSS 2021) on a TorchRec Two-Tower model trained on MovieLens-1M.

### Run the full pipeline

```bash
cd dlattack_research
python main.py --phase all --epochs 30 --rounds 5
```

This will:
1. Download MovieLens-1M dataset
2. Train a clean baseline Two-Tower model
3. Run the DLAttack (generate fake users, poison training data)
4. Evaluate attack impact (overall HR@10 and target item HR@10)
5. Run TIA detection to identify fake users

### Run individual phases

```bash
# Train baseline only
python main.py --phase baseline --epochs 50

# Run attack only (requires trained baseline in checkpoints/)
python main.py --phase attack --rounds 5 --m 5

# Evaluate only (requires both checkpoints)
python main.py --phase evaluate
```

### Pipeline options

| Flag | Default | Description |
|------|---------|-------------|
| `--phase` | `all` | `all`, `baseline`, `attack`, or `evaluate` |
| `--epochs` | `30` | Training epochs for baseline |
| `--rounds` | `5` | Number of DLAttack rounds |
| `--m` | `5` | Fake users injected per round |
| `--embed_dim` | `64` | Embedding dimension |
| `--lr` | `0.001` | MLP learning rate (embeddings use 0.1) |

### Run with EmbdGuard monitoring

To monitor training for poisoning in real-time, add EmbdGuard to the training loop:

```python
import sys
sys.path.insert(0, "..")
from guard import EmbdGuard
from detectors.gradient_anomaly import GradientAnomalyDetector
from detectors.access_frequency import AccessFrequencyDetector

from src.model import build_ebc, TwoTower, TwoTowerTrainTask, make_kjt, make_optimizer
from src.dataset import download_ml1m, load_ratings, split_data
import torch

# Setup
download_ml1m()
df, n_users, n_items, _, _ = load_ratings()
train_df, test_df = split_data(df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ebc = build_ebc(n_users, n_items, 64, device=device)
model = TwoTowerTrainTask(TwoTower(ebc, layer_sizes=[128, 64], device=device))
optimizer = make_optimizer(model)

# Attach EmbdGuard
guard = EmbdGuard(model, log_path="embdguard_log.jsonl")
guard.add_detector(GradientAnomalyDetector(threshold_z=3.0, min_steps=20))
guard.add_detector(AccessFrequencyDetector(concentration_threshold=5.0, min_steps=10))

# Training loop
for epoch in range(30):
    for batch_start in range(0, len(train_df), 4096):
        batch = train_df.iloc[batch_start:batch_start + 4096]
        users = torch.tensor(batch["user_id"].values, device=device)
        items = torch.tensor(batch["item_id"].values, device=device)
        labels = torch.ones(len(batch), device=device)
        kjt = make_kjt(users, items)

        loss, _ = model(kjt, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        alerts = guard.step()
        for alert in alerts:
            print(f"[Step {guard.step_count}] {alert.severity}: {alert.message}")

guard.detach()
```

## Running Tests

```bash
pytest tests/ -v
```

All 33 tests cover stats, hooks, detectors, and end-to-end guard integration.

## How It Works

EmbdGuard hooks into each `nn.EmbeddingBag` module inside a TorchRec EBC using standard PyTorch hook APIs:

- **Forward pre-hook**: captures which embedding rows are accessed (input indices)
- **Backward hook**: captures gradient norms on embedding outputs

Per-step statistics are stored in a fixed-size ring buffer (`StatAccumulator`, bounded memory) and fed to pluggable detectors at configurable intervals.

### Detectors

| Detector | What it catches | How |
|---|---|---|
| `GradientAnomalyDetector` | Gradient spikes from poisoned data | Z-score of `grad_norm` vs rolling window |
| `AccessFrequencyDetector` | Concentrated access on target items | `max_count / mean_count` ratio over access history |
| `TIADetector` | Fake users via embedding similarity | Cosine similarity overlap with target item neighbors |

### Overhead

< 0.1ms per training step. Gradient norm computation and index capture are cheap tensor operations that don't block the training loop.

## Project Structure

```
guard.py                # EmbdGuard orchestrator
hooks.py                # EBC hook attachment + stat collection
stats.py                # StatAccumulator ring buffer
alerts.py               # Alert dataclass
log.py                  # JSONL structured logger
detectors/
  __init__.py           # BaseDetector ABC
  gradient_anomaly.py   # Z-score gradient spike detection
  access_frequency.py   # Access concentration detection
  tia.py                # Target Item Analysis detection
tests/
  conftest.py           # Shared fixtures (small EBC, TwoTower)
  test_stats.py         # Ring buffer tests
  test_hooks.py         # Hook attachment tests
  test_detectors.py     # Detector logic tests
  test_guard.py         # End-to-end integration tests
dlattack_research/      # DLAttack replication
  src/
    model.py            # TorchRec Two-Tower model
    train.py            # Training loop
    attack.py           # DLAttack implementation
    evaluate.py         # HR@K evaluation
    detect.py           # TIA post-hoc detection
    dataset.py          # MovieLens-1M download + preprocessing
  main.py               # Full pipeline CLI
```

## References

- Huang et al., "Data Poisoning Attacks to Deep Learning Based Recommender Systems," NDSS 2021
