# EmbdGuard

Embedding-level poisoning detection for TorchRec recommender systems.

EmbdGuard attaches lightweight hooks to TorchRec `EmbeddingBagCollection` modules to capture embedding-update statistics during training and runs pluggable anomaly detectors that catch data poisoning attacks (e.g., DLAttack) as they happen.

## Quick Start

```python
from embdguard import EmbdGuard, GradientAnomalyDetector, AccessFrequencyDetector

guard = EmbdGuard(model)
guard.add_detector(GradientAnomalyDetector())
guard.add_detector(AccessFrequencyDetector())

for batch in dataloader:
    loss = model(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    alerts = guard.step()
    for alert in alerts:
        print(alert)

guard.detach()
```

## How It Works

EmbdGuard hooks into the `nn.EmbeddingBag` modules inside TorchRec's EBC using standard PyTorch hook APIs:

- **Forward pre-hook**: captures which embedding rows are accessed (input indices)
- **Backward hook**: captures gradient norms on embedding outputs

These per-step statistics are stored in a ring buffer (`StatAccumulator`) and fed to pluggable detectors.

### Detectors

| Detector | What it catches | Method |
|---|---|---|
| `GradientAnomalyDetector` | Gradient spikes from poisoned data | Z-score of `grad_norm` vs rolling window |
| `AccessFrequencyDetector` | Concentrated access on target items | `max_count / mean_count` ratio |
| `TIADetector` | Fake users via embedding similarity | Target Item Analysis (Huang et al.) |

### Overhead

< 0.1ms per training step — gradient norm and index capture are cheap tensor ops.

## Installation

```bash
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
embdguard/              # package
  guard.py              # EmbdGuard orchestrator
  hooks.py              # EBC hook attachment + stat collection
  stats.py              # StatAccumulator ring buffer
  alerts.py             # Alert dataclass
  log.py                # JSONL structured logger
  detectors/
    gradient_anomaly.py
    access_frequency.py
    tia.py
tests/
dlattack_research/      # DLAttack replication (TorchRec Two-Tower on MovieLens-1M)
```
