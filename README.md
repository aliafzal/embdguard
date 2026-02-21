# EmbdGuard

Real-time embedding-level poisoning detection for TorchRec recommender systems. Attaches PyTorch hooks to `EmbeddingBagCollection` modules, captures per-step gradient and access statistics, and runs pluggable anomaly detectors during training.

## Setup

```bash
pip install torch torchrec numpy pandas scikit-learn pytest
```

## Usage

```python
from src.guard import EmbdGuard
from src.detectors.gradient_anomaly import GradientAnomalyDetector
from src.detectors.access_frequency import AccessFrequencyDetector

guard = EmbdGuard(model, log_path="embdguard_log.jsonl")
guard.add_detector(GradientAnomalyDetector(threshold_z=3.0, min_steps=20))
guard.add_detector(AccessFrequencyDetector(concentration_threshold=5.0, min_steps=10))

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

## Detection output

When a detector fires, it returns `Alert` objects:

```
Alert(step=247, detector='gradient_anomaly', severity='warning',
      table='t_item_id', message='Gradient norm z-score=4.12 exceeds threshold 3.0')

Alert(step=310, detector='access_frequency', severity='warning',
      table='t_item_id', message='Row 1121 accessed 8.3x above mean (count=94, mean=11.3)')
```

## Logging

With `log_path` set, EmbdGuard writes JSONL — one line per event:

```jsonl
{"type": "stats", "step": 1, "table": "t_user_id", "data": {"grad_norm": 0.042, "grad_max": 0.018, "n_accessed": 847.0}}
{"type": "stats", "step": 1, "table": "t_item_id", "data": {"grad_norm": 0.037, "grad_max": 0.015, "n_accessed": 1203.0}}
{"type": "alert", "step": 247, "detector": "gradient_anomaly", "severity": "warning", "table": "t_item_id", "message": "Gradient norm z-score=4.12 exceeds threshold 3.0", "details": {"z_score": 4.12, "value": 0.189, "rolling_mean": 0.041, "rolling_std": 0.036}}
```

## Tests

```bash
pytest tests/ -v
```

## DLAttack pipeline

The `dlattack_research/` directory contains a full DLAttack replication (Huang et al., NDSS 2021) on MovieLens-1M with a TorchRec Two-Tower model:

```bash
cd dlattack_research
python main.py --phase all --epochs 30 --rounds 5
```

## Project structure

```
src/
  guard.py              EmbdGuard orchestrator
  hooks.py              EBC hook attachment + stat collection
  stats.py              StatAccumulator ring buffer
  alerts.py             Alert dataclass
  log.py                JSONL structured logger
  detectors/
    __init__.py          BaseDetector ABC
    gradient_anomaly.py  Z-score gradient spike detection
    access_frequency.py  Access concentration detection
    tia.py               Target Item Analysis detection
tests/
dlattack_research/
```
