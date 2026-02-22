# EmbdGuard

Real-time embedding-level poisoning detection and defense for TorchRec recommender systems. Attaches PyTorch hooks to `EmbeddingBagCollection` modules, captures per-step gradient and access statistics, runs pluggable anomaly detectors, and activates defenses during training.

## Setup

```bash
pip install torch torchrec numpy pandas scikit-learn pytest
```

## Usage

```python
from src.guard import EmbdGuard
from src.detectors.gradient_anomaly import GradientAnomalyDetector
from src.detectors.access_frequency import AccessFrequencyDetector
from src.detectors.embedding_drift import EmbeddingDriftDetector
from src.detectors.temporal_access import TemporalAccessDetector
from src.defenses.row_freeze import RowFreezeDefense

guard = EmbdGuard(model, log_path="embdguard_log.jsonl")
guard.add_detector(GradientAnomalyDetector(threshold_z=3.0, min_steps=20))
guard.add_detector(AccessFrequencyDetector(concentration_threshold=5.0, min_steps=10))
guard.add_detector(EmbeddingDriftDetector(drift_threshold_z=3.0, min_steps=10))
guard.add_detector(TemporalAccessDetector(burst_window=5, burst_threshold=0.8, min_steps=10))
guard.add_defense(RowFreezeDefense())

for batch in dataloader:
    loss = model(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    alerts = guard.step()  # collects stats, runs detectors, activates defenses
    for alert in alerts:
        print(alert)

guard.detach()
```

## Detectors

| Detector | Signal | What it catches |
|----------|--------|-----------------|
| `GradientAnomalyDetector` | Z-score of gradient norm | Sudden gradient spikes from poisoned batches |
| `AccessFrequencyDetector` | Max/mean access count ratio | One item accessed disproportionately often |
| `EmbeddingDriftDetector` | Per-row L2 drift from reference | Slow-and-steady weight shifts on target items |
| `GradientDistributionDetector` | Gradient kurtosis + concentration | Gradient energy concentrated in few rows |
| `TemporalAccessDetector` | Jaccard overlap + burst score | Same item appearing in top-K every step |
| `TIADetector` | User profile similarity to target | Fake users with anomalous interaction profiles |

## Defenses

| Defense | Mechanism | Use case |
|---------|-----------|----------|
| `GradientClipDefense` | Clips per-row gradients to max_norm | Reduce attacker influence while allowing some updates |
| `RowFreezeDefense` | Zeros gradients on flagged rows | Completely block suspicious weight updates |
| `InteractionFilterDefense` | Removes flagged items from batch | Prevent poisoned data from reaching the model |

All defenses auto-expire after a configurable duration and integrate with EmbdGuard via `guard.add_defense()`.

## Detection results

`demo.py` trains a Two-Tower model on clean data for 25 steps, then injects poisoned batches where fake users all interact with a target item. EmbdGuard catches the attack with multiple detectors:

```
── Phase 1: Clean training (25 steps) ──
  Step 1: loss=1.8463 — no alerts
  Step 11: loss=0.8115 — no alerts
  Step 21: loss=0.7112 — no alerts

── Phase 2: Poisoned training targeting item 42 ──
  [Step 29] WARNING: Row 42 in top-5 accessed for 4/5 consecutive steps (burst=0.80)
  [Step 33] WARNING: Row 42 accessed 5.5x above mean (count=11, mean=2.0)
  [Step 40] WARNING: Row 42 accessed 8.6x above mean (count=18, mean=2.1)
  [Step 48] WARNING: Row 42 drifted 3.0σ from reference (L2=6.27, mean=2.48)
  [Step 50] WARNING: Row 42 accessed 12.8x above mean (count=28, mean=2.2)
```

The `TemporalAccessDetector` fires first at step 29 (4 steps into poisoning), followed by `AccessFrequencyDetector` at step 33 and `EmbeddingDriftDetector` at step 48.

![Training loss and detection timeline](demo_plot.png)

**Top**: training loss drops sharply once poisoned data is injected (step 26) — the model overfits to the fake interactions. **Bottom**: each red bar is an alert from the `AccessFrequencyDetector`. The concentration ratio starts at 5.5x and climbs to 12.8x, well above the 5.0x threshold (orange dashed line).

## Evaluation framework

Compare detector performance on synthetic attack scenarios:

```bash
python run_evaluation.py --detectors all
```

```
config       | precision | recall | f1     | detection_latency | true_positives | false_positives
access_freq  | 1.0       | 1.0    | 1.0    | 7                 | 18             | 0
emb_drift    | 1.0       | 1.0    | 1.0    | 14                | 74             | 0
temporal     | 0.9583    | 1.0    | 0.9787 | 2                 | 23             | 1
all_combined | 0.9914    | 1.0    | 0.9957 | 2                 | 115            | 1
```

Run parameter sweeps:

```bash
python run_evaluation.py --sweep access_frequency
```

## Logging

With `log_path` set, EmbdGuard writes JSONL — one line per event:

```jsonl
{"type": "stats", "step": 1, "table": "t_user_id", "data": {"n_accessed": 116.0, "grad_norm": 0.0887, "grad_max": 0.0091}}
{"type": "alert", "step": 33, "detector": "access_frequency", "severity": "warning", "table": "t_item_id", "message": "Row 42 accessed 5.5x above mean (count=11, mean=2.0)"}
```

## Tests

```bash
pytest tests/ -v  # 67 tests
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
  guard.py                EmbdGuard orchestrator + defense integration
  hooks.py                EBC hook attachment (gradients + access stats)
  stats.py                StatAccumulator ring buffer
  alerts.py               Alert dataclass
  log.py                  JSONL structured logger
  detectors/
    __init__.py            BaseDetector ABC
    gradient_anomaly.py    Z-score gradient spike detection
    access_frequency.py    Access concentration detection
    embedding_drift.py     Weight drift from reference snapshot
    gradient_distribution.py  Gradient shape analysis (kurtosis + concentration)
    temporal_access.py     Temporal access patterns (Jaccard + burst)
    tia.py                 Target Item Analysis detection
  defenses/
    __init__.py            BaseDefense ABC
    gradient_clip.py       Per-row gradient clipping
    row_freeze.py          Zero gradients on flagged rows
    interaction_filter.py  Batch-level item filtering
  evaluation/
    __init__.py
    harness.py             EvalRun + EvalResult metrics
    sensitivity.py         Parameter sweep analysis
    compare.py             Detector comparison framework
tests/
run_evaluation.py          Top-level evaluation script
eval_movielens.py          MovieLens-1M evaluation script
demo.py                    Detection + defense demo
notebooks/
  demo.ipynb               Jupyter notebook demo with plots
  validate_movielens.ipynb MovieLens-1M validation notebook
dlattack_research/
```
