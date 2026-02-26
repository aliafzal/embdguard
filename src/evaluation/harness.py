"""Evaluation harness for measuring detector performance.

Runs the full attack-and-detect pipeline on synthetic data and reports
precision, recall, F1, and detection latency.
"""
from dataclasses import dataclass, field

import torch
import numpy as np

from src.models import build_ebc, TwoTower, TwoTowerTrainTask, make_kjt, make_optimizer
from src.guard import EmbdGuard


@dataclass
class EvalResult:
    """Metrics from a single evaluation run."""
    true_positives: int = 0
    false_positives: int = 0
    total_attack_steps: int = 0
    detection_latency: int | None = None  # steps from attack start to first alert
    alert_timeline: list = field(default_factory=list)

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        return 1.0 if self.true_positives > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def total_alerts(self) -> int:
        return self.true_positives + self.false_positives

    def summary_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "detection_latency": self.detection_latency,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "total_alerts": self.total_alerts,
        }


@dataclass
class DataConfig:
    """Configuration for synthetic data generation."""
    n_users: int = 1000
    n_items: int = 2000
    batch_size: int = 128
    embedding_dim: int = 32
    layer_sizes: list = field(default_factory=lambda: [64, 32])


@dataclass
class AttackConfig:
    """Configuration for the simulated attack."""
    target_item: int = 42
    poison_ratio: float = 0.8
    clean_steps: int = 25
    attack_steps: int = 25


class EvalRun:
    """Runs a clean→attack training loop with EmbdGuard attached.

    Usage::

        run = EvalRun(
            detectors=[GradientAnomalyDetector(), AccessFrequencyDetector()],
            data_config=DataConfig(),
            attack_config=AttackConfig(),
        )
        result = run.execute()
        print(result.summary_dict())
    """

    def __init__(
        self,
        detectors: list,
        data_config: DataConfig | None = None,
        attack_config: AttackConfig | None = None,
        seed: int = 42,
    ):
        self._detectors = detectors
        self._data = data_config or DataConfig()
        self._attack = attack_config or AttackConfig()
        self._seed = seed

    def execute(self) -> EvalResult:
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        device = torch.device("cpu")

        # Build model
        ebc = build_ebc(
            self._data.n_users, self._data.n_items,
            self._data.embedding_dim, device=device,
        )
        two_tower = TwoTower(ebc, layer_sizes=self._data.layer_sizes, device=device)
        model = TwoTowerTrainTask(two_tower)
        optimizer = make_optimizer(model)

        # Attach guard with detectors
        guard = EmbdGuard(model)
        for det in self._detectors:
            guard.add_detector(det)

        result = EvalResult(total_attack_steps=self._attack.attack_steps)
        attack_start = self._attack.clean_steps + 1

        # Phase 1: Clean training
        for step in range(1, self._attack.clean_steps + 1):
            self._train_step_clean(model, optimizer, guard, step, result)

        # Phase 2: Attack training
        for step in range(attack_start, attack_start + self._attack.attack_steps):
            self._train_step_attack(model, optimizer, guard, step, result, attack_start)

        guard.detach()
        return result

    def _train_step_clean(self, model, optimizer, guard, step, result):
        bs = self._data.batch_size
        users = torch.randint(0, self._data.n_users, (bs,))
        items = torch.randint(0, self._data.n_items, (bs,))
        labels = torch.cat([torch.ones(bs // 2), torch.zeros(bs // 2)])

        kjt = make_kjt(users, items)
        loss, _ = model(kjt, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        alerts = guard.step()
        for a in alerts:
            result.false_positives += 1
            result.alert_timeline.append((step, a.detector, a.message))

    def _train_step_attack(self, model, optimizer, guard, step, result, attack_start):
        bs = self._data.batch_size
        users = torch.randint(0, self._data.n_users, (bs,))
        n_target = int(bs * self._attack.poison_ratio)
        items = torch.cat([
            torch.full((n_target,), self._attack.target_item, dtype=torch.long),
            torch.randint(0, self._data.n_items, (bs - n_target,)),
        ])
        labels = torch.ones(bs)

        kjt = make_kjt(users, items)
        loss, _ = model(kjt, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        alerts = guard.step()
        for a in alerts:
            result.true_positives += 1
            result.alert_timeline.append((step, a.detector, a.message))
            if result.detection_latency is None:
                result.detection_latency = step - attack_start
