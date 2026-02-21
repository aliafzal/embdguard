"""EmbdGuard orchestrator — the main entry point for embedding instrumentation."""
import torch.nn as nn

from hooks import EBCHooks
from stats import StatAccumulator
from log import JSONLLogger


def _find_ebcs(model: nn.Module) -> dict:
    """Find all EmbeddingBagCollection modules in the model tree."""
    try:
        from torchrec.modules.embedding_modules import EmbeddingBagCollection
    except ImportError:
        EmbeddingBagCollection = None

    ebcs = {}
    for name, module in model.named_modules():
        if EmbeddingBagCollection is not None and isinstance(module, EmbeddingBagCollection):
            ebcs[name] = module
        elif hasattr(module, "embedding_bags") and isinstance(module.embedding_bags, nn.ModuleDict):
            ebcs[name] = module
    return ebcs


class EmbdGuard:
    """Embedding-level robustness instrumentation for TorchRec models.

    Usage::

        guard = EmbdGuard(model)
        guard.add_detector(GradientAnomalyDetector())

        for batch in dataloader:
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            alerts = guard.step()

        guard.detach()
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        table_names: list[str] | None = None,
        log_path: str | None = None,
        window_size: int = 100,
        check_interval: int = 1,
    ):
        self._model = model
        self._window_size = window_size
        self._check_interval = check_interval
        self._step_count = 0
        self._detectors = []
        self._table_stats: dict[str, StatAccumulator] = {}
        self._logger = JSONLLogger(log_path) if log_path else None

        # Find EBCs and attach hooks
        self._hooks_list: list[EBCHooks] = []
        ebcs = _find_ebcs(model)
        if not ebcs:
            raise ValueError(
                "No EmbeddingBagCollection found in model. "
                "EmbdGuard requires a model with TorchRec EBC modules."
            )

        for ebc_name, ebc in ebcs.items():
            hooks = EBCHooks(ebc, table_names=table_names)
            hooks.attach()
            self._hooks_list.append(hooks)
            for tname in hooks._table_names:
                self._table_stats[tname] = StatAccumulator(window_size)

    def add_detector(self, detector) -> "EmbdGuard":
        """Register a detector. Returns self for chaining."""
        self._detectors.append(detector)
        return self

    def step(self) -> list:
        """Call after optimizer.step(). Collects stats, runs detectors.

        Returns list of Alert objects (empty if nothing anomalous).
        """
        self._step_count += 1

        # Collect stats from all hooks
        for hooks in self._hooks_list:
            raw = hooks.collect()
            for table_name, stats in raw.items():
                if table_name in self._table_stats:
                    # Store accessed_ids separately (list, not float)
                    ids = stats.pop("accessed_ids", None)
                    self._table_stats[table_name].push(stats)
                    if ids is not None:
                        # Push as a single object in a special buffer
                        self._table_stats[table_name].push({"accessed_ids": ids})

        # Log stats
        if self._logger:
            snapshot = {}
            for tname, acc in self._table_stats.items():
                snapshot[tname] = {
                    s: float(acc.get(s)[-1]) for s in acc.stat_names
                    if len(acc.get(s)) > 0 and s != "accessed_ids"
                }
            self._logger.log_stats(self._step_count, snapshot)

        # Run detectors on schedule
        alerts = []
        if self._step_count % self._check_interval == 0:
            for detector in self._detectors:
                new_alerts = detector.check(
                    self._step_count, self._table_stats, self._model
                )
                alerts.extend(new_alerts)

        # Log alerts
        if self._logger and alerts:
            for alert in alerts:
                self._logger.log_alert(alert)
            self._logger.flush()

        return alerts

    def run_detector(self, detector) -> list:
        """Run a specific detector immediately (outside normal schedule)."""
        return detector.check(self._step_count, self._table_stats, self._model)

    def get_stats(self, table_name: str) -> StatAccumulator:
        """Access raw stat history for a specific embedding table."""
        return self._table_stats[table_name]

    def summary(self) -> dict:
        """Return a summary dict of all collected stats across tables."""
        result = {}
        for tname, acc in self._table_stats.items():
            result[tname] = {
                "steps": len(acc),
                "stats": {
                    s: {
                        "mean": acc.rolling_mean(s),
                        "std": acc.rolling_std(s),
                        "latest": float(acc.get(s)[-1]) if len(acc.get(s)) > 0 else None,
                    }
                    for s in acc.stat_names if s != "accessed_ids"
                },
            }
        return result

    def detach(self) -> None:
        """Remove all hooks from the model."""
        for hooks in self._hooks_list:
            hooks.detach()
        self._hooks_list.clear()
        if self._logger:
            self._logger.close()
            self._logger = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()

    @property
    def step_count(self) -> int:
        return self._step_count
