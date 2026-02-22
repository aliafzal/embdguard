"""Row freeze defense for flagged embedding rows.

Zeros gradients on flagged rows, completely preventing weight updates.
More aggressive than gradient clipping — use when high confidence that
the row is under attack.
"""
import torch
import torch.nn as nn

from src.defenses import BaseDefense


class RowFreezeDefense(BaseDefense):
    """Freezes (zero gradient) flagged embedding rows."""

    def __init__(self):
        # {table_name: {row_id: remaining_steps}}
        self._flagged: dict[str, dict[int, int]] = {}
        self._hooks: list = []
        self._model = None

    @property
    def name(self) -> str:
        return "row_freeze"

    @property
    def active_rows(self) -> dict[str, list[int]]:
        return {t: list(rows.keys()) for t, rows in self._flagged.items() if rows}

    def activate(self, table_name: str, row_ids: list[int], duration: int = 10) -> None:
        if table_name not in self._flagged:
            self._flagged[table_name] = {}
        for rid in row_ids:
            self._flagged[table_name][rid] = duration

    def apply(self, model: nn.Module) -> None:
        self._model = model
        two_tower = model.two_tower if hasattr(model, "two_tower") else model
        if not hasattr(two_tower, "ebc"):
            return

        for tname in two_tower.ebc.embedding_bags:
            weight = two_tower.ebc.embedding_bags[tname].weight
            handle = weight.register_hook(self._make_hook(tname))
            self._hooks.append(handle)

    def _make_hook(self, table_name: str):
        def hook(grad):
            if table_name not in self._flagged or not self._flagged[table_name]:
                return grad
            grad = grad.clone()
            for row_id in self._flagged[table_name]:
                if row_id < grad.shape[0]:
                    grad[row_id] = 0.0
            return grad
        return hook

    def step(self) -> None:
        for table_name in list(self._flagged.keys()):
            expired = []
            for rid, remaining in self._flagged[table_name].items():
                self._flagged[table_name][rid] = remaining - 1
                if remaining - 1 <= 0:
                    expired.append(rid)
            for rid in expired:
                del self._flagged[table_name][rid]

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._flagged.clear()
        self._model = None
