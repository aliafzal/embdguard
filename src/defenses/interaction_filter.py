"""Interaction filter defense.

Operates at the batch level — filters out interactions involving flagged
items before they reach the model. Unlike gradient-based defenses, this
prevents the poisoned data from influencing the model at all.
"""
import torch

from src.defenses import BaseDefense


class InteractionFilterDefense(BaseDefense):
    """Filters out interactions with flagged items from training batches."""

    def __init__(self):
        # {table_name: {row_id: remaining_steps}}
        self._flagged: dict[str, dict[int, int]] = {}

    @property
    def name(self) -> str:
        return "interaction_filter"

    @property
    def active_rows(self) -> dict[str, list[int]]:
        return {t: list(rows.keys()) for t, rows in self._flagged.items() if rows}

    def activate(self, table_name: str, row_ids: list[int], duration: int = 10) -> None:
        if table_name not in self._flagged:
            self._flagged[table_name] = {}
        for rid in row_ids:
            self._flagged[table_name][rid] = duration

    def apply(self, model) -> None:
        # No hooks needed — filtering happens at batch level via filter_batch()
        pass

    def filter_batch(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        labels: torch.Tensor,
        item_table: str = "t_item_id",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove interactions with flagged items from the batch.

        Args:
            users: User ID tensor.
            items: Item ID tensor.
            labels: Label tensor.
            item_table: Table name to check for flagged items.

        Returns:
            Filtered (users, items, labels) tensors.
        """
        if item_table not in self._flagged or not self._flagged[item_table]:
            return users, items, labels

        flagged_set = set(self._flagged[item_table].keys())
        mask = torch.tensor(
            [int(item.item()) not in flagged_set for item in items],
            dtype=torch.bool,
        )

        if mask.all():
            return users, items, labels

        return users[mask], items[mask], labels[mask]

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
        self._flagged.clear()
