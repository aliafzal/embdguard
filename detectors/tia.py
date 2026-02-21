"""TIA (Target Item Analysis) detector.

Ported from dlattack_research/src/detect.py. Detects fake users by
measuring what fraction of each user's interactions are with items
similar to watched target items in embedding space.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from alerts import Alert
from detectors import BaseDetector
from stats import StatAccumulator


class TIADetector(BaseDetector):
    """Target Item Analysis: detect users with anomalous interaction profiles.

    Unlike per-step detectors, TIA is a batch detector that examines the
    full training DataFrame against current embeddings. Call via
    ``guard.run_detector(tia)`` or set a large ``check_interval``.
    """

    def __init__(
        self,
        watch_items: list[int],
        train_df: pd.DataFrame,
        top_similar: int = 50,
        threshold_percentile: float = 95.0,
    ):
        self._watch_items = watch_items
        self._train_df = train_df
        self._top_similar = top_similar
        self._threshold_percentile = threshold_percentile

    @property
    def name(self) -> str:
        return "tia"

    def check(self, step, table_stats, model) -> list[Alert]:
        two_tower = model.two_tower if hasattr(model, "two_tower") else model
        if not hasattr(two_tower, "ebc"):
            return []

        alerts = []
        with torch.no_grad():
            item_embs = two_tower.ebc.embedding_bags["t_item_id"].weight.data

            for target_id in self._watch_items:
                if target_id >= len(item_embs):
                    continue
                target_emb = item_embs[target_id]
                sims = nn.functional.cosine_similarity(
                    item_embs, target_emb.unsqueeze(0)
                ).cpu().numpy()
                sim_items = set(np.argsort(sims)[-self._top_similar:].tolist())

                scores = {}
                for uid, group in self._train_df.groupby("user_id"):
                    items = set(group["item_id"].tolist())
                    scores[uid] = len(items & sim_items) / max(len(items), 1)

                scores_arr = np.array(list(scores.values()))
                threshold = np.percentile(scores_arr, self._threshold_percentile)
                flagged = [uid for uid, s in scores.items() if s >= threshold]

                if flagged:
                    alerts.append(Alert(
                        step=step,
                        detector=self.name,
                        severity="warning",
                        table="t_item_id",
                        message=(f"TIA flagged {len(flagged)} users for "
                                 f"target item {target_id} (threshold={threshold:.4f})"),
                        details={
                            "target_item": target_id,
                            "n_flagged": len(flagged),
                            "flagged_user_ids": flagged[:50],
                            "threshold": float(threshold),
                        },
                    ))
        return alerts
