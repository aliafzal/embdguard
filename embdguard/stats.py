"""Ring-buffer statistics accumulator for per-step embedding metrics."""
import collections
import numpy as np


class StatAccumulator:
    """Fixed-size ring buffer for per-step statistics.

    Stores the last ``window_size`` values for each stat name.
    Supports efficient rolling mean, std, and z-score computation.
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._buffers: dict[str, collections.deque] = {}
        self._count = 0

    def push(self, stats: dict) -> None:
        """Record one step's statistics."""
        for name, value in stats.items():
            if name not in self._buffers:
                self._buffers[name] = collections.deque(maxlen=self._window_size)
            try:
                self._buffers[name].append(float(value))
            except (TypeError, ValueError):
                # Non-numeric values (e.g. list of accessed_ids) stored as-is
                self._buffers[name].append(value)
        self._count += 1

    def get(self, stat_name: str) -> np.ndarray:
        """Return array of recent values for a stat (oldest first)."""
        if stat_name not in self._buffers:
            return np.array([])
        return np.array(self._buffers[stat_name])

    def rolling_mean(self, stat_name: str, n: int = None) -> float:
        """Mean of last n values (default: all in window)."""
        arr = self.get(stat_name)
        if len(arr) == 0:
            return 0.0
        if n is not None:
            arr = arr[-n:]
        return float(arr.mean())

    def rolling_std(self, stat_name: str, n: int = None) -> float:
        """Standard deviation of last n values."""
        arr = self.get(stat_name)
        if len(arr) < 2:
            return 0.0
        if n is not None:
            arr = arr[-n:]
        return float(arr.std())

    def z_score(self, stat_name: str) -> float:
        """Z-score of the most recent value vs the rolling distribution."""
        arr = self.get(stat_name)
        if len(arr) < 2:
            return 0.0
        mean = arr[:-1].mean()
        std = arr[:-1].std()
        if std < 1e-10:
            return 0.0
        return float((arr[-1] - mean) / std)

    @property
    def stat_names(self) -> list[str]:
        return list(self._buffers.keys())

    def __len__(self) -> int:
        return self._count
