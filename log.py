"""Structured JSONL logger for embedding stats and alerts."""
import json


class JSONLLogger:
    """Append-only JSONL logger for stats and alerts.

    Each line is a JSON object with:
        {"type": "stats"|"alert", "step": int, "data": {...}}
    """

    def __init__(self, path: str):
        self._path = path
        self._file = open(path, "a")

    def log_stats(self, step: int, table_stats: dict) -> None:
        for table_name, stats in table_stats.items():
            # Filter out non-serializable fields
            clean = {}
            for k, v in stats.items():
                if k == "accessed_ids":
                    continue  # skip large lists in stats log
                clean[k] = v
            line = json.dumps({
                "type": "stats",
                "step": step,
                "table": table_name,
                "data": clean,
            })
            self._file.write(line + "\n")

    def log_alert(self, alert) -> None:
        line = json.dumps({"type": "alert", **alert.to_dict()})
        self._file.write(line + "\n")

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()
