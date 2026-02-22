"""Detector comparison framework.

Runs multiple detector configurations on the same attack scenario and
produces a side-by-side comparison of detection quality.
"""
from src.evaluation.harness import EvalRun, EvalResult, DataConfig, AttackConfig


def compare(
    detector_configs: list[dict],
    data_config: DataConfig | None = None,
    attack_config: AttackConfig | None = None,
    seed: int = 42,
) -> list[dict]:
    """Run each detector config on the same scenario and compare.

    Args:
        detector_configs: List of dicts, each with:
            - "name": str label for this config
            - "detectors": list of detector instances
        data_config: Data configuration (uses defaults if None).
        attack_config: Attack configuration (uses defaults if None).
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with config name + metrics.
    """
    results = []

    for config in detector_configs:
        run = EvalRun(
            detectors=config["detectors"],
            data_config=data_config,
            attack_config=attack_config,
            seed=seed,
        )
        eval_result = run.execute()

        row = {"config": config["name"], **eval_result.summary_dict()}
        results.append(row)

    return results


def format_comparison(results: list[dict]) -> str:
    """Format comparison results as a readable table."""
    if not results:
        return "No results."

    keys = list(results[0].keys())
    col_widths = {k: max(len(k), max(len(str(r[k])) for r in results)) for k in keys}

    header = " | ".join(k.ljust(col_widths[k]) for k in keys)
    sep = "-+-".join("-" * col_widths[k] for k in keys)
    rows = [
        " | ".join(str(r[k]).ljust(col_widths[k]) for k in keys)
        for r in results
    ]

    return "\n".join([header, sep] + rows)
