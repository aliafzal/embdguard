"""Parameter sweep for detector sensitivity analysis.

Runs EvalRun across a grid of detector parameters and collects metrics.
"""
import itertools
from src.evaluation.harness import EvalRun, EvalResult, DataConfig, AttackConfig


def sweep(
    detector_class,
    param_grid: dict[str, list],
    data_config: DataConfig | None = None,
    attack_config: AttackConfig | None = None,
    seed: int = 42,
) -> list[dict]:
    """Run EvalRun for each combination of parameters in param_grid.

    Args:
        detector_class: The detector class to instantiate.
        param_grid: Dict of param_name -> list of values to try.
            Example: {"threshold_z": [2.0, 3.0, 4.0], "min_steps": [10, 20]}
        data_config: Data configuration (uses defaults if None).
        attack_config: Attack configuration (uses defaults if None).
        seed: Random seed for reproducibility.

    Returns:
        List of dicts, each with param values + EvalResult metrics.
    """
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    results = []

    for combo in itertools.product(*value_lists):
        params = dict(zip(keys, combo))
        detector = detector_class(**params)

        run = EvalRun(
            detectors=[detector],
            data_config=data_config,
            attack_config=attack_config,
            seed=seed,
        )
        eval_result = run.execute()

        row = {**params, **eval_result.summary_dict()}
        results.append(row)

    return results


def format_sweep_results(results: list[dict]) -> str:
    """Format sweep results as a readable table."""
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
