"""Run the EmbdGuard evaluation suite.

Evaluates detectors on a synthetic DLAttack scenario and reports metrics.

Usage:
    python run_evaluation.py
    python run_evaluation.py --detectors access_frequency gradient_anomaly
    python run_evaluation.py --attack-ratio 0.5 --attack-steps 50
    python run_evaluation.py --sweep gradient_anomaly
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detectors.gradient_anomaly import GradientAnomalyDetector
from src.detectors.access_frequency import AccessFrequencyDetector
from src.detectors.embedding_drift import EmbeddingDriftDetector
from src.detectors.gradient_distribution import GradientDistributionDetector
from src.detectors.temporal_access import TemporalAccessDetector
from src.evaluation.harness import EvalRun, DataConfig, AttackConfig
from src.evaluation.sensitivity import sweep, format_sweep_results
from src.evaluation.compare import compare, format_comparison

DETECTOR_REGISTRY = {
    "gradient_anomaly": lambda: GradientAnomalyDetector(threshold_z=3.0, min_steps=20),
    "access_frequency": lambda: AccessFrequencyDetector(concentration_threshold=5.0, min_steps=10),
    "embedding_drift": lambda: EmbeddingDriftDetector(drift_threshold_z=3.0, min_steps=10),
    "gradient_distribution": lambda: GradientDistributionDetector(kurtosis_z=3.0, concentration_threshold=10.0, min_steps=20),
    "temporal_access": lambda: TemporalAccessDetector(burst_window=5, burst_threshold=0.8, min_steps=10),
}


def run_single(args):
    """Run individual detectors and compare them."""
    data_config = DataConfig(
        n_users=args.n_users,
        n_items=args.n_items,
        batch_size=args.batch_size,
    )
    attack_config = AttackConfig(
        target_item=args.target_item,
        poison_ratio=args.attack_ratio,
        clean_steps=args.clean_steps,
        attack_steps=args.attack_steps,
    )

    names = args.detectors if args.detectors != ["all"] else list(DETECTOR_REGISTRY.keys())

    # Individual detector evaluation
    configs = []
    for name in names:
        if name not in DETECTOR_REGISTRY:
            print(f"Unknown detector: {name}")
            continue
        configs.append({
            "name": name,
            "detectors": [DETECTOR_REGISTRY[name]()],
        })

    # Also test "all combined"
    if len(names) > 1:
        configs.append({
            "name": "all_combined",
            "detectors": [DETECTOR_REGISTRY[n]() for n in names],
        })

    print(f"Evaluating {len(configs)} configurations...")
    print(f"  Data: {data_config.n_users} users, {data_config.n_items} items, batch={data_config.batch_size}")
    print(f"  Attack: target={attack_config.target_item}, ratio={attack_config.poison_ratio}, "
          f"clean={attack_config.clean_steps}, attack={attack_config.attack_steps}")
    print()

    results = compare(configs, data_config=data_config, attack_config=attack_config)
    print(format_comparison(results))


def run_sweep(args):
    """Run parameter sweep on a single detector."""
    data_config = DataConfig(
        n_users=args.n_users,
        n_items=args.n_items,
        batch_size=args.batch_size,
    )
    attack_config = AttackConfig(
        target_item=args.target_item,
        poison_ratio=args.attack_ratio,
        clean_steps=args.clean_steps,
        attack_steps=args.attack_steps,
    )

    name = args.sweep
    sweep_grids = {
        "gradient_anomaly": {
            "threshold_z": [2.0, 2.5, 3.0, 3.5, 4.0],
            "min_steps": [10, 20],
        },
        "access_frequency": {
            "concentration_threshold": [3.0, 4.0, 5.0, 7.0, 10.0],
            "min_steps": [5, 10, 15],
        },
        "embedding_drift": {
            "drift_threshold_z": [2.0, 2.5, 3.0, 4.0],
            "min_steps": [5, 10, 20],
        },
        "gradient_distribution": {
            "kurtosis_z": [2.0, 3.0, 4.0],
            "concentration_threshold": [5.0, 10.0, 20.0],
            "min_steps": [10, 20],
        },
        "temporal_access": {
            "burst_threshold": [0.6, 0.7, 0.8, 0.9],
            "burst_window": [3, 5, 7],
            "min_steps": [5, 10],
        },
    }

    if name not in sweep_grids:
        print(f"No sweep grid defined for: {name}")
        return

    detector_classes = {
        "gradient_anomaly": GradientAnomalyDetector,
        "access_frequency": AccessFrequencyDetector,
        "embedding_drift": EmbeddingDriftDetector,
        "gradient_distribution": GradientDistributionDetector,
        "temporal_access": TemporalAccessDetector,
    }

    print(f"Sweeping {name}...")
    print(f"  Grid: {sweep_grids[name]}")
    print()

    results = sweep(
        detector_classes[name],
        sweep_grids[name],
        data_config=data_config,
        attack_config=attack_config,
    )
    print(format_sweep_results(results))


def main():
    parser = argparse.ArgumentParser(description="EmbdGuard evaluation suite")
    parser.add_argument("--detectors", nargs="+", default=["all"],
                        help="Detectors to evaluate (or 'all')")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Run parameter sweep on this detector")
    parser.add_argument("--n-users", type=int, default=1000)
    parser.add_argument("--n-items", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-item", type=int, default=42)
    parser.add_argument("--attack-ratio", type=float, default=0.8)
    parser.add_argument("--clean-steps", type=int, default=25)
    parser.add_argument("--attack-steps", type=int, default=25)
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
