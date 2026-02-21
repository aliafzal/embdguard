"""EmbdGuard: Embedding-level poisoning detection for TorchRec."""
from embdguard.guard import EmbdGuard
from embdguard.alerts import Alert
from embdguard.detectors import BaseDetector
from embdguard.detectors.gradient_anomaly import GradientAnomalyDetector
from embdguard.detectors.access_frequency import AccessFrequencyDetector
from embdguard.detectors.tia import TIADetector

__version__ = "0.1.0"

__all__ = [
    "EmbdGuard",
    "Alert",
    "BaseDetector",
    "GradientAnomalyDetector",
    "AccessFrequencyDetector",
    "TIADetector",
]
