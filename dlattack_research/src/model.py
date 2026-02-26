"""Backward-compat shim — re-exports model symbols from src.models (EmbdGuard)."""
import sys, os

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_saved_src = sys.modules.get("src")
_saved_src_subs = {k: v for k, v in sys.modules.items() if k.startswith("src.")}

if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
for k in list(sys.modules.keys()):
    if k == "src" or k.startswith("src."):
        del sys.modules[k]

from src.models import (  # noqa: E402
    build_ebc, make_kjt, _build_mlp,
    TwoTower, TwoTowerTrainTask, make_optimizer,
    DLRM, DLRMTrainTask, make_dlrm_optimizer,
)

# Restore dlattack's src module
if _saved_src is not None:
    sys.modules["src"] = _saved_src
for k, v in _saved_src_subs.items():
    sys.modules[k] = v

__all__ = [
    "build_ebc", "make_kjt", "_build_mlp",
    "TwoTower", "TwoTowerTrainTask", "make_optimizer",
    "DLRM", "DLRMTrainTask", "make_dlrm_optimizer",
]
