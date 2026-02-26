"""
Backward-compatibility shim — all model code now lives in src.models.

All existing ``from src.model import ...`` statements continue to work.
"""
from .models.common import (  # noqa: F401
    EmbeddingBagConfig,
    EmbeddingBagCollection,
    KeyedJaggedTensor,
    _TORCHREC_AVAILABLE,
    build_ebc,
    make_kjt,
    _build_mlp,
)
from .models.two_tower import TwoTower, TwoTowerTrainTask, make_optimizer  # noqa: F401
from .models.dlrm import DLRM, DLRMTrainTask, make_dlrm_optimizer  # noqa: F401
