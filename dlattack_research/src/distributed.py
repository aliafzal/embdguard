"""
DistributedModelParallel (DMP) utilities for TorchRec.

DMP triggers fbgemm's fused Table Batched Embedding (TBE) kernels even on
single GPU, which batch multiple embedding lookups into a single CUDA kernel
and fuse the optimizer step into backward.
"""
import os
import torch
import torch.distributed as dist
from torch.optim import Adam
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import ShardingEnv

from src.model import TwoTower, TwoTowerTrainTask


def init_process_group(device: torch.device):
    """Initialize torch.distributed for single-GPU DMP usage."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)


def wrap_with_dmp(
    train_task: TwoTowerTrainTask,
    device: torch.device,
    lr: float = 0.001,
):
    """Wrap a TwoTowerTrainTask with DistributedModelParallel.

    1. Applies fused RowWiseAdagrad to EBC parameters (optimizer in backward)
    2. Wraps with DMP for fused TBE kernels
    3. Creates KeyedOptimizerWrapper with Adam for dense (MLP) parameters

    Returns:
        (dmp_model, dense_optimizer)
    """
    from torchrec.optim.apply_optimizer_in_backward import (
        apply_optimizer_in_backward,
    )

    # Fuse sparse optimizer into backward pass BEFORE DMP wrapping
    apply_optimizer_in_backward(
        RowWiseAdagrad,
        train_task.two_tower.ebc.parameters(),
        {"lr": lr},
    )

    # Wrap with DMP
    dmp_model = DistributedModelParallel(
        module=train_task,
        device=device,
    )

    # Dense optimizer for MLP parameters (skip fused-optimized EBC params)
    dense_optimizer = KeyedOptimizerWrapper(
        dict(
            in_backward_optimizer_filter(dmp_model.named_parameters())
        ),
        lambda params: Adam(params, lr=lr),
    )

    return dmp_model, dense_optimizer


def in_backward_optimizer_filter(named_params):
    """Yield only parameters that do NOT have a fused in-backward optimizer."""
    for name, param in named_params:
        if not hasattr(param, "_in_backward_optimizers"):
            yield name, param


def unwrap_model(dmp_model) -> TwoTower:
    """Extract the inner TwoTower from a DMP-wrapped model."""
    return dmp_model.module.two_tower


def deshard_state_dict(state: dict) -> dict:
    """Convert any ShardedTensor values in a state dict to regular tensors.

    After DMP wrapping, embedding parameters become ShardedTensors.
    For single-GPU, each ShardedTensor has one local shard containing the
    full tensor. This function extracts those regular tensors so that
    state dicts can be saved/loaded with plain torch.save/load.
    """
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    clean = {}
    for k, v in state.items():
        if isinstance(v, ShardedTensor):
            clean[k] = v.local_shards()[0].tensor
        else:
            clean[k] = v
    return clean


def extract_state_dict(dmp_model) -> dict:
    """Get the state dict from the inner module for checkpointing.

    Automatically deshards ShardedTensors to regular tensors so that
    checkpoints are portable and can be loaded into plain (non-DMP) models.
    """
    return deshard_state_dict(dmp_model.module.state_dict())
