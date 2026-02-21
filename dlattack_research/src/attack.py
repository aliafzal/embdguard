"""
DLAttack implementation for Two-Tower recommender with DMP support.

Algorithm (from Huang et al. NDSS 2021, adapted for Two-Tower):

For R rounds:
  1. Extract weights from DMP model, create plain TwoTower surrogate
  2. For each of the m fake users: optimize continuous weight vector, discretize
  3. Inject discrete interactions into training data
  4. Create new TwoTower with expanded user embedding, re-wrap with DMP, retrain

Cannot deepcopy DMP-wrapped modules (they contain ProcessGroup objects)
and cannot introspect ShardedEBC configs. So we pass model config explicitly
and rebuild fresh TwoTowers with weight copying via state_dict.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.model import TwoTower, TwoTowerTrainTask, build_ebc, make_kjt
from src.train import train
from src.distributed import (
    wrap_with_dmp,
    unwrap_model,
    extract_state_dict,
)


def attack_loss(user_continuous_emb: torch.Tensor,
                target_item_emb: torch.Tensor,
                all_item_embs: torch.Tensor,
                lambda_reg: float = 1e-2) -> torch.Tensor:
    u_norm = nn.functional.normalize(user_continuous_emb.unsqueeze(0), dim=-1).squeeze(0)
    t_norm = nn.functional.normalize(target_item_emb, dim=-1)
    promote_loss = -(u_norm * t_norm).sum()
    reg = user_continuous_emb.norm(p=2)
    return promote_loss + lambda_reg * reg


def _build_plain_two_tower(n_users, n_items, embedding_dim, layer_sizes, device):
    """Build a fresh plain TwoTower on a real device (not meta)."""
    ebc = build_ebc(n_users, n_items, embedding_dim, device=torch.device(device))
    return TwoTower(ebc, layer_sizes=layer_sizes, device=torch.device(device))


def _deshard_state_dict(state):
    """Convert any ShardedTensor values in a state dict to regular tensors."""
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    clean = {}
    for k, v in state.items():
        if isinstance(v, ShardedTensor):
            # Single-GPU: one local shard contains the full tensor
            clean[k] = v.local_shards()[0].tensor
        else:
            clean[k] = v
    return clean


def _copy_weights_to_plain(dmp_model, plain_two_tower):
    """Copy weights from DMP-wrapped model to a plain TwoTower via state_dict."""
    src_state = _deshard_state_dict(extract_state_dict(dmp_model))
    tt_state = {}
    for k, v in src_state.items():
        if k.startswith("two_tower."):
            tt_state[k[len("two_tower."):]] = v
    if tt_state:
        plain_two_tower.load_state_dict(tt_state, strict=False)
    else:
        plain_two_tower.load_state_dict(src_state, strict=False)


def _build_surrogate_from_dmp(dmp_model, n_users, n_items, embedding_dim, layer_sizes, device):
    """Build a fresh plain TwoTower and copy weights from DMP model."""
    surrogate = _build_plain_two_tower(n_users, n_items, embedding_dim, layer_sizes, device)
    _copy_weights_to_plain(dmp_model, surrogate)
    return surrogate


def optimize_fake_user(
    surrogate: TwoTower,
    target_item_id: int,
    n_items: int,
    n_filler: int = 30,
    n_optim_steps: int = 200,
    lr_attack: float = 0.05,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate one fake user interaction vector via gradient optimization.
    Uses a plain TwoTower (not DMP-wrapped) for direct weight access.
    """
    surrogate.eval()

    all_item_embs = surrogate.ebc.embedding_bags["t_item_id"].weight.detach()
    target_item_emb = all_item_embs[target_item_id].detach()

    w = torch.zeros(n_items, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=lr_attack)

    for _ in range(n_optim_steps):
        optimizer.zero_grad()
        w_softmax = torch.softmax(w, dim=0)
        fake_user_emb = (w_softmax.unsqueeze(1) * all_item_embs).sum(0)
        loss = attack_loss(fake_user_emb, target_item_emb, all_item_embs)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        w_vals = w.detach().cpu().numpy()
        probs = torch.softmax(torch.tensor(w_vals), dim=0).numpy()
        cands = np.array([i for i in range(n_items) if i != target_item_id])
        filler = np.random.choice(
            cands,
            size=min(n_filler, len(cands)),
            replace=False,
            p=probs[cands] / probs[cands].sum()
        )
        interaction_vec = np.zeros(n_items, dtype=np.float32)
        interaction_vec[filler] = 1.0
        interaction_vec[target_item_id] = 1.0

    return interaction_vec


def generate_fake_users(
    surrogate: TwoTower,
    target_item_id: int,
    n_users: int,
    n_items: int,
    m: int = 5,
    n_filler: int = 30,
    n_optim_steps: int = 200,
    fake_user_id_start: int = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Generate m fake users using a plain TwoTower surrogate."""
    if fake_user_id_start is None:
        fake_user_id_start = n_users

    rows = []
    for i in tqdm(range(m), desc="  Optimizing fake users"):
        vec = optimize_fake_user(surrogate, target_item_id, n_items,
                                  n_filler=n_filler,
                                  n_optim_steps=n_optim_steps,
                                  device=device)
        fake_uid = fake_user_id_start + i
        item_ids = np.where(vec > 0)[0]
        for iid in item_ids:
            rows.append({"user_id": fake_uid, "item_id": int(iid), "label": 1})

    fake_df = pd.DataFrame(rows)
    print(f"  Generated {m} fake users with {len(fake_df)} total interactions "
          f"(avg {len(fake_df)/m:.1f}/user)")
    return fake_df


def run_dlattack(
    dmp_model,
    dense_optimizer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    target_item_id: int,
    embedding_dim: int = 64,
    layer_sizes: list = None,
    rounds: int = 5,
    m: int = 5,
    n_filler: int = 30,
    n_optim_steps: int = 200,
    retrain_epochs: int = 10,
    lr: float = 0.001,
    device: str = "cpu",
    eval_fn=None,
) -> tuple:
    """
    Full DLAttack loop with DMP support.

    Args:
        dmp_model: DMP-wrapped TwoTowerTrainTask
        dense_optimizer: KeyedOptimizerWrapper for MLP params
        embedding_dim: embedding dimension (passed explicitly, cannot introspect DMP)
        layer_sizes: MLP layer sizes (passed explicitly, cannot introspect DMP)

    Returns:
        (results_dict, poisoned_train_df, dmp_model, dense_optimizer)
    """
    if layer_sizes is None:
        layer_sizes = [128, 64]

    poisoned_train = train_df.copy()
    fake_user_id_start = n_users
    results = {}

    # Evaluate before any attack
    if eval_fn:
        two_tower = _build_surrogate_from_dmp(
            dmp_model, n_users, n_items, embedding_dim, layer_sizes, device
        )
        metrics = eval_fn(two_tower)
        results["round_0_clean"] = metrics
        print(f"\n  [Before Attack] HR@{metrics['K']}={metrics['HR@K']:.4f} | "
              f"NDCG@{metrics['K']}={metrics['NDCG@K']:.4f}")

    for r in range(1, rounds + 1):
        print(f"\n{'='*50}")
        print(f"  ATTACK ROUND {r}/{rounds}  |  target_item={target_item_id}")
        print(f"{'='*50}")

        # Step 1: Create plain TwoTower surrogate from DMP model
        max_user_id = int(poisoned_train["user_id"].max()) + 1
        surrogate = _build_surrogate_from_dmp(
            dmp_model, max_user_id, n_items, embedding_dim, layer_sizes, device
        )

        # Retrain surrogate via DMP
        print(f"  Retraining surrogate on {len(poisoned_train)} interactions...")
        surr_task = TwoTowerTrainTask(surrogate)
        surr_dmp, surr_opt = wrap_with_dmp(surr_task, torch.device(device), lr=lr)
        train(
            surr_dmp, surr_opt, poisoned_train, n_items,
            epochs=retrain_epochs, batch_size=2048,
            device=device, eval_fn=None, save_path=None,
        )
        # Extract retrained surrogate as plain TwoTower
        surrogate = _build_plain_two_tower(max_user_id, n_items, embedding_dim, layer_sizes, device)
        _copy_weights_to_plain(surr_dmp, surrogate)

        # Step 2: Generate fake users using plain surrogate
        print(f"  Generating {m} fake users...")
        n_current_fake = (r - 1) * m
        fake_df = generate_fake_users(
            surrogate, target_item_id,
            n_users=n_users + n_current_fake,
            n_items=n_items,
            m=m, n_filler=n_filler,
            n_optim_steps=n_optim_steps,
            fake_user_id_start=fake_user_id_start,
            device=device,
        )
        fake_user_id_start += m

        # Step 3: Inject
        poisoned_train = pd.concat([poisoned_train, fake_df]).reset_index(drop=True)
        n_total_fake = len(poisoned_train) - len(train_df)
        print(f"  Poisoned training set: {len(poisoned_train)} interactions "
              f"({n_total_fake} from fake users)")

        # Step 4: Build new expanded TwoTower, copy weights, re-wrap with DMP, retrain
        max_user_id = int(poisoned_train["user_id"].max()) + 1
        new_two_tower = _build_plain_two_tower(
            max_user_id, n_items, embedding_dim, layer_sizes, device
        )
        _copy_weights_to_plain(dmp_model, new_two_tower)

        print(f"  Retraining main model on poisoned data...")
        new_task = TwoTowerTrainTask(new_two_tower)
        dmp_model, dense_optimizer = wrap_with_dmp(
            new_task, torch.device(device), lr=lr
        )
        train(dmp_model, dense_optimizer, poisoned_train, n_items,
              epochs=retrain_epochs, batch_size=2048,
              device=device, eval_fn=None, save_path=None)

        # Evaluate
        if eval_fn:
            eval_tt = _build_surrogate_from_dmp(
                dmp_model, max_user_id, n_items, embedding_dim, layer_sizes, device
            )
            metrics = eval_fn(eval_tt)
            results[f"round_{r}"] = metrics
            print(f"  [After Round {r}] HR@{metrics['K']}={metrics['HR@K']:.4f} | "
                  f"NDCG@{metrics['K']}={metrics['NDCG@K']:.4f}")

    return results, poisoned_train, dmp_model, dense_optimizer
