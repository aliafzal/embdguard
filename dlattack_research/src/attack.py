"""
DLAttack implementation for Two-Tower recommender with DMP support.

Algorithm (from Huang et al. NDSS 2021, adapted for Two-Tower):

For R rounds:
  1. Extract weights from DMP model, create plain TwoTower surrogate
  2. For each of the m fake users:
       a. Initialize continuous interaction weight vector w in R^{n_items}
       b. Optimize w via gradient ascent on attack_loss
       c. Select top-n filler items + target item
       d. Threshold w to get discrete binary interaction vector
  3. Inject discrete interactions into training data
  4. Create new TwoTower with expanded user embedding, re-wrap with DMP, retrain

Key insight: the surrogate model used for gradient optimization does NOT need
DMP -- it's a plain TwoTower for direct weight access and deepcopy.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy
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
    """
    Attack objective: maximize score between fake user embedding and target item.

    user_continuous_emb: (embed_dim,) -- continuous approximation of fake user embedding
    target_item_emb:     (embed_dim,) -- target item embedding (detached from surrogate)
    all_item_embs:       (n_items, embed_dim) -- all item embeddings (for L2 reg)
    lambda_reg:          weight for embedding magnitude regularization
    """
    u_norm = nn.functional.normalize(user_continuous_emb.unsqueeze(0), dim=-1).squeeze(0)
    t_norm = nn.functional.normalize(target_item_emb, dim=-1)

    # Primary: maximize dot product with target item (minimize negative)
    promote_loss = -(u_norm * t_norm).sum()

    # Regularization: keep fake user embedding within normal distribution
    reg = user_continuous_emb.norm(p=2)

    return promote_loss + lambda_reg * reg


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

    Returns:
        binary interaction vector of shape (n_items,)
        where entry i = 1 means fake user "interacted with" item i.
        Always includes target_item_id in interactions.
    """
    surrogate.eval()

    # Get item embeddings from the surrogate (detach so we don't update them)
    all_item_embs = surrogate.ebc.embedding_bags["t_item_id"].weight.detach()
    target_item_emb = all_item_embs[target_item_id].detach()

    # Initialize: continuous weight vector over items
    w = torch.zeros(n_items, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=lr_attack)

    for _ in range(n_optim_steps):
        optimizer.zero_grad()

        # Aggregate item embeddings weighted by w (soft bag-of-items)
        w_softmax = torch.softmax(w, dim=0)
        fake_user_emb = (w_softmax.unsqueeze(1) * all_item_embs).sum(0)

        loss = attack_loss(fake_user_emb, target_item_emb, all_item_embs)
        loss.backward()
        optimizer.step()

    # Discretize: select top-n_filler items + target item
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
    """
    Generate m fake users using a plain TwoTower surrogate.
    Returns DataFrame ready to concat with training data.
    """
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


def _build_surrogate_from_dmp(dmp_model, device) -> TwoTower:
    """Extract weights from DMP model and create a plain TwoTower surrogate."""
    inner_two_tower = unwrap_model(dmp_model)
    # Deep copy the plain TwoTower (not the DMP wrapper)
    surrogate = deepcopy(inner_two_tower)
    return surrogate


def _rebuild_dmp_model(two_tower, device, lr):
    """Create a new DMP-wrapped model from a plain TwoTower."""
    train_task = TwoTowerTrainTask(two_tower)
    dmp_model, dense_optimizer = wrap_with_dmp(train_task, device, lr=lr)
    return dmp_model, dense_optimizer


def run_dlattack(
    dmp_model,
    dense_optimizer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    target_item_id: int,
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

    For each round:
      1. Extract weights from DMP model, create plain TwoTower surrogate
      2. Generate m fake users via gradient optimization on surrogate
      3. Inject fake users into training data
      4. Create new TwoTower with expanded user embedding, re-wrap with DMP, retrain

    Args:
        dmp_model: DMP-wrapped TwoTowerTrainTask
        dense_optimizer: KeyedOptimizerWrapper for MLP params
        ...other args as before...

    Returns:
        (results_dict, poisoned_train_df, dmp_model, dense_optimizer)
    """
    from src.evaluate import evaluate

    poisoned_train = train_df.copy()
    fake_user_id_start = n_users
    results = {}

    # Evaluate before any attack
    if eval_fn:
        two_tower = unwrap_model(dmp_model)
        metrics = eval_fn(two_tower)
        results["round_0_clean"] = metrics
        print(f"\n  [Before Attack] HR@{metrics['K']}={metrics['HR@K']:.4f} | "
              f"NDCG@{metrics['K']}={metrics['NDCG@K']:.4f}")

    for r in range(1, rounds + 1):
        print(f"\n{'='*50}")
        print(f"  ATTACK ROUND {r}/{rounds}  |  target_item={target_item_id}")
        print(f"{'='*50}")

        # Step 1: Create plain TwoTower surrogate from DMP model
        surrogate = _build_surrogate_from_dmp(dmp_model, device)
        max_user_id = int(poisoned_train["user_id"].max()) + 1
        surrogate.resize_user_embedding(max_user_id)

        # Retrain surrogate (plain TwoTower, no DMP needed for surrogate)
        print(f"  Retraining surrogate on {len(poisoned_train)} interactions...")
        surr_task = TwoTowerTrainTask(surrogate)
        surr_dmp, surr_opt = wrap_with_dmp(surr_task, torch.device(device), lr=lr)
        train(
            surr_dmp, surr_opt, poisoned_train, n_items,
            epochs=retrain_epochs, batch_size=2048,
            device=device, eval_fn=None, save_path=None,
        )
        # Extract retrained surrogate
        surrogate = unwrap_model(surr_dmp)

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

        # Step 4: Create new TwoTower with expanded user embedding, re-wrap with DMP
        max_user_id = int(poisoned_train["user_id"].max()) + 1
        inner = unwrap_model(dmp_model)
        # Build a fresh TwoTower, copy weights, expand user embeddings
        embedding_dim = inner.ebc.embedding_bag_configs()[0].embedding_dim
        layer_sizes = []
        for module in inner.user_proj:
            if isinstance(module, nn.Linear):
                layer_sizes.append(module.out_features)
        new_ebc = build_ebc(max_user_id, n_items, embedding_dim, torch.device(device))
        new_two_tower = TwoTower(new_ebc, layer_sizes, torch.device(device))

        # Copy weights from current model
        with torch.no_grad():
            old_user_w = inner.ebc.embedding_bags["t_user_id"].weight.data
            new_two_tower.ebc.embedding_bags["t_user_id"].weight[:old_user_w.shape[0]] = old_user_w
            new_two_tower.ebc.embedding_bags["t_item_id"].weight.copy_(
                inner.ebc.embedding_bags["t_item_id"].weight.data
            )
            # Copy MLP weights
            new_two_tower.user_proj.load_state_dict(inner.user_proj.state_dict())
            new_two_tower.item_proj.load_state_dict(inner.item_proj.state_dict())

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
            two_tower = unwrap_model(dmp_model)
            metrics = eval_fn(two_tower)
            results[f"round_{r}"] = metrics
            print(f"  [After Round {r}] HR@{metrics['K']}={metrics['HR@K']:.4f} | "
                  f"NDCG@{metrics['K']}={metrics['NDCG@K']:.4f}")

    return results, poisoned_train, dmp_model, dense_optimizer
