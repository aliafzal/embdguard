"""
Amazon Reviews 2023 dataset loader.

Uses HuggingFace `datasets` to load Amazon Reviews 2023 (5-core, rating only).
Remaps string user/item IDs to 0-indexed integers for embedding tables.

Reuses split_data() from src.dataset (already generic, uses timestamp ordering).
"""
import pandas as pd


def load_amazon_reviews(
    category: str = "Video_Games",
    kcore: int = 5,
    min_interactions: int = 5,
    cache_dir: str = "data/amazon",
) -> tuple:
    """
    Load Amazon Reviews 2023 from HuggingFace.

    Args:
        category: Amazon product category (e.g. "Video_Games")
        kcore: k-core filtering level (5 = 5-core)
        min_interactions: minimum interactions per user to keep
        cache_dir: directory to cache the downloaded dataset

    Returns:
        (df, n_users, n_items, user_map, item_map)
        df has columns: user_id (int), item_id (int), label (int), timestamp (int)
    """
    from datasets import load_dataset

    config = f"{kcore}core_rating_only_{category}"
    print(f"Loading Amazon Reviews 2023: {config} ...")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        config,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    # The dataset has a single "full" split
    split_name = list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    print(f"  Raw: {len(df)} interactions")

    # Standardize column names
    df = df.rename(columns={"parent_asin": "item_id"})
    # Keep only needed columns
    df = df[["user_id", "item_id", "rating", "timestamp"]].copy()

    # Filter users with fewer than min_interactions
    counts = df.groupby("user_id")["item_id"].count()
    active = counts[counts >= min_interactions].index
    df = df[df["user_id"].isin(active)].copy()

    # Remap string IDs to 0-indexed integers
    user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
    item_map = {v: i for i, v in enumerate(df["item_id"].unique())}
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)

    # Implicit feedback: any review = positive interaction
    df["label"] = 1

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    density = len(df) / (n_users * n_items) * 100
    print(f"  Users: {n_users}, Items: {n_items}, "
          f"Interactions: {len(df)}, Density: {density:.4f}%")
    return df, n_users, n_items, user_map, item_map
