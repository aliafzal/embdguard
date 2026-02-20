import os
import requests
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR  = "data"

def download_ml1m():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-1m.zip")
    if not os.path.exists(zip_path):
        print("Downloading MovieLens-1M...")
        r = requests.get(ML1M_URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    print("Extracted.")


def load_ratings(min_interactions=20):
    """
    Load ml-1m ratings, remap user/item IDs to 0-indexed integers,
    convert to implicit feedback (any rating = interaction).
    Returns DataFrame with columns: user_id, item_id, label (1=interacted).
    """
    path = os.path.join(DATA_DIR, "ml-1m", "ratings.dat")
    df   = pd.read_csv(path, sep="::", engine="python",
                       names=["user_id", "item_id", "rating", "timestamp"])

    # Keep only users with >= min_interactions
    counts = df.groupby("user_id")["item_id"].count()
    active = counts[counts >= min_interactions].index
    df = df[df["user_id"].isin(active)].copy()

    # Remap to 0-indexed
    user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
    item_map = {v: i for i, v in enumerate(df["item_id"].unique())}
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)
    df["label"]   = 1  # implicit feedback

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    print(f"Users: {n_users}, Items: {n_items}, Interactions: {len(df)}")
    return df, n_users, n_items, user_map, item_map


def split_data(df, test_ratio=0.2, seed=42):
    """Leave-one-out style: for each user, hold out last interaction."""
    df = df.sort_values(["user_id", "timestamp"]) if "timestamp" in df.columns \
         else df.sort_values("user_id")
    train_rows, test_rows = [], []
    for uid, group in df.groupby("user_id"):
        if len(group) < 2:
            train_rows.append(group)
        else:
            train_rows.append(group.iloc[:-1])
            test_rows.append(group.iloc[[-1]])
    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


def negative_sample(df, n_items, n_neg=4, seed=42):
    """
    For each positive (user, item) pair in df, sample n_neg negatives.
    Returns DataFrame with columns: user_id, item_id, label.
    Vectorized with numpy for speed.
    """
    rng = np.random.default_rng(seed)
    n_pos = len(df)
    user_arr = df["user_id"].values
    item_arr = df["item_id"].values

    # Build set of positive pairs for collision check
    pos_set = set(zip(user_arr.tolist(), item_arr.tolist()))

    # Vectorized: sample all negatives at once
    neg_users = np.repeat(user_arr, n_neg)
    neg_items = rng.integers(0, n_items, size=n_pos * n_neg)

    # Rejection-resample collisions (rare: n_items >> avg user history)
    for idx in range(len(neg_users)):
        while (neg_users[idx], neg_items[idx]) in pos_set:
            neg_items[idx] = rng.integers(0, n_items)

    neg_df = pd.DataFrame({
        "user_id": neg_users,
        "item_id": neg_items,
        "label": np.zeros(len(neg_users), dtype=np.int32),
    })
    combined = pd.concat([df[["user_id", "item_id", "label"]], neg_df])
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)
