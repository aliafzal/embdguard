# test_data.py
from src.dataset import download_ml1m, load_ratings, split_data, negative_sample

download_ml1m()
df, n_users, n_items, umap, imap = load_ratings()

assert n_users > 1000,  f"Expected >1000 users, got {n_users}"
assert n_items > 1000,  f"Expected >1000 items, got {n_items}"
assert len(df)  > 100000, f"Expected >100k interactions, got {len(df)}"

train_df, test_df = split_data(df)
assert len(test_df) == n_users, \
    f"Expected {n_users} test rows (one per user), got {len(test_df)}"

sampled = negative_sample(train_df.head(200), n_items, n_neg=4)
assert set(sampled["label"].unique()) == {0, 1}, "Missing positive or negative labels"

print(f"\nTrain: {len(train_df)} rows, Test: {len(test_df)} rows")
print(f"Sampled batch shape: {sampled.shape}")
print("✓ data loading OK")
