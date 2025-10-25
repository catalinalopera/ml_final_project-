# src/data.py
import os
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import argparse

# This script generates a synthetic classification dataset, splits it into train/val/test,
# and saves CSV files to the data/processed directory.

def generate_dataset(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42):
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_informative, n_redundant=n_redundant,
                               random_state=random_state)
    columns = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["y"] = y
    return df

def split_and_save(df, test_size=0.2, val_size=0.1, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    # split off test set
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["y"], random_state=42)
    # compute relative validation size
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val_size, stratify=train_val["y"], random_state=42)

    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    df.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)

    print(f"Saved files in {out_dir}: train.csv, val.csv, test.csv, dataset.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--out-dir", type=str, default="data/processed")
    args = parser.parse_args()

    df = generate_dataset(n_samples=args.n_samples, n_features=args.n_features, random_state=42)
    split_and_save(df, test_size=args.test_size, val_size=args.val_size, out_dir=args.out_dir)