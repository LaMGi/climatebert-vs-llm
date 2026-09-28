"""
prepare_data.py
===============
Download the `climatebert/netzero_reduction_data` dataset from Hugging Face,
assign a fixed `id` per sentence, create a stratified 70/15/15 split and
compute the majority-class baseline on the test set.

The dataset only ships a single "train" split, so the split is created here
and saved as train.csv / val.csv / test.csv in data/.

Usage:
    python src/prepare_data.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
DATASET_NAME = "climatebert/netzero_reduction_data"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Candidate column names; the first match in the raw dataset is used
TEXT_COL_CANDIDATES = ["text", "TEXT", "sentence", "Sentence"]
LABEL_COL_CANDIDATES = ["label", "LABEL", "target", "class"]


def find_column(columns, candidates, kind):
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"No {kind} column found. Available columns: {list(columns)}. "
        f"Adjust TEXT_COL_CANDIDATES / LABEL_COL_CANDIDATES."
    )


def load_raw():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "Package 'datasets' is missing. Run 'pip install -r requirements.txt' first."
        )

    print(f"Loading dataset '{DATASET_NAME}' from Hugging Face ...")
    try:
        ds = load_dataset(DATASET_NAME)
    except Exception as e:  # e.g. no connection, rate limit, renamed dataset
        sys.exit(
            f"Could not load dataset ({e!r}).\n"
            "Check your connection and the dataset name: "
            f"https://huggingface.co/datasets/{DATASET_NAME}"
        )

    print("Available splits:", list(ds.keys()))
    if len(ds) != 1:
        print(
            "Warning: dataset has more than one split, expected only 'train'. "
            "Concatenating all splits."
        )
        split = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
        label_feature = ds[list(ds.keys())[0]].features
    else:
        only_split = list(ds.keys())[0]
        hf_split = ds[only_split]
        label_feature = hf_split.features
        split = hf_split.to_pandas()

    return split, label_feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df, features = load_raw()
    print("Columns in raw dataset:", list(df.columns))
    print("Number of rows:", len(df))

    text_col = find_column(df.columns, TEXT_COL_CANDIDATES, "text")
    label_col = find_column(df.columns, LABEL_COL_CANDIDATES, "label")
    print(f"Using text column '{text_col}' and label column '{label_col}'.")

    # Decode int-encoded ClassLabel into readable label strings
    label_feature = features.get(label_col) if features is not None else None
    if label_feature is not None and hasattr(label_feature, "int2str"):
        df[label_col] = df[label_col].apply(label_feature.int2str)
        print("Label classes (from ClassLabel):", label_feature.names)

    print("Class distribution:")
    print(df[label_col].value_counts())

    df = df.rename(columns={text_col: "text", label_col: "label"})[["text", "label"]]
    df = df.reset_index(drop=True)
    df.insert(0, "id", df.index)

    # Stratified 70/15/15 split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=args.seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=args.seed
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print(f"\nSplit saved to {DATA_DIR}:")
    print(f"  train: {len(train_df)} rows")
    print(f"  val:   {len(val_df)} rows")
    print(f"  test:  {len(test_df)} rows")

    # Majority-class baseline (majority class taken from train)
    majority_class = train_df["label"].value_counts().idxmax()
    baseline_preds = [majority_class] * len(test_df)
    macro_f1 = f1_score(test_df["label"], baseline_preds, average="macro")

    print(f"\nMajority class (from train): {majority_class}")
    print(f"Baseline Macro-F1 on test: {macro_f1:.4f}")
    print(classification_report(test_df["label"], baseline_preds, zero_division=0))


if __name__ == "__main__":
    main()
