"""
train_climatebert.py
====================
Fine-tune climatebert/distilroberta-base-climate-f on the splits created by
prepare_data.py (one run per seed). Test-set predictions are written to
preds/ in the same format as run_llm.py
(id, gold, pred, model, seed, shots, prompt, raw_output).

Usage:
    python src/train_climatebert.py --seed 42

Quick pipeline check (runs on CPU):
    python src/train_climatebert.py --smoke-test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREDS_DIR = PROJECT_ROOT / "preds"
MODEL_NAME_DEFAULT = "climatebert/distilroberta-base-climate-f"

# Fixed class order -> consistent label <-> id mapping across all scripts
CLASSES = ["net-zero", "reduction", "none"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def load_csvs():
    paths = {s: DATA_DIR / f"{s}.csv" for s in ("train", "val", "test")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        sys.exit(f"Missing files: {missing}. Run 'python src/prepare_data.py' first.")
    return {s: pd.read_csv(p) for s, p in paths.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default=MODEL_NAME_DEFAULT)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--out-dir", default=str(PREDS_DIR))
    parser.add_argument("--smoke-test", action="store_true",
                         help="60 rows per split and 1 epoch for a quick pipeline check (runs on CPU)")
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from sklearn.metrics import f1_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as e:
        sys.exit(f"Missing package ({e}). Run 'pip install -r requirements.txt' first.")

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + ("" if device == "cuda" else " (no CUDA found, running on CPU)"))

    dfs = load_csvs()
    if args.smoke_test:
        dfs = {s: d.sample(min(len(d), 60), random_state=args.seed).reset_index(drop=True) for s, d in dfs.items()}
        args.epochs = 1
        print("Smoke test: 60 rows per split, 1 epoch.")

    for split, df in dfs.items():
        unknown = set(df["label"]) - set(CLASSES)
        if unknown:
            sys.exit(f"Unknown labels in {split}.csv: {unknown}. Check CLASSES in train_climatebert.py.")
        df["label_id"] = df["label"].map(LABEL2ID)

    print(f"Loading tokenizer/model '{args.model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    ds = {
        split: Dataset.from_pandas(df[["id", "text", "label_id"]], preserve_index=False)
        for split, df in dfs.items()
    }
    ds = {split: d.map(tokenize, batched=True) for split, d in ds.items()}
    for split in ds:
        ds[split] = ds[split].rename_column("label_id", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(CLASSES), id2label=ID2LABEL, label2id=LABEL2ID,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_score(labels, preds, average="macro")}

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / f"seed{args.seed}"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_metrics = trainer.evaluate(ds["val"])
    print(f"\nValidation Macro-F1 (best checkpoint): {val_metrics['eval_macro_f1']:.4f}")

    test_output = trainer.predict(ds["test"])
    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_pred_labels = [ID2LABEL[p] for p in test_preds]

    test_df = dfs["test"]
    macro_f1_test = f1_score(test_df["label_id"], test_preds, average="macro")
    print(f"Test Macro-F1: {macro_f1_test:.4f}")

    out_df = pd.DataFrame({
        "id": test_df["id"],
        "gold": test_df["label"],
        "pred": test_pred_labels,
        "model": "climatebert",
        "seed": args.seed,
        "shots": "",
        "prompt": "",
        "raw_output": "",
    })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoketest" if args.smoke_test else ""
    out_path = out_dir / f"climatebert_seed{args.seed}{suffix}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
