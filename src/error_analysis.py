"""
error_analysis.py
=================
Qualitative error analysis: joins the sentence text (data/test.csv) with the
predictions of the three key runs (same selection as the McNemar tests in
evaluate.py: ClimateBERT seed 42, best Ollama run, best OpenAI run) and writes
all sentences misclassified by at least one of them.

Output columns:
    id, claim, gold,
    pred_climatebert,
    pred_ollama, raw_ollama,
    pred_openai, raw_openai,
    n_wrong   (0-3, number of models that disagree with the gold label)

Rows are sorted by n_wrong in descending order.

Usage:
    python src/error_analysis.py
    python src/error_analysis.py --climatebert-run climatebert_seed1 \
        --ollama-run llm_ollama_llama3.1-8b_concise_0shot \
        --openai-run llm_openai_gpt-6-luna_concise_0shot
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREDS_DIR = PROJECT_ROOT / "preds"
OUT_DIR = PROJECT_ROOT / "results" / "error_analysis"

# Same key runs as evaluate.py::pick_key_runs()
DEFAULT_CLIMATEBERT_RUN = "climatebert_seed42"
DEFAULT_OLLAMA_RUN = "llm_ollama_llama3.1-8b_structured_1shot"
DEFAULT_OPENAI_RUN = "llm_openai_gpt-6-luna_structured_0shot"


def load_run(run_name: str) -> pd.DataFrame:
    path = PREDS_DIR / f"{run_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Run '{run_name}' not found at {path}. "
            "See results/tables/summary.csv for available runs."
        )
    return pd.read_csv(path)


def build_table(climatebert_run: str, ollama_run: str, openai_run: str) -> pd.DataFrame:
    test = pd.read_csv(DATA_DIR / "test.csv")[["id", "text", "label"]]
    test = test.rename(columns={"text": "claim", "label": "gold"})

    cb = load_run(climatebert_run)[["id", "pred"]].rename(columns={"pred": "pred_climatebert"})
    ol = load_run(ollama_run)[["id", "pred", "raw_output"]].rename(
        columns={"pred": "pred_ollama", "raw_output": "raw_ollama"})
    oa = load_run(openai_run)[["id", "pred", "raw_output"]].rename(
        columns={"pred": "pred_openai", "raw_output": "raw_openai"})

    df = test.merge(cb, on="id", how="inner").merge(ol, on="id", how="inner").merge(oa, on="id", how="inner")

    pred_cols = ["pred_climatebert", "pred_ollama", "pred_openai"]
    df["n_wrong"] = sum((df[c] != df["gold"]).astype(int) for c in pred_cols)

    df = df[[
        "id", "claim", "gold",
        "pred_climatebert",
        "pred_ollama", "raw_ollama",
        "pred_openai", "raw_openai",
        "n_wrong",
    ]]
    return df.sort_values(["n_wrong", "id"], ascending=[False, True]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--climatebert-run", default=DEFAULT_CLIMATEBERT_RUN,
                         help=f"file name (without .csv) in preds/, default: {DEFAULT_CLIMATEBERT_RUN}")
    parser.add_argument("--ollama-run", default=DEFAULT_OLLAMA_RUN,
                         help=f"file name (without .csv) in preds/, default: {DEFAULT_OLLAMA_RUN}")
    parser.add_argument("--openai-run", default=DEFAULT_OPENAI_RUN,
                         help=f"file name (without .csv) in preds/, default: {DEFAULT_OPENAI_RUN}")
    args = parser.parse_args()

    full = build_table(args.climatebert_run, args.ollama_run, args.openai_run)
    misclassified = full[full["n_wrong"] > 0].reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "misclassified.csv"
    misclassified.to_csv(out_path, index=False)

    n_total = len(full)
    n_wrong = len(misclassified)
    counts = misclassified["n_wrong"].value_counts().sort_index(ascending=False)
    print(f"Sentences present in all three runs: {n_total}")
    print(f"Misclassified by at least one model: {n_wrong} ({n_wrong / n_total:.1%})")
    for n, count in counts.items():
        print(f"  n_wrong={n}: {count} sentences")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
