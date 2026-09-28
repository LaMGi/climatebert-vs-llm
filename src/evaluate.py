"""
evaluate.py
===========
Compute metrics, tables and figures from all prediction files in preds/.
The majority-class baseline is computed directly from data/train.csv and
data/test.csv.

Outputs:
- results/tables/summary.tex             (+ .csv)  one row per run
- results/tables/summary_aggregated.tex  (+ .csv)  mean +/- std per group
                                                     (ClimateBERT over seeds,
                                                     LLM over prompt variants
                                                     per shot level)
- results/tables/summary_by_approach.tex (+ .csv)  one row per approach
- results/tables/llm_grid.tex            (+ .csv)  Macro-F1 per prompt x shots
- results/tables/mcnemar.tex             (+ .csv)  pairwise McNemar tests and
                                                     bootstrap CIs for the key runs
- results/figures/confusion_<run>.pdf     confusion matrix per run
- results/figures/macro_f1_comparison.pdf Macro-F1 of all runs
- results/figures/shots_curve.pdf         Macro-F1 vs. number of shots

Usage:
    python src/evaluate.py
"""

import glob
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from statsmodels.stats.contingency_tables import mcnemar

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREDS_DIR = PROJECT_ROOT / "preds"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

CLASSES = ["net-zero", "reduction", "none"]

# Color palette: categorical colors per model group, text and grid colors
COLOR_BASELINE = "#2a78d6"      # blue
COLOR_CLIMATEBERT = "#eb6834"   # orange
COLOR_LLM = "#1baf7a"           # aqua
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]  # light -> dark

GROUP_COLOR = {"baseline": COLOR_BASELINE, "climatebert": COLOR_CLIMATEBERT, "llm": COLOR_LLM}
GROUP_LABEL = {"baseline": "Baseline", "climatebert": "ClimateBERT", "llm": "LLM"}

# Shots curve: color encodes prompt variant, line style/marker encodes backend
PROMPT_COLOR = {"concise": "#2a78d6", "detailed": "#eb6834", "structured": "#1baf7a"}
BACKEND_STYLE = {"openai": {"linestyle": "-", "marker": "o"}, "ollama": {"linestyle": "--", "marker": "s"}}

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK_PRIMARY,
    "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
})


def run_group(model_name: str) -> str:
    if model_name == "baseline":
        return "baseline"
    if model_name == "climatebert":
        return "climatebert"
    return "llm"


def load_baseline() -> pd.DataFrame:
    train_path, test_path = DATA_DIR / "train.csv", DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        return pd.DataFrame(columns=["id", "gold", "pred", "model"])
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    majority = train["label"].value_counts().idxmax()
    return pd.DataFrame({
        "id": test["id"], "gold": test["label"], "pred": majority, "model": "baseline",
    })


def load_all_runs() -> dict:
    runs = {}
    baseline = load_baseline()
    if not baseline.empty:
        runs["baseline"] = baseline
    for path in sorted(glob.glob(str(PREDS_DIR / "*.csv"))):
        name = Path(path).stem
        df = pd.read_csv(path)
        if not {"id", "gold", "pred", "model"}.issubset(df.columns):
            print(f"Skipping {path}: expected columns missing.")
            continue
        runs[name] = df
    return runs


def metrics_for(df: pd.DataFrame) -> dict:
    gold = df["gold"]
    pred = df["pred"].fillna("invalid")
    macro_f1 = f1_score(gold, pred, labels=CLASSES, average="macro", zero_division=0)
    per_class = f1_score(gold, pred, labels=CLASSES, average=None, zero_division=0)
    acc = (gold == pred).mean()
    n_invalid = int((pred == "invalid").sum())
    out = {"n": len(df), "accuracy": acc, "macro_f1": macro_f1, "n_invalid": n_invalid}
    out.update({f"f1_{c}": v for c, v in zip(CLASSES, per_class)})
    return out


def plot_confusion(df: pd.DataFrame, name: str) -> None:
    gold = df["gold"]
    pred = df["pred"].fillna("invalid")
    labels = CLASSES + (["invalid"] if (pred == "invalid").any() else [])
    cm = confusion_matrix(gold, pred, labels=labels)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(name, color=INK_PRIMARY, fontsize=9)
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > vmax * 0.6 else INK_PRIMARY
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"confusion_{name}.pdf")
    plt.close(fig)


def plot_macro_f1_bar(summary: pd.DataFrame) -> None:
    summary = summary.sort_values("macro_f1")
    groups = [run_group(m) for m in summary["model"]]
    colors = [GROUP_COLOR[g] for g in groups]

    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.28 * len(summary))))
    bars = ax.barh(summary["run"], summary["macro_f1"], color=colors, height=0.6, zorder=3)
    ax.set_xlabel("Macro-F1")
    ax.set_xlim(0, 1)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    for bar, val in zip(bars, summary["macro_f1"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                va="center", fontsize=7, color=INK_SECONDARY)

    present_groups = list(dict.fromkeys(groups))  # unique, order preserved
    handles = [mpatches.Patch(color=GROUP_COLOR[g], label=GROUP_LABEL[g]) for g in present_groups]
    if len(handles) > 1:
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "macro_f1_comparison.pdf")
    plt.close(fig)


def plot_shots_curve(summary: pd.DataFrame) -> None:
    """Macro-F1 vs. number of shots, one line per backend x prompt variant,
    with ClimateBERT (mean over seeds) and baseline as reference lines."""
    llm = summary.dropna(subset=["shots", "prompt", "backend"])
    if llm.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (backend, prompt), grp in llm.groupby(["backend", "prompt"]):
        grp = grp.sort_values("shots")
        style = BACKEND_STYLE.get(backend, {"linestyle": "-", "marker": "o"})
        ax.plot(grp["shots"], grp["macro_f1"],
                color=PROMPT_COLOR.get(prompt, INK_MUTED),
                linestyle=style["linestyle"], marker=style["marker"],
                markersize=5, linewidth=1.8, label=f"{backend} / {prompt}")

    x_max = llm["shots"].max()
    cb = summary[summary["model"] == "climatebert"]
    if not cb.empty:
        cb_mean = cb["macro_f1"].mean()
        ax.axhline(cb_mean, color=COLOR_CLIMATEBERT, linestyle=":", linewidth=1.5, zorder=1)
        ax.text(x_max, cb_mean, f" ClimateBERT mean {cb_mean:.3f}", va="bottom", ha="right",
                color=COLOR_CLIMATEBERT, fontsize=8)
    # Zoom y-axis to the LLM range; baseline is only drawn if it lies within
    y_vals = list(llm["macro_f1"]) + ([cb_mean] if not cb.empty else [])
    y_lo = np.floor((min(y_vals) - 0.01) * 100) / 100
    y_hi = np.ceil((max(y_vals) + 0.01) * 100) / 100
    base = summary[summary["model"] == "baseline"]
    if not base.empty:
        base_val = base["macro_f1"].iloc[0]
        if y_lo <= base_val <= y_hi:
            ax.axhline(base_val, color=COLOR_BASELINE, linestyle=":", linewidth=1.5, zorder=1)
            ax.text(x_max, base_val, f" Baseline {base_val:.2f}", va="bottom", ha="right",
                    color=COLOR_BASELINE, fontsize=8)

    ax.set_xlabel("Number of shots")
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(sorted(llm["shots"].unique()))
    ax.set_ylim(y_lo, y_hi)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "shots_curve.pdf")
    plt.close(fig)


def aggregated_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std per group: ClimateBERT over seeds, LLM over prompt variants
    per backend x shot level. The baseline is a single deterministic run."""
    rows = []
    base = summary[summary["model"] == "baseline"]
    if not base.empty:
        rows.append({"group": "Baseline", "n_runs": 1,
                     "macro_f1_mean": base["macro_f1"].iloc[0], "macro_f1_std": 0.0})

    cb = summary[summary["model"] == "climatebert"]
    if not cb.empty:
        rows.append({"group": "ClimateBERT (all seeds)", "n_runs": len(cb),
                     "macro_f1_mean": cb["macro_f1"].mean(), "macro_f1_std": cb["macro_f1"].std(ddof=0)})

    llm = summary.dropna(subset=["shots", "prompt", "backend"])
    for (backend, shots), grp in llm.groupby(["backend", "shots"]):
        rows.append({
            "group": f"LLM {backend}, {int(shots)}-shot (all prompts)",
            "n_runs": len(grp),
            "macro_f1_mean": grp["macro_f1"].mean(),
            "macro_f1_std": grp["macro_f1"].std(ddof=0),
        })
    return pd.DataFrame(rows)


BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42


def bootstrap_delta_ci(merged: pd.DataFrame, n_samples: int = BOOTSTRAP_SAMPLES,
                       seed: int = BOOTSTRAP_SEED, alpha: float = 0.05):
    """Paired bootstrap (Berg-Kirkpatrick et al. 2012): resample test sentences
    with replacement, compute Macro-F1(A) - Macro-F1(B) per sample and return
    the percentile interval."""
    gold = merged["gold"].to_numpy()
    pa = merged["pred_a"].fillna("invalid").to_numpy()
    pb = merged["pred_b"].fillna("invalid").to_numpy()
    rng = np.random.default_rng(seed)
    n = len(gold)
    deltas = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, n)
        deltas[i] = (f1_score(gold[idx], pa[idx], labels=CLASSES, average="macro", zero_division=0)
                     - f1_score(gold[idx], pb[idx], labels=CLASSES, average="macro", zero_division=0))
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def approach_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per approach (baseline, ClimateBERT, each LLM backend):
    Macro-F1 mean/std/best, per-class F1 and accuracy averaged over all runs."""
    def approach(row):
        if row["model"] in ("baseline", "climatebert"):
            return row["model"]
        return f"llm_{row['backend']}"
    s = summary.assign(approach=summary.apply(approach, axis=1))
    rows = []
    for name, grp in s.groupby("approach", sort=False):
        out = {"approach": name, "n_runs": len(grp),
               "macro_f1_mean": grp["macro_f1"].mean(),
               "macro_f1_std": grp["macro_f1"].std(ddof=0),
               "macro_f1_best": grp["macro_f1"].max(),
               "best_run": grp.loc[grp["macro_f1"].idxmax(), "run"]}
        for c in CLASSES:
            out[f"f1_{c}_mean"] = grp[f"f1_{c}"].mean()
        out["accuracy_mean"] = grp["accuracy"].mean()
        rows.append(out)
    return pd.DataFrame(rows)


def llm_grid(summary: pd.DataFrame) -> pd.DataFrame:
    """Macro-F1 per backend x prompt variant x shot level, plus mean and
    range (best minus worst prompt) per shot level."""
    llm = summary.dropna(subset=["shots", "prompt", "backend"]).copy()
    llm["shots"] = llm["shots"].astype(int)
    frames = []
    for backend, grp in llm.groupby("backend"):
        piv = grp.pivot_table(index="prompt", columns="shots", values="macro_f1")
        piv["mean"] = piv.mean(axis=1)
        mean_row = piv.drop(columns="mean").mean(axis=0).rename("mean")
        range_row = (piv.drop(columns="mean").max(axis=0) - piv.drop(columns="mean").min(axis=0)).rename("range")
        piv = pd.concat([piv, mean_row.to_frame().T, range_row.to_frame().T])
        piv.insert(0, "backend", backend)
        frames.append(piv.reset_index().rename(columns={"index": "prompt"}))
    return pd.concat(frames, ignore_index=True)


def pairwise_mcnemar(runs: dict, key_runs: list) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(key_runs, 2):
        if a not in runs or b not in runs:
            continue
        da, db = runs[a], runs[b]
        merged = da[["id", "gold", "pred"]].merge(db[["id", "pred"]], on="id", suffixes=("_a", "_b"))
        if len(merged) < 10:
            continue
        correct_a = merged["pred_a"] == merged["gold"]
        correct_b = merged["pred_b"] == merged["gold"]
        table = pd.crosstab(correct_a, correct_b).reindex(index=[False, True], columns=[False, True], fill_value=0)
        n_discordant = table.loc[False, True] + table.loc[True, False]
        if n_discordant == 0:
            # No discordant pairs: chi2 undefined, no difference -> p = 1
            statistic, p_value = 0.0, 1.0
        else:
            result = mcnemar(table.values, exact=(table.values.sum() < 25))
            statistic, p_value = result.statistic, result.pvalue
        # n_a / n_b = sentences where only A / only B is wrong
        n_a = int(table.loc[False, True])
        n_b = int(table.loc[True, False])
        f1_a = f1_score(merged["gold"], merged["pred_a"].fillna("invalid"), labels=CLASSES, average="macro", zero_division=0)
        f1_b = f1_score(merged["gold"], merged["pred_b"].fillna("invalid"), labels=CLASSES, average="macro", zero_division=0)
        ci_lo, ci_hi = bootstrap_delta_ci(merged)
        rows.append({
            "run_a": a, "run_b": b, "n_overlap": len(merged),
            "n_a": n_a, "n_b": n_b,
            "statistic": statistic, "p_value": p_value,
            "delta_macro_f1": f1_a - f1_b, "ci_low": ci_lo, "ci_high": ci_hi,
        })
    return pd.DataFrame(rows)


def pick_key_runs(summary: pd.DataFrame) -> list:
    """Key runs for the McNemar tests: baseline, ClimateBERT seed 42
    (fallback: first seed) and the best LLM run per backend."""
    key = []
    if "baseline" in summary["run"].values:
        key.append("baseline")

    cb = summary[summary["model"] == "climatebert"]
    if not cb.empty:
        cb_42 = cb[cb["run"].str.contains("seed42") & ~cb["run"].str.contains("smoketest")]
        key.append(cb_42["run"].iloc[0] if not cb_42.empty else cb["run"].iloc[0])

    llm = summary.dropna(subset=["backend"])
    for backend, grp in llm.groupby("backend"):
        best = grp.sort_values("macro_f1", ascending=False).iloc[0]
        key.append(best["run"])
    return key


def main():
    runs = load_all_runs()
    if not runs:
        print("No runs found. Run prepare_data.py / train_climatebert.py / run_llm.py first.")
        return
    print(f"Found runs ({len(runs)}): {list(runs.keys())}")

    summary_rows = []
    for name, df in runs.items():
        m = metrics_for(df)
        extra = {
            "shots": df["shots"].iloc[0] if "shots" in df.columns and pd.notna(df["shots"].iloc[0]) else None,
            "prompt": df["prompt"].iloc[0] if "prompt" in df.columns and pd.notna(df["prompt"].iloc[0]) else None,
            "backend": df["backend"].iloc[0] if "backend" in df.columns else None,
        }
        summary_rows.append({"run": name, "model": df["model"].iloc[0], **m, **extra})
        plot_confusion(df, name)

    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)

    # Exclude smoke-test runs (60-sample debug runs) from all results
    smoketest_mask = summary["run"].str.contains("smoketest")
    if smoketest_mask.any():
        print(f"\nExcluded debug runs: "
              f"{summary.loc[smoketest_mask, 'run'].tolist()}")
        summary = summary[~smoketest_mask].reset_index(drop=True)

    print("\n=== Summary (one row per run) ===")
    print(summary.drop(columns=["shots", "prompt", "backend"]).to_string(index=False))

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE_DIR / "summary.csv", index=False)
    with open(TABLE_DIR / "summary.tex", "w", encoding="utf-8") as f:
        f.write(summary.drop(columns=["shots", "prompt", "backend"]).round(3).to_latex(index=False))

    agg = aggregated_summary(summary)
    print("\n=== Aggregated summary (mean +/- std) ===")
    print(agg.round(4).to_string(index=False))
    agg.to_csv(TABLE_DIR / "summary_aggregated.csv", index=False)
    with open(TABLE_DIR / "summary_aggregated.tex", "w", encoding="utf-8") as f:
        f.write(agg.round(3).to_latex(index=False))

    by_approach = approach_summary(summary)
    print("\n=== Summary per approach ===")
    print(by_approach.round(4).to_string(index=False))
    by_approach.to_csv(TABLE_DIR / "summary_by_approach.csv", index=False)
    with open(TABLE_DIR / "summary_by_approach.tex", "w", encoding="utf-8") as f:
        f.write(by_approach.drop(columns=["best_run"]).round(3).to_latex(index=False))

    grid = llm_grid(summary)
    if not grid.empty:
        print("\n=== LLM grid: prompt x shots ===")
        print(grid.round(3).to_string(index=False))
        grid.to_csv(TABLE_DIR / "llm_grid.csv", index=False)
        with open(TABLE_DIR / "llm_grid.tex", "w", encoding="utf-8") as f:
            f.write(grid.round(3).to_latex(index=False))

    plot_macro_f1_bar(summary)
    plot_shots_curve(summary)

    key_runs = pick_key_runs(summary)
    mcnemar_df = pairwise_mcnemar(runs, key_runs)
    if not mcnemar_df.empty:
        print(f"\n=== McNemar tests (key runs: {key_runs}) ===")
        print(mcnemar_df.to_string(index=False))
        mcnemar_df.round(4).to_csv(TABLE_DIR / "mcnemar.csv", index=False)
        with open(TABLE_DIR / "mcnemar.tex", "w", encoding="utf-8") as f:
            f.write(mcnemar_df.round(4).to_latex(index=False))
    else:
        print("\nMcNemar skipped: fewer than two runs with >= 10 overlapping test IDs.")

    print(f"\nTables -> {TABLE_DIR}\nFigures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
