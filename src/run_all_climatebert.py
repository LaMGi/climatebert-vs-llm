"""
run_all_climatebert.py
======================
Train ClimateBERT sequentially for several seeds (full runs, no --smoke-test).

Seeds whose preds/climatebert_seed<N>.csv already exists are skipped, so an
interrupted run can simply be restarted.

Usage:
    python src/run_all_climatebert.py --seeds 42 1 2
    python src/run_all_climatebert.py --seeds 42 1 2 --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDS_DIR = PROJECT_ROOT / "preds"
PYTHON = sys.executable


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    todo = [s for s in args.seeds if not (PREDS_DIR / f"climatebert_seed{s}.csv").exists()]
    done = len(args.seeds) - len(todo)
    print(f"{len(args.seeds)} seeds in total, {done} already done, {len(todo)} to run: {todo}")
    if args.dry_run or not todo:
        return

    for i, seed in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] Seed {seed}")
        t0 = time.time()
        cmd = [
            PYTHON, str(PROJECT_ROOT / "src" / "train_climatebert.py"),
            "--seed", str(seed), "--epochs", str(args.epochs),
        ]
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(
                f"ERROR for seed {seed} (exit code {result.returncode}).\n"
                "Restart the script to resume; finished seeds are skipped."
            )
            sys.exit(result.returncode)
        print(f"  done in {elapsed / 60:.1f} min")

    print("\nAll seeds done.")


if __name__ == "__main__":
    main()
