"""
run_all_llm.py
==============
Run the full LLM grid (backend x shots x prompt variant) on the test set by
calling run_llm.py once per combination.

Combinations whose preds/llm_<backend>_<model>_<prompt>_<shots>shot.csv
already exists are skipped, so an interrupted run can simply be restarted.
Within a run, the response cache (llm_cache.json) avoids repeated API calls.

Usage:
    # Ollama only (5 shot levels x 3 prompt variants):
    python src/run_all_llm.py --backend ollama --ollama-model llama3.1:8b

    # OpenAI only:
    python src/run_all_llm.py --backend openai --openai-model gpt-6-luna

    # Both backends:
    python src/run_all_llm.py --backend both --openai-model gpt-6-luna --ollama-model llama3.1:8b

    # List pending combinations without running them:
    python src/run_all_llm.py --backend both --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDS_DIR = PROJECT_ROOT / "preds"
PYTHON = sys.executable

SHOTS = [0, 1, 3, 5, 10]
PROMPTS = ["concise", "detailed", "structured"]


def pred_path(backend: str, model: str, prompt: str, shots: int) -> Path:
    model_safe = model.replace(":", "-").replace("/", "-")
    return PREDS_DIR / f"llm_{backend}_{model_safe}_{prompt}_{shots}shot.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["openai", "ollama", "both"], required=True)
    parser.add_argument("--openai-model", default="gpt-6-luna")
    parser.add_argument("--ollama-model", default="llama3.1:8b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.0, help="passed through to run_llm.py --sleep")
    parser.add_argument("--dry-run", action="store_true", help="only list pending combinations")
    args = parser.parse_args()

    backends = ["openai", "ollama"] if args.backend == "both" else [args.backend]
    model_for = {"openai": args.openai_model, "ollama": args.ollama_model}

    jobs = [(b, model_for[b], p, s) for b in backends for p in PROMPTS for s in SHOTS]
    todo = [j for j in jobs if not pred_path(*j).exists()]
    done = len(jobs) - len(todo)

    print(f"{len(jobs)} combinations in total, {done} already done, {len(todo)} to run.")
    if args.dry_run:
        for b, m, p, s in todo:
            print(f"  would run: --backend {b} --model {m} --prompt {p} --shots {s}")
        return
    if not todo:
        print("Nothing to do.")
        return

    for i, (backend, model, prompt, shots) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] backend={backend} model={model} prompt={prompt} shots={shots}")
        t0 = time.time()
        cmd = [
            PYTHON, str(PROJECT_ROOT / "src" / "run_llm.py"),
            "--backend", backend, "--model", model,
            "--shots", str(shots), "--prompt", prompt,
            "--split", "test", "--seed", str(args.seed),
        ]
        if args.sleep:
            cmd += ["--sleep", str(args.sleep)]
        result = subprocess.run(cmd)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(
                f"ERROR for backend={backend} prompt={prompt} shots={shots} (exit code {result.returncode}).\n"
                "Restart the script to resume; finished combinations are skipped."
            )
            sys.exit(result.returncode)
        print(f"  done in {elapsed:.0f}s")

    print("\nAll combinations done.")


if __name__ == "__main__":
    main()
