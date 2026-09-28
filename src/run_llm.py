"""
run_llm.py
==========
Classify sentences into {net-zero, reduction, none} via LLM prompting, with
few-shot levels (0/1/3/5/10) and several prompt variants.

Works with both the OpenAI API and a local Ollama model, since Ollama exposes
an OpenAI-compatible Chat Completions API: only --backend (base URL, API key)
and --model differ.

Examples:
    # Check prompts without API calls (no key or Ollama needed):
    python src/run_llm.py --backend openai --model gpt-6-luna --shots 3 --prompt concise --limit 5 --dry-run

    # Quick test: 20 sentences, zero-shot, OpenAI
    python src/run_llm.py --backend openai --model gpt-6-luna --shots 0 --prompt concise --limit 20

    # Same with a local Ollama model
    python src/run_llm.py --backend ollama --model llama3.1:8b --shots 0 --prompt concise --limit 20
"""

import argparse
import hashlib
import json
import random
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PREDS_DIR = PROJECT_ROOT / "preds"
CACHE_PATH = PROJECT_ROOT / "llm_cache.json"

CLASSES = ["net-zero", "reduction", "none"]

# ---------------------------------------------------------------------------
# Prompt variants: each defines only the system message. Few-shot examples
# and the target sentence are appended identically as chat turns (see
# build_messages), so prompt variant and number of shots are independent.
# ---------------------------------------------------------------------------

PROMPTS = {
    "concise": (
        "You classify sentences from corporate reports by the type of "
        "emission target they contain.\n\n"
        "Classes:\n"
        "- net-zero: the sentence states a target to reach net zero emissions "
        "(or carbon neutrality).\n"
        "- reduction: the sentence states a target to reduce emissions by a "
        "certain amount, but not to net zero.\n"
        "- none: the sentence contains no emission target.\n\n"
        "Answer with exactly one word: net-zero, reduction, or none."
    ),
    "detailed": (
        "You are an expert annotator for corporate climate disclosures. "
        "Your task is to read a single sentence taken from a company report "
        "and decide which type of emission target, if any, it expresses.\n\n"
        "Definitions:\n"
        "- net-zero: the sentence commits to reaching net-zero greenhouse gas "
        "emissions, carbon neutrality, or removing as much carbon as the "
        "company emits, at some point in time.\n"
        "- reduction: the sentence commits to cutting emissions by some "
        "amount or percentage (e.g. \"reduce emissions by 30% by 2030\"), "
        "without claiming net-zero or full carbon neutrality.\n"
        "- none: the sentence does not commit to any emission target — this "
        "includes general sustainability talk, past achievements, or "
        "unrelated content.\n\n"
        "Think about which definition fits best, then answer with exactly "
        "one word and nothing else: net-zero, reduction, or none."
    ),
    "structured": (
        "Task: emission-target classification.\n"
        "Input: one sentence from a corporate report.\n"
        "Output: exactly one label, no punctuation, no explanation.\n\n"
        "Label set:\n"
        "  net-zero  -> target is net-zero / carbon neutrality\n"
        "  reduction -> target is a partial emission reduction\n"
        "  none      -> no emission target stated\n\n"
        "Respond with only the label."
    ),
}

# ---------------------------------------------------------------------------


def load_split(split: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split}.csv"
    if not path.exists():
        sys.exit(f"{path} missing. Run 'python src/prepare_data.py' first.")
    return pd.read_csv(path)


def select_few_shot(train_df: pd.DataFrame, shots: int, seed: int):
    """Select `shots` class-balanced examples from the training set.
    Deterministic per (shots, seed), so all models see the same examples."""
    if shots == 0:
        return []

    rng = random.Random(seed)
    per_class = {c: train_df[train_df["label"] == c].sample(
        frac=1.0, random_state=seed).to_dict("records") for c in CLASSES}

    examples = []
    class_cycle = list(CLASSES)
    rng.shuffle(class_cycle)
    idx_per_class = {c: 0 for c in CLASSES}
    i = 0
    while len(examples) < shots:
        c = class_cycle[i % len(class_cycle)]
        pool = per_class[c]
        if idx_per_class[c] < len(pool):
            examples.append(pool[idx_per_class[c]])
            idx_per_class[c] += 1
        i += 1
        if i > 10_000:  # safeguard in case a class is too small
            break
    rng.shuffle(examples)
    return examples[:shots]


def build_messages(system_prompt: str, few_shot, sentence: str):
    messages = [{"role": "system", "content": system_prompt}]
    for ex in few_shot:
        messages.append({"role": "user", "content": f"Sentence: {ex['text']}\nAnswer:"})
        messages.append({"role": "assistant", "content": ex["label"]})
    messages.append({"role": "user", "content": f"Sentence: {sentence}\nAnswer:"})
    return messages


def extract_label(raw: str) -> str:
    if raw is None:
        return "invalid"
    text = raw.strip().strip(".").strip('"').strip("'").lower()
    for c in CLASSES:
        if text == c:
            return c
    for c in CLASSES:
        if c in text:
            return c
    return "invalid"


def cache_key(backend: str, model: str, prompt_id: str, shots: int, seed: int, sentence_id) -> str:
    raw = json.dumps(
        {"backend": backend, "model": model, "prompt": prompt_id,
         "shots": shots, "seed": seed, "id": sentence_id},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: Path, cache: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def make_client(backend: str, api_key: str, base_url: str):
    from openai import OpenAI

    if backend == "openai":
        base_url = base_url or None  # None -> default api.openai.com
        api_key = api_key or None    # None -> SDK reads OPENAI_API_KEY
    elif backend == "ollama":
        base_url = base_url or "http://localhost:11434/v1"
        api_key = api_key or "ollama"  # Ollama ignores the key but requires a string
    else:
        raise ValueError(backend)

    return OpenAI(api_key=api_key, base_url=base_url)


# Parameters rejected per model (e.g. temperature on reasoning models),
# learned at runtime and omitted in all subsequent calls
_UNSUPPORTED_PARAMS: dict[str, set[str]] = {}


def _bad_param_from_error(exc: Exception) -> str | None:
    """Extract the name of the rejected parameter from an OpenAI error.
    Its location differs across SDK versions, so several sources are checked."""
    # 1) direct attribute
    param = getattr(exc, "param", None)
    if param:
        return param

    # 2) .body as dict or JSON string
    body = getattr(exc, "body", None)
    if isinstance(body, str):
        import json as _json
        try:
            body = _json.loads(body)
        except ValueError:
            body = None
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("param"):
            return err["param"]

    # 3) parse the error message, e.g. "'param': 'reasoning_effort'"
    match = re.search(r"'param':\s*'([^']+)'", str(exc))
    if match:
        return match.group(1)
    return None


def call_model(client, backend: str, model: str, messages, temperature: float,
                max_tokens: int, reasoning_effort: str | None = None) -> str:
    # Newer OpenAI models require 'max_completion_tokens', Ollama expects 'max_tokens'
    token_kwarg = "max_completion_tokens" if backend == "openai" else "max_tokens"
    skip = _UNSUPPORTED_PARAMS.get(model, set())

    kwargs = {token_kwarg: max_tokens}
    if "temperature" not in skip:
        kwargs["temperature"] = temperature
    if backend == "openai" and reasoning_effort and "reasoning_effort" not in skip:
        kwargs["reasoning_effort"] = reasoning_effort

    raw_output = None
    for _ in range(len(kwargs) + 1):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
            raw_output = resp.choices[0].message.content
            break
        except Exception as e:
            bad_param = _bad_param_from_error(e)
            if bad_param and bad_param in kwargs:
                print(f"Note: model '{model}' does not support '{bad_param}', omitting it from now on.")
                _UNSUPPORTED_PARAMS.setdefault(model, set()).add(bad_param)
                kwargs.pop(bad_param)
                continue
            raise

    if not raw_output:
        # Reasoning models may spend the whole token budget internally
        print(
            f"Warning: empty response from '{model}' (max_tokens={max_tokens} may be too small "
            "for a reasoning model; increase --max-tokens or set --reasoning-effort minimal)."
        )
    return raw_output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["openai", "ollama"], required=True)
    parser.add_argument("--model", required=True, help="e.g. gpt-6-luna or llama3.1:8b")
    parser.add_argument("--shots", type=int, choices=[0, 1, 3, 5, 10], default=0)
    parser.add_argument("--prompt", choices=list(PROMPTS.keys()), default="concise")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                         help="val for prompt development, test for final evaluation only")
    parser.add_argument("--limit", type=int, default=None, help="only the first N sentences (for quick tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=300,
                         help="response token budget (includes hidden reasoning tokens for reasoning models)")
    parser.add_argument("--sleep", type=float, default=0.0, help="pause in seconds between calls")
    parser.add_argument("--reasoning-effort", default="low",
                         choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                         help="OpenAI only: limits hidden reasoning (cost, latency); "
                         "omitted automatically if unsupported")
    parser.add_argument("--api-key", default=None, help="overrides OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None, help="overrides the default endpoint")
    parser.add_argument("--dry-run", action="store_true",
                         help="build prompts and cache keys without calling a model (no key needed)")
    parser.add_argument("--out-dir", default=str(PREDS_DIR))
    args = parser.parse_args()

    train_df = load_split("train")
    eval_df = load_split(args.split)
    if args.limit:
        eval_df = eval_df.head(args.limit)

    few_shot = select_few_shot(train_df, args.shots, args.seed)
    system_prompt = PROMPTS[args.prompt]

    print(f"Backend={args.backend} Model={args.model} Shots={args.shots} "
          f"Prompt={args.prompt} Split={args.split} n={len(eval_df)} "
          f"Dry-Run={args.dry_run}")
    if few_shot:
        print("Few-shot examples:")
        for ex in few_shot:
            print(f"  [{ex['label']}] {ex['text'][:80]}...")

    client = None
    if not args.dry_run:
        if args.backend == "openai" and not (args.api_key or __import__("os").environ.get("OPENAI_API_KEY")):
            sys.exit(
                "No OpenAI API key found. Set OPENAI_API_KEY or pass --api-key "
                "(use --dry-run to test without a key)."
            )
        client = make_client(args.backend, args.api_key, args.base_url)

    cache = load_cache(CACHE_PATH)
    rows = []
    n_cache_hits = 0
    n_calls = 0

    progress = tqdm(eval_df.iterrows(), total=len(eval_df), unit="sentence", disable=args.dry_run)
    for _, row in progress:
        messages = build_messages(system_prompt, few_shot, row["text"])
        key = cache_key(args.backend, args.model, args.prompt, args.shots, args.seed, row["id"])

        if args.dry_run:
            raw_output = None
        elif key in cache:
            raw_output = cache[key]
            n_cache_hits += 1
        else:
            raw_output = call_model(client, args.backend, args.model, messages, args.temperature,
                                     args.max_tokens, args.reasoning_effort)
            cache[key] = raw_output
            n_calls += 1
            if n_calls % 10 == 0:
                # Checkpoint cache so an interruption does not lose paid API calls
                save_cache(CACHE_PATH, cache)
            if args.sleep:
                time.sleep(args.sleep)

        pred = extract_label(raw_output) if raw_output is not None else None
        rows.append({
            "id": row["id"],
            "gold": row["label"],
            "pred": pred,
            "model": args.model,
            "seed": args.seed,
            "shots": args.shots,
            "prompt": args.prompt,
            "raw_output": raw_output,
            "backend": args.backend,      # extra column, not part of the common format
            "run_date": date.today().isoformat(),  # API outputs are not reproducible -> record date
        })
        n_invalid_so_far = sum(1 for r in rows if r["pred"] == "invalid")
        progress.set_postfix(new=n_calls, cache=n_cache_hits, invalid=n_invalid_so_far)

        if args.dry_run and len(rows) >= (args.limit or 5):
            print("\n--- Example messages (first sentence) ---")
            for m in messages:
                print(f"[{m['role']}] {m['content'][:200]}")
            break

    if not args.dry_run:
        save_cache(CACHE_PATH, cache)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_safe = args.model.replace(":", "-").replace("/", "-")
        out_path = out_dir / f"llm_{args.backend}_{model_safe}_{args.prompt}_{args.shots}shot.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)

        n_invalid = sum(1 for r in rows if r["pred"] == "invalid")
        print(f"\n{len(rows)} predictions saved to {out_path}")
        print(f"API calls: {n_calls} new, {n_cache_hits} from cache, {n_invalid} invalid responses")
    else:
        print("\nDry run finished: nothing saved, no model called.")


if __name__ == "__main__":
    main()
