# ClimateBERT-NetZero vs. LLM Few-Shot

Comparison of fine-tuned ClimateBERT and few-shot prompting
(llama3.1:8b via Ollama, gpt-6-luna via the OpenAI API) for classifying
emission targets (net-zero / reduction / none) on
[`climatebert/netzero_reduction_data`](https://huggingface.co/datasets/climatebert/netzero_reduction_data).

## Structure

```
environment-claims/
├── src/
│   ├── prepare_data.py          # load dataset, 70/15/15 split, baseline
│   ├── train_climatebert.py     # fine-tuning, one run per seed
│   ├── run_all_climatebert.py   # all seeds (42, 1, 2)
│   ├── run_llm.py               # LLM classification (Ollama / OpenAI)
│   ├── run_all_llm.py           # full grid: backend x prompt x shots
│   ├── evaluate.py              # metrics, significance tests, tables, figures
│   └── error_analysis.py        # misclassifications of the three key runs
├── data/                        # train.csv, val.csv, test.csv
├── preds/                       # predictions, one CSV per run (34 runs)
├── results/
│   ├── tables/                  # .csv and .tex
│   ├── figures/                 # .pdf
│   └── error_analysis/          # misclassified.csv
├── llm_cache.json               # cached LLM responses
└── requirements.txt
```

`data/`, `preds/` and `llm_cache.json` are committed, so the evaluation can be
reproduced without a GPU or API key.

The folders `literature/` (source PDFs) and `paper/` (LaTeX sources) were only
used for writing the paper and are excluded via `.gitignore`.

## Reproduction

```bash
pip install -r requirements.txt
python src/evaluate.py         # generates results/ from preds/
```

For a full rerun: `prepare_data.py` → `run_all_climatebert.py` →
`run_all_llm.py --backend both` → `evaluate.py`
(OpenAI requires `OPENAI_API_KEY`, Ollama a locally pulled `llama3.1:8b`).
