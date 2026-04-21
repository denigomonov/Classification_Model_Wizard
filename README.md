#  Classification Wizard (sklearn)

Terminal-based engine that scans your CSV, asks a few questions, and benchmarks all major sklearn classifiers to recommend the top 3 for your specific problem — with reasoning and ready-to-use code.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      
pip install -r requirements.txt
```

Python 3.9+

## Usage

```bash
python main.py path/to/your_data.csv
```

---

## How It Works

```
1. Scanner      Profiles columns, detects all valid target candidates (binary vs. multiclass),
                flags missing data and class imbalance before any question is asked.

2. Wizard       You pick the target column — shown with problem type + class list so the
                choice is informed. Then 8 questions about goals, data size, noise, etc.

3. Round 1      All 16 models run in parallel (joblib) on a 30% stratified subsample.
                Bottom 50% eliminated immediately.

4. Round 2      Survivors tuned with Optuna (TPE sampler + median pruner, 20 trials,
                5-fold CV on full data). Bad hyperparameter regions pruned mid-trial.

5. Output       Top 3 ranked panels: why each fits your answers, trade-off warnings,
                best params found, and a copy-paste code snippet for your top pick.
```

---

## Files

```
main.py          Orchestrator
scanner.py       CSV profiler — detects all target candidates with per-column stats
questionnaire.py Guided Q&A — informed target selection + 8 context questions
benchmark.py     Tournament engine — joblib parallelism + Optuna tuning
recommender.py   Scoring, reasoning, rich terminal output
```

---

## Tuning

| What | Where | Default |
|---|---|---|
| Optuna trials per model | `benchmark.py → run_round2()` | `n_trials=20` |
| Round 1 subsample size | `benchmark.py → run_round1()` | `test_size=0.3` |
| Survivor count | `benchmark.py → run_round1()` | `max(4, n // 2)` |

To add a classifier: register it in `CANDIDATE_MODELS`, `FAST_DEFAULTS`, `_get_search_space()` in `benchmark.py` and `MODEL_TRAITS` in `recommender.py`.

---

## Limitations

- Tabular classification only (no regression, NLP, images)
- Scores reflect your sample — validate on full data before committing
- Optuna is stochastic — re-runs may surface slightly different params

## What to Expect

As of 2026-04-21
### Part 1
<img width="730" height="485" alt="first_part_output" src="https://github.com/user-attachments/assets/d124ed90-9a2a-41cd-9321-b569c99a15cf" />

### Part 2
<img width="730" height="461" alt="second_part_output" src="https://github.com/user-attachments/assets/c4221741-a4c5-4498-9513-03181efb1c01" />

### Part 3
<img width="730" height="348" alt="thrid_part_output" src="https://github.com/user-attachments/assets/0e2c08b2-596c-4840-80a3-a69eea8da1d8" />

### Final Part
Final part is too large to provide here but it is a output of **TOP 1-3** suggestion with the best parameters suggested for the implementation




