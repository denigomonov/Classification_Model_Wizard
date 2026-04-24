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
1. Scanner      Profiles columns. Detects valid target candidates (2–50 unique values,
                non-float). Shows problem type (binary / N-class), imbalance flag, and
                a class preview. High-cardinality text columns flagged and excluded.
 
2. Wizard       If multiple target candidates exist, each is shown with its problem type
                and class preview so the choice is informed — not a blind confirmation.
                Then 8 questions about your goals, data size, noise tolerance, etc.
 
3. Round 1      All 16 models run in parallel (joblib/loky) on a stratified subsample
                capped at 10,000 rows. Worker pool explicitly shut down after to free
                memory. Bottom 50% of models eliminated immediately.
 
4. Round 2      Survivors tuned with Optuna (TPE sampler + median pruner, 20 trials,
                5-fold CV) on a subsample capped at 20,000 rows. Each model gets its
                own isolated Optuna study — no parameter namespace collisions.
 
5. Output       Top 3 ranked panels: why each fits your answers, trade-off warnings,
                best hyperparameters found, and a copy-paste code snippet for the top pick.
```
 
---
 
## Target Column Detection
 
The scanner qualifies a column as a classification target if it meets all of:
- Between **2 and 50 unique values** (configurable via `MAX_TARGET_CARDINALITY` in `scanner.py`)
- **Not a pure float** column (continuous variables are not classification targets)
- Any dtype: int, bool, object, or category
High-cardinality text columns (e.g. artist names, track titles) are excluded from candidates and dropped from the feature set before benchmarking.
 
---
 
## Large Dataset Behaviour
 
| Stage | Row cap | Why |
|---|---|---|
| Round 1 subsample | 10,000 | Keeps parallel elimination fast across 16 models |
| Round 2 tuning | 20,000 | 5-fold CV × 20 Optuna trials per model stays under ~5 min |
 
Both caps use stratified sampling to preserve class distribution. Scores on large datasets reflect the sample — validate your final model on the full data before committing.
 
---

## GPU & LightGBM

- `⚡ Apple MPS detected — LightGBM running on Metal GPU` — source build with `-DUSE_GPU=1`
- `⚡ Apple MPS detected — LightGBM falling back to CPU (pip wheel has no GPU support)` — standard pip install
LightGBM on CPU is still faster than sklearn's `GradientBoosting` due to histogram-based splitting and full `n_jobs=-1` parallelism across all cores.

 
## Memory Management
 
After `main.py` exits, run these in the terminal to fully reclaim memory:
 
```bash
pkill -f "loky"                                        # kill any lingering worker processes
python -c "import gc; gc.collect(); print('gc done')"  # flush Python memory cache
sudo purge                                             # flush macOS unified memory page cache
```
 
The loky worker pool (spawned during Round 1 parallelism) is shut down explicitly inside `benchmark.py` after Round 1 completes. The raw DataFrame is also freed immediately after `X` and `y` are extracted, before Round 2 begins.
 
---
 
## Files
 
```
main.py          Orchestrator
scanner.py       CSV profiler — target detection, cardinality filtering, class previews
questionnaire.py Guided Q&A — informed target selection + 8 context questions
benchmark.py     Tournament engine — adaptive subsampling, joblib/loky, Optuna per-model studies
recommender.py   Scoring, context-aware reasoning, rich terminal output + quick-start code
```
 
---
 
## Flow
 
 
| What | Where | Default |
|---|---|---|
| Max target cardinality | `scanner.py → MAX_TARGET_CARDINALITY` | `50` |
| Round 1 row cap | `benchmark.py → run_round1()` | `10,000` |
| Round 2 row cap | `benchmark.py → run_round2()` | `20,000` |
| Optuna trials per model | `benchmark.py → run_round2()` | `20` |
| Optuna parallel trials | `benchmark.py → _optuna_optimize()` | `n_jobs=5` |
| Survivor count into Round 2 | `benchmark.py → run_round1()` | `max(4, n // 2)` |
 
To add a classifier: register it in `CANDIDATE_MODELS`, `FAST_DEFAULTS`, and `_get_search_space()` in `benchmark.py`, and `MODEL_TRAITS` in `recommender.py`.
 
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
Final part is too large to provide here but it is an output of **TOP 1-3** suggestion with the best parameters suggested for the implementation:
<img width="730" height="450" alt="final_part_output" src="https://github.com/user-attachments/assets/06b6d9f9-b042-498c-9732-5d668757fd1b" />






