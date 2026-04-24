"""
benchmark.py — Tournament-style fast classification benchmark engine.

Speed strategy:
  Round 1 (Elimination): All models scored on a stratified 30% subsample using 3-fold CV.
                          Bottom 50% eliminated immediately.
  Round 2 (Finals):      Survivors get full stratified 5-fold CV with Optuna hyperparameter
                         search (20 trials, pruning enabled). Parallelized via joblib.

This avoids wasting compute on poor candidates and focuses tuning effort on contenders.
"""

import gc
import time
import platform
import warnings
import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed, parallel_backend
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# --- Classifiers ---
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    _LGBM_GPU_OK = False

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

console = Console()


# ── Device detection ──────────────────────────────────────────────────────────

def _detect_device() -> str:
    """Detect best available compute device. Returns 'mps', 'cuda', or 'cpu'."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            # torch installed but mps check failed, still Apple Silicon
            return "mps"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

DEVICE = _detect_device()

# Probe LightGBM GPU support now that DEVICE is known.
# Must run after DEVICE assignment — pip wheels are CPU-only builds.
_LGBM_GPU_OK = False
if HAS_LGBM and DEVICE in ("mps", "cuda"):
    import os, sys
    try:
        # Suppress LightGBM's Fatal stderr messages during the probe —
        # it writes directly to stderr before raising, so redirect fd-level.
        _devnull = open(os.devnull, "w")
        _old_stderr_fd = os.dup(2)
        os.dup2(_devnull.fileno(), 2)
        try:
            _probe = lgb.LGBMClassifier(n_estimators=1, device="gpu", gpu_use_dp=False, verbose=-1)
            _probe.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
            _LGBM_GPU_OK = True
        finally:
            os.dup2(_old_stderr_fd, 2)
            os.close(_old_stderr_fd)
            _devnull.close()
    except Exception:
        _LGBM_GPU_OK = False


def print_device_banner():
    """Print MPS/CUDA/CPU detection result before Round 1."""
    if DEVICE == "mps":
        if _LGBM_GPU_OK:
            console.print("[bold green]⚡ Apple MPS detected — LightGBM running on Metal GPU[/bold green]")
        else:
            console.print("[bold yellow]⚡ Apple MPS detected — LightGBM falling back to CPU (pip wheel has no GPU support)[/bold yellow]")
    elif DEVICE == "cuda":
        if _LGBM_GPU_OK:
            console.print("[bold green]⚡ CUDA detected — LightGBM running on GPU[/bold green]")
        else:
            console.print("[bold yellow]⚡ CUDA detected — LightGBM falling back to CPU (pip wheel has no GPU support)[/bold yellow]")
    else:
        console.print("[dim]🖥  No GPU acceleration detected — running on CPU[/dim]")


# ── Candidate model registry ──────────────────────────────────────────────────

def _build_candidate_models() -> dict:
    """Build CANDIDATE_MODELS dict, injecting LightGBM with correct device if available."""
    models = {
        "RandomForest": RandomForestClassifier,
        "ExtraTrees": ExtraTreesClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "HistGradientBoosting": HistGradientBoostingClassifier,
        "AdaBoost": AdaBoostClassifier,
        "LogisticRegression": LogisticRegression,
        "RidgeClassifier": RidgeClassifier,
        "SGD": SGDClassifier,
        "DecisionTree": DecisionTreeClassifier,
        "KNN": KNeighborsClassifier,
        "SVC": SVC,
        "LinearSVC": LinearSVC,
        "GaussianNB": GaussianNB,
        "LDA": LinearDiscriminantAnalysis,
        "QDA": QuadraticDiscriminantAnalysis,
        "MLP": MLPClassifier,
    }
    if HAS_LGBM:
        models["LightGBM"] = lgb.LGBMClassifier
    return models

CANDIDATE_MODELS = _build_candidate_models()

# Fast default params for Round 1 (speed > accuracy)
def _build_fast_defaults() -> dict:
    defaults = {
        "RandomForest": {"n_estimators": 50, "max_depth": 8, "n_jobs": -1, "random_state": 42},
        "ExtraTrees": {"n_estimators": 50, "max_depth": 8, "n_jobs": -1, "random_state": 42},
        "GradientBoosting": {"n_estimators": 50, "max_depth": 4, "random_state": 42},
        "HistGradientBoosting": {"max_iter": 50, "random_state": 42},
        "AdaBoost": {"n_estimators": 50, "random_state": 42},
        "LogisticRegression": {"max_iter": 300, "random_state": 42},
        "RidgeClassifier": {},
        "SGD": {"max_iter": 100, "random_state": 42},
        "DecisionTree": {"max_depth": 8, "random_state": 42},
        "KNN": {"n_neighbors": 7},
        "SVC": {"kernel": "rbf", "C": 1.0, "probability": False, "random_state": 42},
        "LinearSVC": {"max_iter": 500, "random_state": 42},
        "GaussianNB": {},
        "LDA": {},
        "QDA": {"reg_param": 0.1},
        "MLP": {"hidden_layer_sizes": (64,), "max_iter": 100, "random_state": 42},
    }
    if HAS_LGBM:
        lgbm_defaults = {
            "n_estimators": 100,
            "num_leaves": 31,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }
        if _LGBM_GPU_OK:
            lgbm_defaults["device"] = "gpu"
            lgbm_defaults["gpu_use_dp"] = False
        defaults["LightGBM"] = lgbm_defaults
    return defaults

FAST_DEFAULTS = _build_fast_defaults()

# Optuna search spaces for Round 2.
# Uses if/elif so only the target model's trial.suggest_* calls execute —
# a dict literal would evaluate ALL branches eagerly, registering every
# model's params on the trial object and causing collision errors.
def _get_search_space(trial, name):
    if name == "RandomForest":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 400),
            "max_depth":         trial.suggest_int("max_depth", 4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "n_jobs": -1, "random_state": 42,
        }
    elif name == "ExtraTrees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth":    trial.suggest_int("max_depth", 4, 20),
            "n_jobs": -1, "random_state": 42,
        }
    elif name == "GradientBoosting":
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 200),
            "max_depth":     trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "random_state": 42,
        }
    elif name == "HistGradientBoosting":
        return {
            "max_iter":      trial.suggest_int("max_iter", 50, 300),
            "max_depth":     trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "random_state": 42,
        }
    elif name == "AdaBoost":
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
            "random_state": 42,
        }
    elif name == "LogisticRegression":
        return {
            "C":      trial.suggest_float("C", 0.001, 100.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": 500, "random_state": 42,
        }
    elif name == "RidgeClassifier":
        return {
            "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
        }
    elif name == "SGD":
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
            "loss":  trial.suggest_categorical("loss", ["hinge", "modified_huber", "log_loss"]),
            "max_iter": 200, "random_state": 42,
        }
    elif name == "DecisionTree":
        return {
            "max_depth":         trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "random_state": 42,
        }
    elif name == "KNN":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
            "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    elif name == "SVC":
        return {
            "C":      trial.suggest_float("C", 0.01, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
            "gamma":  trial.suggest_categorical("gamma", ["scale", "auto"]),
            "random_state": 42,
        }
    elif name == "LinearSVC":
        return {
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
            "max_iter": 1000, "random_state": 42,
        }
    elif name == "GaussianNB":
        return {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-10, 1e-6, log=True),
        }
    elif name == "LDA":
        return {
            "solver": trial.suggest_categorical("solver", ["svd", "lsqr"]),
        }
    elif name == "QDA":
        return {
            "reg_param": trial.suggest_float("reg_param", 0.0, 1.0),
        }
    elif name == "MLP":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", [(64,), (128,), (64, 32), (128, 64)]
            ),
            "alpha":    trial.suggest_float("alpha", 1e-5, 0.01, log=True),
            "max_iter": 300, "random_state": 42,
        }
    elif name == "LightGBM":
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }
        if _LGBM_GPU_OK:
            params["device"] = "gpu"
            params["gpu_use_dp"] = False
        return params
    return {}


# ── Preprocessing helper ───────────────────────────────────────────────────────

def build_preprocessor(X: pd.DataFrame):
    """Build a ColumnTransformer that handles numeric + categorical features."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))

    return ColumnTransformer(transformers, remainder="drop")


# ── Round 1: Elimination ───────────────────────────────────────────────────────

def _round1_score(name, clf_class, params, X_sub, y_sub, cv):
    """Score a single model on the subsample. Returns (name, score, time)."""
    try:
        clf = Pipeline([
            ("pre", build_preprocessor(X_sub)),
            ("clf", clf_class(**params)),
        ])
        t0 = time.perf_counter()
        scores = cross_val_score(clf, X_sub, y_sub, cv=cv, scoring="f1_weighted", n_jobs=1)
        elapsed = time.perf_counter() - t0
        return name, float(np.mean(scores)), elapsed
    except Exception as e:
        return name, -1.0, 0.0


def run_round1(X: pd.DataFrame, y: np.ndarray, n_classes: int) -> list:
    console.rule("[bold yellow]⚡ Round 1: Elimination (subsample)[/bold yellow]")

    from sklearn.model_selection import train_test_split

    # Adaptive subsample cap: never exceed 10k rows in Round 1.
    # On 114k rows, 30% = 34k which is still too slow for 16 parallel models.
    n_total = len(X)
    MAX_R1_ROWS = 10_000
    if n_total > MAX_R1_ROWS:
        sub_frac = MAX_R1_ROWS / n_total
        _, X_sub, _, y_sub = train_test_split(X, y, test_size=sub_frac, stratify=y, random_state=42)
    else:
        _, X_sub, _, y_sub = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    console.print(f"  Subsample size: {len(X_sub):,} / {n_total:,} rows | Running {len(CANDIDATE_MODELS)} models in parallel...\n")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = []
    with parallel_backend("loky", n_jobs=-1):
        raw = Parallel()(
            delayed(_round1_score)(name, cls, FAST_DEFAULTS.get(name, {}), X_sub, y_sub, cv)
            for name, cls in CANDIDATE_MODELS.items()
        )
    results = raw

    # Explicitly shut down loky worker pool and free subsample memory
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)
    del X_sub, y_sub
    gc.collect()

    results = [(n, s, t) for n, s, t in results if s >= 0]
    results.sort(key=lambda x: -x[1])

    # Eliminate bottom 50%
    cutoff = max(4, len(results) // 2)
    survivors = results[:cutoff]

    from rich.table import Table
    from rich import box
    t = Table(box=box.SIMPLE, header_style="bold")
    t.add_column("Model", style="cyan")
    t.add_column("F1 (subsample)", justify="right")
    t.add_column("Time (s)", justify="right")
    t.add_column("Status", justify="center")

    for i, (name, score, elapsed) in enumerate(results):
        status = "[green]✓ Survivor[/green]" if i < cutoff else "[red]✗ Eliminated[/red]"
        t.add_row(name, f"{score:.4f}", f"{elapsed:.2f}s", status)

    console.print(t)
    console.print(f"\n[green]✓ {len(survivors)} models advance to Round 2.[/green]\n")
    return [name for name, _, _ in survivors]


# ── Round 2: Finals with Optuna ────────────────────────────────────────────────

def _optuna_optimize(name, clf_class, X, y, n_trials=20):
    """Run Optuna hyperparameter search for a single model. Returns best score + params.
    Each model gets its own isolated in-memory study to prevent parameter name collisions.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = _get_search_space(trial, name)
        try:
            clf = Pipeline([
                ("pre", build_preprocessor(X)),
                ("clf", clf_class(**params)),
            ])
            scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted", n_jobs=1)
            return float(np.mean(scores))
        except Exception:
            return 0.0

    study = optuna.create_study(
        study_name=f"study_{name}",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=5, show_progress_bar=False)
    return name, study.best_value, study.best_params


def run_round2(survivor_names: list, X: pd.DataFrame, y: np.ndarray, n_trials: int = 20) -> list:
    console.rule("[bold yellow]🏆 Round 2: Finals (full data + Optuna tuning)[/bold yellow]")

    # Adaptive cap for Round 2: 5-fold CV on 114k rows per trial is prohibitive.
    # Cap at 20k rows — enough for reliable ranking without melting RAM.
    MAX_R2_ROWS = 20_000
    n_total = len(X)
    if n_total > MAX_R2_ROWS:
        from sklearn.model_selection import train_test_split
        frac = MAX_R2_ROWS / n_total
        _, X_r2, _, y_r2 = train_test_split(X, y, test_size=frac, stratify=y, random_state=42)
        console.print(
            f"  [dim]Large dataset detected ({n_total:,} rows) — "
            f"Round 2 uses a {len(X_r2):,}-row stratified sample for speed.[/dim]"
        )
    else:
        X_r2, y_r2 = X, y

    console.print(f"  Tuning {len(survivor_names)} models with {n_trials} Optuna trials each...\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Optimizing...", total=len(survivor_names))
        results = []
        for name in survivor_names:
            progress.update(task, description=f"Tuning [bold]{name}[/bold]")
            clf_class = CANDIDATE_MODELS[name]
            n, score, params = _optuna_optimize(name, clf_class, X_r2, y_r2, n_trials=n_trials)
            results.append((n, score, params))
            progress.advance(task)

    results.sort(key=lambda x: -x[1])

    from rich.table import Table
    from rich import box
    t = Table(box=box.SIMPLE, header_style="bold")
    t.add_column("Rank", justify="center")
    t.add_column("Model", style="cyan")
    t.add_column("Best F1 (5-fold)", justify="right")

    for i, (name, score, _) in enumerate(results):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"  {i+1}."
        t.add_row(medal, name, f"{score:.4f}")

    console.print(t)
    if n_total > MAX_R2_ROWS:
        console.print(f"[dim]  Note: scores based on {MAX_R2_ROWS:,}-row sample — validate final model on full data.[/dim]\n")
    return results


# ── Main entry ─────────────────────────────────────────────────────────────────

def run_benchmark(profile: dict, ctx: dict) -> list:
    """Full benchmark pipeline. Returns ranked list of (name, score, params)."""
    df = profile["df"]
    target_col = ctx["target_col"]

    # Drop columns that are pure text/ID — not useful as features
    text_cols = [
        c for c in df.columns
        if c != target_col and df[c].dtype == object and df[c].nunique() > 50
    ]
    if text_cols:
        console.print(f"[dim]Dropping high-cardinality text columns (not usable as features): {text_cols}[/dim]\n")
        df = df.drop(columns=text_cols)

    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    # Free the raw dataframe — no longer needed once X and y are built
    del df, y_raw
    profile["df"] = None
    gc.collect()

    console.print(f"[dim]Target: '{target_col}' | Classes: {n_classes} | Features: {X.shape[1]} | Rows: {len(X):,}[/dim]\n")

    print_device_banner()
    survivors = run_round1(X, y, n_classes)
    final_results = run_round2(survivors, X, y)

    # Final cleanup
    del X, y
    gc.collect()

    return final_results[:3]
