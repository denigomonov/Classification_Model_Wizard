"""
recommender.py — Context-aware scoring and rich terminal output of top 3 recommendations.

Takes benchmark results + wizard context and adds qualitative reasoning:
  - Why this model fits your stated goals
  - Trade-offs to be aware of
  - Best hyperparameters found
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# ── Model trait registry ────────────────────────────────────────────────────

MODEL_TRAITS = {
    "RandomForest": {
        "interpretable": False,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": True,
        "good_imbalanced": True,
        "good_nonlinear": True,
        "good_small_data": True,
        "good_large_data": True,
        "description": "Ensemble of decision trees. Robust, parallelizable, low variance.",
    },
    "ExtraTrees": {
        "interpretable": False,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": True,
        "good_imbalanced": True,
        "good_nonlinear": True,
        "good_small_data": True,
        "good_large_data": True,
        "description": "Extremely randomized trees — faster training than RF, similar accuracy.",
    },
    "GradientBoosting": {
        "interpretable": False,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": False,
        "good_imbalanced": True,
        "good_nonlinear": True,
        "good_small_data": True,
        "good_large_data": False,
        "description": "Sequential boosting. High accuracy but slower to train.",
    },
    "HistGradientBoosting": {
        "interpretable": False,
        "fast_inference": True,
        "handles_missing": True,
        "handles_categorical": True,
        "robust_noise": False,
        "good_imbalanced": True,
        "good_nonlinear": True,
        "good_small_data": False,
        "good_large_data": True,
        "description": "Histogram-based boosting. sklearn's fastest tree ensemble; handles missing natively.",
    },
    "AdaBoost": {
        "interpretable": False,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": False,
        "good_imbalanced": False,
        "good_nonlinear": True,
        "good_small_data": True,
        "good_large_data": False,
        "description": "Adaptive boosting. Sensitive to noise and outliers.",
    },
    "LogisticRegression": {
        "interpretable": True,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": True,
        "good_imbalanced": True,
        "good_nonlinear": False,
        "good_small_data": True,
        "good_large_data": True,
        "description": "Linear probabilistic classifier. Fast, explainable, well-calibrated.",
    },
    "RidgeClassifier": {
        "interpretable": True,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": True,
        "good_imbalanced": False,
        "good_nonlinear": False,
        "good_small_data": True,
        "good_large_data": True,
        "description": "Ridge-regularized linear classifier. Very fast, good baseline.",
    },
    "SGD": {
        "interpretable": True,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": False,
        "good_imbalanced": True,
        "good_nonlinear": False,
        "good_small_data": False,
        "good_large_data": True,
        "description": "Stochastic gradient descent. Scales well to massive datasets.",
    },
    "DecisionTree": {
        "interpretable": True,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": False,
        "good_imbalanced": False,
        "good_nonlinear": True,
        "good_small_data": True,
        "good_large_data": False,
        "description": "Single tree. Highly interpretable (can be visualized). Prone to overfitting.",
    },
    "KNN": {
        "interpretable": True,
        "fast_inference": False,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": False,
        "good_imbalanced": False,
        "good_nonlinear": True,
        "good_small_data": True,
        "good_large_data": False,
        "description": "Instance-based. No training phase, but slow at inference on large datasets.",
    },
    "SVC": {
        "interpretable": False,
        "fast_inference": False,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": True,
        "good_imbalanced": True,
        "good_nonlinear": True,
        "good_small_data": True,
        "good_large_data": False,
        "description": "Support Vector Machine. Strong for small/medium data, slow at scale.",
    },
    "LinearSVC": {
        "interpretable": True,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": True,
        "good_imbalanced": False,
        "good_nonlinear": False,
        "good_small_data": True,
        "good_large_data": True,
        "description": "Linear SVM via liblinear. Fast, scalable, good for high-dimensional data.",
    },
    "GaussianNB": {
        "interpretable": True,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": False,
        "good_imbalanced": False,
        "good_nonlinear": False,
        "good_small_data": True,
        "good_large_data": True,
        "description": "Naive Bayes. Blazing fast, works well as a baseline or when features are independent.",
    },
    "LDA": {
        "interpretable": True,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": False,
        "good_imbalanced": False,
        "good_nonlinear": False,
        "good_small_data": True,
        "good_large_data": True,
        "description": "Linear Discriminant Analysis. Good for linearly separable classes, also a dimensionality reducer.",
    },
    "QDA": {
        "interpretable": True,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": False,
        "good_imbalanced": False,
        "good_nonlinear": True,
        "good_small_data": True,
        "good_large_data": False,
        "description": "Quadratic Discriminant Analysis. Allows non-linear boundaries; sensitive to small samples.",
    },
    "MLP": {
        "interpretable": False,
        "fast_inference": True,
        "handles_missing": False,
        "handles_categorical": False,
        "robust_noise": False,
        "good_imbalanced": True,
        "good_nonlinear": True,
        "good_small_data": False,
        "good_large_data": True,
        "description": "Multi-layer perceptron. Captures complex patterns; needs more data and careful tuning.",
    },
}

MEDALS = ["🥇", "🥈", "🥉"]


def _build_reasons(name: str, ctx: dict, traits: dict) -> list[str]:
    reasons = []
    warnings_list = []

    goal = ctx.get("goal", "")
    if "accuracy" in goal and traits.get("good_nonlinear"):
        reasons.append("Strong at capturing non-linear patterns → high F1 potential")
    if "interpretability" in goal and traits.get("interpretable"):
        reasons.append("Highly interpretable — coefficients or tree rules can be explained to stakeholders")
    if "interpretability" in goal and not traits.get("interpretable"):
        warnings_list.append("⚠ Low interpretability — consider pairing with SHAP for explanations")
    if "fast inference" in goal and traits.get("fast_inference"):
        reasons.append("Fast at inference — suitable for production latency requirements")
    if "fast inference" in goal and not traits.get("fast_inference"):
        warnings_list.append("⚠ Slower inference — may need optimization for real-time use")

    if ctx.get("has_missing") and traits.get("handles_missing"):
        reasons.append("Natively handles missing values — no imputation pipeline needed")
    elif ctx.get("has_missing") and not traits.get("handles_missing"):
        warnings_list.append("⚠ Does not handle missing values — imputation applied automatically")

    if ctx.get("has_categorical") and traits.get("handles_categorical"):
        reasons.append("Handles categorical features natively")

    if ctx.get("is_imbalanced") and traits.get("good_imbalanced"):
        reasons.append("Handles class imbalance well (supports class_weight='balanced')")
    elif ctx.get("is_imbalanced") and not traits.get("good_imbalanced"):
        warnings_list.append("⚠ May struggle with imbalanced classes — consider class_weight tuning")

    if ctx.get("noisy_data") and traits.get("robust_noise"):
        reasons.append("Robust to label noise and outliers")
    elif ctx.get("noisy_data") and not traits.get("robust_noise"):
        warnings_list.append("⚠ Sensitive to noisy data — consider data cleaning steps")

    size = ctx.get("expected_size", "")
    if "Large" in size and traits.get("good_large_data"):
        reasons.append("Scales well to large datasets")
    elif "Large" in size and not traits.get("good_large_data"):
        warnings_list.append("⚠ May not scale well to large data — monitor training time")

    if not reasons:
        reasons.append("Competitive benchmark score on your specific data profile")

    return reasons, warnings_list


def render_recommendations(top3: list, ctx: dict):
    """Render the final top-3 recommendations panel in the terminal."""
    from rich.rule import Rule
    console.rule("[bold green]🎯 Top 3 Recommendations[/bold green]")
    console.print()

    for i, (name, score, best_params) in enumerate(top3):
        traits = MODEL_TRAITS.get(name, {})
        reasons, warnings_list = _build_reasons(name, ctx, traits)
        medal = MEDALS[i] if i < 3 else f"#{i+1}"

        # Panel content
        lines = []
        lines.append(f"[bold]{traits.get('description', '')}[/bold]\n")
        lines.append(f"[green]✓ Why it fits your use-case:[/green]")
        for r in reasons:
            lines.append(f"  • {r}")
        if warnings_list:
            lines.append(f"\n[yellow]Considerations:[/yellow]")
            for w in warnings_list:
                lines.append(f"  {w}")

        lines.append(f"\n[cyan]Best hyperparameters found:[/cyan]")
        for k, v in best_params.items():
            lines.append(f"  {k}: [bold]{v}[/bold]")

        lines.append(f"\n[dim]Benchmark F1 (5-fold CV): {score:.4f}[/dim]")

        content = "\n".join(lines)
        console.print(Panel(
            content,
            title=f"{medal}  [bold cyan]{name}[/bold cyan]",
            border_style="cyan" if i == 0 else "blue" if i == 1 else "magenta",
            padding=(1, 2),
        ))
        console.print()

    # Quick-start code snippet for #1
    best_name, best_score, best_params = top3[0]
    _render_quickstart(best_name, best_params, ctx)


def _render_quickstart(name: str, params: dict, ctx: dict):
    from sklearn.ensemble import RandomForestClassifier  # just for type hint in display
    console.rule("[bold dim]📋 Quick-Start Code (Top Pick)[/bold dim]")

    cls_import_map = {
        "RandomForest": "from sklearn.ensemble import RandomForestClassifier",
        "ExtraTrees": "from sklearn.ensemble import ExtraTreesClassifier",
        "GradientBoosting": "from sklearn.ensemble import GradientBoostingClassifier",
        "HistGradientBoosting": "from sklearn.ensemble import HistGradientBoostingClassifier",
        "AdaBoost": "from sklearn.ensemble import AdaBoostClassifier",
        "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
        "RidgeClassifier": "from sklearn.linear_model import RidgeClassifier",
        "SGD": "from sklearn.linear_model import SGDClassifier",
        "DecisionTree": "from sklearn.tree import DecisionTreeClassifier",
        "KNN": "from sklearn.neighbors import KNeighborsClassifier",
        "SVC": "from sklearn.svm import SVC",
        "LinearSVC": "from sklearn.svm import LinearSVC",
        "GaussianNB": "from sklearn.naive_bayes import GaussianNB",
        "LDA": "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis",
        "QDA": "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis",
        "MLP": "from sklearn.neural_network import MLPClassifier",
    }

    cls_name_map = {
        "RandomForest": "RandomForestClassifier",
        "ExtraTrees": "ExtraTreesClassifier",
        "GradientBoosting": "GradientBoostingClassifier",
        "HistGradientBoosting": "HistGradientBoostingClassifier",
        "AdaBoost": "AdaBoostClassifier",
        "LogisticRegression": "LogisticRegression",
        "RidgeClassifier": "RidgeClassifier",
        "SGD": "SGDClassifier",
        "DecisionTree": "DecisionTreeClassifier",
        "KNN": "KNeighborsClassifier",
        "SVC": "SVC",
        "LinearSVC": "LinearSVC",
        "GaussianNB": "GaussianNB",
        "LDA": "LinearDiscriminantAnalysis",
        "QDA": "QuadraticDiscriminantAnalysis",
        "MLP": "MLPClassifier",
    }

    target = ctx.get("target_col", "target")
    imp = cls_import_map.get(name, f"from sklearn import {name}")
    cls = cls_name_map.get(name, name)
    param_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())

    code = f"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
{imp}

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["{target}"])
y = df["{target}"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", {cls}({param_str})),
])

model.fit(X_train, y_train)
print("Test accuracy:", model.score(X_test, y_test))
"""
    from rich.syntax import Syntax
    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))
    console.print()
