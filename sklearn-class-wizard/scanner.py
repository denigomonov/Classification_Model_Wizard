"""
scanner.py — Auto-profiles any CSV for classification readiness.
Detects target column, class balance, feature types, missing data, scale, and size.
"""

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def scan_csv(filepath: str) -> dict:
    """Load and profile a CSV. Returns a structured profile dict."""
    console.rule("[bold cyan]📂 Dataset Scanner[/bold cyan]")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        console.print(f"[red]❌ Failed to load CSV: {e}[/red]")
        raise

    n_rows, n_cols = df.shape
    console.print(f"[green]✓ Loaded:[/green] {n_rows} rows × {n_cols} columns\n")

    # --- Candidate target columns: integer or object dtype, low cardinality ---
    target_candidates = []
    for col in df.columns:
        nunique = df[col].nunique()
        if df[col].dtype in [object, "category"] or (
            pd.api.types.is_integer_dtype(df[col]) and 2 <= nunique <= 20
        ):
            problem_type = "binary" if nunique == 2 else f"{nunique}-class"
            target_candidates.append((col, nunique, problem_type))

    # --- Feature summary table ---
    table = Table(title="Column Summary", box=box.SIMPLE_HEAVY, header_style="bold magenta")
    table.add_column("Column", style="cyan", no_wrap=True)
    table.add_column("Dtype", style="yellow")
    table.add_column("Unique", justify="right")
    table.add_column("Missing", justify="right")
    table.add_column("Role Hint", style="green")

    candidate_cols = {tc[0] for tc in target_candidates}
    for col in df.columns:
        nunique = df[col].nunique()
        missing = df[col].isna().sum()
        missing_str = f"[red]{missing}[/red]" if missing > 0 else "0"
        if col in candidate_cols:
            ptype = next(tc[2] for tc in target_candidates if tc[0] == col)
            role = f"[bold yellow]🎯 target? ({ptype})[/bold yellow]"
        elif pd.api.types.is_numeric_dtype(df[col]):
            role = "feature"
        else:
            role = "categorical?"
        table.add_row(col, str(df[col].dtype), str(nunique), missing_str, role)

    console.print(table)

    # --- Profile ALL candidate targets so the wizard can show meaningful context ---
    target_profiles = {}
    for col, nunique, ptype in target_candidates:
        counts = df[col].value_counts()
        total = len(df)
        balance = {str(k): round(v / total, 3) for k, v in counts.items()}
        min_ratio = min(balance.values())
        target_profiles[col] = {
            "n_classes": nunique,
            "problem_type": ptype,
            "class_balance": balance,
            "is_imbalanced": min_ratio < 0.15,
            "classes": list(counts.index.astype(str)),
        }

    if target_profiles:
        console.print("\n[bold]Detected target column candidates:[/bold]")
        for col, tp in target_profiles.items():
            imb_flag = " [yellow]⚠ imbalanced[/yellow]" if tp["is_imbalanced"] else ""
            console.print(
                f"  [cyan]{col}[/cyan] → {tp['problem_type']}{imb_flag}  "
                f"classes: {tp['classes']}"
            )
        console.print()

    profile = {
        "filepath": filepath,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_any": int(df.isna().sum().sum()),
        "target_candidates": [tc[0] for tc in target_candidates],
        "target_profiles": target_profiles,          # per-column profiles
        "n_classes": None,
        "class_balance": None,
        "is_imbalanced": False,
        "n_features": None,
        "has_categorical": any(df[col].dtype == object for col in df.columns),
        "scale_variance": None,
        "df": df,
    }

    return profile



