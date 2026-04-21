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

# Max unique values for a column to be considered a classification target.
# Keeps high-cardinality text columns (artists, track names, etc.) out of candidates.
MAX_TARGET_CARDINALITY = 20


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

    # --- Candidate target columns ---
    # Must have between 2 and MAX_TARGET_CARDINALITY unique values.
    # Pure float columns are excluded (continuous → not a classifier target).
    # High-cardinality text columns (names, IDs) are excluded.
    target_candidates = []
    for col in df.columns:
        nunique = df[col].nunique()
        dtype = df[col].dtype

        is_float_only = pd.api.types.is_float_dtype(dtype)
        within_cardinality = 2 <= nunique <= MAX_TARGET_CARDINALITY

        if not is_float_only and within_cardinality:
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
        elif nunique > MAX_TARGET_CARDINALITY:
            role = "[dim]high-cardinality / text[/dim]"
        else:
            role = "categorical?"
        table.add_row(col, str(df[col].dtype), str(nunique), missing_str, role)

    console.print(table)

    # --- Profile all valid target candidates ---
    # Print only count + imbalance flag — never dump full class lists to terminal.
    target_profiles = {}
    for col, nunique, ptype in target_candidates:
        counts = df[col].value_counts()
        total = len(df)
        balance = {str(k): round(v / total, 3) for k, v in counts.items()}
        min_ratio = min(balance.values())
        is_imbalanced = min_ratio < 0.15
        target_profiles[col] = {
            "n_classes": nunique,
            "problem_type": ptype,
            "class_balance": balance,
            "is_imbalanced": is_imbalanced,
            "classes": list(counts.index.astype(str)),
        }

    if target_profiles:
        console.print("\n[bold]Detected target column candidates:[/bold]")
        for col, tp in target_profiles.items():
            imb_flag = " [yellow]⚠ imbalanced[/yellow]" if tp["is_imbalanced"] else ""
            # Show class count and a short preview (max 6 values), not the full list
            classes = tp["classes"]
            preview = classes[:6]
            suffix = f" … +{len(classes) - 6} more" if len(classes) > 6 else ""
            console.print(
                f"  [cyan]{col}[/cyan] → {tp['problem_type']}{imb_flag}  "
                f"classes: {preview}{suffix}"
            )
        console.print()
    else:
        console.print(
            f"[yellow]⚠ No clear target column detected (checked for 2–{MAX_TARGET_CARDINALITY} unique values, "
            f"non-float). You will be asked to enter the column name manually.[/yellow]\n"
        )

    profile = {
        "filepath": filepath,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_any": int(df.isna().sum().sum()),
        "target_candidates": [tc[0] for tc in target_candidates],
        "target_profiles": target_profiles,
        "n_classes": None,
        "class_balance": None,
        "is_imbalanced": False,
        "n_features": None,
        "has_categorical": any(df[col].dtype == object for col in df.columns),
        "scale_variance": None,
        "df": df,
    }

    return profile

