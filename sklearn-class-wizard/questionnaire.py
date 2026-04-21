"""
questionnaire.py — Interactive terminal wizard to capture user knowledge about their data.
Uses questionary for styled prompts. Outputs a context dict used by the benchmark engine.
"""

import questionary
from rich.console import Console
from rich.rule import Rule

console = Console()


def run_questionnaire(profile: dict) -> dict:
    """Run the guided wizard and return a context dict."""
    console.print(Rule("[bold cyan]🧙 Classification Wizard[/bold cyan]"))
    console.print(
        "[dim]Answer a few questions so the engine can tailor its recommendations.\n"
        "Press Enter to accept defaults.[/dim]\n"
    )

    ctx = {}

    # --- 1. Select target column — show problem type context per candidate ---
    candidates = profile.get("target_candidates", [])
    target_profiles = profile.get("target_profiles", {})

    if not candidates:
        ctx["target_col"] = questionary.text(
            "No obvious target column detected. Enter the target column name:"
        ).ask()
    elif len(candidates) == 1:
        ctx["target_col"] = candidates[0]
        console.print(f"[dim]Auto-selected only target candidate: [cyan]{candidates[0]}[/cyan][/dim]\n")
    else:
        # Build rich choice labels so user understands what each option means
        choices = []
        for col in candidates:
            tp = target_profiles.get(col, {})
            ptype = tp.get("problem_type", "?")
            classes = tp.get("classes", [])
            imb = " ⚠ imbalanced" if tp.get("is_imbalanced") else ""
            label = f"{col}  [{ptype}{imb}]  classes: {classes}"
            choices.append(questionary.Choice(title=label, value=col))

        ctx["target_col"] = questionary.select(
            "Multiple target column candidates found — which are you trying to predict?",
            choices=choices,
        ).ask()

    # Update profile with chosen target's stats for downstream use
    chosen = ctx["target_col"]
    tp = target_profiles.get(chosen, {})
    profile["n_classes"] = tp.get("n_classes")
    profile["class_balance"] = tp.get("class_balance")
    profile["is_imbalanced"] = tp.get("is_imbalanced", False)
    profile["inferred_target"] = chosen
    profile["feature_cols"] = [c for c in profile["columns"] if c != chosen]
    profile["n_features"] = len(profile["feature_cols"])

    # Print chosen target's class balance
    if tp:
        from rich.table import Table
        from rich import box
        balance_table = Table(
            title=f"Class Balance → '{chosen}'", box=box.SIMPLE, header_style="bold blue"
        )
        balance_table.add_column("Class", style="cyan")
        balance_table.add_column("Ratio", justify="right")
        for cls, ratio in tp.get("class_balance", {}).items():
            color = "red" if ratio < 0.15 else "green"
            balance_table.add_row(str(cls), f"[{color}]{ratio:.1%}[/{color}]")
        console.print(balance_table)
        if tp.get("is_imbalanced"):
            console.print("[yellow]⚠️  Class imbalance detected — will factor into recommendations.[/yellow]\n")
        else:
            console.print("[green]✓ Classes are reasonably balanced.[/green]\n")

    # --- 2. Primary goal ---
    ctx["goal"] = questionary.select(
        "What's your primary goal?",
        choices=[
            "Best accuracy / F1 score (I care about performance above all)",
            "Fast inference in production (model must predict quickly)",
            "Interpretability (I need to explain decisions to stakeholders)",
            "Balance of all three",
        ],
    ).ask()

    # --- 3. Dataset size expectation ---
    ctx["expected_size"] = questionary.select(
        "In production, how large do you expect your dataset to grow?",
        choices=[
            "Small (< 10k rows)",
            "Medium (10k – 500k rows)",
            "Large (500k+ rows)",
        ],
        default="Small (< 10k rows)",
    ).ask()

    # --- 4. Training time budget ---
    ctx["train_time"] = questionary.select(
        "How much training time is acceptable?",
        choices=[
            "Seconds (quick iteration / prototyping)",
            "Minutes (standard ML workflow)",
            "Hours (large-scale, offline training)",
        ],
        default="Minutes (standard ML workflow)",
    ).ask()

    # --- 5. Missing data expectation ---
    has_missing = profile.get("missing_any", 0) > 0
    ctx["has_missing"] = questionary.confirm(
        "Does your data contain (or will it contain) missing values?",
        default=has_missing,
    ).ask()

    # --- 6. Feature types ---
    has_cat = profile.get("has_categorical", False)
    ctx["has_categorical"] = questionary.confirm(
        "Does your data include categorical / text features?",
        default=has_cat,
    ).ask()

    # --- 7. Class imbalance ---
    is_imb = profile.get("is_imbalanced", False)
    ctx["is_imbalanced"] = questionary.confirm(
        "Is your target class heavily imbalanced (e.g., rare fraud, rare faults)?",
        default=is_imb,
    ).ask()

    # --- 8. Feature interactions ---
    ctx["feature_interactions"] = questionary.confirm(
        "Do you expect complex non-linear relationships between features?",
        default=True,
    ).ask()

    # --- 9. Noise tolerance ---
    ctx["noisy_data"] = questionary.confirm(
        "Is your data noisy or prone to labeling errors?",
        default=False,
    ).ask()

    console.print("\n[green]✓ Wizard complete. Running benchmark engine...[/green]\n")
    return ctx
