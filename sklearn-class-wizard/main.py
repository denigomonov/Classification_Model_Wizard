"""
main.py — sklearn Classification Wizard
Entry point. Run: python main.py path/to/your_data.csv

Tournament benchmark engine:
  Round 1 → Eliminate bottom 50% on 30% subsample (fast, parallel)
  Round 2 → Tune survivors with Optuna (20 trials, 5-fold CV)
  Output  → Top 3 recommendations with reasoning + quick-start code
"""

import sys
import time
from rich.console import Console
from rich.rule import Rule
from rich import print as rprint

from scanner import scan_csv
from questionnaire import run_questionnaire
from benchmark import run_benchmark
from recommender import render_recommendations

console = Console()

BANNER = """
[bold cyan]
 ╔══════════════════════════════════════════════════════╗
 ║   ¯\_(ツ)_/¯ sklearn Classification Wizard  v0.1     ║
 ║     Tournament-style model selection engine          ║
 ╚══════════════════════════════════════════════════════╝
[/bold cyan]"""


def main():
    console.print(BANNER)

    # ── 1. Get CSV path ──────────────────────────────────────────────────────
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python main.py <path_to_csv>[/yellow]")
        console.print("[dim]Example: python main.py data/antenna_dataset_sample.csv[/dim]")
        sys.exit(1)

    csv_path = sys.argv[1]

    # ── 2. Scan dataset ──────────────────────────────────────────────────────
    profile = scan_csv(csv_path)

    # ── 3. Run wizard questionnaire ──────────────────────────────────────────
    ctx = run_questionnaire(profile)

    # ── 4. Run tournament benchmark ──────────────────────────────────────────
    t0 = time.perf_counter()
    top3 = run_benchmark(profile, ctx)
    elapsed = time.perf_counter() - t0

    console.print(f"\n[dim]⏱ Total benchmark time: {elapsed:.1f}s[/dim]\n")

    # ── 5. Render recommendations ────────────────────────────────────────────
    render_recommendations(top3, ctx)

    console.rule("[bold green]✅ Done[/bold green]")
    console.print("[dim]Tip: Pass a different CSV to re-run on new data.\n[/dim]")


if __name__ == "__main__":
    main()
