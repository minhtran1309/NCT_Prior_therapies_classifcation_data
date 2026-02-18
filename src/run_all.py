"""
Orchestrator — Run all LLM models concurrently and produce a comparison report.

Each model runs in its own thread so you don't wait for one to finish
before the next starts. Results are saved independently per model.

Usage:
    conda activate oncogpt && python -m src.run_all [--num_rows N] [--models gpt4o,gpt5,...] [--dry-run]
"""

import os
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.config import MODELS, OUTPUT_DIR

# Import each runner
from src.run_gpt4o import run as run_gpt4o
from src.run_gpt5 import run as run_gpt5
from src.run_ministral import run as run_ministral
from src.run_qwen import run as run_qwen


RUNNERS = {
    "gpt4o": run_gpt4o,
    "gpt5": run_gpt5,
    "ministral": run_ministral,
    "qwen": run_qwen,
}


def _run_model(model_key, num_rows, dry_run):
    """Wrapper to run a single model and catch exceptions."""
    try:
        RUNNERS[model_key](num_rows=num_rows, dry_run=dry_run)
        return model_key, True, None
    except Exception as e:
        return model_key, False, str(e)


def compare_results():
    """Load all output CSVs and print a comparison table."""
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON")
    print("=" * 80)

    header = f"{'Model':<25} {'Total':>8} {'Valid':>8} {'Correct':>8} {'Accuracy':>10}"
    print(header)
    print("-" * 80)

    for model_key in RUNNERS:
        csv_path = os.path.join(OUTPUT_DIR, f"{model_key}.csv")
        if not os.path.exists(csv_path):
            print(f"{model_key:<25} {'(no results)':>8}")
            continue

        df = pd.read_csv(csv_path)
        total = len(df)
        valid_mask = df["llm_prediction"].isin([0, 1, 0.0, 1.0])
        valid = valid_mask.sum()
        correct = (df.loc[valid_mask, "label"] == df.loc[valid_mask, "llm_prediction"]).sum()
        accuracy = correct / valid if valid > 0 else 0.0

        print(f"{model_key:<25} {total:>8} {valid:>8} {correct:>8} {accuracy:>10.2%}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run all LLM models for prior therapy classification"
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=100,
        help="Number of rows to process (default: 100)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to run (default: all). "
             f"Available: {', '.join(RUNNERS.keys())}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompts without making API calls",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only show comparison of existing results, don't run models",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run models one at a time instead of concurrently",
    )
    args = parser.parse_args()

    if args.compare_only:
        compare_results()
        return

    # Determine which models to run
    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        for m in selected:
            if m not in RUNNERS:
                print(f"❌ Unknown model: '{m}'. Available: {', '.join(RUNNERS.keys())}")
                sys.exit(1)
    else:
        selected = list(RUNNERS.keys())

    mode = "sequential" if args.sequential else "concurrent"

    print("=" * 60)
    print("  CLINICAL TRIAL PRIOR THERAPY CLASSIFIER")
    print("=" * 60)
    print(f"  Models   : {', '.join(selected)}")
    print(f"  Rows     : {args.num_rows}")
    print(f"  Mode     : {mode}")
    print(f"  Dry run  : {args.dry_run}")
    print(f"  Output   : {OUTPUT_DIR}")
    print("=" * 60)

    if args.sequential:
        # Run one at a time (original behavior)
        for model_key in selected:
            RUNNERS[model_key](num_rows=args.num_rows, dry_run=args.dry_run)
    else:
        # Run all models concurrently in separate threads
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = {
                executor.submit(_run_model, key, args.num_rows, args.dry_run): key
                for key in selected
            }

            for future in as_completed(futures):
                model_key, success, error = future.result()
                if success:
                    print(f"\n  ✅ {model_key} completed successfully")
                else:
                    print(f"\n  ❌ {model_key} failed: {error}")

    # Show comparison if we have real results
    if not args.dry_run:
        compare_results()

    print("\n🏁 All runs complete.")


if __name__ == "__main__":
    main()
