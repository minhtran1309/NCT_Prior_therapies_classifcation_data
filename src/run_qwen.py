"""
Runner for Qwen3-32b via OpenRouter.

Saves results incrementally to output/qwen.csv after EACH row so no
progress is lost if interrupted. Rerun to continue from where you left off.

Usage:
    conda activate oncogpt && python -m src.run_qwen [--num_rows N] [--dry-run]
"""

import os
import argparse

import pandas as pd

from src.config import MODELS, OUTPUT_DIR
from src.data_loader import load_data
from src.prompt_builder import build_prompt
from src.openrouter_client import OpenRouterClient


MODEL_KEY = "qwen"
MODEL_ID = MODELS[MODEL_KEY]
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{MODEL_KEY}.csv")
TAG = f"[{MODEL_KEY}]"
ERROR_FILE = os.path.join(OUTPUT_DIR, f"error_output_{MODEL_KEY}.csv")

COLUMNS = [
    "drug_name",       # 1. Drug name from source filename
    "label",           # 2. Ground-truth label (0 or 1)
    "trial_id",        # 3. Clinical trial ID
    "section",         # 4. INCL or EXCL
    "criteria_text",   # 5. Eligibility criteria sentence
    "llm_prediction",  # 6. LLM output: 0 (exclude) or 1 (include)
    "llm_reason",      # 7. LLM reasoning
]


def _load_cache() -> pd.DataFrame:
    """Load existing results CSV if it exists."""
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print(f"  {TAG} 📦 Cache loaded: {len(df)} rows from {OUTPUT_FILE}")
        return df
    return pd.DataFrame(columns=COLUMNS)


def _is_cached(cache_df: pd.DataFrame, row: dict) -> bool:
    """Check if a row already has a valid prediction in the cache."""
    if cache_df.empty:
        return False
    mask = (
        (cache_df["trial_id"] == row["trial_id"])
        & (cache_df["criteria_text"] == row["criteria_text"])
        & (cache_df["llm_prediction"].notna())
        & (cache_df["llm_prediction"].isin([0, 1, 0.0, 1.0]))
    )
    return mask.any()


def _save_df(df: pd.DataFrame):
    """Save main output and error output to CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # Also save error rows (failed/missing predictions) incrementally
    error_mask = ~df["llm_prediction"].isin([0, 1, 0.0, 1.0])
    error_df = df.loc[error_mask, ["drug_name", "label", "trial_id", "section", "criteria_text"]].copy()
    error_df.to_csv(ERROR_FILE, index=False)


def _append_row(cache_df: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    """
    Append a new result row to the cache DataFrame.
    If the row already exists (same trial_id + criteria_text), replace it.
    """
    # Remove any existing entry for this row
    if not cache_df.empty:
        mask = (
            (cache_df["trial_id"] == new_row["trial_id"])
            & (cache_df["criteria_text"] == new_row["criteria_text"])
        )
        cache_df = cache_df[~mask]

    new_df = pd.DataFrame([new_row], columns=COLUMNS)
    return pd.concat([cache_df, new_df], ignore_index=True)


def run(num_rows: int = 100, dry_run: bool = False):
    """Run classification with Qwen3-32b."""
    print(f"\n{TAG} 🚀 Running classification with {MODEL_ID}")
    print(f"{TAG}    Rows: {num_rows} | Dry run: {dry_run}\n")

    # 1. Load data
    data = load_data(num_rows=num_rows)
    print(f"  {TAG} ✅ Loaded {len(data)} records from input data")

    # 2. Load cache
    cache_df = _load_cache()

    if dry_run:
        cached = sum(1 for r in data if _is_cached(cache_df, r))
        print(f"  {TAG} 📊 Already cached: {cached}/{len(data)}")
        print(f"  {TAG} 📊 Need to process: {len(data) - cached}")
        for i, record in enumerate(data[:3]):
            sys_msg, usr_msg = build_prompt(record)
            print(f"\n--- Record {i+1} ---")
            print(f"Drug: {record['drug_name']} | Section: {record['section']} | Label: {record['label']}")
            print(f"Cached: {_is_cached(cache_df, record)}")
            print(f"Prompt preview: {usr_msg[:200]}...")
        print(f"\n  {TAG} 🏁 Dry run complete. No API calls made.")
        return

    # 3. Classify each record, saving after every row
    client = OpenRouterClient()
    skipped = 0
    processed = 0

    for i, record in enumerate(data):
        if _is_cached(cache_df, record):
            skipped += 1
            print(f"  {TAG} [{i+1}/{len(data)}] {record['trial_id']} — ⏭ cached")
            continue

        sys_msg, usr_msg = build_prompt(record)
        print(f"  {TAG} [{i+1}/{len(data)}] {record['trial_id']} ({record['drug_name']})…", end=" ")

        response = client.classify(MODEL_ID, sys_msg, usr_msg)

        pred = response["prediction"]
        reason = response["reason"]
        error = response["error"]

        if error:
            print(f"❌ ERROR: {error}")
            pred_val = None
            reason_val = f"ERROR: {error}"
        else:
            status = "✅" if pred == record["label"] else "❌"
            print(f"pred={pred} truth={record['label']} {status}")
            pred_val = pred
            reason_val = reason

        new_row = {
            "drug_name": record["drug_name"],
            "label": record["label"],
            "trial_id": record["trial_id"],
            "section": record["section"],
            "criteria_text": record["criteria_text"],
            "llm_prediction": pred_val,
            "llm_reason": reason_val,
        }

        # Save immediately after each row
        cache_df = _append_row(cache_df, new_row)
        _save_df(cache_df)
        processed += 1

    # 4. Summary
    print(f"\n  {TAG} 💾 Saved {len(cache_df)} total rows to {OUTPUT_FILE}")

    valid = cache_df["llm_prediction"].isin([0, 1, 0.0, 1.0])
    correct = cache_df.loc[valid, "label"] == cache_df.loc[valid, "llm_prediction"]
    total_valid = valid.sum()
    total_correct = correct.sum()

    print(f"  {TAG} 📊 This run: {processed} new | {skipped} cached")
    if total_valid > 0:
        print(f"  {TAG} 📊 Accuracy: {total_correct}/{total_valid} = {total_correct/total_valid:.2%}")

    # 5. Error summary (error file is saved incrementally by _save_df)
    error_mask = ~cache_df["llm_prediction"].isin([0, 1, 0.0, 1.0])
    print(f"  {TAG} ⚠️  {error_mask.sum()} failed rows in {ERROR_FILE}")


def main():
    parser = argparse.ArgumentParser(description=f"Run {MODEL_ID} classification")
    parser.add_argument("--num_rows", type=int, default=100, help="Number of rows to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview prompts without API calls")
    args = parser.parse_args()

    run(num_rows=args.num_rows, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
