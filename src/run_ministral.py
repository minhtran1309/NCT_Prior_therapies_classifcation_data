"""
Runner for Mistral AI Ministral-14b via OpenRouter.

Saves results incrementally to output/ministral.csv after EACH row so no
progress is lost if interrupted. Rerun to continue from where you left off.

Usage:
    conda activate oncogpt && python -m src.run_ministral [--num_rows N] [--dry-run]
"""

import os
import argparse
import pandas as pd

from src.config import MODELS, OUTPUT_DIR
from src.data_loader import load_data
from src.prompt_builder import build_prompt
from src.openrouter_client import OpenRouterClient


MODEL_KEY = "ministral"
MODEL_ID = MODELS[MODEL_KEY]
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{MODEL_KEY}.csv")
ERROR_FILE = os.path.join(OUTPUT_DIR, f"error_output_{MODEL_KEY}.csv")

COLUMNS = [
    "drug_name", "label", "trial_id", "section", "criteria_text",
    "llm_prediction", "llm_reason",
]


def _load_cache():
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print(f"  📦 Cache loaded: {len(df)} rows from {OUTPUT_FILE}")
        return df
    return pd.DataFrame(columns=COLUMNS)


def _is_cached(cache_df, row):
    if cache_df.empty:
        return False
    mask = (
        (cache_df["trial_id"] == row["trial_id"])
        & (cache_df["criteria_text"] == row["criteria_text"])
        & (cache_df["llm_prediction"].notna())
        & (cache_df["llm_prediction"].isin([0, 1, 0.0, 1.0]))
    )
    return mask.any()


def _save_df(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    error_mask = ~df["llm_prediction"].isin([0, 1, 0.0, 1.0])
    error_df = df.loc[error_mask, ["drug_name", "label", "trial_id", "section", "criteria_text"]].copy()
    error_df.to_csv(ERROR_FILE, index=False)


def _append_row(cache_df, new_row):
    if not cache_df.empty:
        mask = (
            (cache_df["trial_id"] == new_row["trial_id"])
            & (cache_df["criteria_text"] == new_row["criteria_text"])
        )
        cache_df = cache_df[~mask]
    new_df = pd.DataFrame([new_row], columns=COLUMNS)
    return pd.concat([cache_df, new_df], ignore_index=True)


def run(num_rows=100, dry_run=False):
    print(f"\n🚀 Running classification with {MODEL_ID}")
    print(f"   Rows: {num_rows} | Dry run: {dry_run}\n")

    data = load_data(num_rows=num_rows)
    print(f"  ✅ Loaded {len(data)} records from input data")
    cache_df = _load_cache()

    if dry_run:
        cached = sum(1 for r in data if _is_cached(cache_df, r))
        print(f"  📊 Already cached: {cached}/{len(data)}")
        print(f"  📊 Need to process: {len(data) - cached}")
        print("\n  🏁 Dry run complete. No API calls made.")
        return

    client = OpenRouterClient()
    skipped = processed = 0

    for i, record in enumerate(data):
        if _is_cached(cache_df, record):
            skipped += 1
            print(f"  [{i+1}/{len(data)}] {record['trial_id']} — ⏭ cached")
            continue

        sys_msg, usr_msg = build_prompt(record)
        print(f"  [{i+1}/{len(data)}] {record['trial_id']} ({record['drug_name']})…", end=" ")

        response = client.classify(MODEL_ID, sys_msg, usr_msg)
        pred, reason, error = response["prediction"], response["reason"], response["error"]

        if error:
            print(f"❌ ERROR: {error}")
        else:
            status = "✅" if pred == record["label"] else "❌"
            print(f"pred={pred} truth={record['label']} {status}")

        new_row = {
            "drug_name": record["drug_name"], "label": record["label"],
            "trial_id": record["trial_id"], "section": record["section"],
            "criteria_text": record["criteria_text"],
            "llm_prediction": pred if not error else None,
            "llm_reason": reason if not error else f"ERROR: {error}",
        }
        cache_df = _append_row(cache_df, new_row)
        _save_df(cache_df)
        processed += 1

    print(f"\n  💾 Saved {len(cache_df)} total rows to {OUTPUT_FILE}")
    valid = cache_df["llm_prediction"].isin([0, 1, 0.0, 1.0])
    correct = cache_df.loc[valid, "label"] == cache_df.loc[valid, "llm_prediction"]
    print(f"  📊 This run: {processed} new | {skipped} cached")
    if valid.sum() > 0:
        print(f"  📊 Accuracy: {correct.sum()}/{valid.sum()} = {correct.sum()/valid.sum():.2%}")

    error_mask = ~cache_df["llm_prediction"].isin([0, 1, 0.0, 1.0])
    print(f"  ⚠️  {error_mask.sum()} failed rows in {ERROR_FILE}")


def main():
    parser = argparse.ArgumentParser(description=f"Run {MODEL_ID} classification")
    parser.add_argument("--num_rows", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(num_rows=args.num_rows, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
