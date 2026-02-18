"""
Evaluation module for the prior therapy classification system.

Computes standard binary-classification metrics and produces
formatted reports and CSV output.
"""

import os
import csv
import json
from typing import List, Dict
from datetime import datetime

from src.config import RESULTS_DIR


def compute_metrics(results: List[Dict]) -> Dict:
    """
    Compute accuracy, precision, recall, F1, and confusion matrix.

    Args:
        results: List of dicts, each with at least 'label' (ground truth)
                 and 'prediction' (model output, int or None).

    Returns:
        Dict with keys: accuracy, precision, recall, f1, confusion_matrix,
                        total, valid, errors
    """
    tp = fp = tn = fn = 0
    errors = 0

    for r in results:
        gt = r["label"]
        pred = r.get("prediction")

        if pred is None:
            errors += 1
            continue

        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 0:
            tn += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1

    total = len(results)
    valid = total - errors

    accuracy = (tp + tn) / valid if valid > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        },
        "total": total,
        "valid": valid,
        "errors": errors,
    }


def print_report(model_name: str, metrics: Dict) -> None:
    """Print a formatted metrics report to stdout."""
    cm = metrics["confusion_matrix"]
    print()
    print("=" * 60)
    print(f"  MODEL: {model_name}")
    print("=" * 60)
    print(f"  Total samples  : {metrics['total']}")
    print(f"  Valid responses : {metrics['valid']}")
    print(f"  Parse errors   : {metrics['errors']}")
    print("-" * 60)
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  F1 Score       : {metrics['f1']:.4f}")
    print("-" * 60)
    print("  Confusion Matrix:")
    print(f"                   Predicted 0    Predicted 1")
    print(f"    Actual 0       {cm['TN']:>8}       {cm['FP']:>8}")
    print(f"    Actual 1       {cm['FN']:>8}       {cm['TP']:>8}")
    print("=" * 60)
    print()


def save_results(
    model_name: str,
    results: List[Dict],
    metrics: Dict,
) -> str:
    """
    Save detailed results to CSV and metrics to JSON.

    Args:
        model_name:  Short model name (e.g. "gpt4o").
        results:     List of dicts with prediction results.
        metrics:     Output from compute_metrics().

    Returns:
        Path to the results directory for this model.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(RESULTS_DIR, f"{model_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # ── Save detailed results as CSV ──────────────────────────────────────
    csv_path = os.path.join(model_dir, "predictions.csv")
    fieldnames = [
        "drug_name",
        "trial_id",
        "section",
        "label",
        "prediction",
        "correct",
        "raw_response",
        "error",
        "criteria_text",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "drug_name": r.get("drug_name", ""),
                    "trial_id": r.get("trial_id", ""),
                    "section": r.get("section", ""),
                    "label": r.get("label", ""),
                    "prediction": r.get("prediction", ""),
                    "correct": (
                        r.get("label") == r.get("prediction")
                        if r.get("prediction") is not None
                        else ""
                    ),
                    "raw_response": r.get("raw_response", ""),
                    "error": r.get("error", ""),
                    "criteria_text": r.get("criteria_text", ""),
                }
            )

    # ── Save metrics as JSON ──────────────────────────────────────────────
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"model": model_name, "timestamp": timestamp, **metrics},
            f,
            indent=2,
        )

    print(f"  📁 Results saved to: {model_dir}")
    return model_dir


def compare_models(all_metrics: Dict[str, Dict]) -> None:
    """
    Print a side-by-side comparison table of all models.

    Args:
        all_metrics: Dict mapping model_name -> metrics dict.
    """
    print()
    print("=" * 80)
    print("  MODEL COMPARISON")
    print("=" * 80)
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    print(header)
    print("-" * 80)
    for name, m in all_metrics.items():
        row = f"{name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}"
        print(row)
    print("=" * 80)
    print()

    # Save comparison to JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    comparison_path = os.path.join(RESULTS_DIR, "comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  📁 Comparison saved to: {comparison_path}")
