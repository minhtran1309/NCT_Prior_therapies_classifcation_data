"""
Data loader for the clinical trial prior therapy annotation dataset.

Parses the tab-separated text file and extracts:
  - drug_name   : extracted from the source filename (e.g. "Abiraterone")
  - label       : ground-truth classification (0 or 1)
  - trial_id    : clinical trial identifier
  - section     : INCL or EXCL
  - criteria_text : eligibility criteria sentence
"""

import re
from typing import List, Dict, Optional

from src.config import DATA_FILE, DEFAULT_NUM_ROWS


def _extract_drug_name(source_filename: str) -> str:
    """
    Extract the drug name from the source filename.

    Pattern: annot-NOT_prior_therapy_<DrugName>.txt
    The drug name can contain hyphens, underscores, and alphanumeric characters.

    Examples:
        annot-NOT_prior_therapy_Abiraterone.txt        -> Abiraterone
        annot-NOT_prior_therapy_anti-CD137_monoclonal_antibody.txt -> anti-CD137_monoclonal_antibody
        annot-NOT_prior_therapy_anti-CTLA-4.txt        -> anti-CTLA-4
        annot-NOT_prior_therapy_AKT_inhibitor.txt      -> AKT_inhibitor
    """
    match = re.match(r"annot-NOT_prior_therapy_(.+)\.txt", source_filename)
    if match:
        return match.group(1)
    return source_filename  # Fallback: return as-is


def load_data(
    filepath: Optional[str] = None,
    num_rows: Optional[int] = None,
) -> List[Dict]:
    """
    Load and parse the annotation data file.

    Args:
        filepath:  Path to the data file. Defaults to config.DATA_FILE.
        num_rows:  Maximum number of valid rows to return. None = all rows.

    Returns:
        List of dicts with keys:
            drug_name, label, trial_id, section, criteria_text
    """
    filepath = filepath or DATA_FILE
    num_rows = num_rows if num_rows is not None else DEFAULT_NUM_ROWS

    records: List[Dict] = []

    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 5:
                continue  # Skip malformed rows

            source_file = parts[0]
            raw_label = parts[1].strip()
            trial_id = parts[2].strip()
            section = parts[3].strip()
            criteria_text = parts[4].strip()

            # Only keep rows with valid label (0 or 1)
            if raw_label not in ("0", "1"):
                continue

            records.append(
                {
                    "drug_name": _extract_drug_name(source_file),
                    "label": int(raw_label),
                    "trial_id": trial_id,
                    "section": section,           # INCL or EXCL
                    "criteria_text": criteria_text,
                }
            )

            if num_rows is not None and len(records) >= num_rows:
                break

    return records


if __name__ == "__main__":
    # Quick sanity check
    data = load_data(num_rows=5)
    for row in data:
        print(row)
