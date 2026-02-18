"""
Prompt builder for the prior therapy classification task.

Constructs structured prompts for the LLM based on:
  - The section the criteria appears in (INCLUSION vs EXCLUSION)
  - The criteria text itself
  - The drug name

The LLM is asked to return a JSON with "prediction" (0 or 1) and "reason".
"""

from typing import Dict


# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert clinical trial eligibility text classification. "
    "Your task is to classify whether a prior therapy eligibility criterion "
    "means the patient MUST NOT have had the drug (strict exclusion, label=0) or "
    "the patient MAY or MAY NOT have had the drug (not a strict requirement, label=1). "
    "You MUST respond in EXACTLY this format on two lines:\n"
    "PREDICTION: <0 or 1>\n"
    "REASON: <one sentence explanation>\n"
    "Do NOT include any other text."
)

# ─── User Prompt Templates ───────────────────────────────────────────────────
INCLUSION_TEMPLATE = """The following clause appears in the "INCLUSION CRITERIA" section of the clinical trial eligibility document.

"{criteria_text}"

Drug of interest: {drug_name}

Classification rules:
- If the sentence contains "NOT prior" or negates prior therapy with {drug_name}:
  0 = To be eligible for the trial, the patient MUST NOT have had {drug_name} prior to enrollment.
  1 = The patient MAY or MAY NOT have had {drug_name} prior to enrollment.

- If the sentence says "Prior {drug_name}" or requires prior exposure:
  1 = To be eligible for the trial, the patient MUST HAVE had {drug_name} prior to enrollment, or the patient will be excluded.
  0 = The patient MAY or MAY NOT have had {drug_name} prior to enrollment.

- "must not have been exposed to {drug_name}" = 0

Respond in EXACTLY this format:
PREDICTION: <0 or 1>
REASON: <one sentence explanation>"""

EXCLUSION_TEMPLATE = """The following clause appears in the "EXCLUSION CRITERIA" section of the clinical trial eligibility document.

"{criteria_text}"

Drug of interest: {drug_name}

Classification rules:
- If the sentence contains "NOT prior" or negates prior therapy with {drug_name}:
  0 = To be eligible for the trial, the patient MUST NOT have had {drug_name} prior to enrollment.
  1 = The patient MAY or MAY NOT have had {drug_name} prior to enrollment.

- If the sentence says "Prior {drug_name}" or requires prior exposure:
  1 = To be eligible for the trial, the patient MUST HAVE had {drug_name} prior to enrollment, or the patient will be excluded.
  0 = The patient MAY or MAY NOT have had {drug_name} prior to enrollment.

- "must not have been exposed to {drug_name}" = 0

Respond in EXACTLY this format:
PREDICTION: <0 or 1>
REASON: <one sentence explanation>"""


def build_prompt(record: Dict) -> tuple:
    """
    Build the (system_message, user_message) prompt pair for a single record.

    Args:
        record: Dict with keys drug_name, section, criteria_text.

    Returns:
        (system_prompt: str, user_prompt: str)
    """
    drug_name = record["drug_name"]
    section = record["section"].upper()
    criteria_text = record["criteria_text"]

    if section == "INCL":
        user_prompt = INCLUSION_TEMPLATE.format(
            criteria_text=criteria_text,
            drug_name=drug_name,
        )
    else:  # EXCL
        user_prompt = EXCLUSION_TEMPLATE.format(
            criteria_text=criteria_text,
            drug_name=drug_name,
        )

    return SYSTEM_PROMPT, user_prompt


if __name__ == "__main__":
    sample = {
        "drug_name": "Abiraterone",
        "section": "INCL",
        "criteria_text": "Prior treatment with Abiraterone, Enzalutamide or both",
    }
    sys_msg, usr_msg = build_prompt(sample)
    print("=== SYSTEM ===")
    print(sys_msg)
    print("\n=== USER ===")
    print(usr_msg)
