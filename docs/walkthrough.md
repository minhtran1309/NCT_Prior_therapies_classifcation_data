# Walkthrough: Clinical Trial Prior Therapy Classification System

## What Was Built

An LLM-based classification system that uses **OpenRouter** to determine whether clinical trial eligibility criteria require a patient to have had (or not had) a prior therapy. The system tests 4 models: GPT-4o, GPT-5, Ministral-14b, and Qwen3-32b.

## Project Structure

```
src/
├── __init__.py
├── config.py              # API config, model IDs, parameters
├── data_loader.py         # Parses tab-separated data, extracts drug names
├── prompt_builder.py      # Builds INCL/EXCL-aware classification prompts
├── openrouter_client.py   # Unified API client with retry & rate-limit handling
├── evaluator.py           # Metrics (accuracy, precision, recall, F1) + CSV/JSON export
├── run_gpt4o.py           # GPT-4o runner
├── run_gpt5.py            # GPT-5 runner
├── run_ministral.py       # Ministral-14b runner
├── run_qwen.py            # Qwen3-32b runner
└── run_all.py             # Orchestrator — runs all models & compares
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Drug name extracted from filename via regex | First column follows pattern `annot-NOT_prior_therapy_<Drug>.txt` |
| Rows with non-0/1 labels are filtered out | Per user requirement for clean data |
| Prompts adapt to INCL vs EXCL section | Context matters for classification meaning |
| `temperature=0.0` | Deterministic classification output |
| Retry with exponential backoff | Handles OpenRouter rate limits gracefully |

## How to Run

```bash
# Set API key
export OPENROUTER_API_KEY="your-key-here"

# Dry run (no API calls, verify prompts)
python -m src.run_all --num_rows 5 --dry-run

# Run a single model
python -m src.run_gpt4o --num_rows 100

# Run all models
python -m src.run_all --num_rows 100

# Run specific models
python -m src.run_all --models gpt4o,qwen --num_rows 50
```

## Verification

Dry-run test passed — all 4 model runners loaded data correctly, extracted drug names, built section-aware prompts, and the orchestrator dispatched to all models successfully.
