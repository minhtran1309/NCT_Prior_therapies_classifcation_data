"""
Configuration for the Clinical Trial Prior Therapy Classification System.

Stores OpenRouter API settings, model identifiers, and shared parameters.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ─── OpenRouter API ───────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ─── Model Identifiers (OpenRouter format) ────────────────────────────────────
MODELS = {
    "gpt4o": "openai/gpt-4o",
    "gpt5": "openai/gpt-5",
    "ministral": "mistralai/ministral-14b-2512",
    "qwen": "qwen/qwen3-32b",
}

# ─── Inference Parameters ─────────────────────────────────────────────────────
TEMPERATURE = 0.0          # Deterministic output for classification
MAX_TOKENS = 512           # Enough for prediction + reasoning
REQUEST_TIMEOUT = 180      # Seconds (some models like Qwen include reasoning steps)
MAX_RETRIES = 3            # Retry on transient failures
RETRY_DELAY = 2            # Seconds between retries

# ─── Data ─────────────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "annot-NOT_prior_therapy-tutti-20250502.txt",
)
DEFAULT_NUM_ROWS = 100

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output",
)
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
)
