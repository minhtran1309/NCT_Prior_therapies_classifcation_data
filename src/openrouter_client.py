"""
Unified OpenRouter API client.

Sends chat-completion requests to any model available on OpenRouter and
parses the response to extract prediction (0/1) and reasoning.
"""

import re
import time
import requests
from typing import Optional

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    TEMPERATURE,
    MAX_TOKENS,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
)


class OpenRouterClient:
    """Client for the OpenRouter chat-completion API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set the OPENROUTER_API_KEY environment variable or add to .env file."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/clinical-trial-classifier",
            "X-Title": "Clinical Trial Prior Therapy Classifier",
        }

    def classify(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """
        Send a classification request and return the result.

        Args:
            model:          OpenRouter model identifier.
            system_prompt:  System-level instruction.
            user_prompt:    The user prompt with criteria text.

        Returns:
            dict with keys:
                prediction   (int or None) : 0, 1, or None if parsing failed
                reason       (str)         : LLM reasoning for the classification
                raw_response (str)         : raw text from the model
                error        (str or None) : error message if request failed
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    OPENROUTER_API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code == 429:
                    wait = RETRY_DELAY * attempt
                    print(f"  ⏳ Rate limited. Waiting {wait}s (attempt {attempt}/{MAX_RETRIES})…")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                # Check for embedded error (OpenRouter returns 200 but error in choices)
                choice = data.get("choices", [{}])[0]
                choice_error = choice.get("error")
                if choice_error:
                    error_msg = choice_error.get("message", str(choice_error))
                    print(f"  ⚠ API error: {error_msg}")
                    return {
                        "prediction": None,
                        "reason": "",
                        "raw_response": "",
                        "error": error_msg,
                    }

                # Extract text from the response
                raw_text = (
                    choice
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )

                prediction, reason = self._parse_response(raw_text)

                return {
                    "prediction": prediction,
                    "reason": reason,
                    "raw_response": raw_text,
                    "error": None,
                }

            except requests.exceptions.RequestException as exc:
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY * attempt
                    print(f"  ⚠ Request error: {exc}. Retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    return {
                        "prediction": None,
                        "reason": "",
                        "raw_response": "",
                        "error": str(exc),
                    }

        return {
            "prediction": None,
            "reason": "",
            "raw_response": "",
            "error": "Max retries exceeded",
        }

    @staticmethod
    def _parse_response(text: str) -> tuple:
        """
        Parse the model's response to extract prediction and reason.

        Expected format:
            PREDICTION: 0
            REASON: The criteria explicitly excludes patients who...

        Returns:
            (prediction: int or None, reason: str)
        """
        prediction = None
        reason = ""

        # Try to find PREDICTION: <0|1>
        pred_match = re.search(r"PREDICTION:\s*([01])", text, re.IGNORECASE)
        if pred_match:
            prediction = int(pred_match.group(1))

        # Try to find REASON: <text>
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()
            # Clean up: take only first line of reason
            reason = reason.split("\n")[0].strip()

        # Fallback: if no structured format, try to extract just a 0 or 1
        if prediction is None:
            for char in text:
                if char in ("0", "1"):
                    prediction = int(char)
                    break
            if not reason:
                reason = text.strip()

        return prediction, reason
