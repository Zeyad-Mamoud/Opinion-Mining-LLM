"""Inference helpers for structured opinion mining output.

The final system should extract sentiment and product features from customer
reviews and return them as reliable JSON instead of free-form text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

import torch


SYSTEM_INSTRUCTION = """You are an opinion mining assistant.
Extract sentiment and product features from a customer review.
Return valid JSON only.
"""


def build_review_prompt(review_text: str) -> str:
    """Create a task prompt that asks the model for structured JSON output."""
    return f"""
{SYSTEM_INSTRUCTION}

Use this schema:
{{
  "review": "<original review>",
  "overall_sentiment": "positive | negative | neutral | mixed",
  "aspects": [
    {{
      "feature": "<product feature>",
      "sentiment": "positive | negative | neutral",
      "opinion": "<opinion phrase from the review>"
    }}
  ]
}}

Review:
{review_text}
""".strip()


def extract_json_block(text: str) -> str:
    """Extract the last JSON object from a model response."""
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if not matches:
        raise ValueError("No JSON object found in model output.")
    return matches[-1]


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse model output into a Python dictionary."""
    return json.loads(extract_json_block(text))


def generate_structured_output(
    model,
    tokenizer,
    review_text: str,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Run generation and return the parsed JSON prediction."""
    prompt = build_review_prompt(review_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_only = output_ids[0][prompt_length:]
    generated_text = tokenizer.decode(generated_only, skip_special_tokens=True)
    return parse_json_response(generated_text)
