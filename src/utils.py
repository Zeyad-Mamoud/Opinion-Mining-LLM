"""Shared utilities for the opinion mining pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def get_output_schema_example() -> Dict[str, Any]:
    """Return a reference JSON structure for sentiment and aspect extraction."""
    return {
        "review": "The battery life is great but the screen is too dim.",
        "overall_sentiment": "mixed",
        "aspects": [
            {
                "feature": "battery life",
                "sentiment": "positive",
                "opinion": "great",
            },
            {
                "feature": "screen",
                "sentiment": "negative",
                "opinion": "too dim",
            },
        ],
    }


def validate_prediction_structure(prediction: Dict[str, Any]) -> bool:
    """Check that the main fields needed for the course output are present."""
    required_top_level = {"review", "overall_sentiment", "aspects"}
    if not required_top_level.issubset(prediction.keys()):
        return False

    if not isinstance(prediction["aspects"], list):
        return False

    for aspect in prediction["aspects"]:
        required_aspect_fields = {"feature", "sentiment", "opinion"}
        if not isinstance(aspect, dict):
            return False
        if not required_aspect_fields.issubset(aspect.keys()):
            return False

    return True


def save_json(data: Dict[str, Any], output_path: str) -> None:
    """Save structured predictions for later evaluation or app demos."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
