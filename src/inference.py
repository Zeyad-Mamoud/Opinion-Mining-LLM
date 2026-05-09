"""Inference helpers for structured opinion mining output.

The final system should extract sentiment and product features from customer
reviews and return them as reliable JSON instead of free-form text.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Dict

import torch


def build_review_prompt(text: str) -> str:
    """Create prompt for domain and aspect-based sentiment extraction."""

    return "\n".join([
        "You extract structured information from text and return only valid JSON.",
        "",
        "No explanation No introduction No conclusion",
        "",
        "Extract the domain and all aspect terms with their sentiment polarity.",
        "- Domains: electronics, restaurants, movies, books, software, general",
        "- Polarity: positive, negative, neutral",
        "- Extract ALL aspects mentioned in the text",
        "",
        "input :",
        text,
        "",
        "output format:",
        '{"domain":"...","aspects":[{"term":"...","polarity":"..."}, ...]}'
    ])

def extract_json_block(text: str) -> str:
    """Extract the last valid JSON object from a model response."""
    # cleaned = text.strip()
    # if cleaned.startswith("```"):
    #     cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
    # if cleaned.endswith("```"):
    #     cleaned = cleaned.removesuffix("```").strip()

    decoder = json.JSONDecoder()
    candidates: list[tuple[dict[str, Any], str]] = []

    for index, char in enumerate(text):
        if char != "{":
            continue

        try:
            parsed, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            candidates.append((parsed, text[index : index + end]))

    if not candidates:
        raise ValueError("No JSON object found in model output.")

    for parsed, raw_json in reversed(candidates):
        if "aspects" in parsed:
            return raw_json

    return candidates[-1][1]


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse model output into a Python dictionary."""
    return json.loads(extract_json_block(text))


def _move_to_device(inputs: Any, device: torch.device | str) -> Any:
    """Move tokenizer outputs to the same device as the model."""
    if hasattr(inputs, "to"):
        return inputs.to(device)

    if _has_input_ids(inputs):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    return inputs


def _build_model_inputs(tokenizer, prompt: str, device: torch.device | str) -> Any:
    """Build model inputs using the chat template when the tokenizer supports it."""
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    return _move_to_device(inputs, device)


def _has_input_ids(inputs: Any) -> bool:
    """Return whether an object behaves like tokenizer keyword inputs."""
    if not hasattr(inputs, "keys") or not hasattr(inputs, "__getitem__"):
        return False
    return "input_ids" in inputs.keys()


def _as_generation_kwargs(inputs: Any) -> dict[str, Any]:
    """Convert tokenizer mapping objects into kwargs for model.generate."""
    if isinstance(inputs, Mapping):
        return dict(inputs)
    if _has_input_ids(inputs):
        return {key: inputs[key] for key in inputs.keys()}
    raise TypeError(f"Unsupported tokenizer input type: {type(inputs).__name__}")


def _input_token_count(inputs: Any) -> int:
    """Return the prompt length so only generated tokens are decoded."""
    if _has_input_ids(inputs):
        return inputs["input_ids"].shape[-1]
    return inputs.shape[-1]


def generate_structured_output(
    model,
    tokenizer,
    review_text: str,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Run generation and return the parsed JSON prediction."""
    prompt = build_review_prompt(review_text)
    inputs = _build_model_inputs(tokenizer, prompt, model.device)

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        if _has_input_ids(inputs):
            output_ids = model.generate(
                **_as_generation_kwargs(inputs),
                **generation_args,
            )
        else:
            output_ids = model.generate(inputs, **generation_args)

    prompt_length = _input_token_count(inputs)
    generated_only = output_ids[0][prompt_length:]
    generated_text = tokenizer.decode(generated_only, skip_special_tokens=True)
    return parse_json_response(generated_text)
