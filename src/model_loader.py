"""Helpers for loading the base LLM and its LoRA adapter.

This module supports the project rule that the same base model must be used
before and after fine-tuning so the comparison stays fair.
"""

from __future__ import annotations

from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def load_tokenizer(base_model_name: str = DEFAULT_BASE_MODEL):
    """Load the tokenizer for the selected base model."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    base_model_name: str = DEFAULT_BASE_MODEL,
    adapter_path: str | None = None,
    device_map: str = "auto",
):
    """Load the original base model or the same base model with a LoRA adapter."""

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        torch_dtype="auto",
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    return model


def resolve_adapter_path(adapter_path: str | None) -> str | None:
    """Normalize the adapter path so the app and notebooks can reuse it safely."""
    if not adapter_path:
        return None

    resolved = Path(adapter_path).expanduser().resolve()
    return str(resolved)
