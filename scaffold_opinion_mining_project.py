"""Create the initial folder structure for the Opinion Mining project.

This script is safe to rerun because it only creates missing files and folders.
It exists to keep the workspace aligned with the course brief.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


README_CONTENT = """# Opinion Mining from Customer Reviews

## 1. Project Idea
This project focuses on opinion mining from customer reviews. The task is to extract sentiment and product features from reviews using a fine-tuned LLM and return the result as structured JSON.

## 2. Dataset Information
Store raw datasets in `data/raw/` and cleaned datasets in `data/processed/`.

## 3. Base Model Information
Choose one base model such as Llama 3 or Mistral and keep the same base model for fine-tuning, evaluation, and comparison.

## 4. Brief Explanation of Fine-Tuning with LoRA
LoRA is used to adapt the chosen base model efficiently without updating all model parameters.

## 5. Evaluation Methods
Compare the base model and the LoRA-tuned model using metrics such as precision, recall, F1-score, and JSON validity.

## 6. Project Results and Plots
Save visual results in `results/plots/` and compare base vs. tuned model behavior.

## 7. Structured Output
The model should return structured outputs for sentiment and product features, preferably in JSON format.
"""


REQUIREMENTS_CONTENT = """transformers
peft
bitsandbytes
datasets
accelerate
streamlit
pandas
matplotlib
plotly
scikit-learn
"""


MODEL_LOADER_CONTENT = '''"""Helpers for loading the base LLM and its LoRA adapter."""'''

INFERENCE_CONTENT = '''"""Inference helpers for structured opinion mining output."""'''

UTILS_CONTENT = '''"""Shared utilities for the opinion mining pipeline."""'''

APP_CONTENT = '''"""Streamlit demo for the opinion mining project."""'''

NOTEBOOK_PLACEHOLDER = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def write_if_missing(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def write_notebook_if_missing(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(NOTEBOOK_PLACEHOLDER, indent=2), encoding="utf-8")


def main() -> None:
    # Data folders support raw review collection and processed splits for the NLU workflow.
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    # Notebooks separate preprocessing, LoRA fine-tuning, and final evaluation.
    write_notebook_if_missing(ROOT / "notebooks" / "01_data_preprocessing.ipynb")
    write_notebook_if_missing(ROOT / "notebooks" / "02_lora_finetuning.ipynb")
    write_notebook_if_missing(ROOT / "notebooks" / "03_evaluation_results.ipynb")

    # Source modules hold reusable code for model loading, inference, and utilities.
    write_if_missing(ROOT / "src" / "model_loader.py", MODEL_LOADER_CONTENT)
    write_if_missing(ROOT / "src" / "inference.py", INFERENCE_CONTENT)
    write_if_missing(ROOT / "src" / "utils.py", UTILS_CONTENT)

    # The Streamlit app is used to demonstrate structured outputs live.
    write_if_missing(ROOT / "app" / "main.py", APP_CONTENT)

    # Results storage keeps evaluation charts and comparative plots organized.
    (ROOT / "results" / "plots").mkdir(parents=True, exist_ok=True)

    # Root files document the project and list required libraries.
    write_if_missing(ROOT / "README.md", README_CONTENT)
    write_if_missing(ROOT / "requirements.txt", REQUIREMENTS_CONTENT)

    print("Opinion mining project scaffold is ready.")


if __name__ == "__main__":
    main()
