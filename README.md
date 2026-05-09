# Opinion Mining with Fine-Tuned LLMs

Aspect-based opinion mining from customer reviews using a base instruction-tuned LLM and a LoRA fine-tuned version of the same model.

The system extracts structured information from a review:

- Overall review sentiment
- Product or service aspects mentioned in the review
- Sentiment polarity for each aspect
- Opinion phrases that explain each sentiment

Predictions are returned as valid JSON so they can be evaluated, visualized, and compared in the Streamlit demo.

## Project Overview

This repository implements an end-to-end opinion mining workflow:

1. Prepare SemEval-style review data for instruction tuning.
2. Evaluate the original base LLM before fine-tuning.
3. Fine-tune the same base model with LoRA.
4. Compare base and fine-tuned model predictions.
5. Run a Streamlit app for live base-vs-LoRA inference.

The configured base model is:

The fine-tuned model uses the same base model plus the local LoRA adapter saved in:

## Repository Structure

## Dataset

The project uses an aspect-based sentiment analysis dataset based on SemEval 2014 Task 4. Examples with one or more `conflict` polarity labels were excluded.

The included data is stored in Alpaca-style instruction format with these fields:

- `instruction`
- `input`
- `output`
- `split`
- `domain`
- `sentence_id`

Dataset splits:

| Split | Examples |
| --- | ---: |
| Train | 5,959 |
| Dev | 200 |
| Test | 1,572 |

See `data/README.md` for the dataset card and citation.

## Output Format

The target model output is JSON:

```json
{
        "domain": "electronics",
        "aspects": [
            {
                "term": "battery life",
                "polarity": "positive",
            },
            {
                "term": "screen",
                "polarity": "negative",
            },
        ],
    }

```

Valid structured output makes it easier to score JSON validity, aspect matching, polarity accuracy, precision, recall, and F1.

## Setup

Create a virtual environment:

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on Linux/macOS:

```sh
source .venv/bin/activate
```

Install dependencies:

```sh
pip install -r requirements.txt
```

Model loading uses Hugging Face Transformers and PEFT. A CUDA-enabled environment is recommended for running the Streamlit comparison because it loads both the base model and the LoRA-adapted model.

## Running the Streamlit Demo

Start the app from the repository root:

```sh
streamlit run app/main.py
```

The app:

- Accepts a customer review.
- Runs the base `Qwen/Qwen2.5-1.5B-Instruct` model.
- Runs the same base model with the `lora_model/` adapter.
- Displays both JSON outputs and aspect tables side by side.

If the model files are not already cached locally, the first run may download the base model from Hugging Face.

## Notebook Workflow

Run the notebooks in this order:

| Notebook | Purpose |
| --- | --- |
|`00_data_preprocessing.ipynb` | Clean data and convert it into instruction format |
|`01_Data_Labeling_using_LLM.ipynb` | Generate or refine labels using an LLM-assisted workflow | 
| `02_lora_finetuning.ipynb` | Fine-tune the selected base model using LoRA |
| `03_evaluation_results.ipynb` | Compare model predictions and visualize results |


## Core Modules

`src/model_loader.py`

- Defines the default base model.
- Loads the tokenizer.
- Loads either the original model or the same model with a LoRA adapter.

`src/inference.py`

- Builds the opinion mining prompt.
- Generates model output.
- Extracts and parses the JSON response.


## Project Goal

The main experimental question is whether LoRA fine-tuning improves structured aspect-based sentiment extraction compared with the original base model.

The comparison keeps the base model fixed:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tuned model: `Qwen/Qwen2.5-1.5B-Instruct` plus the trained LoRA adapter in `lora_model/`

