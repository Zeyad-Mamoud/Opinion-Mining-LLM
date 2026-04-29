# Opinion Mining LLM

Aspect-based opinion mining from customer reviews using a base instruction-tuned LLM and a LoRA fine-tuned version of the same model.

The project extracts structured information from reviews:

- Overall review sentiment
- Product or service aspects mentioned in the review
- Sentiment polarity for each aspect
- Opinion phrases that explain each sentiment

The target output is valid JSON so predictions can be evaluated, visualized, and used in a simple demo app.

## Project Overview

This repository follows a complete opinion mining workflow:

1. Label and prepare review data for instruction tuning.
2. Evaluate the base LLM before fine-tuning.
3. Fine-tune the same base model with LoRA.
4. Compare base and fine-tuned predictions.
5. Present the extraction task through a Streamlit demo.

The current base model configured in the code is:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The model loading logic is implemented in `src/model_loader.py`, and the prompt-generation and JSON-parsing helpers are implemented in `src/inference.py`.

## Repository Structure

```text
.
|-- app/
|   `-- main.py                         # Streamlit demo scaffold
|-- data/
|   |-- README.md                       # Dataset card and citation
|   |-- train-00000-of-00001.parquet
|   |-- dev-00000-of-00001.parquet
|   |-- test-00000-of-00001.parquet
|   |-- Labeled-data/                   # LLM-assisted labeled splits
|   |-- LLM responses for evaluation/   # Saved model responses
|   `-- processed/                      # Cleaned and instruction-format data
|-- notebooks/
|   |-- 00_baseline_evaluation.ipynb
|   |-- 01_Data_Labeling_using_LLM.ipynb
|   |-- 01_data_preprocessing.ipynb
|   |-- 02_lora_finetuning.ipynb
|   `-- 03_evaluation_results.ipynb
|-- results/
|   |-- baseline_evaluation.csv
|   |-- baseline_metrics.json
|   `-- plots/
|       `-- baseline_evaluation_plots.png
|-- src/
|   |-- inference.py                    # Prompting, generation, JSON parsing
|   |-- model_loader.py                 # Base model and LoRA adapter loading
|   `-- utils.py                        # Shared schema and validation helpers
|-- requirements.txt
`-- README.md
```

## Dataset

The project uses an aspect-based sentiment analysis dataset based on SemEval 2014 Task 4.

The included dataset contains review examples in Alpaca-style instruction format with these main fields:

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

Examples containing the `conflict` polarity were excluded, as noted in `data/README.md`.

## Output Format

The system is designed to return JSON in the following structure:

```json
{
  "review": "The battery life is great but the screen is too dim.",
  "overall_sentiment": "mixed",
  "aspects": [
    {
      "feature": "battery life",
      "sentiment": "positive",
      "opinion": "great"
    },
    {
      "feature": "screen",
      "sentiment": "negative",
      "opinion": "too dim"
    }
  ]
}
```

This format makes model outputs easier to validate, compare, and visualize.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you plan to load quantized models with `bitsandbytes`, use a compatible CUDA-enabled environment.

## Running the Demo

Start the Streamlit app:

```bash
streamlit run app/main.py
```

The current app shows the review input flow and expected structured output. It is prepared to be connected to `src/model_loader.py` and `src/inference.py` for real model predictions.

## Notebook Workflow

Run the notebooks in this order:

| Notebook | Purpose |
| --- | --- |
| `00_baseline_evaluation.ipynb` | Evaluate the original base model before fine-tuning |
| `01_Data_Labeling_using_LLM.ipynb` | Generate or refine labels using an LLM-assisted workflow |
| `01_data_preprocessing.ipynb` | Clean data and convert it into instruction format |
| `02_lora_finetuning.ipynb` | Fine-tune the selected base model using LoRA |
| `03_evaluation_results.ipynb` | Compare model predictions and visualize results |

## Baseline Results

The saved baseline run in `results/baseline_metrics.json` evaluates `Qwen/Qwen2.5-1.5B-Instruct` on 20 samples:

| Metric | Value |
| --- | ---: |
| JSON validity rate | 0.15 |
| Average precision | 0.00 |
| Average recall | 0.00 |
| F1 score | 0.00 |
| Polarity accuracy | 0.00 |
| Matched aspects | 0 |
| Ground-truth aspects | 6 |
| Predicted aspects | 10 |

These baseline results provide the comparison point for the LoRA fine-tuned model.

## Core Modules

`src/model_loader.py`

- Defines the default base model.
- Loads the tokenizer.
- Loads either the original model or the same model with a LoRA adapter.
- Supports optional 4-bit quantization.

`src/inference.py`

- Builds the opinion mining prompt.
- Generates model output.
- Extracts and parses the JSON response.

`src/utils.py`

- Provides a reference output schema.
- Validates prediction structure.
- Saves JSON outputs.

## Project Goal

The main experimental question is whether LoRA fine-tuning improves structured aspect-based sentiment extraction compared with the original base model.

The comparison should use the same base model in both cases:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tuned model: `Qwen/Qwen2.5-1.5B-Instruct` plus the trained LoRA adapter

## Citation

The dataset is based on SemEval 2014 Task 4:

```bibtex
@inproceedings{pontiki_semeval-2014_2014,
  title = {{SemEval}-2014 {Task} 4: {Aspect} {Based} {Sentiment} {Analysis}},
  doi = {10.3115/v1/S14-2004},
  booktitle = {Proceedings of the 8th {International} {Workshop} on {Semantic} {Evaluation} ({SemEval} 2014)},
  publisher = {Association for Computational Linguistics},
  author = {Pontiki, Maria and Galanis, Dimitris and Pavlopoulos, John and Papageorgiou, Harris and Androutsopoulos, Ion and Manandhar, Suresh},
  year = {2014},
  pages = {27--35}
}
```
