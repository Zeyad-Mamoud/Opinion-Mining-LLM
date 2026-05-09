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

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The fine-tuned model uses the same base model plus the local LoRA adapter saved in:

```text
lora_model/
```

## Repository Structure

```text
.
|-- app/
|   `-- main.py                         # Streamlit live comparison app
|-- data/
|   |-- README.md                       # Dataset card and citation
|   |-- train-00000-of-00001.parquet
|   |-- dev-00000-of-00001.parquet
|   |-- test-00000-of-00001.parquet
|   |-- Labeled-data/                   # LLM-assisted labeled splits
|   |-- LLM responses for evaluation/   # Saved model responses
|   `-- processed/                      # Cleaned and instruction-format data
|-- lora_model/
|   |-- adapter_config.json
|   |-- adapter_model.safetensors
|   |-- tokenizer.json
|   `-- tokenizer_config.json
|-- notebooks/
|   |-- 00_baseline_evaluation.ipynb
|   |-- 01_Data_Labeling_using_LLM.ipynb
|   |-- 01_data_preprocessing.ipynb
|   |-- 02_lora_finetuning.ipynb
|   `-- 03_evaluation_results.ipynb
|-- results/
|   |-- baseline_evaluation.csv
|-- src/
|   |-- inference.py                    # Prompting, generation, JSON parsing
|   |-- model_loader.py                 # Base model and LoRA adapter loading
|   `-- utils.py                        # Shared schema and validation helpers
|-- requirements.txt
`-- README.md
```

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

Valid structured output makes it easier to score JSON validity, aspect matching, polarity accuracy, precision, recall, and F1.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Model loading uses Hugging Face Transformers and PEFT. A CUDA-enabled environment is recommended for running the Streamlit comparison because it loads both the base model and the LoRA-adapted model.

## Running the Streamlit Demo

Start the app from the repository root:

```bash
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
| `00_baseline_evaluation.ipynb` | Evaluate the original base model before fine-tuning |
| `01_Data_Labeling_using_LLM.ipynb` | Generate or refine labels using an LLM-assisted workflow |
| `01_data_preprocessing.ipynb` | Clean data and convert it into instruction format |
| `02_lora_finetuning.ipynb` | Fine-tune the selected base model using LoRA |
| `03_evaluation_results.ipynb` | Compare model predictions and visualize results |

## Baseline Results

The saved baseline run in `results/baseline_evaluation.csv` evaluates `Qwen/Qwen2.5-1.5B-Instruct` on 20 samples:

| Metric | Value |
| --- | ---: |
| JSON validity rate | 1.00 |
| Average precision | 0.75 |
| Average recall | 0.75 |
| F1 score | 0.75 |
| Polarity accuracy | 1.00 |
| Matched aspects | 15 |
| Ground-truth aspects | 20 |
| Predicted aspects | 20 |

These results are the baseline comparison point for the LoRA fine-tuned model.

## Core Modules

`src/model_loader.py`

- Defines the default base model.
- Loads the tokenizer.
- Loads either the original model or the same model with a LoRA adapter.

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

The comparison keeps the base model fixed:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tuned model: `Qwen/Qwen2.5-1.5B-Instruct` plus the trained LoRA adapter in `lora_model/`

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
