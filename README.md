# BESSTIE Sarcasm Detection - COMM061 NLP Coursework

University of Surrey.

Cross-variety sarcasm detection on the [BESSTIE-CW-26 dataset](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26), comparing two fine-tuning approaches across three English varieties (en-UK, en-AU, en-IN):

- **RoBERTa-base** with full fine-tuning
- **TinyLlama-1.1B** with LoRA (parameter-efficient fine-tuning)

A live demo is deployed as a Gradio app on Hugging Face Spaces (see `space_app/`).

## What's in here

| Path | What it is |
|------|------------|
| `main.ipynb` | Full implementation - the notebook runs top to bottom |
| `space_app/` | Gradio demo app deployed to Hugging Face Spaces |
| `outputs/` | Generated figures, confusion matrices, and results CSVs |
| `outputs/lime_explanations/` | LIME interpretability outputs for selected predictions |
| `requirements.txt` | Dependencies for running the notebook |

## Tasks covered

- **Q1** — Dataset analysis: label distributions, vocabulary overlap (Jaccard + TF-IDF cosine), and POS analysis across varieties
- **Q2.1** — Baseline vs PTLM: TF-IDF with LR/SVM against RoBERTa-base, multi-seed
- **Q2.2** — Cross-variety evaluation matrix with RoBERTa-base, plus LIME interpretability
- **Q2.3** — LoRA fine-tuning on TinyLlama-1.1B for cross-variety transfer, multi-seed
- **Q3** — Consolidated evaluation across
