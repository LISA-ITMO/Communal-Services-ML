# Application of Machine Learning Methods for Classification of Appeals in the Field of Housing and Communal Inspection

This repository explores the application of both transformer-based and classical machine learning methods for automatic classification of citizen appeals. The focus is on appeals within the domain of housing and communal services, aiming to route them accurately by identifying their **detailed topics**.

The project integrates modern NLP pipelines based on BERT (via Hugging Face Transformers) alongside traditional multi-label classifiers. It includes tools for data preprocessing, model training, evaluation, and prediction, with support for handling imbalanced datasets through label reduction. It has both, full pipeline training and lightweight usage via precomputed model outputs.

## Repository Structure

```
project-root/
│
├── database/                   # Dataset storage
│   ├── raw_data.csv            # Original unprocessed dataset
│   └── clean_data.csv          # Cleaned and label-reduced dataset
│
├── assets/                     # Precomputed outputs and converters
│   ├── converter.json          # Label ↔ ID mappings used by models
│   ├── classic_ml_outputs.pkl  # Evaluation and predictions from classic models
│   └── transformer_outputs.pkl # Evaluation and predictions from transformer models
│
├── preprocessing/
│   └── preprocess.py           # Converts raw data into usable format
│
├── model pipelines/
│   ├── transformers_gpu.ipynb          # Transformer training and evaluation
│   ├── classic_ml_fasttext.ipynb       # Classic ML and FastText training
│   └── train_knn_only.ipynb            # (WIP) KNN-only training notebook
│
├── analysis/
│   └── results_analysis.ipynb          # Visualization and performance metrics
│
├── app/
│   └── main.py                         # Quick demo: load assets, predict single text
│
└── README.md
```

## How to Run

1. **Create a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Usage (Assets Already Ready)

With `assets/*.pkl` already available:

1. Analytics Notebook  
   Run full comparison and visualization in **Google Colab** *(CPU runtime setting)* or from command line:

   ```bash
   jupyter notebook analysis/results_analysis.ipynb
   ```

3. Fast Inference Demo (Console):

   ```bash
   python app/main.py
   ```

   Paste a text and receive the model’s prediction (one at a time).

## Full Pipeline – Step-by-Step

### 1. Load your dataset

Format example (`raw_data.csv` in `database/`):

```
topic_id;president_topic;detailed_topic;appeal
0;heating;central heating;The central heating doesn't work in the region
```

### 2. Run Preprocessing

```bash
python preprocessing/preprocess.py
```

This generates:
- `database/clean_data.csv` → cleaned, label-reduced dataset  
- `assets/converter.json` → contains mappings like:

```json
{
  "label2id": {"central heating": 0, "...": "..."},
  "id2label": {"0": "central heating", "...": "..."},

  "label2id_reduced": {"central heating": 0, "...": "..."},
  "id2label_reduced": {"0": "central heating", "...": "..."}
}
```

### 3. Train Models (via Notebooks)

Best run in **Google Colab** *(T4 GPU for transformers, CPU for others)*

#### 3.1 Transformers (2 models)

```bash
jupyter notebook models/transformers_gpu.ipynb
```

- Loads `clean_data.csv` and `converter.json`
- Trains 2 transformer models (change `repo_prefix` for your HF account)
- Pushes to HF Hub
- Evaluates and stores predictions/stats to:
  - `assets/transformer_outputs.pkl`

Format:
```python
{
  "bert_base": {
        "predictions": [...],
        "basic_stats": {...},
        "val_labels": [...]
    },
    "ruroberta": {
        "predictions": [...],
        "basic_stats": {...},
    }
}
```

#### 3.2 Classic ML + FastText

```bash
jupyter notebook models/classic_ml_fasttext.ipynb
```

- Loads `clean_data.csv` and `converter.json`
- Trains 4 models: Naive Bayes, Decision Tree, SVM, FastText
- Saves:
  - Trained models → `assets/models/*.joblib` or `.bin` (not included in repo)
  - Evaluation results → `assets/classic_ml_outputs.pkl`

### 4. Analyze Results

```bash
jupyter notebook analysis/results_analysis.ipynb
```

- Loads:
  - `converter.json`
  - `transformer_outputs.pkl`
  - (Optionally) `classic_ml_outputs.pkl`, `KNN_preds.csv`

Shows:
- Top-1/3/5 Accuracy, F1, ROC-AUC
- Per-class top-k breakdown (Transformers)
- Transformer learning curves
- Side-by-side comparison of all models

### Run Tests

```bash
python -m unittest discover -s Tests
```

## Results

| Model Name         | Accuracy@1 | Accuracy@3 | Accuracy@5 | F1 Score | ROC-AUC |
|--------------------|------------|------------|------------|----------|---------|
| BERT Base          |            |            |            |          |         |
| RoBERTa            |            |            |            |          |         |
| Naive Bayes        |            |            |            |          |         |
| SVM                |            |            |            |          |         |
| Decision Tree      |            |            |            |          |         |
| FastText           |            |            |            |          |         |
| KNN (optional)     |            |            |            |          |         |

## Example Usage (from code)

```python
from app.main import get_classifier

texts = ["There’s been no hot water in our building for a week", ...]
clf = get_classifier()
basic_classifier.print_prediction(texts, k=5)
```