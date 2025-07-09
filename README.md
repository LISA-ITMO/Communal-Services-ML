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
| BERT Base          |   66.9%    |    85.5%   |  90.5%     |  64.2%   |   92.5% |
| RuRoBERTa            |   69.9%    |    89.6%   |  95.0%     |  68.5%   |   93.9% |
| Naive Bayes        |   47.0%    |    71.4%   |  75.4%     |  36.4%   |   76.9% |
| SVM                |   65.1%    |    82.8%   |  88.2%     |  63.6%   |   90.6% |
| Decision Tree      |   46.5%    |    54.2%   |  55.0%     |  43.4%   |   70.9% |
| FastText           |   15.0%    |    26.2%   |  34.3%     |  10.3%   |   62.7% |
| KNN                |   45.24%   |    71.8%   |  80.2%     |  44.2%   |   -     |

## Example Usage (from code)

```python
from app.main import get_classifier

texts = ["There’s been no hot water in our building for a week", ...]
clf = get_classifier()
basic_classifier.print_prediction(texts, k=5)
```

## Example Service
The [GitLab project](https://gitlab.com/lisa-itmo/auto-replier) includes a simple interface for testing the model's predictions on individual user-submitted appeals.

![image](https://github.com/user-attachments/assets/63882903-e7e8-4ca4-bf96-4f7d20459265)

### How It Works
- The user enters an appeal in free form (as text) into the input field.

- The model processes the text and returns top-5 predicted categories with their corresponding probabilities.

- The predictions are displayed in a table that includes:

   - Category name (label predicted by the model)

   - Probability (how confident the model is in each label)

## Publications

1. Skvortsov D.A., Lemanov A.A. (supervised by Fedorov D.A.)
Comparison of Approaches to Improve the Accuracy of Complaint Classification Algorithms Using Machine Learning Methods
// Proceedings of the Congress of Young Scientists. Electronic Edition. – St. Petersburg: ITMO University, [2025].
https://kmu.itmo.ru/digests/article/15916

2. Lemanov A.A., Skvortsov D.A. (supervised by Fedorov D.A.)
Exploring the RAG Approach in Intelligent Question-Answering Systems
// Proceedings of the Congress of Young Scientists. Electronic Edition. – St. Petersburg: ITMO University, [2025].
https://kmu.itmo.ru/digests/article/16001

3. Lemanov A.A., Skvortsov D.A. (supervised by Fedorov D.A.)
Designing Agent-Based Dialogue Systems Using RAG and LLM Technologies for Text2SQL Tasks on Government Databases
// Proceedings of the Congress of Young Scientists. Electronic Edition. – St. Petersburg: ITMO University, [2025].
https://kmu.itmo.ru/digests/article/14107

## Публикации

1. Скворцов Д.А., Леманов А.А. (науч. рук. Федоров Д.А.) Сравнение подходов для повышения точности алгоритмов классификации жалоб на основе методов машинного обучения // Сборник тезисов докладов конгресса молодых ученых. Электронное издание. – СПб: Университет ИТМО, [2025]. URL: https://kmu.itmo.ru/digests/article/15916
2. Леманов А.А., Скворцов Д.А. (науч. рук. Федоров Д.А.) Исследование подхода RAG в интеллектуальных вопросно-ответных системах // Сборник тезисов докладов конгресса молодых ученых. Электронное издание. – СПб: Университет ИТМО, [2025]. URL: https://kmu.itmo.ru/digests/article/16001
3. Леманов А.А., Скворцов Д.А. (науч. рук. Федоров Д.А.) Проектирование агентных диалоговых систем на примере государственных баз данных с применением технологии rag и llm в рамках задачи text2sql. Электронное издание. – СПб: Университет ИТМО, [2025]. URL: https://kmu.itmo.ru/digests/article/14107

## Exhibition & Media
This project was presented at the ITMO Scientific Developments Exhibition, showcasing innovations in machine learning for public service applications.

![image](https://github.com/user-attachments/assets/f8a93a35-1bd2-4f27-a8d1-9cef668f3c43)
![image](https://github.com/user-attachments/assets/c15ca0f3-9117-4749-9e66-a9348e92e6d6)
[Exhibition Coverage (Telegram)](https://t.me/lisaitmo/165)
