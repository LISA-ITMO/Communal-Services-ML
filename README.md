# Application of Machine Learning Methods for Classification of Appeals in the Field of Housing and Communal Inspection

This repository explores the application of both transformer-based and classical machine learning methods for automatic classification of citizen appeals. The focus is on appeals within the domain of housing and communal services, aiming to route them accurately by identifying their **detailed topics**.

The project integrates modern NLP pipelines based on BERT (via Hugging Face Transformers) alongside traditional multi-label classifiers. It includes tools for data preprocessing, model training, evaluation, and prediction, with support for handling imbalanced datasets through label reduction.

---

## Repository Structure

```
Data_Processing/
│
├── Database/db_2.csv          # Input dataset (appeal; topic_id; detailed_topic; etc.)
├── create_converters.py       # Builds label converters
├── get_stats.py               # Computes topic distribution stats (used in training/evaluation)
├── load_data.py               # Loads appeal data from PostgreSQL
│
Models/
├── bert_pipeline.py           # Transformer-based classification (Model class)
├── classic_pipeline.py        # Classical ML pipelines (ClassicModels class)
│
Tests/
├── test_texts.py              # Dictionary with example test cases
├── tests.py                   # Unit tests for model logic
│
main.py                        # Main entry point for running predictions
```

---

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

3. **Prepare your dataset:**

Place the CSV file (e.g., `db_2.csv`) into `Data_Processing/Database/`. It should have the following columns:

| topic_id | president_topic | detailed_topic     | appeal                            |
|----------|------------------|---------------------|------------------------------------|
| 102      | Social           | Infrastructure Issue| "The pavement near our house..."   |

4. **Run the model:**
Edit and execute `main.py` to load the model and predict topics for test appeals:
```bash
python main.py
```

---

## Available Components

### Transformer-Based Classification

- File: `Models/bert_pipeline.py`
- Class: `Model`
- Features:
  - Fine-tunes a pre-trained RoBERTa model
  - Pushes trained model to Hugging Face Hub
  - Predicts top-k detailed topics with probability thresholding

### Classical ML Models

- File: `Models/classic_pipeline.py`
- Class: `ClassicModels`
- Models included:
  - Naive Bayes
  - Decision Tree
  - Support Vector Machine (SVC)
  - k-Nearest Neighbors (MLkNN)

These models use TF-IDF vectorization and optional SVD for dimensionality reduction.

---

## Label Converter

- File: `Data_Processing/create_converters.py`
- Class: `Converter`
- Features:
  - Converts topic names to numerical IDs and vice versa
  - Optional label reduction by frequency threshold
  - Handles mappings consistently between training and prediction

---

## Example Usage

```python
from Models.bert_pipeline import Model
from Data_Processing.create_converters import Converter

converter = Converter(db_name="db_2.csv", reducing_threshold=10)
model = Model("Goshective/lab_comm_services_detailed_sber", converter)
model.print_prediction(["The heating hasn't worked for days..."], k=5)
```

---

## Tests

Run included unit tests using:

```bash
python -m unittest discover -s Tests
```

This validates label mappings and prediction logic using provided test cases in `test_texts.py`.

---