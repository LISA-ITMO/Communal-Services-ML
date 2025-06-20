import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer,
    TrainingArguments,
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification, 
    RobertaTokenizer, 
    AutoTokenizer)

from Data_Processing.create_converters import Converter
from Data_Processing.get_stats import compute_metrics_top_k


MODEL_NAME = "sberbank-ai/ruRoberta-large"


class BertModel:
    '''
    A wrapper class for a transformer-based classifier that predicts detailed topics
    from text inputs using a pretrained Hugging Face model.
    '''

    def __init__(self, repo_name, converter: Converter, model_name=MODEL_NAME, eval=True, probability_threshold=0.1):
        '''
        Initializes the model with tokenizer, converter, and optional probability threshold.

                Parameters:
                        repo_name (str): Hugging Face model repository name
                        converter (Converter): A converter class to map between labels and IDs
                        probability_threshold (float): Minimum probability to accept a prediction
        '''
        self.repo_name = repo_name
        self.model_name = model_name
        self.probability_threshold = probability_threshold


        self.is_reducing = converter.is_reduced

        self.converter = converter

        if not eval:
            self.train_model()

        self.load_model()

    def train_model(self):
        '''
        Fine-tunes a Roberta-based classifier on labeled appeal texts and pushes the model to Hugging Face Hub.

                Steps:
                    - Converts detailed topics to numeric labels using the converter
                    - Splits data into training and validation sets
                    - Encodes text data with RobertaTokenizer
                    - Defines a custom Dataset class for PyTorch
                    - Initializes the RobertaForSequenceClassification model
                    - Configures Hugging Face Trainer with training arguments
                    - Trains the model and evaluates on validation data
                    - Pushes tokenizer and model to the Hugging Face Hub

                Returns:
                    None (prints evaluation metrics to stdout)
        '''
        # topic_id -> sequential label index
        data = self.converter.df
        data["label"] = data["detailed_topic"].map(self.converter.get_id)

        # Split data into training & validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            data["appeal"].tolist(), data["label"].tolist(), test_size=0.2, random_state=42
        )
        num_labels = self.converter.get_num_labels()

        # Load tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(self.model_name)

        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

        class AppealDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        # PyTorch datasets
        train_dataset = AppealDataset(train_encodings, train_labels)
        val_dataset = AppealDataset(val_encodings, val_labels)

        model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

        repo_dir = self.repo_name[11:]

        training_args = TrainingArguments(
            output_dir=repo_dir,
            push_to_hub=True,
            hub_model_id=self.repo_name,
            eval_strategy="epoch",  # Evaluate only at the end of each epoch
            num_train_epochs=4,  # Total number of training epochs
            per_device_train_batch_size=8,  # Increase if GPU memory allows
            per_device_eval_batch_size=8,
            weight_decay=0.01,
            logging_dir="./logs",
            fp16=True,  # Mixed precision for speedup
            gradient_accumulation_steps=2,  # Simulates a larger batch size
            learning_rate=2e-5,  # Fine-tuned learning rate for stability
            warmup_ratio=0.1,  # Gradual learning rate increase at the start
            save_strategy="epoch",  # Save model at the end of each epoch
            save_total_limit=2,  # Keep only 2 most recent checkpoints
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_top_k,
        )

        trainer.train()

        df_stat = pd.DataFrame.from_dict(trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val"), orient='index')
        
        tokenizer.push_to_hub(self.repo_name)

        print(df_stat)


    def load_model(self):
        '''
        Loads the transformer model and tokenizer from Hugging Face repository.
        '''
        self.model = AutoModelForSequenceClassification.from_pretrained(self.repo_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_name)

    def predict_topics_batch(self, appeal_texts, k=5, batch_size=16):
        '''
        Predicts top-k detailed topic classes for a batch of texts.

                Parameters:
                        appeal_texts (list[str]): A list of input strings (appeals)
                        k (int): Number of top predictions to return
                        batch_size (int): Number of samples per batch

                Returns:
                        all_top_k_classes (list[list[int]]): Top-k class indices per sample
                        all_top_k_probs (list[list[float]]): Probabilities for each top-k class
        '''
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        all_top_k_classes, all_top_k_probs = [], []

        for i in range(0, len(appeal_texts), batch_size):
            batch_texts = appeal_texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

            # Get top-k predictions
            top_k_indices = np.argsort(probs, axis=1)[:, ::-1][:, :k]
            top_k_probs = np.take_along_axis(probs, top_k_indices, axis=1)

            all_top_k_classes.extend(top_k_indices.tolist())
            all_top_k_probs.extend(top_k_probs.tolist())

        return all_top_k_classes, all_top_k_probs

    def predict(self, texts, k=3):
        '''
        Converts prediction indices into human-readable labels with probability filtering.

                Parameters:
                        texts (list[str]): A list of input texts
                        k (int): Number of top predictions to return

                Returns:
                        res (list[list[tuple[str, float]]]): List of predicted labels with probabilities
        '''
        response, probs = self.predict_topics_batch(texts, k=k)
        res = []
        for i, pred_k in enumerate(response):
            res.append([])
            for j, pred_class in enumerate(pred_k):
                detailed_topic = self.converter.get_label(pred_class)
                probability = probs[i][j]
                if probability > self.probability_threshold:
                    res[-1].append((detailed_topic, probability))
                else:
                    break
        return res

    def print_prediction(self, texts, k=3):
        '''
        Prints predictions with probabilities for each input text.

                Parameters:
                        texts (list[str]): A list of input texts
                        k (int): Number of top predictions to return
        '''
        predictions = self.predict(texts, k=k)
        for i, pred_k in enumerate(predictions):
            print(f"{i+1}.", texts[i][:100] + '...')
            for _, (detailed_topic, probability) in enumerate(pred_k):
                print(f"\t Probability: {probability:.4f}")
                print(f"\t Predicted Detailed Topic: {detailed_topic}.")
                print()