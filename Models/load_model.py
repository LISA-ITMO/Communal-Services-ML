import os
import sys
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Data_Processing.create_converters import Converter


class Model:
    def __init__(self, repo_name, converter: Converter, probability_threshold=0.1):
        self.repo_name = repo_name
        self.probability_threshold = probability_threshold

        self.is_reducing = converter.is_reduced

        self.converter = converter

        self.load_model()

    def load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.repo_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_name)

    def predict_topics_batch(self, appeal_texts, k=5, batch_size=16):
        """
        Predicts top-k classes using a single detailed classifier.
        """
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
        predictions = self.predict(texts, k=k)
        for i, pred_k in enumerate(predictions):
            print(f"{i+1}.", texts[i][:100] + '...')
            for j, (detailed_topic, probability) in enumerate(pred_k):
                print(f"\t Probability: {probability:.4f}")
                print(f"\t Predicted Detailed Topic: {detailed_topic}.")
                print()