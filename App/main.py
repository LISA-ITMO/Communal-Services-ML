import os
import numpy as np
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Preprocessing.preprocess import preprocess_main


class BertModel:
    '''
    A wrapper class for a transformer-based classifier that predicts detailed topics
    from text inputs using a pretrained Hugging Face model.
    '''
    def __init__(self, repo_name, converter, probability_threshold=0.1):
        '''
        Initializes the model with tokenizer, converter, and optional probability threshold.

                Parameters:
                        repo_name (str): Hugging Face model repository name
                        converter (dict): A converter to map between labels and IDs
                        probability_threshold (float): Minimum probability to accept a prediction
        '''
        
        self.repo_name = repo_name
        self.label2id = {k: int(v) for k, v in converter["label2id_reduced"].items()}
        self.id2label = {int(k): v for k, v in converter["id2label_reduced"].items()}
        self.probability_threshold = probability_threshold

        self.load_model()

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
                detailed_topic = self.id2label[pred_class]
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


def get_classifier():
    '''
    The intermediate function to initialize the classifier pipeline.
    '''
    NEW_DB_NAME = "clean_data.csv"
    REPO_PREFIX = "Goshective"
    REPO_DIR = "lab_comm_services_detailed_sber"
    REPO_NAME = f"{REPO_PREFIX}/{REPO_DIR}"
    CONVERTER_PATH = 'Assets/converter.json'

    if os.path.exists(CONVERTER_PATH):
        with open(CONVERTER_PATH, 'r', encoding='utf-8') as f:
            converter = json.load(f)
    else:
        preprocess_main(NEW_DB_NAME)

    return BertModel(REPO_NAME, converter)



def main():
    '''
    The main function to print predictions on given input texts
    '''
    basic_classifier = get_classifier()
    while True:
        print("Write Text:")
        print("(Enter - Exit)")
        print()
        
        inp = input()

        if inp != '':
            texts = [inp]
            basic_classifier.print_prediction(texts, k=5)
        else:
            print("End of program.")
            break
        print()
    
    return


if __name__ == "__main__":
    print("Classification of Appeals in the Field of Housing and Communal Inspection.")
    print("--------------------------------------------------------------------------")
    main()