import os
from Data_Processing.create_converters import Converter
from Models.load_model import Model
from Data_Processing.texts import texts

def read_texts(texts_name):
    PATH = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(PATH, 'Data_Processing', texts_name), 'r') as f:
        texts = f.readlines()
    return texts


if __name__ == "__main__":
    DB_NAME = "db_2.csv"
    REPO_AUTHOR = "Goshective"
    REPO_DIR = "lab_comm_services_detailed_sber"
    REPO_NAME = f"{REPO_AUTHOR}/{REPO_DIR}"
    TEXTS_FILENAME = "test_texts.txt"

    basic_converter = Converter(DB_NAME)
    basic_classifier = Model(REPO_NAME, basic_converter)
    # texts = read_texts(TEXTS_FILENAME)

    basic_classifier.print_prediction(texts, k=5)