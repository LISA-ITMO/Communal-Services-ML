import os
from Data_Processing.create_converters import Converter
from Models.load_model import Model
from Data_Processing.texts import texts

def main():
    '''
    The main function to initialize the classifier pipeline and print predictions on given input texts.
    '''
    DB_NAME = "db_2.csv"
    REPO_AUTHOR = "Goshective"
    REPO_DIR = "lab_comm_services_detailed_sber"
    REPO_NAME = f"{REPO_AUTHOR}/{REPO_DIR}"

    basic_converter = Converter(DB_NAME)
    basic_classifier = Model(REPO_NAME, basic_converter)

    basic_classifier.print_prediction(texts, k=5)

if __name__ == "__main__":
    main()