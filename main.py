from Data_Processing.create_converters import Converter
from Models.bert_pipeline import BertModel
from Data_Processing.texts import texts_loaded

def main(texts, eval=True):
    '''
    The main function to initialize the classifier pipeline and print predictions on given input texts.
    '''
    DB_NAME = "db_2.csv"
    REPO_AUTHOR = "Goshective"
    REPO_DIR = "lab_comm_services_detailed_sber"
    REPO_NAME = f"{REPO_AUTHOR}/{REPO_DIR}"

    basic_converter = Converter(DB_NAME)
    basic_classifier = BertModel(REPO_NAME, basic_converter, eval=eval)

    basic_classifier.print_prediction(texts, k=5)


if __name__ == "__main__":
    while True:
        print("Choose mode:")
        print("1 - Your own input")
        print("2 - Test input")
        print("3 - Train model")
        print("All other - Exit")
        print()
        
        eval = True
        mode = input()
        if mode == '1':
            texts = [input()]
        elif mode == '2':
            texts = texts_loaded
        elif mode == '3':
            texts = texts_loaded
            eval = False
        else:
            print("End of program.")
            break
            
        main(texts, eval=eval)

        print()