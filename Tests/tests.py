import unittest

from Data_Processing.create_converters import Converter
from Models.load_model import Model
from Tests.test_texts import test_texts_dict

REPO_NAME = "Goshective/lab_comm_services_detailed_sber"
DB_NAME = "db_2.csv"

class ModelsTestCase(unittest.TestCase):
    def setUp(self):
        self.predictions = {}
        basic_converter = Converter(DB_NAME)
        basic_classifier = Model(REPO_NAME, basic_converter)

        for label, texts in test_texts_dict.items():
            self.predictions[label] = basic_classifier.predict(texts, k=5)

    def test_prediction(self):
        PERCENTAGE = 80

        right_counter = 0
        full_counter = 0

        for label, preds in self.predictions.items():
            for pred_k in preds:
                full_counter += 1
                right_counter += any([x[0] == label for x in pred_k])

        acc = right_counter / full_counter
        self.assertGreaterEqual(acc * 100, PERCENTAGE)

    def test_number_of_predictions(self):
        is_below_max_pred_mumber = []
        for preds in self.predictions.values():
            for pred_k in preds:
                is_below_max_pred_mumber.append(len(pred_k) <= 5)
        
        self.assertTrue(all(is_below_max_pred_mumber))

    def test_base_threshold(self):
        is_under_threshold_probability = []
        for preds in self.predictions.values():
            for pred_k in preds:
                is_under_threshold_probability.append(min([x[1] for x in pred_k]) >= 0.1)
        
        self.assertTrue(all(is_under_threshold_probability))

    def test_modes_correlation(self):
        basic_converter = Converter(DB_NAME, reducing_threshold=10)
        basic_classifier = Model(REPO_NAME, basic_converter)

        self.assertTrue(basic_converter.is_reduced)
        self.assertTrue(basic_classifier.is_reducing)

    

if __name__ == "__main__":
    unittest.main()