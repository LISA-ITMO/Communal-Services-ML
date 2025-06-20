import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.adapt import MLkNN

from Data_Processing.create_converters import Converter


def prepare_data(converter: Converter):
    '''
    Prepares training and validation data for classical models by vectorizing and reducing dimensionality.

            Parameters:
                    converter (Converter): Object that contains the labeled DataFrame

            Returns:
                    dict: Dictionary containing original and reduced feature matrices and label arrays
    '''
    data = converter.df

    # topic_id -> sequential label index
    data["label"] = converter.get_id(data["detailed_topic"])

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data["appeal"].tolist(), data["label"].tolist(), test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(val_texts)

    svd = TruncatedSVD(n_components=1000)  # Reduce to less-dimensional space
    X_train_reduced = svd.fit_transform(X_train)
    X_test_reduced = svd.transform(X_test)

    train_labels_mlb = [[i] for i in train_labels]

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_labels_mlb)
    y_test = np.array(val_labels)

    return {
        "X": {
            "Train": X_train,
            "Test": X_test
        },
        "X_Reduced": {
            "Train": X_train_reduced,
            "Test": X_test_reduced
        },
        "Y": {
            "Train": y_train,
            "Test": y_test
        }
    }


class ClassicModel:
    '''
    A class to train and store predictions from multiple classical ML classifiers.
    '''

    def __init__(self, converter):
        '''
        Initializes and trains multiple classifiers using prepared data.

                Parameters:
                        converter (Converter): Converter object to prepare labeled data
        '''
        self.data = prepare_data(converter)
        self.predictions = {}

        self.train_NB()
        self.train_DT()
        self.train_SV()
        self.train_KNN()

    def train_NB(self):
        '''
        Trains a Multinomial Naive Bayes classifier using Binary Relevance wrapper.
        '''
        classifier = BinaryRelevance(MultinomialNB())
        classifier.fit(self.data["X"]["Train"], self.data["Y"]["Train"])
        self.predictions["NB"] = classifier.predict_proba(self.data["X"]["Test"])

    def train_DT(self):
        '''
        Trains a Decision Tree classifier using Binary Relevance wrapper.
        '''
        classifier = BinaryRelevance(DecisionTreeClassifier())
        classifier.fit(self.data["X"]["Train"], self.data["Y"]["Train"])
        self.predictions["DT"] = classifier.predict_proba(self.data["X"]["Test"])

    def train_SV(self):
        '''
        Trains an SVM classifier (with probability output) on reduced feature space.
        '''
        classifier = BinaryRelevance(SVC(probability=True))
        classifier.fit(self.data["X_Reduced"]["Train"], self.data["Y"]["Train"])
        self.predictions["SVC"] = classifier.predict_proba(self.data["X_Reduced"]["Test"])

    def train_KNN(self):
        '''
        Trains a K-Nearest Neighbors classifier using MLkNN.
        '''
        classifier = MLkNN(k=5)
        classifier.fit(self.data["X"]["Train"], self.data["Y"]["Train"])
        self.predictions["KNN"] = classifier.predict_proba(self.data["X"]["Test"])