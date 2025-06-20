import pandas as pd
import os

class Converter:
    '''
    Handles the loading, preprocessing, and optional reduction of label sets for classification tasks.
    Converts between labels and sequential numeric IDs.
    '''

    def __init__(self, db_name='db.csv', sep=';', encoding='utf-8', reducing_threshold=0):
        '''
        Initializes the Converter object by loading the dataset and optionally reducing rare labels.

                Parameters:
                        db_name (str): Name of the CSV file containing labeled data
                        sep (str): Separator used in the CSV file
                        encoding (str): Encoding of the CSV file
                        reducing_threshold (int): Minimum frequency a label must have to be retained
        '''
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.reducing_threshold = reducing_threshold
        self.is_reduced = (reducing_threshold > 0)

        self._load_base(db_name, sep, encoding)

        if self.is_reduced:
            self._reduce_labels()

    def _construct_converter(self, dataset):
        '''
        Builds mapping dictionaries and frequency DataFrame for labels.

                Parameters:
                        dataset (Series): A frequency mapping of labels (e.g., from value_counts)

                Returns:
                        id2label (dict): Maps index IDs to label names
                        label2id (dict): Maps label names to index IDs
                        freq_full (DataFrame): Frequency table of labels
        '''
        label2id = {}
        id2label = {}
        temp_dct = {
            'topic_id': [],
            'size': []
        }

        for i, detailed_topic in enumerate(dataset.keys()):
            new_id = i
            label2id[detailed_topic] = new_id
            id2label[new_id] = detailed_topic

            size = dataset[detailed_topic]
            temp_dct['topic_id'].append(new_id)
            temp_dct['size'].append(size)

        freq_full = pd.DataFrame(temp_dct).set_index('topic_id')

        return id2label, label2id, freq_full

    def _load_base(self, db_name, sep, encoding):
        '''
        Loads the original labeled dataset and initializes full label mapping.

                Parameters:
                        db_name (str): CSV filename
                        sep (str): Separator used in the CSV
                        encoding (str): File encoding
        '''
        self.df = pd.read_csv(os.path.join(self.PATH, 'Database', db_name), sep=sep, encoding=encoding)
        self.df.dropna(inplace=True)

        freq2 = self.df.groupby(['detailed_topic']).size().sort_values(ascending=False)

        id2label, label2id, freq_full2 = self._construct_converter(freq2)

        self.full_label2id = label2id
        self.full_id2label = id2label
        self.freq_full2 = freq_full2

    def _reduce_labels(self):
        '''
        Reduces the label set by merging infrequent labels into an 'undefined' category,
        and rebuilds the label mappings accordingly.
        '''
        labels_reduce = set()

        for i in self.full_id2label.keys():
            if self.freq_full2.loc[i]['size'] < self.reducing_threshold:
                labels_reduce.add(i)

        reduced_df = self.df.copy()
        reduced_df["detailed_topic"] = reduced_df["detailed_topic"].map(
            lambda x: x if self.full_label2id[x] not in labels_reduce else "undefined"
        )

        freq3 = reduced_df.groupby(['detailed_topic']).size().sort_values(ascending=False)

        id2label, label2id, _ = self._construct_converter(freq3)

        self.reduced_label2id = label2id
        self.reduced_id2label = id2label

    def get_label(self, seq_id):
        '''
        Converts a numeric label ID to its corresponding detailed topic label.

                Parameters:
                        seq_id (int): Sequential numeric label ID

                Returns:
                        label (str): Corresponding detailed topic string
        '''
        if self.is_reduced:
            return self.reduced_id2label.get(seq_id, "Неизвестная тема")

        return self.full_id2label.get(seq_id, "Неизвестная тема")
    
    def get_id(self, label):
        '''
        Converts a detailed topic label to its corresponding numeric ID.

                Parameters:
                        label (str): Detailed topic label

                Returns:
                        seq_id (int or str): Corresponding numeric ID or default error label
        '''
        if self.is_reduced:
            return self.reduced_label2id.get(label, "Неизвестная тема")

        return self.full_label2id.get(label, "Неизвестная тема")
    
    def get_num_labels(self):
        '''
        Returns the total number of unique labels based on whether reduction is applied.

                Returns:
                        num_labels (int): Number of label classes currently in use
        '''
        if self.is_reduced:
            return len(self.reduced_label2id)

        return len(self.full_label2id)