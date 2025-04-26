import pandas as pd
import os
import sys


PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PATH, '..'))


class Converter:
    def __init__(self, db_name='db.csv', sep=';', encoding='utf-8', reducing_threshold=0):
        self.reducing_threshold = reducing_threshold
        self.is_reduced = (reducing_threshold > 0)

        self._load_base(db_name, sep, encoding)

        if self.is_reduced:
            self._reduce_labels()

    def _construct_converter(self, dataset):
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
        self.df = pd.read_csv(os.path.join(PATH, 'Database', db_name), sep=sep, encoding=encoding)
        self.df.dropna(inplace=True)

        freq2 = self.df.groupby(['detailed_topic']).size().sort_values(ascending=False)

        id2label, label2id, freq_full2 = self._construct_converter(freq2)

        self.full_label2id = label2id
        self.full_id2label = id2label
        self.freq_full2 = freq_full2

    def _reduce_labels(self):
        labels_reduce = set()

        for i in self.full_id2label.keys():
            if self.freq_full2.loc[i]['size'] < self.frequency_threshold:
                labels_reduce.add(i)

        reduced_df = self.df.copy()
        reduced_df["detailed_topic"] = reduced_df["detailed_topic"].map(lambda x: x if self.full_label2id[x] not in labels_reduce else "undefined")

        freq3 = reduced_df.groupby(['detailed_topic']).size().sort_values(ascending=False)

        id2label, label2id, _ = self.construct_converter(freq3)

        self.reduced_label2id = label2id
        self.reduced_id2label = id2label

    def get_label(self, seq_id):
        if self.is_reduced:
            return self.reduced_id2label.get(seq_id, "Неизвестная тема")

        return self.full_id2label.get(seq_id, "Неизвестная тема")
    
    def get_id(self, label):
        if self.is_reduced:
            return self.reduced_label2id.get(label, "Неизвестная тема")

        return self.full_label2id.get(label, "Неизвестная тема")