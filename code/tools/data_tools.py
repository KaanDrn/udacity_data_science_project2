import re

import nltk
import pandas as pd
from nltk.corpus import stopwords


# Labels
class LabelPreparation:
    def __init__(self, label_data):
        self.label_data = label_data

    def prepare_labels(self):
        self.get_labels()

        self.drop_useless_labels()

        self.transform_labels()

    def get_labels(self):
        self.label_data.set_index('id', inplace=True)
        labels = pd.DataFrame()

        icount = 0
        max_count = len(self.label_data)
        for iid, icategories in self.label_data.iterrows():
            if icount % 500 == 0:
                print("%i/%i" % (icount, max_count))
            temp_categories = self.split_categories(icategories)
            for ilabels in temp_categories:
                temp_categories = self.split_labels(iid, ilabels)

            labels = pd.concat([labels, temp_categories], axis=0)
            icount += 1

        self.label_data = labels

    @staticmethod
    def split_labels(label_id, ilabels):
        temp_categories = [icategories.split('-') for icategories in ilabels]
        temp_categories = pd.DataFrame(temp_categories,
                                       columns=['label_names', label_id])
        temp_categories.set_index('label_names', inplace=True)
        temp_categories = temp_categories.T
        return temp_categories

    @staticmethod
    def split_categories(icategories):
        temp_categories = [ilabel.split(';') for ilabel in icategories]
        return temp_categories

    def drop_useless_labels(self):
        a = 0
        for icolumn, idata in self.label_data.items():
            if len(idata.value_counts()) < 2:
                self.label_data.drop(icolumn, inplace=True, axis=1)

    def transform_labels(self):
        return self.label_data.astype(int)


# NLP Preparation
class TextPreparation:
    def __init__(self, text_data):
        self.data = text_data
        self.pre_clean_data()

    def pre_clean_data(self):
        self.data.set_index('id', inplace=True)
        self.data.drop(['original'], axis=1)

    def prepare_data(self):
        self.normalize_me()
        self.tokenize_me()
        self.remove_stopwords()

    def normalize_me(self):
        self.data.message = self.data.message.apply(str.lower)
        self.data.message = self.data.message.apply(self.clean_punctuation)

    @staticmethod
    def clean_punctuation(text):
        return re.sub(r"[^a-zA-Z0-9]", " ", text)

    def tokenize_me(self):
        self.data.message = self.data.message.apply(nltk.tokenize.word_tokenize)

    def remove_stopwords(self):
        return [w for w in self.data.message if w not in stopwords.words("english")]
