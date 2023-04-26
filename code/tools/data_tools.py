import re
import nltk
from nltk.corpus import stopwords

import pandas as pd


# Labels
def prepare_labels(label_data):
    label_data = get_labels(label_data)

    label_data = drop_useless_labels(label_data)

    label_data = transform_labels(label_data)

    return label_data


def get_labels(categories):
    categories.set_index('id', inplace=True)
    labels = pd.DataFrame()

    icount = 0
    max_count = len(categories)
    for iid, icategories in categories.iterrows():
        if icount % 500 == 0:
            print("%i/%i" % (icount, max_count))
        temp_categories = split_categories(icategories)
        for ilabels in temp_categories:
            temp_categories = split_labels(iid, ilabels)

        labels = pd.concat([labels, temp_categories], axis=0)
        icount += 1

    return labels


def split_labels(label_id, ilabels):
    temp_categories = [icategories.split('-') for icategories in ilabels]
    temp_categories = pd.DataFrame(temp_categories, columns=['label_names', label_id])
    temp_categories.set_index('label_names', inplace=True)
    temp_categories = temp_categories.T
    return temp_categories


def split_categories(icategories):
    temp_categories = [ilabel.split(';') for ilabel in icategories]
    return temp_categories


def drop_useless_labels(data):
    a = 0
    for icolumn, idata in data.items():
        if len(idata.value_counts()) < 2:
            data.drop(icolumn, inplace=True, axis=1)
    return data


def transform_labels(labels):
    return labels.astype(int)


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
