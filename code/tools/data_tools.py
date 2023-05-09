import re
import sqlite3

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


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


class SqlPreprocessor:
    def __init__(self, data, path_database):
        self.connection = None
        self.data = data
        self.database = path_database

        self.engine = None

    def process_data(self):
        self.handle_wrong_labels()
        self.drop_duplicates()

        self.connect_to_db()
        self.write_to_db()
        self.connection.close()

    def handle_wrong_labels(self):
        """
        I investigated the column `related` in the categories.csv dataset,
        but it was
        not clear, what a `2` means. UdacityGPT also stated that it thinks
        that this
        is a wrong label. I than analysed, how often a `2` appears in the
        dataset. It
        was 193 out of more than 26 000 labels, therefore I decided to just
        drop them.
        """
        condition = self.data.related == '2'
        indices = self.data.index[condition]

        self.data.drop(indices, axis=0, inplace=True)

    def drop_duplicates(self):
        index_duplicates = self.data.index[
            self.data.message.duplicated(keep='first')]
        self.data.drop(index_duplicates, inplace=True)

    def write_to_db(self):
        """
        The index is not needed anymore, since the data is merged now,
        therefore we can drop the index here.
        :return:
        """
        self.data.to_sql(name='disaster_data', con=self.connection,
                         if_exists='replace', index=False)

    def connect_to_db(self):
        self.connection = sqlite3.connect(self.database)


# Implemented this class on my way through the videos and realized
# that the TF-IDF Vectorizer can do all this as well... so this deprecated
# before I needed it... But still wanted to show my progress here
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
        self.remove_my_stopwords()
        self.lemmatize_me()

    def normalize_me(self):
        self.data.message = self.data.message.apply(str.lower)
        self.data.message = self.data.message.apply(self._clean_punctuation)

    @staticmethod
    def _clean_punctuation(text):
        return re.sub(r"[^a-zA-Z0-9]", " ", text)

    def tokenize_me(self):
        self.data.message = self.data.message.apply(
                nltk.tokenize.word_tokenize)

    def remove_my_stopwords(self):
        self.data.message = self.data.message.apply(self._remove_stopwords)

    @staticmethod
    def _remove_stopwords(text):
        return [w for w in text if w not in stopwords.words("english")]

    def lemmatize_me(self):
        self.data.message = self.data.message.apply(self._lemmatize_text)

    @staticmethod
    def _lemmatize_text(text):
        return [WordNetLemmatizer().lemmatize(w) for w in text]
