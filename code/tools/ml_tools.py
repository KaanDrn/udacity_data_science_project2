import sqlite3

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


class TrainingPreprocessor:
    def __init__(self, path_db):
        self.database = path_db
        self.data = None
        self.connection = None

        self.train_data = None
        self.test_data = None

    def transform(self, seed=None, split_rate=0.7):
        self.connect_to_db()
        self.load_data()
        self.connection.close()

        temp_data = shuffle(self.data, random_state=seed)
        self._split_it(temp_data, split_rate)

        return self.seperate_labels()

    def connect_to_db(self):
        self.connection = sqlite3.connect(self.database)

    def load_data(self):
        self.data = pd.read_sql('SELECT * FROM disaster_data', self.connection)

    def _split_it(self, data, split_rate):
        self.train_data = data[:int(data.shape[0] * split_rate)]
        self.test_data = data[int(data.shape[0] * split_rate):]

    def seperate_labels(self):
        """
        :return: train_features, test_features, train_target, test_target

        """
        return self.train_data.iloc[:, -2:], self.test_data.iloc[:, -2:], \
               self.train_data.iloc[:, :-2], self.test_data.iloc[:, :-2]


def setup_pipeline():
    pipeline = Pipeline([
        ('feature_extractor', TfidfVectorizer(max_features=10**4)),
        ('scaler', StandardScaler(with_mean=False)),
        ('multiclass_classifier',
         MultiOutputClassifier(GradientBoostingClassifier()))
    ])
    return pipeline


def get_classified_labels(y_pred, label_names):
    bool_list = y_pred.astype(int).astype(bool).tolist()[0]
    return label_names[bool_list]


# Unnecessary... I could also just use the TF-IDF Vectorizer as the
#  estimator istead of creating my own. The estimator does not have any
#  specialties compared to the TF-IDF Vectorizer provided by sklearn. BUt
#  this is the way to create a homemade estimator, right?
class FeatureExtractor(BaseEstimator):
    def __init__(self, corpus, feature_extractor=TfidfVectorizer):
        self.data = corpus
        self.extractor = feature_extractor()

        self.features = None

    def fit(self):
        pass

    def transform(self):
        self.features = self.extractor.fit_transform(self.data)

    def get_features(self):
        return self.features.toarray()
