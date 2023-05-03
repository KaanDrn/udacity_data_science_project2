from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle


class Preprocessor(BaseEstimator):
    def __init__(self, data):
        self.data = data

        self.train_data = None
        self.test_data = None

    def transform(self, seed=None, split_rate=0.6):
        self.handle_wrong_labels()

        temp_data = shuffle(self.data, random_state=seed)
        self._split_it(temp_data, split_rate)

        return self.seperate_labels()

    def _split_it(self, data, split_rate):
        self.train_data = data[:int(data.shape[0] * split_rate)]
        self.test_data = data[int(data.shape[0] * split_rate):]

    def seperate_labels(self):
        return self.train_data.iloc[:, -2:], self.test_data.iloc[:, -2:], \
               self.train_data.iloc[:, :-2], self.test_data.iloc[:, :-2]

    def handle_wrong_labels(self):
        """
        I investigated the column `related` in the categories.csv dataset, but it was
        not clear, what a `2` means. UdacityGPT also stated that it thinks that this
        is a wrong label. I than analysed, how often a `2` appears in the dataset. It
        was 193 out of more than 26 000 labels, therefore I decided to just drop them.
        """
        condition = self.data.related == '2'
        indices = self.data.index[condition]

        self.data.drop(indices, axis=0, inplace=True)


# TODO: Unnecessary? I could also just use the TF-IDF Vectorizer as the estimator
#  istead of creating my own. The estimator does not have any specialties compared to
#  the TF-IDF Vectorizer provided by sklearn.
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
