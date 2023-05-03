import argparse
import logging

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import multilabel_confusion_matrix

from tools import data_tools, ml_tools
from tools.ml_tools import FeatureExtractor

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('texts')
parser.add_argument('categories')
parser.add_argument('database')

args = parser.parse_args()

path_texts = args.texts
path_categories = args.categories
path_database = args.database

print('reading in texts')
data_texts = pd.read_csv(path_texts)

print('reading in categories')
data_categories = pd.read_csv(path_categories)

# TODO: DEBUG just got a slice, which should be all in the prod version
data_categories = data_categories.iloc[:500, :]
data_texts = data_texts.iloc[:500, :]
# TODO: DEBUG END
data_texts.drop('original', inplace=True, axis=1)
data_texts.set_index('id', inplace=True)

# clean categories
print('preparing labels')
label_obj = data_tools.LabelPreparation(data_categories)
label_obj.prepare_labels()

data_all = pd.concat([label_obj.label_data, data_texts], axis=1)

preprocessor = ml_tools.Preprocessor(data_all)
train_data, test_data, train_labels, test_labels = preprocessor.transform()

# TODO: DEBUG will be replaced by the pipeline
feature_extractor = TfidfVectorizer()
feature_extractor = feature_extractor.fit(train_data.message)
train_features = feature_extractor.transform(train_data.message)
test_features = feature_extractor.transform(test_data.message)

model = GradientBoostingClassifier(loss="log_loss",
                                   learning_rate=0.1,
                                   n_estimators=200,
                                   min_samples_leaf=5)
multiclass_gb = MultiOutputClassifier(model, n_jobs=2)

multiclass_gb.fit(train_features, train_labels)

y_pred = multiclass_gb.predict(test_features)



# TODO: DEBUG END

from sklearn.metrics import classification_report
report = classification_report(test_labels.astype(int), y_pred.astype(int))

confusion_matrix = multilabel_confusion_matrix(test_labels.astype(int),
                                               y_pred.astype(int))

print('end script')
