import argparse

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools import ml_tools

parser = argparse.ArgumentParser()

parser.add_argument('database')
parser.add_argument('model_output')

args = parser.parse_args()

path_database = args.database
path_model_output = args.model_output

preprocessor_obj = ml_tools.TrainingPreprocessor(path_database)
# TODO: remove seed here
train_features, test_features, train_target, test_target = \
    preprocessor_obj.transform(seed=1)


# TODO: DEBUG just a slice
train_features = train_features.iloc[:1000, :]
test_features = test_features.iloc[:1000, :]
train_target = train_target.iloc[:1000, :]
test_target = test_target.iloc[:1000, :]
# TODO: DEBUG END


# TODO: DEBUG will be replaced by the pipeline
feature_extractor = TfidfVectorizer()
feature_extractor = feature_extractor.fit(train_features.message)
train_features = feature_extractor.transform(train_features.message)
test_features = feature_extractor.transform(test_features.message)

# scale data
scaler = StandardScaler(with_mean=False)
scaler = scaler.fit(train_features)
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)

model = GradientBoostingClassifier(loss="log_loss",
                                   learning_rate=0.1,
                                   n_estimators=50,
                                   min_samples_leaf=5)
multiclass_gb = MultiOutputClassifier(model)

multiclass_gb.fit(train_features, train_target)

y_pred = multiclass_gb.predict(test_features)

# TODO: DEBUG END

# TODO: DEBUG
from sklearn.metrics import classification_report, multilabel_confusion_matrix

report = classification_report(test_target.astype(int), y_pred.astype(int))
print(report)

confusion_matrix = multilabel_confusion_matrix(test_target.astype(int),
                                               y_pred.astype(int))

# TODO: DEBUG END
model = GradientBoostingClassifier(loss="log_loss",
                                   learning_rate=0.1,
                                   n_estimators=50,
                                   min_samples_leaf=5)
pipeline_tweets = Pipeline([
    ('feature_extractor', TfidfVectorizer()),
    ('scaler', StandardScaler(with_mean=False)),
    ('multiclass_classifier', MultiOutputClassifier(model))
])




# get labels
bool_list = y_pred.astype(int).astype(bool).tolist()[0]
train_target.columns[bool_list]
