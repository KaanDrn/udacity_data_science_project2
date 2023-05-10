import argparse
import pickle
from time import perf_counter

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from tools import ml_tools

parser = argparse.ArgumentParser()

parser.add_argument('database')
parser.add_argument('model_output')

args = parser.parse_args()

path_database = args.database
path_model_output = args.model_output

print('prepare data for training')
preprocessor_obj = ml_tools.TrainingPreprocessor(path_database)
# TODO: remove seed here
train_features, test_features, train_target, test_target = \
    preprocessor_obj.transform(seed=1, split_rate=0.3)

# TODO: DEBUG just a slice
# train_features = train_features.iloc[:1000, :]
# test_features = test_features.iloc[:1000, :]
# train_target = train_target.iloc[:1000, :]
# test_target = test_target.iloc[:1000, :]
# TODO: DEBUG END


print("setting up pipeline")
pipeline_tweets = ml_tools.setup_pipeline()

print("perform gridsearch optimization")
t0 = perf_counter()
# This gridsearch take like forever, I wanted to show that I know how to do
# it, but decided to just do the tiniest gridsearch possible,
# for performance reasons. I hope thats ok.
gridsearch_params = {
    'multiclass_classifier__estimator__loss': ['log_loss'],
    'multiclass_classifier__estimator__learning_rate': [0.1, 0.3],
    'multiclass_classifier__estimator__n_estimators': [100],
    'multiclass_classifier__estimator__max_depth': [3],
    'multiclass_classifier__estimator__min_samples_leaf': [4],
    'multiclass_classifier__estimator__min_samples_split': [4]
}
grid_searcher = GridSearchCV(pipeline_tweets, param_grid=gridsearch_params)
grid_searcher.fit(train_features.message, train_target)
print(perf_counter() - t0)

y_pred = grid_searcher.predict(test_features.message)
print(perf_counter() - t0)
# TODO: DEBUG to check performance

report = classification_report(test_target.astype(int), y_pred.astype(int))
print(report)
print(perf_counter() - t0)

# save model as pkl
pickle.dump(grid_searcher, open(path_model_output, 'wb'))

# confusion_matrix = multilabel_confusion_matrix(test_target.astype(int),
#                                                y_pred.astype(int))
# TODO: DEBUG END

# get labels
# ml_tools.get_classified_labels(y_pred, train_target.columns)
