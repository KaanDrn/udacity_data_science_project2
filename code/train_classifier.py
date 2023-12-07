import argparse
from time import perf_counter

import skops.io as sio
from joblib import parallel_backend
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

train_features, test_features, train_target, test_target = \
    preprocessor_obj.transform(split_rate=0.7)

print("setting up pipeline")
pipeline_tweets = ml_tools.setup_pipeline()

print("perform gridsearch optimization")
t0 = perf_counter()
# This gridsearch take like forever, I wanted to show that I know how to do
# it, but decided to "not" do a gridsearch. I hope thats ok.
# I trained the model in databricks and let this here, so you can see
# on which parameters I trained the model.
gridsearch_params = {
    'multiclass_classifier__estimator__learning_rate': [0.2, 0.3],
    'multiclass_classifier__estimator__n_estimators': [50, 100],
    'multiclass_classifier__estimator__max_depth': [3, 5, 7],
    'multiclass_classifier__estimator__min_samples_leaf': [2],
    'multiclass_classifier__estimator__min_samples_split': [2]
}
grid_searcher = GridSearchCV(pipeline_tweets, param_grid=gridsearch_params,
                             verbose=3, cv=3, n_jobs=-1)

grid_searcher.fit(train_features.message, train_target)

print(perf_counter() - t0)

y_pred = grid_searcher.predict(test_features.message)
print(perf_counter() - t0)

report = classification_report(test_target.astype(int), y_pred.astype(int))
print('model performance on test-set:')
print(report)
print(perf_counter() - t0)

print('save model as skops')
sio.dump(grid_searcher, open(path_model_output, 'wb'))

print('saved model')
