import argparse

import pandas as pd

from tools import data_tools

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

# # TODO: DEBUG just got a slice, which should be all in the prod version
# data_categories = data_categories.iloc[:2000, :]
# data_texts = data_texts.iloc[:2000, :]
# # TODO: DEBUG END
data_texts.drop('original', inplace=True, axis=1)
data_texts.set_index('id', inplace=True)

# clean categories
print('preparing labels')
label_obj = data_tools.LabelPreparation(data_categories)
label_obj.prepare_labels()

# join data
print('join data')
data_all = pd.concat([label_obj.label_data, data_texts], axis=1)

# process data and write to db
print('preprocess data and write to database')
preprocessor_obj = data_tools.SqlPreprocessor(data_all, path_database)
preprocessor_obj.process_data()

print('data is prepared')
