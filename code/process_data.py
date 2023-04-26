import argparse
import logging

import pandas as pd

from tools import data_tools

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

# TODO: DEBUG
data_categories = data_categories.iloc[:5000, :]
# TODO: DEBUG END


# clean categories
print('preparing labels')
data_obj = data_tools.LabelPreparation(data_categories)
data_obj.prepare_labels()

print('preparing texts')
texts_obj = data_tools.TextPreparation(data_texts)
texts_obj.prepare_data()

print('end script')
