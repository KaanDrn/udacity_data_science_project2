# udacity_data_science_project2

# Motivation
In this repository, a disaster response pipeline is beeing developed. The aim is to create an supervised NLP Machine Learning (ML) pipeline, that is trainend to classify tweets from disasters to find out, if the tweet is asking for help and what help the tweeter needs. This should help disaster relief forces to plan their rescue/help mission better.

The pipeline is currently trained with a `GradientBoostingClassifier` from `sklearn` as a classifier and makes use of the `TfidfVectorizer` to generate input for it.
The data provided consists of about 26k tweets and their labels.

A pretrained model exists in this repo as well, which can either be used or further be trained. A small gridsearch was peformed on the modell to finetune the parameters.


# Libraries
The analysis makes use of the most known python libraries, namely:
- `pandas`
- `numpy`
- `plotly`
- `sqlite3`


The following ML libraries were used:
- `sklearn`
- `nltk`

In addition `pickle` and `skops` are used to save the trained model and `flask` is used to visualize the results.

# Files
The repo contains three directories, namely: `./code`, `./app` and  `./resources`.
The `./code` directory contains the code to:
1. preprocess the data (`process_data.py`)
2. train the NLP model and setup a ML-Pipeline and save the trained model as a pickle/skops file (`train_classifier.py`)

In addition, code that was used can be found in `./code/tools`.

The data used in this analysis can be found in the `./resources` directory. The trained modell (`tweet_classifier.pkl`, `tweet_classifier.skops`) can be found here as well.

The code used to launch the flask-app can be found in `./app`, where all the resources can be found to launch the app.

# Usage
## Preprocessing data
If new data is provided and the data base has to be cleaned and preprocessed, the `process_data.py` function can be used. Do do so, you can start the script from the cosole using the following code:
```cmd
python process_data.py <path_to_raw_text_data.csv> <path_to_categories_data.csv> <path_to_sql_database_to_write_to.db>
```
If you execute the code in the resource folder, thios is what the call looks like:
```cmd
python process_data.py ../resources/messages.csv ../resources/categories.csv ../resources/DisasterResponse.db
```

## Training the model
After the data is pre-processed, the model can be trained using the `train_classifier.py` script, which is used like this:
```cmd
python train_classifier.py ../resources/DisasterResponse.db ../resources/tweet_classifier.skops
```
Please note that the model outputs a `.skops` file, which is a more robust format of a `.pkl` file. If you prefere `pickle` instead, you can modifie the `line 60` in `train_calssifier.py` to change the output format.

## Running the app
To run the app and makle use of the model you can simply type in
```cmd
python run.py
```
Make sure that you are in the folder, where the `run.py` file is and make sure that you installed the `tools` package provided in this repo (or use a IDE, which recognizes the package in the project folder like Pycharm).

# Summary
The model optimizing is not as good as it could be since it takes quite a time to train the model, therefore the model performance is not the best. If you want to use this please take that into account or retrain the model.

# Acknowledgments
Thanks to Udacity for providing the most part of the flask app.