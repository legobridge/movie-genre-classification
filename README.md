# IMDb Movie Genre Classification


This project uses the following dataset (freely available on Kaggle) to predict the primary genre of a movie, relying only on the natural language description of the movie on IMDb.

[IMDb Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb "IMDb Genre Classification Dataset")

The project directory structure is described below.

## Data

The _data_ directory houses the original Kaggle data at the root level, and the preprocessed data under the _processed_ subdirectory. The preprocessed data is created from the basic data using __preprocessing.py__ (see below under _scripts_)

## Notebooks

The _notebooks_ directory contains Jupyter notebooks, which we used for EDA, preliminary preprocessing, training models, and other experiments. 

A brief description of each file follows:

- __eda.ipynb__ : This notebook contains most of the EDA we did, including genre distributions, trends over time, word clouds, etc.
- __tfidf.ipynb__ : This notebook was used for training and validating the performance of our TF-IDF based models.
- __basic_neural_models.ipynb__ : This notebook was used for training and validating the performance of our neural network models like the simple neural network and RNNs.
- __small_bert.ipynb__ : This notebook was used for training and validating the performance of our Small BERT model.
- __training.ipynb__ : This notebook was used to train all of our models on the entire training set, test them on the test set, and save the models for inference.
- __class_imbalance_experiments.ipynb__ : This notebook was used for experiments regarding our attempt to handle class imbalance via oversampling and SMOTE.

The directory also contains some _deprecated_ notebooks, which are preserved for legacy reasons, but are not part of the submission.

## Scripts

The _scripts_ directory is a Python package containing our data preprocessing scripts and the server-client modules that can be used to run a small inference demo. 

A brief description of each file follows:

- __language_identification.py__ : This module contains functions that can identify the language a piece of text is written in.
- __text_processing.py__ : This module contains functions (such as stemming / stopword removal) that can preprocess text to clean/simplify it.
- __preprocessing.py__ : This module utilizes __language_identification.py__ and __text_processing.py__ to create the preprocessed data we use to train and test all our models. Running this script creates all the data found under _data/processed/_.
- __server.py__ : This module whips up a minimal server (using Flask Restful) at 127.0.0.1:5000 that provides a GET endpoint for inference over the trained Small BERT model.
- __tkinter_client.py__ : This module brings up a (very) simple GUI where one can input movie descriptions and get the probability distribution over genres. __server.py__ must be running for this to work.


## Models

The _models_ directory contains saved instances of the trained models.

## Metrics

The _metrics_ directory contains classification reports of the models we trained and tested.
