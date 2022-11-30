import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from text_processing import get_year, remove_year, remove_punctuation, tokenize, remove_stop_words, stemming
from language_identification import identify_languages

PROCESSED_DATA_FILEDIR = '../data/processed/'


def preprocess_descriptions(descriptions: pd.Series) -> pd.Series:
    # Clean and tokenize the description
    processed_descriptions = descriptions.str.lower()
    processed_descriptions = processed_descriptions.apply(remove_punctuation)
    processed_descriptions = processed_descriptions.apply(tokenize)

    # Remove stop words from the tokenized description
    stop_words = set(stopwords.words('english'))
    processed_descriptions = processed_descriptions.apply(remove_stop_words, stop_words=stop_words)

    # Stem the tokens in the tokenized description
    porter_stemmer = PorterStemmer()
    processed_descriptions = processed_descriptions.apply(stemming, porter_stemmer=porter_stemmer)

    # Join the tokenized description back together for use in text models
    return processed_descriptions.str.join(' ')


def preprocess_dataset(dataset: pd.DataFrame, processed_data_filename):
    # Format the DataFrame
    df_columns = {0: 'id', 1: 'title', 2: 'genre', 3: 'description'}
    dataset = dataset.rename(columns=df_columns)
    dataset = dataset.set_index('id')

    # Extract years from titles
    dataset['year'] = dataset['title'].apply(get_year)
    dataset['title'] = dataset['title'].apply(remove_year)

    # Clean up whitespace around genres
    dataset['genre'] = dataset['genre'].str.strip()

    # Drop the "short" genre, since it is not semantically related to the description
    dataset = dataset[dataset["genre"] != 'short']

    # Identify the languages the descriptions are written in
    dataset['language'] = identify_languages(dataset['description'])
    print(dataset['language'].value_counts())
    dataset = dataset[dataset['language'] == 'English']

    # Preprocess the descriptions
    dataset['processed_description_string'] = preprocess_descriptions(dataset['description'])

    # Save as a CSV file
    processed_data_filepath = os.path.join(PROCESSED_DATA_FILEDIR, processed_data_filename)
    dataset.to_csv(processed_data_filepath)

    # Save fewer label versions
    top_genres = dataset.genre.value_counts().index
    num_classes = [3, 4, 5, 6, 7]
    for n in num_classes:
        genres = top_genres[:n].to_list()
        new_data = dataset[dataset['genre'].isin(genres)]
        fewer_labels_data_filename = '{}_genres_{}'.format(n, processed_data_filename)
        fewer_labels_data_filepath = os.path.join(PROCESSED_DATA_FILEDIR, 'fewer_labels', fewer_labels_data_filename)
        new_data.to_csv(fewer_labels_data_filepath, index=False)


if __name__ == '__main__':
    # Download NLTK packages
    nltk.download('punkt')
    nltk.download('stopwords')

    # Load data
    train_df = pd.read_table('../data/train_data.txt', sep=':::', header=None, engine='python')
    test_df = pd.read_table('../data/test_data.txt', sep=':::', header=None, engine='python')
    test_data_labels = pd.read_table("../data/test_data_solution.txt", sep=":::", header=None, engine='python')
    test_df[3] = test_df[2]
    test_df[2] = test_data_labels[2]

    # Preprocess train dataset
    preprocess_dataset(train_df, 'train_data_processed.csv')

    # Preprocess test dataset
    preprocess_dataset(test_df, 'test_data_processed.csv')

