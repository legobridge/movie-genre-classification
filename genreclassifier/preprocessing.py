import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from text_processing import get_year, remove_year, remove_punctuation, tokenize, remove_stop_words, stemming
from language_identification import identify_languages


def preprocess_dataset(raw_data_filepath, processed_data_filepath, is_test_dataset):
    # Load data
    df = pd.read_table(raw_data_filepath, sep=':::', header=None, engine='python')

    # Format DataFrame
    if is_test_dataset:
        df_columns = {0: 'id', 1: 'title', 2: 'description'}
    else:
        df_columns = {0: 'id', 1: 'title', 2: 'genre', 3: 'description'}
    df = df.rename(columns=df_columns)
    df = df.set_index('id')

    # Extract year from title
    df['year'] = df['title'].apply(get_year)
    df['title'] = df['title'].apply(remove_year)

    # Clean up whitespace around genre
    if not is_test_dataset:
        df['genre'] = df['genre'].str.strip()

    # Clean and tokenize the description
    df['processed_description'] = df['description'].str.lower()
    df['processed_description'] = df['processed_description'].apply(remove_punctuation)
    df['processed_description'] = df['processed_description'].apply(tokenize)

    # Remove stop words from the tokenized description
    stop_words = set(stopwords.words('english'))
    df['processed_description'] = df['processed_description'].apply(remove_stop_words, stop_words=stop_words)

    # Stem the tokens in the tokenized description
    porter_stemmer = PorterStemmer()
    df['processed_description'] = df['processed_description'].apply(stemming, porter_stemmer=porter_stemmer)

    # Join the tokenized description back together for use in text models
    df['processed_description_string'] = df['processed_description'].str.join(' ')

    # Identify which language a description is written in
    df['language'] = identify_languages(df['description'])
    print(df['language'].value_counts())

    df.to_csv(processed_data_filepath)


if __name__ == '__main__':
    # Download NLTK packages
    nltk.download('punkt')
    nltk.download('stopwords')

    # Preprocess train dataset
    preprocess_dataset('../data/train_data.txt',
                       '../data/processed/train_data_processed.csv',
                       False)

    # Preprocess test dataset
    preprocess_dataset('../data/test_data.txt',
                       '../data/processed/test_data_processed.csv',
                       True)
