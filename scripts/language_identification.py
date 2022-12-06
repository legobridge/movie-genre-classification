import fasttext
import pandas as pd


class LanguageIdentification:

    def __init__(self):
        PRETRAINED_LANG_MODEL_FILEPATH = '../models/lid.176.ftz'
        self.model = fasttext.load_model(PRETRAINED_LANG_MODEL_FILEPATH)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1)
        return predictions


LANG_ENC_FILEPATH = '../data/language_encodings.txt'


# Reads the text file containing language encodings
def load_language_encodings_dict():
    language_encodings_df = pd.read_csv(LANG_ENC_FILEPATH, index_col='iso_639-1')
    language_encodings_dict = language_encodings_df.to_dict()['language']
    return language_encodings_dict


# Maps the ISO code of a language to its name
def map_to_language(iso_code, language_encodings_dict):
    return language_encodings_dict[iso_code[0][0][-2:]]


# Returns the language for a single description
def get_language(description, model, language_encodings_dict):
    return map_to_language(model.predict_lang(description), language_encodings_dict)


# Entry point into this module, accepts a series of descriptions and returns a series of languages
def identify_languages(descriptions: pd.Series) -> pd.Series:
    language_id_model = LanguageIdentification()
    language_encodings_dict = load_language_encodings_dict()
    return descriptions.apply(get_language, model=language_id_model, language_encodings_dict=language_encodings_dict)
