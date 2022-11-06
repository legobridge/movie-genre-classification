from nltk import word_tokenize


def get_year(title):
    return title[-6:-2]


def remove_year(title):
    return title[:-7]


def remove_punctuation(description):
    description = description.replace('.', '')
    description = description.replace(',', '')
    description = description.replace('!', '')
    description = description.replace('?', '')
    description = description.replace('(', '')
    description = description.replace(')', '')
    description = description.replace('$', '')
    description = description.replace('%', '')
    description = description.replace('&', '')
    description = description.replace('*', '')
    description = description.replace('\'', '')
    description = description.replace('"', '')
    return description


def tokenize(description):
    return word_tokenize(description)


def remove_stop_words(description, stop_words):
    return [w for w in description if w not in stop_words]


def stemming(text, porter_stemmer):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
