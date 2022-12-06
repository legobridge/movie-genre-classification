from nltk import word_tokenize


# Extract year of release from movie title
def get_year(title):
    return title[-6:-2]


# Remove year of release from movie title
def remove_year(title):
    return title[:-7]


# Removes punctuation from the description
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


# Tokenizes descriptions using NLTK
def tokenize(description):
    return word_tokenize(description)


# Removes stop words
def remove_stop_words(description, stop_words):
    return [w for w in description if w not in stop_words]


# Stems text using the Porter Stemmer
def stemming(text, porter_stemmer):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
