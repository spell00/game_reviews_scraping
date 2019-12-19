import re

from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


class Tools():
    def __init__(self):
        pass

    def preprocessing(self, corpus):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS = set(stopwords.words('english'))

        def clean_text(text):
            text = BeautifulSoup(text, "lxml").text  # HTML decoding
            text = text.lower()  # lowercase text
            text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
            text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
            text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
            return text

        try:
            corpus['blog'] = corpus['blog'].apply(clean_text)
            return corpus
        except:
            print("Corpus is not in the right format. It must have two colomns, first, named blog and second, named label.")
