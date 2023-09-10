import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
# nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sent):
    return nltk.word_tokenize(sent)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sent, vocab):
    words = [stem(word) for word in tokenized_sent]
    bag = np.zeros(len(vocab), dtype=np.float32)
    for idx, word in enumerate(vocab):
        if word in words:
            bag[idx] = 1.0

    return bag