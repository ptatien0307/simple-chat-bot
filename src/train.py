import json
import numpy as np
from utils import tokenize, stem, bag_of_words

# Read intents.json file
with open('./intents.json') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    # Extract tag
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        # Tokenize the sentence
        tokenized = tokenize(pattern)

        # Add tokenized to all_words  to create vocabulary
        all_words.extend(tokenized)

        # Data and label for training
        xy.append((tokenized, tag))

# Ignore some punctuation
ignore_words = ['?', '!', '.', ',']
all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for sent, tag in xy:
    bag = bag_of_words(sent, all_words)
    X_train.append(bag)
    y_train = tags.index(tag)


X_train = np.array(X_train)
y_train = np.array(y_train)

