import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import tokenize, stem, bag_of_words
from model import MyNeuralNetwork

def create_data():

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

        label = tags.index(tag)
        y_train.append(label)


    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train, tags, all_words


class MyDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    X_train, y_train, tags, all_words = create_data()

    batch_size = 8
    input_size = len(X_train[0])
    hidden_size = 8
    num_classes = len(tags)
    learning_rate = 0.001
    num_epochs = 500

    dataset = MyDataset(X_train, y_train)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = MyNeuralNetwork(input_size, hidden_size, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device, dtype=torch.int64)

            # Forward
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch : {epoch + 1}/{num_epochs}, loss={loss.item():.4f}')


    data = {
        'model_state': model.state_dict(),
        'input_size': input_size,
        'output_size': num_classes,
        'hidden_size': hidden_size,
        'vocabulary': all_words,
        'tags': tags
    }

    CONFIG = 'config.pth'
    torch.save(data, CONFIG)

