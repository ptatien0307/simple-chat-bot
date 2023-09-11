import random
import json
import torch
from model import MyNeuralNetwork
from utils import tokenize, bag_of_words, stem

# Read intents.json file
with open('./intents.json') as f:
    intents = json.load(f)

config = torch.load('config.pth')

input_size = config['input_size']
hidden_size = config['hidden_size']
output_size = config['output_size']
vocabulary = config['vocabulary']
tags = config['tags']
model_state = config['model_state']


model = MyNeuralNetwork(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bot_name = 'Erris'

print('Type: "quit" to exit')
while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break

    tokenized = tokenize(sentence)
    X = bag_of_words(tokenized, vocabulary)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.5:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f'{bot_name}: {random.choice(intent["responses"])}')
    else:
        print(f'{bot_name}: I do not understand...')