import numpy as np
import json 
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from Neural_Networkx import bag_of_words , tokenize , stem
from Neural_network import NeuralNet
from main import Speak

def trainFriday():
    with open("C:\\Users\\Lenovo\\bigboss\\jarvis.ai\\Jarvis Source Code\\AI Jarvis Using Python Tut\\Data\\Tasks.json",'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
            tag = intent['tag']
            tags.append(tag)

            for pattern in intent['patterns']:
                w = tokenize(pattern)
                all_words.extend(w)
                xy.append((w,tag))

    ignore_words = [',','?','/','.','!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    x_train = []
    y_train = []

    for (pattern_sentence,tag) in xy:
        bag = bag_of_words(pattern_sentence,all_words)
        x_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    num_epochs = 1000
    batch_size = 8
    learning_rate = 0.001
    input_size = len(x_train[0])
    hidden_size = 8
    output_size = len(tags)

    print(">> Training Friday:- Working ")
    Speak("Training FRIDAY:- Working")
    class ChatDataset(Dataset):

            def __init__(self):
                self.n_samples = len(x_train)
                self.x_data = x_train
                self.y_data = y_train

            def __getitem__(self,index):
                return self.x_data[index],self.y_data[index]

            def __len__(self):
                return self.n_samples
            
    dataset = ChatDataset()

    train_loader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size,hidden_size,output_size).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
            for (words,labels)  in train_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)
                outputs = model(words)
                loss = criterion(outputs,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch+1) % 100 ==0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final Loss : {loss.item():.4f}')
    data = {
        "model_state":model.state_dict(),
        "input_size":input_size,
        "hidden_size":hidden_size,
        "output_size":output_size,
        "all_words":all_words,
        "tags":tags
        }

    FILE = "C:\\Users\\Lenovo\\bigboss\\jarvis.ai\\Jarvis Source Code\\AI Jarvis Using Python Tut\\DataBase\\Tasks.pth"
    
    torch.save(data,FILE)
    print(f"Training Complete sir, File Saved To {FILE}")
    Speak("Training Complete sir...file saved")

def train_DUM_E():
    with open("Real_projects\\DUM-E\\Commands.json",'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
            tag = intent['tag']
            tags.append(tag)

            for pattern in intent['patterns']:
                w = tokenize(pattern)
                all_words.extend(w)
                xy.append((w,tag))

    ignore_words = [',','?','/','.','!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    x_train = []
    y_train = []

    for (pattern_sentence,tag) in xy:
        bag = bag_of_words(pattern_sentence,all_words)
        x_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    num_epochs = 1000
    batch_size = 8
    learning_rate = 0.001
    input_size = len(x_train[0])
    hidden_size = 8
    output_size = len(tags)

    print(">> Training DUM-E:- Working ")
    Speak("Training DUMMY:- Working")
    class ChatDataset(Dataset):

            def __init__(self):
                self.n_samples = len(x_train)
                self.x_data = x_train
                self.y_data = y_train

            def __getitem__(self,index):
                return self.x_data[index],self.y_data[index]

            def __len__(self):
                return self.n_samples
            
    dataset = ChatDataset()

    train_loader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size,hidden_size,output_size).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
            for (words,labels)  in train_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)
                outputs = model(words)
                loss = criterion(outputs,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch+1) % 100 ==0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final Loss : {loss.item():.4f}')
    data = {
        "model_state":model.state_dict(),
        "input_size":input_size,
        "hidden_size":hidden_size,
        "output_size":output_size,
        "all_words":all_words,
        "tags":tags
        }

    FILE = "Real_projects\\DUM-E\\Functions.pth"
    
    torch.save(data,FILE)
    print(f"Training Complete sir, File Saved To {FILE}")
    Speak("Trained DUMMY BOSS , file saved")
train_DUM_E()