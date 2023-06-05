import random 
import json
import torch
from Neural_network import NeuralNet
from Neural_Networkx import bag_of_words , tokenize 
import speech_recognition as sr
def Listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source=source , duration=1)
        audio = r.listen(source)
    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language='en-in') #Using google for voice recognition.
        print(f"Boss you said: {query}\n")  #User query will be printed.

    except Exception as e:
       print("Say that again please...")   #Say that again will be printed in case of improper voice 
       return "None" #None string will be returned
    query = str(query)
    return query.lower()
   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('jarvis.ai\\jarvis.ai\\Data\\intents.json','r') as json_data:
    intents = json.load(json_data)

FILE = "jarvis.ai\\jarvis.ai\\Data\\Tasks.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size , hidden_size , output_size).to(device)
model.load_state_dict(model_state)
model.eval()

NAME = 'YOUR AI MODEL'

def FridayRun():
    query = Listen()
    sentence = str(query)

    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)

    _ , predicted = torch.max(output,dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:

        for intent in intents['intents']:

            if tag == intent['tag']:
                reply = random.choice(intent["responses"])
                reply = str(reply).lower()
                print(reply)
            #You can now add functions by using reply variable which predicts the output of your json data
            '''for example :
            if 'play songs' in reply:
                play_songs(spotify)'''
        
while True:
    FridayRun()
