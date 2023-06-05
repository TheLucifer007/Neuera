import random 
import json
import torch
from Neural_network import NeuralNet
from Neural_Networkx import bag_of_words , tokenize 
from main import Speak
import speech_recognition as sr
import sysfunc
import search 
import powerful_functions as pf
from sysfunc import intercheck
from ipfinder import urlipfinder , myipfinder
from subdomain_scanner import subdomain_scanner
import basic_functions
def offline_Speech():
    from vosk import Model, KaldiRecognizer
    import pyaudio
    try:
        model = Model("C:\\Users\\Lenovo\\bigboss\\Friday.ai\\Friday.ai\\all\\vosk-model-en-in-0.5")
        recognizer = KaldiRecognizer(model, 16000)
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        stream.start_stream()
        print("")
        print("Listening...")
        print("")

        while True:

            data = stream.read(4096)

            if recognizer.AcceptWaveform(data):
                text = recognizer.Result()
                
                print(f"You Said : {text}")

                return text
    except Exception:
        text= "None"
        return text
    
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

NAME = 'Friday'

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
            
                if "date" in reply:
                    sysfunc.date()
                elif "wikipedia" in reply:
                    inputexecution(reply,sentence)            
                elif "yes" in reply:
                    pass
                elif "no" in reply:
                    pass
                elif "google_search" in reply:
                    inputexecution(reply , sentence)
                elif "youtube search" in reply:
                    inputexecution(reply , sentence)
                elif "news" in reply:
                    pf.news("india")
                elif "run_window" in reply:
                    sysfunc.run_window()
                elif "spotify_search" in reply:
                    inputexecution(reply , sentence)
                elif "hidden_menu" in reply:
                    sysfunc.hiddenmenu()
                elif "task_manager" in reply:
                    sysfunc.taskmanager()
                elif "close_this_app" in reply:
                    sysfunc.closetheapp()
                elif "volume_up" in reply:
                    sysfunc.volumeup()
                elif "volume_down" in reply:
                    sysfunc.volumedown()
                elif "mute" in reply:
                    sysfunc.mute()
                elif "snip" in reply:
                    sysfunc.snip()
                elif "lockscreen" in reply:
                    sysfunc.lockscreen()
                elif "new_virtual_desktop" in reply:
                    sysfunc.new_virtual_desktop()
                elif "check_internet" in reply:
                    sysfunc.intercheck(host='http://google.com')
                elif "system_info" in reply:
                    sysfunc.sysinfo()
                elif "time" in reply :
                    sysfunc.timex()
                elif "youtube_search" in reply :
                    inputexecution(reply , sentence)
                elif "news" in reply :
                    inputexecution(reply , sentence)
                elif "wolframalpha" in reply :
                    inputexecution(reply , sentence)
                elif "virtual_mouse" in reply:
                    Speak('Starting virtual mouse Boss')
                    pf.virtual_mouse()
                elif "distance_measure" in reply:
                    Speak("Measuring distance froom camera to your face Boss")
                    pf.distancemeasure()
                elif "image_to_text" in reply:
                    Speak("starting image to text Boss : ")
                    pf.imgtotxt()
                elif "screenshot" in reply:
                    sysfunc.screenshot()
                elif "settings" in reply:
                    sysfunc.settings()
                elif "my_ip_finder" in reply:
                    myipfinder()
                elif "url_ip_finder" in reply:
                    urlipfinder()
                elif "open_ai_brain" in reply:
                    import brain
                    brain.anyreply("sentence")
                elif "subdomain_scanner" in reply:
                    subdomain_scanner()
                elif "speedtest" in reply:
                    basic_functions.speedtest()
                elif "qr_code_generator" in reply:
                    basic_functions.qr_code_generator()
                elif "pdf_reader" in reply:
                    basic_functions.pdf_reader()
                elif "audio_extractor" in reply:
                    basic_functions.audio_extractor()
                elif "qr_code_generator" in reply:
                    basic_functions.qr_code_generator()
                elif "translator" in reply:
                    basic_functions.translate()
                elif "playing_music_BOSS" in reply:
                    Speak(reply)
                    basic_functions.play_music()
                elif "bluetooth_scanning" in query:
                    basic_functions.bluettoth_scanner()
                elif "wifi_scanner" in reply:
                    basic_functions.wifi_scanner()
                elif "wifi_connect" in reply:
                    basic_functions.wifi_connect()
                elif "maximize" in reply:
                    sysfunc.maximize()
                elif "minimize" in reply:
                    basic_functions.minimize()
                elif "face_distance_measure" in reply:
                    pf.distancemeasure()
                elif "youtube_video_downloader" in reply:
                    pf.youtube_video_Downloader()
                elif "number_scanner" in reply:
                    pf.num_scanner()
                elif "my_youtube_channel" in reply:
                    search.my_youtube_channel()
                elif "esatsang" in reply:
                    search.esatsang()
                else:
                    Speak(reply)
def inputexecution(tag, query):
    query = str(query)
    if 'wikipedia' in tag:
        query = str(query)
        query = query.replace("who is" , "")
        search.wikipedia(query)
    elif 'google search' in tag :
        query = query.replace("google" , "")
        query = query.replace("on" , "")
        query = query.replace("do a" , "")
        query = query.replace("search" , "")
        query = query.replace("par" , "")
        query = query.replace("karna" , "")
        query = query.replace("Friday" , "")
        query = query.replace("execute a")
        search.googlesearch(query)
    elif 'spotify_search' in tag:
        search.spotify_search(query)
    elif 'youtube_search' in tag:
        query = query.replace("search", "")
        query = query.replace("on", "")
        query = query.replace("youtube", "")
        query = query.replace("Friday", "")
        search.YouTubesearch(query)
    elif 'news' in tag:
        query = query.replace("today", "")
        query = query.replace("news", "")
        query = query.replace("from", "")
        query = query.replace("play", "")
        query = query.replace("aaj", "")
        query = query.replace("ki", "")
        query = query.replace("headlines", "")
        query = query.replace("khabhar", "")
        query = query.replace("Friday", "")
        query = query.replace("taaza", "")
        pf.news(query)
    elif 'wolframalpha' in tag:
        query = query.replace("Friday", "")
        query = query.replace("solve", "")
        query = query.replace("this", "")
        query = query.replace("question", "")
        query = query.replace("answer", "")
        query = query.replace("karna", "")
        query = query.replace("iss", "")
        query = query.replace("maths", "")
        query = query.replace("science", "")
        query = query.replace("doubt", "")
        query = query.replace("ko", "")
        query = query.replace("fix", "")
        query = query.replace("problem", "")
        query = query.replace("my", "")
        pf.wolframalphax(query)
    elif 'translator' in tag:
        query = query.replace("translate" , "")
        query = query.replace("translator")
        query = query.replace("this")
        query = query.replace("iss ko translate")
        query = query.replace("karna" , "")
        query = query.replace("convert" , "")
        query = query.replace("into" , "")
        query = query.replace("english" , "")
        query = query.replace("language" , "")
        basic_functions.translate(query)
while True:
    FridayRun()