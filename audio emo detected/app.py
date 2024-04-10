from flask import Flask, render_template, request, jsonify 
import pickle
import soundfile
import numpy as np
import librosa
import os
from pydub import AudioSegment  
from pydub.playback import play 
import glob
from tqdm import tqdm
import pandas as pd
from scipy.io import wavfile
import json
import random
from bs4 import BeautifulSoup as SOUP
import re
import requests as HTTP


with open(f'model/Emotion_Voice_Detection_Model.pkl', 'rb') as f: #opening the trained model
    model = pickle.load(f) #implementing binary protocol (trained model) deserialize the data stream
app = Flask(__name__, template_folder='templates') #for rendering templates (html pages) for using render_template function

def extract_feature(file_name, mfcc, chroma, mel): #feature extraction function
    with soundfile.SoundFile(file_name) as sound_file: #opening the audio file as SoundFile object
        X = sound_file.read(dtype="float32") #reading the features of sound file and returning it as floating point audio data
        sample_rate=sound_file.samplerate #sample rate of the SoundFile object
        if chroma: #chroma - descriptor which represents the tonal content of a musical audio signal in a condensed form
            stft=np.abs(librosa.stft(X)) #creating stft (short time fourier series) by calculating absolute value on each element in the short time fourier series of floating point audio data
        result=np.array([]) #returns nd array - multidimensional homogeneous array of fixed size items
        if mfcc: #mfcc - scales the frequency in order to match more closely what the human ear can hear
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0) #compute the arithmetic mean of mfcc sequence (with specified arguments) along the specified axis - returning nd array
        result=np.hstack((result, mfccs)) #stacks array sequence horizontally - array formed by stacking the given arrays - returns stacked nd array 
        if chroma: #calculating arithmetic mean of already computed stft(short time fourier series) and stacking the array
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0) #computing the arithmetic mean of chroma stft (with specified arguments) along the specified axis
        result=np.hstack((result, chroma)) #stacks array sequence horizontally - array formed by stacking the given arrays - returns stacked nd array
        if mel: #applies the Mel-frequency scaling, which is perceptual scale that helps to simulate the way human ear works
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0) #computing arithmetic mean of mel spectogram (with specified arguments) along the specified axis
        result=np.hstack((result, mel)) #stacks array sequence horizontally - array formed by stacking the given arrays - returns stacked nd array
    return result #returns the result nd array
#Now Cleaning Step is Performed where:
#DOWN SAMPLING OF AUDIO FILES IS DONE  AND PUT MASK OVER IT AND DIRECT INTO CLEAN FOLDER
#MASK IS TO REMOVE UNNECESSARY EMPTY VOIVES AROUND THE MAIN AUDIO VOICE 
def envelope(y , rate, threshold): 
    mask=[] #new array
    y=pd.Series(y).apply(np.abs) #apply() - invoke passed function (np.abs - numpy.absolute : calculate absolute value element wise) on each element of given series 
    #returns data frame if function returns series object
    y_mean = y.rolling(window=int(rate/10) ,  min_periods=1 , center = True).mean() #calculating the mean over each window of selected size (rate/10)
    for mean in y_mean: #checking the calculated mean for every rate/10 elements
        if mean>threshold: #if mean > 0.0005 the audio part will be added
            mask.append(True) #appending true to mask array to add the audio
        else: #else the audio part will be removed
            mask.append(False) #appending false to remove the audio part
    return mask #return the mask array

#for index page route
@app.route('/')
def home():
    return render_template('main.html') #whenever project run, main.html will be loaded (path='/')

def my_form_post():
    filee = './songs/output10.wav' #pathname for the recorded and saved audio file 
    for file in tqdm(glob.glob(r'C:/Users/m.vishnu.priya/Desktop/audio-emotion detected/songs//*.wav')): #tqdm - prgress bar for opening the file; glob.glob - used to retrieve files or pathnames matching the path
        file_name = os.path.basename(file) #returns the tail part after spliting the specified path into (head,tail) [retrieving the file name]
        signal , rate = librosa.load(file, sr=16000) #audio file can be resampled to given rate and returning nd array of floating point values
        mask = envelope(signal,rate, 0.0005) #calling the masking function with threshold value = 0.0005
        #saving the cleaned and masked file in the songs folder in the same name
        wavfile.write(filename= r'C:/Users/m.vishnu.priya/Desktop/audio-emotion detected/songs//'+str(file_name), rate=rate,data=signal[mask]) #writing a numpy array as wavfile and signal[mask] - masking the values in signal nd array using mask array from envelope function


    #wav_file = AudioSegment.from_file(file, format="wav")
    ans =[] #new array
    new_feature = extract_feature(filee, mfcc=True, chroma=True, mel=True) #calling feature extraction function (all 3 features - mfcc,chroma,mel)
    ans.append(new_feature) #appending extracted feature into ans array
    ans = np.array(ans) #Numpy nd array 
    result = model.predict(ans)[0] #predicting the emotion using trained model from ans array 
    return result #detected emotion (String)

#route for recording voice html page
@app.route('/record', methods=['GET','POST'])
def record():
    import pyaudio
    import wave

    CHUNK = 1024 #number of frames in a buffer
    FORMAT = pyaudio.paInt16 #paInt8 16-bit binary string
    CHANNELS = 1 #each frame has single sample
    RATE = 44100 #sample rate
    RECORD_SECONDS = 4 #recording time
    WAVE_OUTPUT_FILENAME = "./songs/output10.wav" #

    p = pyaudio.PyAudio() #To use PyAudio, first instantiate PyAudio using pyaudio.PyAudio(), which sets up the portaudio system.
    
    #sample rate, channels and format of the stream have to match the wav parameters
    #To record or play audio, open a stream on the desired device with the desired audio parameters using pyaudio.
    #Return type: A PortAudio Sample Format constant
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) #buffer

    print("* recording") #print statement
    frames = [] #new array

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)): #for loop for reading and appending frame
        data = stream.read(CHUNK) #read audio data from the stream using pyaudio.Stream.read()
        frames.append(data) # 2 bytes(16 bits) per channel

    print("* done recording")

    stream.stop_stream() #Use pyaudio.Stream.stop_stream() to pause playing/recording
    stream.close() #pyaudio.Stream.close() to terminate the stream
    p.terminate() #terminate the portaudio session using pyaudio.PyAudio.terminate()

    #The wb indicates that the file is opened for writing in binary mode. When writing in binary mode, Python makes no changes to data as it is written to the file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb') #if file is a string, open the file by that name, otherwise treat it as a file-like object. mode of 'wb' returns a Wave_write object
    wf.setnchannels(CHANNELS) #Set the number of channels
    wf.setsampwidth(p.get_sample_size(FORMAT)) #Set the sample width to n bytes
    wf.setframerate(RATE) #Set the frame rate to n
    wf.writeframes(b''.join(frames)) #Write audio frames and make sure nframes is correct. It will raise an error if the output stream is not seekable and the total number of frames that have been written after data has been written does not match the previously set value for nframes.
    wf.close() #Make sure nframes is correct, and close the file if it was opened by wave. This method is called upon object collection. It will raise an exception if the output stream is not seekable and nframes does not match the number of frames actually written.
    result = my_form_post() #calling function to retrieve the result
    #display(result)
    if result=="happy" or result=="disgust":
        return emotion(result)
    elif result=="fearful" or result=="calm":
        temp=movie(result)
        output=temp.split("/")
        return render_template('result.html',data=result,category="Movie Recommendation",m1=output[0],m2=output[1],m3=output[2],m4=output[3])
    else:
        return render_template('result.html',data=result)


def emotion(emo):    
    num = random.randint(1,50)
    result1=""
    result2=""
    result3=""
    cat=""
    if emo=="happy":
        data_file = open('happy.json',)
        intents = json.load(data_file)
        for i in intents["happy"]:
            #print(i)
            if i["number"]==num:
               if i["category"]=="sentence":
                    result1=i["data"] 
                    #return render_template('result.html',data=result,data1=result1)
               if i["category"]=="youtube":
                    cat="Video"
                    result2=i["link"]
               if i["category"]=="activity":
                    cat="Activity"
                    x=i["data"]
                    y=x.split("--")
                    result1=y[0]
                    result2=y[1]
               if i["category"]=="poem":
                    cat="Poem"
                    result1=i["data"]
               if i["category"]=="picture":
                    cat="Picture"
                    result3=i["link"]
               return render_template('result.html',data=emo,data1=result1,data2=result2,data3=result3,category=cat) #redirecting the result to result.html page
    if emo=="disgust":
        data_file = open('anger.json',encoding='utf-8')
        intents = json.loads(data_file.read())
        for i in intents["anger"]:
            if i["number"]==num:
                if i["category"]=="sentence":
                    result1=i["data"]
                if i["category"]=="link":
                    cat="Video"
                    result2=i["data"]
                if i["category"]=="fact":
                    cat="Fact"
                    result1=i["data"]
                if i["category"]=="joke":
                    cat="Joke"
                    result1=i["data"]
                if i["category"]=="exercise":
                    cat="Exercise"
                    result1=i["data"]
                if i["category"]=="yoga":
                    cat="Yoga"
                    result1=i["data"]
                if i["category"]=="food":
                    cat="Food"
                    result1=i["data"]
                return render_template('result.html',data=emo,data1=result1,data2=result2,data3=result3,category=cat)

    else:
        return render_template('result.html',data=emo)




def webscrap(emotion):
  
    # IMDb Url for Drama genre of
    # movie against emotion Sad
    if(emotion == "Sad"):
        urlhere = 'http://www.imdb.com/search/title?genres=drama&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Musical genre of
    # movie against emotion Disgust
    elif(emotion == "Disgust"):
        urlhere = 'http://www.imdb.com/search/title?genres=musical&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Family genre of
    # movie against emotion Anger
    elif(emotion == "Anger"):
        urlhere = 'http://www.imdb.com/search/title?genres=family&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Thriller genre of
    # movie against emotion Anticipation
    elif(emotion == "Anticipation"):
        urlhere = 'http://www.imdb.com/search/title?genres=thriller&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Sport genre of
    # movie against emotion Fear
    elif(emotion == "fearful"):
        urlhere = 'http://www.imdb.com/search/title?genres=sport&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Thriller genre of
    # movie against emotion Enjoyment
    elif(emotion == "Enjoyment"):
        urlhere = 'http://www.imdb.com/search/title?genres=thriller&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Western genre of
    # movie against emotion Trust
    elif(emotion == "calm"):
        urlhere = 'http://www.imdb.com/search/title?genres=western&title_type=feature&sort=moviemeter, asc'
  
    # IMDb Url for Film_noir genre of
    # movie against emotion Surprise
    elif(emotion == "Surprise"):
        urlhere = 'http://www.imdb.com/search/title?genres=film_noir&title_type=feature&sort=moviemeter, asc'
  
    # HTTP request to get the data of
    # the whole page
    response = HTTP.get(urlhere)
    data = response.text
  
    # Parsing the data using
    # BeautifulSoup
    soup = SOUP(data, "lxml")
  
    # Extract movie titles from the
    # data using regex
    title = soup.find_all("a", attrs = {"href" : re.compile(r'\/title\/tt+\d*\/')})
    return title


def movie(emotion):
    a = webscrap(emotion)
    temp=""
    j=0
    count = 0
  
    if(emotion == "Disgust" or emotion == "Anger"
                           or emotion=="Surprise"):
  
        for i in a:
  
            # Splitting each line of the
            # IMDb data to scrape movies
            tmp = str(i).split('>;')
  
            if(len(tmp) == 3):
                temp = temp+tmp[1][:-3]
                temp=temp+"/"
                #j=j+1
  
            if(count > 13):
                break
            count += 1

    else:
        for i in a:
            tmp = str(i).split('>')
  
            if(len(tmp) == 3):
                #temp=temp+"/"
                temp = temp+tmp[1][:-3]
                temp=temp+"/"
                #j=j+1
  
            if(count > 11):
                break
            count+=1  

    return temp  

    



    

#app.py is made to run whenever flask starts to run (whenever main.py runs as main program app.py is made to run)
if __name__ == '__main__':
    app.run() 