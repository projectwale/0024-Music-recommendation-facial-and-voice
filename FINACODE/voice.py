
import librosa
import librosa.display
import numpy as np

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
from matplotlib.pyplot import specgram
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.keras import regularizers
from tensorflow.keras.models import model_from_json
import os
import pyaudio
import wave
import pandas as pd
from filechunktextgenerate import filechunkandgenartetext
from pydub import AudioSegment 
import speech_recognition as sr 
import os
import time
from tensorflow.keras import optimizers
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

opt = optimizers.RMSprop(lr=0.00001, decay=1e-6)

mylist= os.listdir('RawData/')
feeling_list=[]        
labels = pd.read_csv('filelist.csv')
df = pd.DataFrame(columns=['feature'])

#rnewdf.to_csv('featurevoiceemotion.csv')
rnewdf= pd.read_csv('featurevoiceemotion.csv')
newdf1 = np.random.rand(len(rnewdf)) < 0.8
train = rnewdf[newdf1]
test = rnewdf[~newdf1]
trainfeatures = train.iloc[:, :-1]
trainlabel = train.iloc[:, -1:]
testfeatures = test.iloc[:, :-1]
testlabel = test.iloc[:, -1:]
opt = optimizers.RMSprop(lr=0.00001, decay=1e-6)
from tensorflow.python.keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))
CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 1 
RATE = 44100 #sample rate
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output12.wav"

def filecallingvoice():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

    print("* done recording")

    #stream.stop_stream()
    stream.close()
    #p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    datalist,textofvoiceis=filechunkandgenartetext()
    return datalist,textofvoiceis
from flask import Flask, request, render_template
import datetime as datetime
import os

from tensorflow.keras import backend as K
import tensorflow as tf
graph = tf.compat.v1.reset_default_graph()
#import detect_blinks
#recordingvoice.recordvoice()

outputobt=[]

   
def loadingmodel(filename):
    #tf.keras.backend.clear_session()
    
    #global loaded_model
    #graph = tf.get_default_graph()

    #model12 = load_model('my_model.h5')


# extracting features from the images using pretrained model
    #with graph.as_default():
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
    print("Loaded model from disk")
 
        # evaluate loaded model on test data
        #loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    X, sample_rate = librosa.load(filename, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)
    livepreds = loaded_model.predict(twodim, 
                         batch_size=32, 
                         verbose=1)
        #K.clear_session()
    liveabc = livepreds.astype(int).flatten()
    livepreds1=livepreds.argmax(axis=1)
    liveabc = livepreds1.astype(int).flatten()
    livepredictions =(lb.inverse_transform((liveabc)))
    print("=================================================================================")
    
    print('op',livepredictions)
    print('op',livepredictions[0])
    
    print("=================================================================================")
        
    return livepredictions[0]
        #return livepredictions[0]
        
def filechunkandgenartetext():
    listofamotionsinchunk=[]
    textofvoiceis=[]
# Input audio file to be sliced 
    audio = AudioSegment.from_wav("output12.wav") 

    n = len(audio) 
  
# Variable to count the number of sliced chunks 
    counter = 1
  
# Text file to write the recognized audio 
    fh = open("recognized.txt", "w+") 
  
# Interval length at which to slice the audio file. 
# If length is 22 seconds, and interval is 5 seconds, 
# The chunks created will be: 
# chunk1 : 0 - 5 seconds 
# chunk2 : 5 - 10 seconds 
# chunk3 : 10 - 15 seconds 
# chunk4 : 15 - 20 seconds 
# chunk5 : 20 - 22 seconds 
    interval = 5 * 1000
  
# Length of audio to overlap.  
# If length is 22 seconds, and interval is 5 seconds, 
# With overlap as 1.5 seconds, 
# The chunks created will be: 
# chunk1 : 0 - 5 seconds 
# chunk2 : 3.5 - 8.5 seconds 
# chunk3 : 7 - 12 seconds 
# chunk4 : 10.5 - 15.5 seconds 
# chunk5 : 14 - 19.5 seconds 
# chunk6 : 18 - 22 seconds 
    overlap = 0.00 * 1000
  
# Initialize start and end seconds to 0 
    start = 0
    end = 0
  
# Flag to keep track of end of file. 
# When audio reaches its end, flag is set to 1 and we break 
    flag = 0
    folderchunk='audiochunk//'
    try:
        os.rmdir(folderchunk)
    except:
        if not os.path.exists(folderchunk):
            os.mkdir(folderchunk)
# Iterate from 0 to end of the file, 
# with increment = interval 
    for i in range(0, 2 * n, interval): 
      
    # During first iteration, 
    # start is 0, end is the interval 
        if i == 0: 
            start = 0
            end = interval 
  
    # All other iterations, 
    # start is the previous end - overlap 
    # end becomes end + interval 
        else: 
            start = end - overlap 
            end = start + interval  
  
    # When end becomes greater than the file length, 
    # end is set to the file length 
    # flag is set to 1 to indicate break. 
        if end >= n: 
            end = n 
            flag = 1
  
    # Storing audio file from the defined start to end 
        chunk = audio[start:end] 
  
    # Filename / Path to store the sliced audio 
        filename = folderchunk+'chunk'+str(counter)+'.wav'
  
    # Store the sliced audio file to the defined path 
        chunk.export(filename, format ="wav") 
    # Print information about the current chunk 
        print("Processing chunk "+str(counter)+". Start = "
                        +str(start)+" end = "+str(end)) 
  
    # Increment counter for the next chunk 
        counter = counter + 1
        try:
            datais=loadingmodel(filename)
            listofamotionsinchunk.append(datais)
        except Exception as e:
            print(e)
            listofamotionsinchunk.append("")
            pass
        
        r = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio1 = r.record(source)
        
        try:
            command = r.recognize_google(audio1)
            print(command)
            textofvoiceis.append(command)
            fh.write(command+"\n")
        #time.sleep(1)
        except:
            textofvoiceis.append("")
            pass
    
    fh.close()
    print(textofvoiceis)
    print(listofamotionsinchunk)
    return listofamotionsinchunk,textofvoiceis
      
    # Slicing of the audio file is done. 
    # Skip the below steps if there is some other usage 
    # for the sliced audio files. 
#loadingmodel()
filecallingvoice()
#data=loadingmodel()