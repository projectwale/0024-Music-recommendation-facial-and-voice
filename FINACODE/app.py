
from flask import Flask,render_template,request,session, url_for, redirect
#from flask_mysqldb import MySQL
import warnings

# from voiceemotiontest import callFuction

from model import FacialExpressionModel
warnings.simplefilter(action='ignore', category=FutureWarning)
#from sklearn import datasets
import pickle
import pymysql
import numpy as np
import math
import pickle
import collections

import numpy as np
import json
import tflearn
import pandas as pd
import nltk
import re
import sys
import warnings
import random
warnings.simplefilter(action='ignore', category=FutureWarning)
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# from voice import filecallingvoice

from werkzeug.utils import secure_filename

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
UPLOAD_FOLDER = 'static/uploadedimages'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','wav','mp3'}



global usernm
usernm = ""


app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'random string'
import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emotionlistforadding=['Angry','Disgust','Fearful','Happy','Neutral','Sad','Surprise']

global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]
global listofemotion
listofclasses=['ENFJ',
 'ENFP',
 'ENTJ',
 'ENTP',
 'ESFJ',
 'ESFP',
 'ESTP',
 'INFJ',
 'INFP',
 'INTJ',
 'INTP',
 'ISFJ',
 'ISFP',
 'ISTJ',
 'ISTP']
Corpus = pd.read_csv(r"processed_data.csv",encoding='latin-1',nrows=10,error_bad_lines=False)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])

##############################################################################################################
                                             #startface emotion base songs recomended
##############################################################################################################

global emotion
emotion=""
import spotipy
#cid = 'eb9d53df11484a72b61c48259ffd7c8b'
#secret = '33f5f097a1404c8082bf1ce05b9c23bf'
cid="0f0b1633f7f74a0c937e46d42de6497c"
secret="4fd5e59894cf459686ff971fd0731b6a"
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials

#spotify = spotipy.Spotify(auth_manager = SpotifyOAuth(client_id = cid,
                                                      #client_secret =secret,
                                                      #redirect_uri = 'http://localhost:5000/callback'))
auth_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
spotify = spotipy.Spotify(auth_manager=auth_manager)
def valid_token(resp):
    return resp is not None and not 'error' in resp


def make_search(search_type, name):
    print("========================")
    print(search_type)
    print(name)
    data = spotify.search(name,limit=40,type="track")   
    api_url = data['tracks']['items']
    items = data['tracks']['items']
    # print(items)
    # print(api_url)
    #print(items)
    p=[]
    for i in range(len(items)):
        b=items[i]['id']
        p.append(b)
    import random
    items = random.sample(items, len(items))
    print("hiiiiiiiiiiiiiii")

    return render_template('testing.html',
                           name=name,
                           results=items,
                           api_url=api_url, 
                           search_type=search_type)


##############################################################################################################
                                             #start search all songes
##############################################################################################################

def make_search2(search_type, name):
    data = spotify.search(name,limit=20,type="track")   
    api_url = data['tracks']['items']
    items = data['tracks']['items']
    #print(items)
    p=[]
    for i in range(len(items)):
        b=items[i]['id']
        p.append(b)
    import random
    items = random.sample(items, len(items))
    dst=os.listdir("static/uploadeddetected")
    k=dst[0]

    return render_template('search2.html',
                           name=name,
                           results=items,
                           api_url=api_url,
                           k=k,
                           search_type=search_type)



##############################################################################################################
                                             #database connection
##############################################################################################################


# import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
import os


import pandas as pd
import cv2
import numpy as np



def dbConnection():
    connection = pymysql.connect(host="localhost", user="root", password="root", database="musicreco")
    return connection

def dbClose():
    dbConnection().close()
    return

con = dbConnection()
cursor = con.cursor()

##############################################################################################################

##############################################################################################################
                                             #import face emotion libary import
##############################################################################################################


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import tensorflow as tf
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res10_300x300_ssd_iter_140000.caffemodel")

UPLOAD_FOLDER = 'static/uploadedimages'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','wav','mp3'}




##############################################################################################################






app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'random string'
@app.route('/')
def index():
    return render_template('login.html')


@app.route("/framehtml",methods=['POST','GET'])
def framehtml():
    return render_template("frames.html")


##############################################################################################################
                                             #import face emotion start camera
##############################################################################################################

@app.route("/imagecaptures1",methods=['POST','GET'])
def imagecaptures1():
    try:
        global emotion
        import cv2
        import imutils
        #from cv2 import *
        from tensorflow.keras.models import model_from_json
        model = model_from_json(open("fer.json", "r").read())
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt","res10_300x300_ssd_iter_140000.caffemodel")
        Emotion_list=[]
    # initialize camera stream
        vid=cv2.VideoCapture(0)
        # loop over the frames from the video stream
        while True:
            ret,frame = vid.read()
            frame = imutils.resize(frame, width=400)
            #Mean subtraction is used to help combat illumination changes in the input images in our dataset
            #view mean subtraction as a technique used to aid our Convolutional Neural Networks.
        
            #Before we even begin training our deep neural network,
            #we first compute the average pixel intensity across all images in the training set for each of the Red, Green, and Blue channels.
            ##1 is scale factor
           ##blob is the binary long object classification
            
         
            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
         
            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()
            #print(detections)
            faceBoxes = []
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
        
                # filter out detections by confidence
                if confidence < 0.7:
                    continue
        
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #print(gray_frame)
                #print(box)
                frameHeight = frame.shape[0]
                frameWidth = frame.shape[1]
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                #faceBoxes=box
                #print(faceBoxes)
                #print(faceBoxes[0][2])
                
                roi_gray_frame = gray_frame[faceBoxes[0][2]:faceBoxes[0][2] + faceBoxes[0][3], faceBoxes[0][0]:faceBoxes[0][0] + faceBoxes[0][3]]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        
                emotionlistforadding=['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']
                #break
                import tensorflow as tf
                #new_model = tf.keras.models.load_model('NewEmotion.hp5')
                #emotion_model.load_weights('emotion_model.h5')
                #emotion_model.load_weights('')
                model.load_weights('fer.h5')
                emotion_prediction = model.predict(cropped_img)
                #print(emotion_prediction)
                maxindex = int(np.argmax(emotion_prediction))
                print(maxindex)
                labelemotion= "{}".format("Emotion : " + emotion_dict[maxindex])
                Emotion_list.append(labelemotion)
            
             
            # draw the bounding box of the face along with the associated
            # probability
                text = "{:.2f}%".format(confidence * 100)
                
                
                ##adding the text for showing emotion
                #it is the coordinates of the bottom-left corner of the text string in the image. The coordinates are represented as tuples of two values
                #Font scale factor that is multiplied by the font-specific base size.(0.8)
                #BGR, we pass a tuple.255 0 0
                #gives the type of the line to be used.
                #2 is thickness
                
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                cv2.putText(frame, labelemotion, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)
            #cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
    
    	# show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
     
        # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        
        vid.release()
    # Destroy all the windows
        cv2.destroyAllWindows()
        from collections import Counter
        occurrences3 = collections.Counter(Emotion_list)
        print(occurrences3)
        Keymax_emotion = max(zip(occurrences3.values(), occurrences3.keys()))[1]
        Final_Emotion=Keymax_emotion[9:]
        emotion = Final_Emotion
        return render_template("showemotion.html", Keymax_emotion=Final_Emotion)
    except cv2.error:
        return "Please look steadily in Camera"
    
        
    
##############################################################################################################


##############################################################################################################
                                             #import upload image
##############################################################################################################

@app.route("/imagecaptures",methods=['POST','GET'])
def imagecaptures():
    import cv2
    import imutils
    #from cv2 import *
    from tensorflow.keras.models import model_from_json
    model = model_from_json(open("fer.json", "r").read())
    videoCaptureObject = cv2.VideoCapture(0)
    i=0
    while(i<=9):
        ret,frame = videoCaptureObject.read()
        cv2.imwrite("static/captureimages/NewPicture"+str(i)+".jpg",frame)
        i=i+1
       # cv2.imshow("images Detector", frame)
        
    videoCaptureObject.release()
    cv2.destroyAllWindows()
    #cv2.imshow("Age-Gender-emotion Detector", frame)
    
    #if cv2.waitKey(1) == ord('q'):
        #break
    emotions=[]
    
    for j in os.listdir("static/captureimages"):
        img = cv2.imread("static/captureimages/"+str(j))
        frame = imutils.resize(img, width=400)
 
    # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
     
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        #print(detections)
        faceBoxes = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
    
            # filter out detections by confidence
            if confidence < 0.7:
                continue
    
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print(gray_frame)
            #print(box)
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            #faceBoxes=box
            #print(faceBoxes)
            #print(faceBoxes[0][2])
            
            roi_gray_frame = gray_frame[faceBoxes[0][2]:faceBoxes[0][2] + faceBoxes[0][3], faceBoxes[0][0]:faceBoxes[0][0] + faceBoxes[0][3]]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
            emotionlistforadding=['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']
            #break
            import tensorflow as tf
            #new_model = tf.keras.models.load_model('emotion_model.h5')
            #emotion_model.load_weights('emotion_model.h5')
            #emotion_model.load_weights('')
            
            
            #load weights
            model.load_weights('fer.h5')
            emotion_prediction = model.predict(cropped_img)
            #print(emotion_prediction)
            maxindex = int(np.argmax(emotion_prediction))
            print(maxindex)
            labelemotion= "{}".format("Emotion : " + emotion_dict[maxindex])
            emotions.append(labelemotion)
            
             
            # draw the bounding box of the face along with the associated
            # probability
            #text = "{:.2f}%".format(confidence * 100)
            
            
            
            
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(frame, labelemotion, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite("static/detectimages/NewPicture"+str(j)+".jpg", frame)
    import collections 
    from collections import Counter
    occurrences3 = collections.Counter(emotions)
    print(occurrences3)
    Keymax_emotion = max(zip(occurrences3.values(), occurrences3.keys()))[1]
    print(Keymax_emotion)
    Keymax_emotion=Keymax_emotion[10:]
    print(Keymax_emotion)
    if Keymax_emotion=="Happy":
        search_type="track"
        name="party"
        return make_search(search_type, name)
    elif Keymax_emotion=="Angry":
        search_type="track"
        name="sad"
        return make_search(search_type, name)
    elif Keymax_emotion=="Disgusted":
        search_type="track"
        name="motivational"
        return make_search(search_type, name)
    elif Keymax_emotion=="Fearful":
        search_type="track"
        name="motivational"
        return make_search(search_type, name)
    elif Keymax_emotion=="Neutral":
        search_type="track"
        name="party"
        return make_search(search_type, name)
    elif Keymax_emotion=="Sad":
        search_type="track"
        name="sad"
        return make_search(search_type, name)
    else:
        search_type="track"
        name="surprised"
        return make_search(search_type, name)
        
        
        
    msg="Something Camera Incompatibility Issue"
    return msg


##############################################################################################################

@app.route("/uploadimage",methods=['POST','GET'])
def uploadimage():
    if request.method == "POST":
        import imutils
        emotions=[]
        file = request.files['file']
        from werkzeug.utils import secure_filename
        from werkzeug.datastructures import  FileStorage
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread("static/uploadedimages/"+str(filename))
        frame = imutils.resize(img, width=400)
 
    # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
     
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        print("++++++++++++++++++++++++++++++++")
        #print(detections)
        faceBoxes = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
    
            # filter out detections by confidence
            if confidence < 0.7:
                continue
    
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print(gray_frame)
            #print(box)
            
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            #faceBoxes=box
            #print(faceBoxes)
            #print(faceBoxes[0][2])
            
            roi_gray_frame = gray_frame[faceBoxes[0][2]:faceBoxes[0][2] + faceBoxes[0][3], faceBoxes[0][0]:faceBoxes[0][0] + faceBoxes[0][3]]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
            emotionlistforadding=['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']
            print("------")
            #break
            import tensorflow as tf
            emotion_model = tf.keras.models.load_model('NewEmotion.hp5')
            print("-------------------")
            # emotion_model.load_weights('emotion_model.h5')
            #emotion_model.load_weights('')
            print("----------------------------------------------")
            emotion_prediction = emotion_model.predict(cropped_img)
            #print(emotion_prediction)
            maxindex = int(np.argmax(emotion_prediction))
            print(maxindex)
            labelemotion= "{}".format("Emotion : " + emotion_dict[maxindex])
            emotions.append(labelemotion)
            
             
            # draw the bounding box of the face along with the associated
            # probability
            #text = "{:.2f}%".format(confidence * 100)
            
            
            print("++++++++++++++++++++++++++++++++")
            
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(frame, labelemotion, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite("static/uploadeddetected/NewPicture"+str(filename)+".jpg", frame)
    import collections 
    from collections import Counter
    occurrences3 = collections.Counter(emotions)
    print(occurrences3)
    Keymax_emotion = max(zip(occurrences3.values(), occurrences3.keys()))[1]
    print(Keymax_emotion)
    Keymax_emotion=Keymax_emotion[10:]
    print(Keymax_emotion)
    return render_template("showemotion.html", Keymax_emotion=Keymax_emotion)
    #if Keymax_emotion=="Happy":
    #    search_type="track"
    #    name="party"
    #    return make_search2(search_type, name)
    #elif Keymax_emotion=="Angry":
    #    search_type="track"
    #    name="sad"
    #    return make_search2(search_type, name)
    #elif Keymax_emotion=="Disgusted":
    #    search_type="track"
    #    name="motivational"
    #    return make_search2(search_type, name)
    #elif Keymax_emotion=="Fearful":
    #    search_type="track"
    #    name="motivational"
    #    return make_search2(search_type, name)
    #elif Keymax_emotion=="Neutral":
    #    search_type="track"
    #    name="party"
    #    return make_search2(search_type, name)
    #elif Keymax_emotion=="Sad":
    #    search_type="track"
    #    name="sad"
    #    return make_search2(search_type, name)
    #else:
    #    search_type="track"
    #    name="love"
    #    return make_search2(search_type, name)
        
        
        
    
    #return render_template('frames.html')

##############################################################################################################
##############################################################################################################



@app.route("/question_new",methods=['POST','GET'])
def question_new():
    if request.method == "POST":
        global emotion
        print(emotion)
        name2=emotion[1:]
        print(name2)
        answer=request.form.get('mood')
        name = request.form.get('mood2')
        print(answer)
        print(name)
        if answer=="Yes":
            print("name2",name2)
            if name2=="Happy":
                search_type="track"
                con = dbConnection()
                cursor = con.cursor()
                cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
                res = cursor.fetchone()
                res=res[0]
                print(res)
                name="Party "+str(res)
                return make_search(search_type, name)
            if name2=="Sad":
                search_type="track"
                con = dbConnection()
                cursor = con.cursor()
                cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
                res = cursor.fetchone()
                res=res[0]
                print(res)
                name="Party "+str(res)
                return make_search(search_type, name)
            if name2=="Surprised":
               search_type="track"
               con = dbConnection()
               cursor = con.cursor()
               cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
               res = cursor.fetchone()
               res=res[0]
               print(res)
               name="Party "+str(res)
               return make_search(search_type, name)
            if name2=="Neutral":
               search_type="track"
               con = dbConnection()
               cursor = con.cursor()
               cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
               res = cursor.fetchone()
               res=res[0]
               print(res)
               name="Party "+str(res)
               return make_search(search_type, name)
            if name2=="Fearful":
               search_type="track"
               con = dbConnection()
               cursor = con.cursor()
               cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (session['user']))
               res = cursor.fetchone()
               res=res[0]
               print(res)
               name="Party "+str(res)
               return make_search(search_type, name)

##############################################################################################################

@app.route("/callback/")
def callback():

    auth_token = request.args['code']
    auth_header = spotify.authorize(auth_token)
    session['auth_header'] = auth_header

    return profile()
@app.route('/profile')
def profile():
    if 'auth_header' in session:
        auth_header = session['auth_header']
        # get profile data
        profile_data = spotify.get_users_profile(auth_header)

        # get user playlist data
        playlist_data = spotify.get_users_playlists(auth_header)

        # get user recently played tracks
        recently_played = spotify.get_users_recently_played(auth_header)
        
        if valid_token(recently_played):
            return render_template("profile.html",
                               user=profile_data,
                               playlists=playlist_data["items"],
                               recently_played=recently_played["items"])

    return render_template('profile.html')

@app.route('/home')
def home():
    email = session['user']
    cursor.execute('SELECT * FROM userdetailes where email= %s;',(email))
    row = cursor.fetchall()

    return render_template('index.html',row=row)

@app.route('/prediction',methods=['POST','GET'])
def prediction():
    return render_template('prediction.html')

@app.route('/contact',methods=['POST','GET'])
def contact():
    if request.method == "POST":
        Name = request.form.get("Name")
        Email = request.form.get("Email")
        Subject = request.form.get("Subject")
        Message = request.form.get("Message")
        
        sql = "INSERT INTO contact (Name,Email,Subject,Message) VALUES (%s,%s,%s, %s)"
        val =  (Name,Email,Subject,Message)
        print(sql," ",val)
        cursor.execute(sql, val)
        con.commit()
     
        msg = "User query was successfully added."+Email
        return render_template("contact.html",msg=msg)
        
        
        
        
    
    
    return render_template('contact.html')

@app.route('/about',methods=['POST','GET'])
def about():
    return render_template('about.html')

@app.route('/captureimages',methods=['POST','GET'])
def captureimages():
    import cv2
    #from cv2 import *
    videoCaptureObject = cv2.VideoCapture(0)
    i=0
    while(i<=9):
        ret,frame = videoCaptureObject.read()
        cv2.imwrite("static/captureimages/NewPicture"+str(i)+".jpg",frame)
        i=i+1
        
    videoCaptureObject.release()
    cv2.destroyAllWindows()
    return render_template('prediction.html')



@app.route('/question',methods=['POST','GET'])
def question():    
    # usersessionformob = usernm 
    usersessionformob = session['user'] 
    print("--------------------------------")
    print(usersessionformob)            
    if request.method == "POST":
        mood  = request.form.get("selectop")
        name  = request.form.get('happyrad1')
        name2 = request.form.get('happyrad2')            
        # usersessionformob = usernm
        print("--------------------------------")
        print(usersessionformob)
        print("--------------------------------")
        print(mood)
        print(name)
        print(name2)
        if name=="Yes":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (str(usersessionformob)))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="Party "+str(res)
            return make_search(search_type, name)
        if name=="No" and name2=="No":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (str(usersessionformob)))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="Happy "+str(res)
            return make_search(search_type, name)
        if name=="No" and name2=="Yes":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (str(usersessionformob)))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="Calm "+str(res)
            return make_search(search_type, name)
        sadname3 = request.form.get('sadrad1')
        sadname4 = request.form.get('sadrad2')
        sadname5 = request.form.get('sadrad3')
        if sadname3=="Yes":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (str(usersessionformob)))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="Sad "+str(res)
            return make_search(search_type, name)
        if sadname3=="No" and sadname4=="Yes":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (str(usersessionformob)))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="Heartbreak "+str(res)
            return make_search(search_type, name)
        if sadname3=="No" and sadname4=="No":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (str(usersessionformob)))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="Sad "+str(res)
            return make_search(search_type, name)
        if sadname3=="No" and sadname4=="No" and sadname5=="Yes":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (usersessionformob))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="Sad "+str(res)
            return make_search(search_type, name)
        if sadname3=="No" and sadname4=="No" and sadname5=="No":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (usersessionformob))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="Sad "+str(res)
            return make_search(search_type, name)
        angryname6 = request.form.get('angryrad1')
        angryname7 = request.form.get('angryrad2')
        if angryname6=="Yes":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (usersessionformob))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="Calm "+str(res)
            return make_search(search_type, name)
        if angryname6=="No" and angryname7=="Yes":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (usersessionformob))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="Motivaltional "+str(res)
            return make_search(search_type, name)
        if angryname6=="No" and angryname7=="No":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (usersessionformob))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            if res=="hindi":
                name="Angry "+str(res)
                return make_search(search_type, name)
            else:
                playlistid=" 0l9dAmBrUJLylii66JOsHB"
                search_type="playlist"
                make_search(search_type, name)
        
        
        if mood=="Romantic":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (str(usersessionformob)))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="romantic "+str(res)
            return make_search(search_type, name)
        if mood=="Demotivated":
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (usersessionformob))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="motivational "+str(res)
            return make_search(search_type, name)
        else:
            search_type="track"
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT lang FROM userdetailes WHERE email = %s', (usersessionformob))
            res = cursor.fetchone()
            res=res[0]
            print(res)
            name="motivational "+str(res)
            return make_search(search_type, name)
    

            
        #if name==""
        
        #return render_template('question.html')
    return render_template('newtest.html')

##############################################################################################################
                        #registration
##############################################################################################################

@app.route('/register',methods=['POST','GET'] )
def register():
    if request.method == "POST":
        try:
            status=""
            fname = request.form.get("Name")
            mobileno = request.form.get("mobileno")
            Email = request.form.get("Email")
            pass1 = request.form.get("pass1")
           
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT * FROM userdetailes WHERE email = %s', (Email))
            res = cursor.fetchone()
            #res = 0
            if not res:
                sql = "INSERT INTO userdetailes (name,phone,email,password) VALUES (%s,%s,%s, %s)"
                val = (fname ,mobileno ,Email,pass1)
                print(sql," ",val)
                cursor.execute(sql, val)
                con.commit()
                status= "success"
                msg = "User successfully added by admin side."+"username is-"+fname
                return render_template("register.html",msg=msg)
            else:
                status = "Already available"
            #return status
            return redirect(url_for('index'))
        except Exception as e:
            print(e)
            print("Exception occured at user registration")
            return redirect(url_for('index'))
        finally:
            dbClose()
    return render_template('register.html')
##############################################################################################################

@app.route("/logout", methods = ['POST', 'GET'])
def logout():
    # username=session.get('uname')
    session.pop('user',None)
    return redirect(url_for('login'))
##############################################################################################################

##############################################################################################################
                                         #login 
##############################################################################################################

@app.route('/login',methods=['POST','GET'])
def login():
    msg = ''
    if request.method == "POST":
        session.pop('user',None)
        mailid = request.form.get("Email")
        password = request.form.get("password")
        print(mailid, password)
        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM userdetailes WHERE email = %s AND password = %s', (mailid, password))
        #a= 'SELECT * FROM userdetails WHERE mobile ='+mobno+'  AND password = '+ password
        print(result_count)
        #result_count=cursor.execute(a)
        # result = cursor.fetchone()
        if result_count>0:
            print(result_count)
            session['user'] = mailid
            return redirect(url_for('home'))
            # return render_template("home.html")
        else:
            print(result_count)
            msg = 'Incorrect username/password!'
            return msg
    return render_template('login.html')


@app.route('/project.html')
def contact1():
    return render_template('project.html')
@app.route('/analysis.html')
def analysis():
   return render_template('analysis.html')
@app.route('/modification.html')
def Modification():
    return render_template('modification.html')


##############################################################################################################

@app.route('/userRegister', methods=['GET', 'POST'])
def userRegister():
    if request.method == 'POST':
        print("GET")        

        username = request.form.get("username")
        addr = request.form.get("addr")        
        mob = request.form.get("mob")
        email = request.form.get("email")  
        passss = request.form.get("pass")   
        
        con = dbConnection()
        cursor = con.cursor()
        cursor.execute('SELECT * FROM userdetailes WHERE email = %s', (email))
        res = cursor.fetchone()
        if not res:
            sql = "INSERT INTO userdetailes (name, address,phone,email,password) VALUES (%s,%s, %s, %s, %s)"
            val = (username ,addr ,mob ,email ,passss)
            cursor.execute(sql, val)
            con.commit()
            return "success"
        else:
            return "fail"
        
##############################################################################################################        
        
@app.route('/userLogin', methods=['GET', 'POST'])
def userLogin():
    if request.method == 'POST':
        global usernm
        print("GET")       
        username = request.form.get("username")
        passw = request.form.get("password")
        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM userdetailes WHERE email = %s AND password = %s', (username, passw))
        if result_count>0:
            print(username)
            session['usernm'] = username            
            usernm += session.get("usernm")
            print("usersessionformob",usernm)
            return "success"
        else:
            return "Fail"
##############################################################################################################

        
@app.route('/uploadfile',methods=['POST','GET'])
def uploadfile():
    if request.method == "POST":
        f2= request.files['bill']
        
        filename_secure = secure_filename(f2.filename)
        print("GGGGGGGGGGGGGGGGGGGGGGG")        
        print(filename_secure)
        print("GGGGGGGGGGGGGGGGGGGGGGG")
        split_filename = filename_secure.split('_')[-1]
        f2.save(os.path.join(app.config['UPLOAD_FOLDER'], split_filename))  
        return "success"
    
import collections 
from collections import Counter
@app.route('/recommendation',methods=['POST','GET'])
def recommendation():
    if request.method == "POST":
        imagename = request.form.get("imagename")
        import imutils 
        emotions=[]  
        
        
        print("GGGGGGGGGGGGGGGGGGGGGGG")        
        print(imagename)
        print("GGGGGGGGGGGGGGGGGGGGGGG")
        
        print("We are in recommendation"+imagename)
        
        img = cv2.imread("static/uploadedimages/"+str(imagename))
        frame = imutils.resize(img, width=400)
 
    # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
     
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        #print(detections)
        faceBoxes = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
    
            # filter out detections by confidence
            if confidence < 0.7:
                continue
    
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print(gray_frame)
            #print(box)
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            #faceBoxes=box
            #print(faceBoxes)
            #print(faceBoxes[0][2])
            
            roi_gray_frame = gray_frame[faceBoxes[0][2]:faceBoxes[0][2] + faceBoxes[0][3], faceBoxes[0][0]:faceBoxes[0][0] + faceBoxes[0][3]]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
            emotionlistforadding=['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']
            #break
            from tensorflow.keras.models import model_from_json
            model = model_from_json(open("fer.json", "r").read())
            #load weights
            model.load_weights('fer.h5')
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
            emotionlistforadding=['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']
            #break
            
           
            #emotion_model.load_weights('emotion_model.h5')
           
            emotion_prediction = model.predict(cropped_img)
            #print(emotion_prediction)
            maxindex = int(np.argmax(emotion_prediction))
            print(maxindex)
            labelemotion= "{}".format("Emotion : " + emotion_dict[maxindex])
            emotions.append(labelemotion)
            
             
            # draw the bounding box of the face along with the associated
            # probability
            #text = "{:.2f}%".format(confidence * 100)
            
            
            
            
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(frame, labelemotion, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite("static/uploadeddetected/NewPicture"+str(imagename), frame)

    occurrences3 = collections.Counter(emotions)
    print(occurrences3)
    Keymax_emotion = max(zip(occurrences3.values(), occurrences3.keys()))[1]
    print(Keymax_emotion)
    Keymax_emotion=Keymax_emotion[10:]
    print(Keymax_emotion)  
    
    if Keymax_emotion=="Happy":
        search_type="track"
        nameofemotion="party"
        print(nameofemotion)
    elif Keymax_emotion=="Angry":
        search_type="track"
        nameofemotion="sad"
        print(nameofemotion)
    elif Keymax_emotion=="Disgusted":
        search_type="track"
        print(nameofemotion)
        nameofemotion="motivational"
    elif Keymax_emotion=="Fearful":
        search_type="track"
        nameofemotion="motivational"
        print(nameofemotion)
    elif Keymax_emotion=="Neutral":
        search_type="track"
        nameofemotion="party"
        print(nameofemotion)
    elif Keymax_emotion=="Sad":
        search_type="track"
        nameofemotion="sad"
        print(nameofemotion)
    else:
        search_type="track"
        nameofemotion="love" 
        print(nameofemotion)
    
    global search_typeformob 
    global name_formob 
    
    search_typeformob = search_type
    name_formob = nameofemotion      
        
    return "success"

##############################################################################################################

@app.route('/makesearchformobile',methods=['POST','GET'])
def make_search_formobile():
    if request.method == "GET":          
        
        global search_typeformob 
        global name_formob 
        
        data = spotify.search(name_formob,limit=40,type="track")   
        api_url = data['tracks']['items']
        items = data['tracks']['items']
        #print(items)
        p=[]
        for i in range(len(items)):
            b=items[i]['id']
            p.append(b)
    
        return render_template('testing.html',
                               name=name_formob,
                               results=items,
                               api_url=api_url, 
                               search_type=search_typeformob)

    

def  search_all(search_type, name):
    print(name) 
    data = spotify.search(name,limit=50,type="track")   

    api_url = data['tracks']['items']
    items = data['tracks']['items']

    p=[]
    for i in range(len(items)):
        b=items[i]['id']
        p.append(b)
    import random
    items = random.sample(items, len(items))

    return name,items,api_url, search_type

@app.route("/searchall", methods=['POST','GET'])
def searchall():
    if request.method=="POST":
        search_type = request.form.get("search_type")
        name = request.form.get("name")

        print(search_type, name)

        name,items,api_url, search_type = search_all(search_type, name)

        return render_template('search.html',name=name,results=items,api_url=api_url, search_type=search_type)
    return render_template('search.html')

##############################################################################################################
                                             #voicespeech
##############################################################################################################
import pyaudio
import wave
import keyboard
import tensorflow.keras
import librosa
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

emotions = {
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}
# Define recording parameters
CHUNK = 1024  # Record buffer size
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Number of channels
RATE = 44100  # Sampling rate

# Initialize PyAudio and Wave objects
p = pyaudio.PyAudio()
wf = wave.open("output.wav", "wb")  # Create WAV file
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)


# Open audio stream
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def extract_feature(data, sr, mfcc, chroma, mel):    
    if chroma:                          
        stft = np.abs(librosa.stft(data))  
    result = np.array([])
    if mfcc:                          
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:                          
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:                             
        mel = np.mean(librosa.feature.melspectrogram(data, sr=sr).T,axis=0)
        result = np.hstack((result, mel))        
    return result 

def load_single_data(file):
    x = []
    data, sr = librosa.load(file)
    feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
    x.append(feature)
    return np.array(x)

################################################################

@app.route('/voicespeech', methods=['POST','GET'])
def voicespeech():
    
    
    return render_template('voicespeech.html')

##############################################################################################################
                                             #voicespeech song s
##############################################################################################################
def make_searchemotion(search_type, name,emotion):
    print("========================")
    print(search_type)
    print(name)
    data = spotify.search(name,limit=40,type="track")   
    api_url = data['tracks']['items']
    items = data['tracks']['items']
    # print(items)
    # print(api_url)
    #print(items)
    p=[]
    for i in range(len(items)):
        b=items[i]['id']
        p.append(b)
    import random
    items = random.sample(items, len(items))
    print("hiiiiiiiiiiiiiii")


    return render_template('SpeechFetchedemotion.html',
                           name=name,
                           results=items,
                           api_url=api_url, 
                           search_type=search_type,emotion=emotion)

# Open Stream
# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)

##############################################################################################################
                                             #startspeechemotion
##############################################################################################################

@app.route('/startspeechemotion', methods = ['GET', 'POST'])
def startspeechemotion():
    if request.method == 'POST':
       p = pyaudio.PyAudio()
       frames = []

       stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
       
       print("* Recording...")
       while True:
           # Read audio data
           data = stream.read(CHUNK)
           frames.append(data)
       
           # Check if 'q' is pressed
           if keyboard.is_pressed('q'):
               break
           
       # Stop recording and close stream
       stream.stop_stream()
       stream.close()
       p.terminate()
       
       # Write audio data to WAV file
       wf = wave.open("output.wav", "wb")
       wf.setnchannels(CHANNELS)
       wf.setsampwidth(p.get_sample_size(FORMAT))
       wf.setframerate(RATE)
       wf.writeframes(b"".join(frames))
       wf.close()
       
       print("* Recording saved as output.wav")
       
       XX = load_single_data("output.wav")
       XXTemp = np.expand_dims(XX, axis=2)
       
       loaded_model = tensorflow.keras.models.load_model("cnn.h5")
       ypred = loaded_model.predict(XXTemp)
       argmax_index = np.argmax(ypred)
       
       emotion = emotions[str(argmax_index + 1).zfill(2)]  # +1 because emotions are 1-indexed
       
       print("Emotion:", emotion)
       print('-----------------------------------------------------')
       print(emotion)
       print('-----------------------------------------------------')
       
       if emotion == "neutral":
           search_type = "track"
           name = "party"
       elif emotion == "calm":
           search_type = "track"
           name = "happy"
       elif emotion == "happy":
           search_type = "track"
           name = "surprised"
       elif emotion == "sad":
           search_type = "track"
           name = "happy"
       elif emotion == "angry":
           search_type = "track"
           name = "calm"
       elif emotion == "fearful":
           search_type = "track"
           name = "happy"
       elif emotion == "disgust":
           search_type = "track"
           name = "sad"
       else:
           search_type = "surprised"
           name = "disgust"
       
       return make_searchemotion(search_type, name, emotion)
            
        # return render_template('SpeechFetchedemotion.html')
    return render_template('voicespeech.html')

##############################################################################################################




from model import FacialExpressionModel
from tensorflow.keras.models import model_from_json
#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Creating an instance of the class with the parameters as model and its weights.
test_model = FacialExpressionModel("model.json", "model_weights.h5")
# Loading the classifier from the file.
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def startcamera():
    global listofemotion
    listofemotion=[]
    cap = cv2.VideoCapture(0)
    while True:
        ret,image=cap.read()# captures frame and returns boolean value and captured image
        print('hello')
        if not ret:
            continue
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Image size is reduced by 30% at each image scale.
        scaleFactor = 1.3

    # 5 neighbors should be present for each rectangle to be retained.
        minNeighbors = 5

    # Detect the Faces in the given Image and store it in faces.
        faces = facec.detectMultiScale(gray_frame, scaleFactor, minNeighbors)

    # When Classifier could not detect any Face.


        for (x, y, w, h) in faces:

        # Taking the Face part in the Image as Region of Interest.
            roi = gray_frame[y:y+h, x:x+w]

        # Let us resize the Image accordingly to use pretrained model.
            roi = cv2.resize(roi, (48, 48))

        # Let us make the Prediction of Emotion present in the Image
            prediction = test_model.predict_emotion(
                roi[np.newaxis, :, :, np.newaxis])

        # Custom Symbols to print with text of emotion.
            Symbols = {"Happy": ":)", "Sad": ":}", "Surprise": "!!",
                       "Angry": "?", "Disgust": "#", "Neutral": ".", "Fear": "~"}
    

        ## based on the prediction recommend music


        # Defining the Parameters for putting Text on Image
            Text = str(prediction)
            Text_Color = (180, 105, 255)

            Thickness = 2
            Font_Scale = 1
            Font_Type = cv2.FONT_HERSHEY_SIMPLEX

        # Inserting the Text on Image
            cv2.putText(image, Text, (x, y), Font_Type,
                    Font_Scale, Text_Color, Thickness)
            listofemotion.append(Text)
        resized_img = cv2.resize(image, (1000, 700))
        cv2.imshow('Facial emotion analysis ',resized_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    
     
##############################################################################################################
                                    #end code
##############################################################################################################


if __name__=="__main__":
    app.run("0.0.0.0")
    
    # app.run(debug=True)