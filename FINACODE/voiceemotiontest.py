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
wf = wave.open("saved_models/output.wav", "wb")  # Create WAV file
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)

loaded_model = tensorflow.keras.models.load_model("saved_models/cnn.h5")

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

def callFuction():
    frames = []
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
    wf.writeframes(b"".join(frames))
    wf.close()
    
    print("* Recording saved as output.wav")
    
    XX = load_single_data("saved_models/output.wav")
    
    # Predict for the test set
    # XXTemp=np.expand_dims(XX, axqis=2)
    XXTemp = np.expand_dims(XX, axis=2)
    
    print('-----------------------------------------------------')
    print(XXTemp)
    print('-----------------------------------------------------')

    ypred = loaded_model.predict(XXTemp)# Find argmaqx index
    argmax_index = np.argmax(ypred)
    
    # Map the index to emotion
    emotion = emotions[str(argmax_index + 1).zfill(2)]  # +1 because emotions are 1-indexed
    
    print("Emotion:", emotion)
    return str(emotion)
    
em=callFuction()
print('--------------------')
print(em)