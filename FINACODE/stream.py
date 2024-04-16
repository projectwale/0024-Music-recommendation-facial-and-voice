# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:49:34 2024

@author: sushant
"""

import pyaudio

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open Stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

try:
    # Read from the stream
    data = stream.read(CHUNK)
    
    # Your processing or handling of the audio data goes here
    
except OSError as e:
    print(f"Error: {e}")
    
finally:
    # Close the stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
