# __author__ ='shengtanwu'
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as plt
import pyaudio
import struct
from . import constants as c
import wave

class record_module:
    def __init__(self,
                 CHUNK = 2048,
                 FORMAT = pyaudio.paInt16,
                 CHANNELS = 1,
                 RATE = c.SAMPLE_RATE
                 ):

        self.CHUNK = CHUNK  # frames per buffer
        self.FORMAT = FORMAT
        self.CHANNELS = CHANNELS
        self.RATE = RATE
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)

    def vad(self, signal, threshold = 20):

        energy = np.mean(np.square(signal))
        if energy >= threshold:
            return energy, True
        else:
            return energy, False

    def recording(self, seconds):
        result = np.zeros((1,2048))

        i = 0
        while i <= self.RATE*seconds/self.CHUNK:
            data = self.stream.read(self.CHUNK, exception_on_overflow= False)
            y = np.fromstring(data, 'int16')
            if self.vad(y):
                result = np.vstack([result,y])
            i += 1
#            if i%(self.RATE//self.CHUNK) == 0:
#                print('{} seconds left'.format(self.seconds-(i//(self.RATE/self.CHUNK))), flush = True)
            #print(y,end = '\r',flush = True)
            #print(y.shape,flush = True)

        return result[1:]

    def streaming(self, n_chunks = 10):
        for k in range(n_chunks):
            raw_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            y = np.fromstring(raw_data, 'int16')
            if k == 0:
                res = y
            else:
                res = np.concatenate((res,y))

        return res

