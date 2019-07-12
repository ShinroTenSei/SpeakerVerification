# __author__ ='shengtanwu'
# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import wave
import pyaudio
import struct
import argparse

def parse_args():
    '''
    parse parameters
    :return: args
    '''
    parser = argparse.ArgumentParser(description = 'Parse Args to initialize prediction')
    parser.add_argument('-file_name', '--file_name', type = str)
    parser.add_argument('-sec' , '--seconds', type = int)
    args = parser.parse_args()
    return args

def get_audio():
    args = parse_args()
    CHUNK = 2048
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = args.seconds

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    print('Start recording for ' + str(RECORD_SECONDS) + ' seconds')
    for i in range(int(RECORD_SECONDS*RATE/CHUNK)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)
    print('End recording.')
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(args.file_name+'.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == '__main__':
    get_audio()