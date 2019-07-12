import platform
import sys
if platform.system() == 'Windows':
    sys.path.append(".\\")
import os
import numpy as np

from . import sigproc
from . import constants as c
import librosa
from scipy.signal import lfilter, butter


'''
read wav files given file path

return numpy dataframe of shape [n, time, dimension]
'''

def load_wav(filename, sample_rate):
    audio, sr = librosa.load(filename, sr = sample_rate, mono = True)
    audio = audio.flatten()
    return audio

def standardization(v, epsilon = 1e-12):
    return (v - np.mean(v))/(np.std(v)+ epsilon)


# slice the data with FRAME_SIZE
def cut_feature(features, f_type):
    if f_type == 'mfcc':
        f_size = c.MFCC_FRAME_SIZE
    if f_type == 'filterbanks':
        f_size = c.FB_FRAME_SIZE
    if f_type == 'spectrum':
        f_size = c.FRAME_SIZE
    return np.stack([features[f_size * j: f_size * (j + 1), :] for j in range(features.shape[0] // f_size)])


def get_fft_spectrum(filename):
    signal = load_wav(filename, c.SAMPLE_RATE)
    signal = sigproc.preemphasis(signal, coeff = c.PREEMPHASIS_ALPHA)
    frames = sigproc.framesig(signal,
                              frame_len=c.FRAME_LEN*c.SAMPLE_RATE,
                              frame_step=c.FRAME_STEP*c.SAMPLE_RATE,
                              winfunc=np.hamming)

    fft = abs(np.fft.rfft(frames, n = c.NFFT))
    fft = ((1.0/c.NFFT)* ((fft)**2)) #power spectrum
    fft_norm = standardization(fft)

    if c.FRAME_SIZE <= fft_norm.shape[0]:
        res = cut_feature(fft_norm, f_type= 'spectrum')
        return np.nan_to_num(res)
    else:
        return None


def get_mfcc(filename):
    signal = load_wav(filename, c.SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(signal, sr = c.SAMPLE_RATE, n_mfcc = c.N_MFCC).T
    assert mfcc.shape[1] == 13
    mfcc = standardization(mfcc)
    if c.MFCC_FRAME_SIZE <= mfcc.shape[0]:
        mfcc = cut_feature(mfcc , f_type = 'mfcc')
        return np.nan_to_num(mfcc)
    else:
        return None

def get_filterbank(filename):
    '''
    generate filterbanks
    ref: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    :param filename:
    :return:
    '''
    signal = load_wav(filename, c.SAMPLE_RATE)
    signal = sigproc.preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA)
    frames = sigproc.framesig(signal,
                              frame_len=c.FRAME_LEN * c.SAMPLE_RATE,
                              frame_step=c.FRAME_STEP * c.SAMPLE_RATE,
                              winfunc=np.hamming)
    fft = abs(np.fft.rfft(frames, n=c.NFFT))
    fft = ((1.0/c.NFFT)* ((fft)**2)) #power spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (c.SAMPLE_RATE/ 2) / 700))  # Convert Hz to Mel

    mel_points = np.linspace(low_freq_mel, high_freq_mel, c.N_FILT + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((c.NFFT + 1) * hz_points / c.SAMPLE_RATE)

    fbank = np.zeros((c.N_FILT, int(np.floor(c.NFFT / 2 + 1))))
    for m in range(1, c.N_FILT + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(fft, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    filter_banks = standardization(filter_banks)

    if c.FRAME_SIZE <= filter_banks.shape[0]:
        res = cut_feature(filter_banks, f_type = 'filterbanks')
        return np.nan_to_num(res)
    else:
        # return padding result
        shape = filter_banks.shape
        res = np.zeros([1, c.FRAME_SIZE, shape[-1]])
        res[:,:shape[0], :] = filter_banks
        return np.nan_to_num(res)

