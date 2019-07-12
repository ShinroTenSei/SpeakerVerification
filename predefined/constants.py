import os
import numpy as np


# fft constants
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NFFT = 512
FRAME_SIZE = 300

# mfcc constants
N_MFCC = 13
MFCC_FRAME_SIZE = 32

# filter banks constants
N_FILT = 40
FB_FRAME_SIZE = 200

EPSILON = 1e-16
