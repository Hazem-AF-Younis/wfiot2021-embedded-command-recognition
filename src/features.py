"""Shared MFCC settings used for training and on-device inference.

16 x 16 MFCC matrix per 1-second clip at 8 kHz: 256 ms Hann windows, 50 ms hop,
26 mel filters, 16 cepstral coefficients (paper, Section IV).
"""
import numpy as np
import python_speech_features

SAMPLE_RATE = 8000
NUM_MFCC = 16
LEN_MFCC = 16


def mfcc(signal, fs=SAMPLE_RATE):
    """Return a (NUM_MFCC, frames) MFCC matrix for a mono signal."""
    m = python_speech_features.base.mfcc(signal,
                                         samplerate=fs,
                                         winlen=0.256,
                                         winstep=0.050,
                                         numcep=NUM_MFCC,
                                         nfilt=26,
                                         nfft=2048,
                                         preemph=0.0,
                                         ceplifter=0,
                                         appendEnergy=False,
                                         winfunc=np.hanning)
    return m.transpose()
