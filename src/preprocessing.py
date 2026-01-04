# src/preprocessing.py

import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(ecg_signal, fs=360, lowcut=0.5, highcut=40):
    """
    Apply zero-phase Butterworth bandpass filter to ECG signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(N=4, Wn=[low, high], btype='band')
    filtered_ecg = filtfilt(b, a, ecg_signal)

    return filtered_ecg
