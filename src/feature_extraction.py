# src/feature_extraction.py

import numpy as np

def extract_time_features(beat):
    """
    Extract time-domain features from ECG beat.
    """
    mean_val = np.mean(beat)
    var_val = np.var(beat)
    rms_val = np.sqrt(np.mean(beat ** 2))
    ptp_val = np.ptp(beat)

    return [mean_val, var_val, rms_val, ptp_val]


def extract_freq_features(beat, num_coeffs=10):
    """
    Extract frequency-domain features using FFT.
    """
    fft_vals = np.fft.fft(beat)
    mag = np.abs(fft_vals)

    return mag[:num_coeffs]
