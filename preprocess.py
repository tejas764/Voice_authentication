import librosa
import numpy as np
from scipy.signal import lfilter
from python_speech_features import sigproc

import parameters as p


def load(filename, sample_rate):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    return audio.flatten()


def normalize_frames(m, epsilon=1e-12):
    arr = np.asarray(m, dtype=np.float32)
    means = np.mean(arr, axis=1, keepdims=True)
    stds = np.std(arr, axis=1, keepdims=True)
    return (arr - means) / np.maximum(stds, epsilon)


# Valuable dc and dither removal function implemented
# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
def remove_dc_and_dither(sin, sample_rate):
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        raise ValueError("Sample rate must be 16kHz or 8kHz only")

    sin = lfilter([1, -1], [1, -alpha], sin)
    dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
    spow = np.std(dither)
    return sin + 1e-6 * spow * dither


def get_fft_spectrum(filename, buckets):
    signal = load(filename, p.SAMPLE_RATE)
    signal *= 2 ** 15

    signal = remove_dc_and_dither(signal, p.SAMPLE_RATE)
    signal = sigproc.preemphasis(signal, coeff=p.PREEMPHASIS_ALPHA)

    frame_len = int(round(p.FRAME_LEN * p.SAMPLE_RATE))
    frame_step = int(round(p.FRAME_STEP * p.SAMPLE_RATE))
    frames = sigproc.framesig(
        signal,
        frame_len=frame_len,
        frame_step=frame_step,
        winfunc=np.hamming,
    )

    fft = np.abs(np.fft.fft(frames, n=p.NUM_FFT))
    fft_norm = normalize_frames(fft.T)

    rsize = max(k for k in buckets if k <= fft_norm.shape[1])
    rstart = int((fft_norm.shape[1] - rsize) / 2)
    return fft_norm[:, rstart:rstart + rsize]
