from functools import lru_cache

import numpy as np
import pandas as pd

from preprocess import get_fft_spectrum
import parameters as p


def buckets(max_time, steptime, frameskip):
    data = {}
    frames_per_sec = int(1 / frameskip)
    end_frame = int(max_time * frames_per_sec)
    step_frame = int(steptime * frames_per_sec)

    for i in range(0, end_frame + 1, step_frame):
        s = i
        s = np.floor((s - 7 + 2) / 2) + 1
        s = np.floor((s - 3) / 2) + 1
        s = np.floor((s - 5 + 2) / 2) + 1
        s = np.floor((s - 3) / 2) + 1
        s = np.floor((s - 3 + 2) / 1) + 1
        s = np.floor((s - 3 + 2) / 1) + 1
        s = np.floor((s - 3 + 2) / 1) + 1
        s = np.floor((s - 3) / 2) + 1
        s = np.floor((s - 1) / 1) + 1
        if s > 0:
            data[i] = int(s)

    return data


@lru_cache(maxsize=16)
def _cached_buckets(max_time, bucket_step, frame_step):
    return buckets(max_time, bucket_step, frame_step)


def get_embedding(model, wav_file, max_time):
    buckets_var = _cached_buckets(max_time, p.BUCKET_STEP, p.FRAME_STEP)
    signal = get_fft_spectrum(wav_file, buckets_var)
    model_input = signal.reshape(1, *signal.shape, 1).astype(np.float32, copy=False)
    return model.predict_embedding(model_input)


def get_embedding_batch(model, wav_files, max_time):
    return [get_embedding(model, wav_file, max_time) for wav_file in wav_files]


def get_embeddings_from_list_file(model, list_file, max_time):
    buckets_var = _cached_buckets(max_time, p.BUCKET_STEP, p.FRAME_STEP)
    result = pd.read_csv(list_file, delimiter=",")
    result["features"] = result["filename"].apply(lambda x: get_fft_spectrum(x, buckets_var))
    result["embedding"] = result["features"].apply(
        lambda x: model.predict_embedding(x.reshape(1, *x.shape, 1).astype(np.float32, copy=False))
    )
    return result[["filename", "speaker", "embedding"]]
