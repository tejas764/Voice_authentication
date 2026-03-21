import os
from dataclasses import dataclass
from typing import Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


tf.get_logger().setLevel("ERROR")


@dataclass
class LoadedModel:
    """Small adapter that gives a unified predict interface."""

    predict_fn: Callable[[np.ndarray], np.ndarray]

    def predict_embedding(self, model_input: np.ndarray) -> np.ndarray:
        output = self.predict_fn(model_input.astype(np.float32, copy=False))
        embedding = np.squeeze(output)
        embedding = embedding.astype(np.float32, copy=False)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding


def _keras_predictor(model) -> Callable[[np.ndarray], np.ndarray]:
    def _predict(arr: np.ndarray) -> np.ndarray:
        out = model(arr, training=False)
        if hasattr(out, "numpy"):
            out = out.numpy()
        return out

    return _predict


def _saved_model_predictor(model_dir: str) -> Callable[[np.ndarray], np.ndarray]:
    loaded = tf.saved_model.load(model_dir)
    infer = loaded.signatures["serving_default"]

    def _predict(arr: np.ndarray) -> np.ndarray:
        output = infer(input_array=tf.convert_to_tensor(arr))
        first_tensor = next(iter(output.values()))
        return first_tensor.numpy()

    return _predict


def load_voice_model(model_dir: str) -> LoadedModel:
    """Load model in a Keras 3 + SavedModel compatible way."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model path not found: {model_dir}")

    try:
        keras_model = tf.keras.models.load_model(model_dir, compile=False)
        return LoadedModel(predict_fn=_keras_predictor(keras_model))
    except Exception:
        return LoadedModel(predict_fn=_saved_model_predictor(model_dir))
