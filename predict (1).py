"""
Predict the emotion of a single speech audio file using the trained model.

Usage:
    python predict.py path/to/audio.wav
(Run emotion_recognition.py first to create emotion_model.keras and emotion_meta.npz)
"""
import sys
import numpy as np
import tensorflow as tf
from emotion_recognition import extract_features


def predict(path):
    model = tf.keras.models.load_model("emotion_model.keras")
    meta = np.load("emotion_meta.npz", allow_pickle=True)
    classes, mean, std = meta["classes"], float(meta["mean"]), float(meta["std"])

    feat = (extract_features(path) - mean) / std
    probs = model.predict(feat[None, ...], verbose=0)[0]

    print(f"\nFile: {path}")
    print(f"Predicted emotion: {classes[int(np.argmax(probs))].upper()}")
    print("\nConfidence per emotion:")
    for c, p in sorted(zip(classes, probs), key=lambda x: -x[1]):
        print(f"  {c:<10} {p * 100:5.1f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict.py path/to/audio.wav")
    predict(sys.argv[1])
