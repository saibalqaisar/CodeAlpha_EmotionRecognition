"""
CodeAlpha ML Internship - Task 2: Emotion Recognition from Speech
Uses MFCC features (librosa) + a deep learning model (Keras CNN).

Dataset: RAVDESS (https://zenodo.org/record/1188976) - download "Audio_Speech_Actors_01-24"
and extract so the structure is:  dataset/Actor_01/03-01-05-01-01-01-01.wav ...

Install:  pip install librosa tensorflow scikit-learn numpy matplotlib
Run:      python emotion_recognition.py --data dataset
"""
import argparse
import glob
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# RAVDESS filename: modality-channel-EMOTION-intensity-statement-rep-actor.wav
EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}

SR = 22050
DURATION = 3.0
N_MFCC = 40
MAX_LEN = 130  # frames for ~3 sec


def extract_features(path):
    y, sr = librosa.load(path, sr=SR, duration=DURATION, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    # pad / trim to fixed length
    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc.T  # (time, n_mfcc)


def augment(y, sr=SR):
    """Simple augmentation: add noise."""
    return y + 0.005 * np.random.randn(len(y))


def load_dataset(data_dir):
    X, labels = [], []
    files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No .wav files found in {data_dir}")
    for f in files:
        code = os.path.basename(f).split("-")[2]
        if code not in EMOTIONS:
            continue
        X.append(extract_features(f))
        labels.append(EMOTIONS[code])
    return np.array(X), np.array(labels)


def build_model(input_shape, n_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Conv1D(128, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    print("Extracting features...")
    X, labels = load_dataset(args.data)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print("Samples:", X.shape, "Classes:", list(le.classes_))

    # normalize
    mean, std = X.mean(), X.std()
    X = (X - mean) / std

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model(X_train.shape[1:], len(le.classes_))
    model.summary()

    cbs = [callbacks.EarlyStopping(patience=10, restore_best_weights=True),
           callbacks.ReduceLROnPlateau(patience=4, factor=0.5)]
    model.fit(X_train, y_train, validation_split=0.15,
              epochs=args.epochs, batch_size=32, callbacks=cbs)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.3f}")

    pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, pred, target_names=le.classes_))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    model.save("emotion_model.keras")
    np.savez("emotion_meta.npz", classes=le.classes_, mean=mean, std=std)
    print("Saved: emotion_model.keras, emotion_meta.npz")


if __name__ == "__main__":
    main()
