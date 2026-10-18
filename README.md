# CodeAlpha_EmotionRecognition

Machine Learning Internship @ CodeAlpha - **Task 2: Emotion Recognition from Speech**

## Objective
Recognize human emotions (happy, angry, sad, etc.) from speech audio using deep learning and speech signal processing.

## Dataset
**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) - 1,440 speech files from 24 actors.
Download: https://zenodo.org/record/1188976 (`Audio_Speech_Actors_01-24.zip`)

Emotions (8 classes): neutral, calm, happy, sad, angry, fearful, disgust, surprised.

## Approach
1. **Feature extraction:** 40 MFCCs (Mel-Frequency Cepstral Coefficients) per audio clip using `librosa`; clips are fixed to ~3 seconds (130 time frames).
2. **Normalization:** global mean/std standardization.
3. **Model:** Conv1D (64) -> Conv1D (128) -> Bidirectional LSTM (64) -> Dense (64) -> Softmax (8 classes), with BatchNormalization and Dropout.
4. **Training:** Adam optimizer, early stopping, learning-rate reduction on plateau, 80/20 stratified split.

## Project Structure
```
CodeAlpha_EmotionRecognition/
|-- emotion_recognition.py   # feature extraction + training + evaluation
|-- predict.py               # predict emotion for a single .wav file
|-- requirements.txt
|-- README.md
```

## How to Run
```bash
pip install -r requirements.txt

# 1. Train (dataset folder contains Actor_01 ... Actor_24)
python emotion_recognition.py --data dataset

# 2. Predict on a new audio file
python predict.py sample.wav
```

## Input / Output
**Input:** a `.wav` speech file (e.g. `03-01-05-01-01-01-01.wav`)

**Output:**
```
Predicted emotion: ANGRY

Confidence per emotion:
  angry      87.3%
  disgust     6.1%
  fearful     3.2%
  ...
```

## Results
<img width="1920" height="1080" alt="Screenshot (59)" src="https://github.com/user-attachments/assets/7fdac882-585f-4a34-9184-ca8496b9a7d3" />
<img width="1920" height="1080" alt="Screenshot (60)" src="https://github.com/user-attachments/assets/1510ecfa-569a-4466-9884-94fb274d9e45" />
<img width="1920" height="1080" alt="Screenshot (61)" src="https://github.com/user-attachments/assets/8bc88f7e-9e5c-4159-af88-330122ef2ba8" />




| Metric | Value |
|--------|-------|
| Test Accuracy | 0.62|

()

## Tech Stack
Python, TensorFlow/Keras, librosa, NumPy, scikit-learn

## Author
Saibal - CodeAlpha ML Intern
