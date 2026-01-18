import numpy as np
import joblib
import librosa
import os

SR = 16000
N_MFCC = 40  # doit correspondre à train.py

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pipeline.pkl")

def load_wav_1s(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr)
    return np.pad(audio, (0, max(0, sr - len(audio))))[:sr]

def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    return mfcc.flatten()

# Charger le modèle UNE FOIS
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
label_encoder = bundle["label_encoder"]

def predict_command(wav_path):
    audio = load_wav_1s(wav_path)
    features = extract_mfcc(audio).reshape(1, -1)
    probs = pipeline.predict_proba(features)[0]
    idx = np.argmax(probs)
    label = label_encoder.inverse_transform([idx])[0]
    return label, probs