import os
import numpy as np
import joblib
from tqdm import tqdm
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score




DATA_DIR = "train/audio"
#MAX_FILES_PER_CLASS = 5000
SR = 16000
N_MFCC = 40
TARGET_WORDS = ["yes","no","up","down","left","right","on","off","stop","go"]
UNKNOWN_TARGET = "unknown"
SILENCE_TARGET = "silence"
All_TARGET = ["yes","no","up","down","left","right","on","off","stop","go", "unknown", "silence"]


def load_wav_1s(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr)
    return np.pad(audio, (0, max(0, sr - len(audio))))[:sr]

def extract_mfcc(audio):
    return librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC).flatten()



X, y = [], []

from collections import defaultdict

class_counts = defaultdict(int)

for root, _, files in os.walk(DATA_DIR):
    label = os.path.basename(root)

    if label not in TARGET_WORDS:
        if label == "_background_noise_":
            label = SILENCE_TARGET
        else:
            label = UNKNOWN_TARGET

    for f in files:
        if not f.endswith(".wav"):
            continue

        #if class_counts[label] >= MAX_FILES_PER_CLASS:
        #    continue

        path = os.path.join(root, f)

        audio = load_wav_1s(path)
        X.append(extract_mfcc(audio))
        y.append(label)

        class_counts[label] += 1

X = np.array(X)
y = np.array(y)
print(set(y))

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)



pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=50)),
    ("svm", SVC(kernel="rbf", C=5, probability=True))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)
acc = accuracy_score(y_val, y_pred)

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

labels = np.arange(len(le.classes_))
ConfusionMatrixDisplay.from_predictions(
    y_val, y_pred,
    labels=labels,
    display_labels=le.classes_,
    cmap="Blues"
)
plt.xticks(rotation=45, ha="right")
plt.show()



print(f"Validation accuracy: {acc:.3f}")

os.makedirs("models", exist_ok=True)

joblib.dump({
    "pipeline": pipeline,
    "label_encoder": le
}, "models/pipeline.pkl")

print("Model saved to models/pipeline.pkl")
