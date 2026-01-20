# Speech Command Recognition — Lightweight ML Demo (MFCC + SVM)
# Speech Command Recognition (ML)

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://speech-command-recognition-ml-7tx4qnctrkbjbvq53wne5a.streamlit.app/)

**Live demo:** https://speech-command-recognition-ml-7tx4qnctrkbjbvq53wne5a.streamlit.app/
> A Streamlit app that classifies short spoken commands using an ML pipeline (audio preprocessing → features → model → prediction).


This project implements a **lightweight speech command recognition system** designed as a **demonstration application**.
It focuses on clarity, fast inference, and end-to-end structure rather than maximizing raw performance.

The pipeline covers:
- audio preprocessing
- feature extraction
- model training and evaluation
- interactive inference through a Streamlit app

---

## Project Motivation

The goal of this project is to demonstrate a **clean and understandable machine learning pipeline** applied to audio data.

Rather than using heavy deep learning models, we deliberately focus on:
- fast inference
- low model footprint
- simple deployment

This makes the project suitable as a **technical demo** and an educational example of applied ML.

---

## Dataset

The project uses the **Google Speech Commands dataset**.

Expected structure:
```bash
train/
  audio/
    yes/
    no/
    up/
    down/
    left/
    right/
    on/
    off/
    stop/
    go/
    learn/
    follow/
    ...
    ...
    background_noise/
```

Each subfolder contains `.wav` files of spoken commands.

---

## Exploratory Analysis & Model Selection
Before implementing the final demo pipeline, we conducted a more advanced experimental study
to benchmark different modeling approaches on the Speech Commands dataset.

The notebook:
notebooks/01_data_exploration_and_model_choice.ipynb

includes:
- detailed analysis of class distributions
- proper construction of a `silence` class using sampled background noise
- comparison between multiple model families:
  - logistic regression
  - SVM (MFCC-based)
  - convolutional neural networks (log-mel spectrograms)
- evaluation of performance, training cost, and inference complexity

This notebook demonstrates that while CNN-based models achieve higher accuracy,
the SVM approach provides a more favorable trade-off for a lightweight demo application.


---

## Model Choice (Why SVM?)
Although convolutional neural networks outperform classical models in terms of raw accuracy,
they introduce additional complexity in terms of:
- model size
- inference latency
- deployment constraints

Since the goal of this project is to deliver a **simple, fast, and easily deployable demo**,
I intentionally chose an **SVM classifier** trained on MFCC features for the Streamlit application.

This choice is not due to technical limitations, but to a **conscious engineering decision**
supported by empirical benchmarking (see notebook).
---

## Audio Processing & Features

- Audio is resampled to **16 kHz**
- Signals are truncated or zero-padded to **1 second**
- **MFCC features (40 coefficients)** are extracted
- MFCCs are flattened to produce fixed-size feature vectors

---

## Model Pipeline

The full training pipeline is:
StandardScaler → PCA (50 components) → SVM (RBF kernel)

- Scaling ensures numerical stability
- PCA reduces dimensionality and noise
- SVM provides non-linear classification

The trained pipeline is serialized using `joblib`.

---

## Labels

Target command classes:
yes, no, up, down, left, right, on, off, stop, go

Additional classes:
- `unknown`: commands outside the target set
- `silence`: background noise samples

## Note on Silence Handling

In the benchmarking notebook, silence is handled properly by sampling fixed-length windows
from background noise recordings, and is included in both classical and CNN-based experiments.

For the final demo application, silence handling is simplified in order to keep the interface
and inference logic minimal.

---

## Training the Model

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:
```bash
python train.py
```

This will:
train the pipeline
evaluate it on a validation split
display a confusion matrix
save the trained model to:
```bash
models/pipeline.pkl
```

## Running the Streamlit Demo

Launch the app:
```bash
streamlit run app.py
```
The app allows you to:
upload a .wav file
run inference using the trained model
display the predicted command
The demo is intentionally simple and designed for clarity.

Repository Structure
```bash
.
├── app.py
├── train.py
├── src/
│   └── inference.py
├── notebooks/
│   └── 01_data_exploration_and_model_choice.ipynb
├── models/
│   └── pipeline.pkl
├── requirements.txt
└── README.md

```
