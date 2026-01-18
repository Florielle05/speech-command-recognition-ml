# Speech Command Recognition — Lightweight ML Demo (MFCC + SVM)

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


Each subfolder contains `.wav` files of spoken commands.

---

## Exploratory Analysis & Model Selection

Before implementing the final pipeline, an exploratory analysis was conducted to better understand the dataset and guide model selection.

The notebook:
notebooks/01_data_exploration_and_model_choice.ipynb
covers:
- class distribution analysis
- inspection of MFCC features
- baseline comparison between:
  - logistic regression (linear baseline)
  - SVM with non-linear kernel
- performance vs. complexity trade-offs

This analysis motivated the choice of an **SVM classifier**, which provides a good balance between expressive power and computational efficiency for a demo-oriented application.

---

## Model Choice (Why SVM?)

We deliberately use an **SVM classifier** trained on **MFCC features** for this project.

This choice offers:
- stronger decision boundaries than linear models (e.g. logistic regression)
- fast inference and low memory footprint
- good performance on small-to-medium audio datasets
- suitability for a simple Streamlit demo without GPU requirements

The objective is **not** to compete with state-of-the-art deep learning models, but to showcase a **well-reasoned engineering trade-off**.

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

### Note on Silence Handling

Silence is handled in a **simplified manner** in this demo.
The application assumes controlled inputs (spoken commands) and does not aim to model real-world silence or noise conditions.

More robust silence modeling is intentionally left as **future work**.

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
