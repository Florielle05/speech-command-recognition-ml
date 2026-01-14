# Speech Command Recognition (Machine Learning)

Machine learning project focused on recognizing short spoken commands (e.g. *yes*, *no*, *up*, *down*, *stop*) from 1-second audio clips.

The project compares classical machine learning models with a Convolutional Neural Network (CNN) using MFCC audio features.

---

## Project Objective

Build and evaluate an end-to-end speech command recognition system:
- audio preprocessing
- MFCC feature extraction
- model training and evaluation
- comparison between classical ML and deep learning approaches

---

## Dataset

**Google / TensorFlow Speech Commands dataset (v0.01)**  
- ~64,000 audio clips  
- 1-second duration, 16 kHz sampling rate  
- 12 classes:
  - 10 commands: *yes, no, up, down, left, right, on, off, go, stop*
  - *unknown* (all other words)
  - *silence* (background noise segments)

---

## Methods

### Feature Extraction
- MFCCs (40 coefficients per frame)
- Flattened MFCC vectors for classical ML
- 2D MFCC matrices for CNN input

### Models Implemented
- Multinomial Logistic Regression (baseline)
- Logistic Regression with class weighting and partial feature selection
- SVM with RBF kernel + PCA
- Convolutional Neural Network on 2D MFCCs

---

## Results

| Model | Validation Accuracy |
|-----|--------------------|
| Logistic Regression | ~0.65 |
| Logistic Regression (balanced) | ~0.36 |
| SVM + PCA | ~0.75 |
| CNN (MFCC 2D) | **~0.89** |

The CNN significantly outperforms classical models by exploiting the time–frequency structure of MFCCs.

---

## Key Learnings

- Audio preprocessing and MFCC feature extraction
- Limitations of linear models under class imbalance
- Benefits of non-linear and deep learning approaches
- Designing and evaluating CNNs for audio classification
- End-to-end ML experimentation and analysis

**LODJO Florielle**  
Engineering student – Data Science & Artificial Intelligence
