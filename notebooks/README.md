# Notebooks — Data Exploration & Model Benchmarking

This folder contains exploratory and experimental notebooks used to guide
modeling decisions for the speech command recognition project.

## Purpose

The notebooks in this folder are **not part of the final demo pipeline**.
They are used to:
- analyze the dataset
- compare different modeling approaches
- justify engineering trade-offs

## Main Notebook

### `speech_command_recognition_experiments.ipynb`

This notebook includes:
- exploratory data analysis of the Speech Commands dataset
- proper construction and handling of a `silence` class
- feature extraction experiments (MFCC, log-mel spectrograms)
- comparison between:
  - logistic regression
  - SVM (MFCC-based)
  - CNN (log-mel spectrograms)
- evaluation of accuracy and complexity trade-offs

Although CNN-based models achieve higher raw performance,
the SVM approach was selected for the final demo due to its lower inference cost
and simpler deployment.

## Relationship to the Main Project

The conclusions of this notebook directly motivate:
- the choice of MFCC features
- the use of an SVM classifier in the Streamlit demo
- the simplified handling of silence in the application
