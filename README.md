# ECG_Arrhythmia_Project
This repository implements ECG heartbeat classification using classical machine learning and deep learning, developed as a self-directed academic project for AI/ML exploration..

## Overview

This project presents an end-to-end implementation of ECG heartbeat classification using both classical machine learning and deep learning approaches. It was developed as a self-directed academic project to build a strong foundation for MSc studies in Artificial Intelligence and Machine Learning, with emphasis on signal processing, model comparison, and critical evaluation.

The objective is to classify ECG heartbeats into normal and abnormal categories using the MIT-BIH Arrhythmia Dataset, while highlighting the strengths and limitations of different modeling approaches in a medical signal-processing context.

## Dataset

- MIT-BIH Arrhythmia Dataset
- Accessed programmatically using the wfdb library
- ECG signals sampled at 360 Hz
- Expert-annotated R-peak locations used for heartbeat segmentation
- Raw dataset files are not included in the repository (loaded dynamically)

## Methodology

### 1. Signal Preprocessing

- Raw ECG signals are preprocessed using a zero-phase Butterworth band-pass filter (0.5–40 Hz)
- This removes baseline drift and high-frequency noise while preserving ECG morphology
- Zero-phase filtering (filtfilt) is used to avoid phase distortion

### 2. Heartbeat Segmentation

- ECG signals are segmented into fixed-length windows centered around annotated R-peaks
- Each heartbeat segment has a uniform length (~1 second)
- Each segment represents one machine learning sample

### 3. Feature Extraction (Classical ML)

- For classical machine learning models, each heartbeat is represented using:
- Time-domain features: mean, variance, RMS, peak-to-peak amplitude
- Frequency-domain features: FFT magnitude coefficients

These features form a compact and interpretable feature vector.

### 4. Classical Machine Learning Models

The following models are trained and evaluated:

- Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest

Feature scaling is applied where required, and performance is evaluated using standard classification metrics.

### 5. Deep Learning Model

- A Long Short-Term Memory (LSTM) neural network is trained directly on raw heartbeat segments
- The LSTM learns temporal representations without handcrafted features
- Results are compared against classical ML baselines

## Results and Analysis

- Model performance is evaluated using precision, recall, F1-score, and confusion matrices
- Particular attention is given to class imbalance, a common challenge in medical datasets
- High overall accuracy is observed due to dominance of normal beats, but minority-class (abnormal) detection is limited
- Error analysis highlights why accuracy alone is misleading for healthcare applications
  
Saved visual results are available in the results/ directory, including:
- ECG signal plots
- Segmented heartbeat example
- Confusion matrices for all models

## Limitations

- Severe class imbalance affects abnormal beat detection
- Binary classification only (normal vs abnormal)
- Limited dataset size for deep learning models

## Future Work

- Address class imbalance using resampling or class-weighted loss
- Extend to multi-class arrhythmia classification
- Explore CNN or attention-based architectures
- Apply explainable AI techniques for clinical interpretability
