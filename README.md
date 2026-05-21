# clickbait-headline-detection
Deep learning and NLP-based clickbait headline detection using LSTM and Logistic Regression with TF-IDF baseline comparison.
Copy
# Clickbait Headline Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-blue?logo=scikit-learn)](https://scikit-learn.org)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

Binary text classification of news headlines as clickbait or non-clickbait,
comparing a TF-IDF + Logistic Regression baseline against an LSTM neural
network on a dataset of ~32,000 headlines.

---

## Overview

Clickbait headlines exploit psychological triggers to drive traffic, often
at the cost of content quality. This project implements and benchmarks
two approaches to automated clickbait detection:

- **Baseline**: TF-IDF vectorization with Logistic Regression
- **Deep Learning**: LSTM with a trainable embedding layer

Both models are evaluated on the same train/val/test split using accuracy,
precision, recall, and F1-score.

---

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression (baseline) | 96.19% | — | — | 96.16% |
| **LSTM** | **97.97%** | **98.30%** | **97.63%** | **97.96%** |

The LSTM outperforms the baseline by **~1.78 percentage points** in F1-score,
demonstrating that sequential modeling captures contextual signals in
headlines that bag-of-words approaches miss.

---

## LSTM Architecture

```
Input → Embedding(10000, 64) → LSTM(128) → Dropout(0.5) → Dense(1, sigmoid)
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Embedding | (None, 30, 64) | 640,000 |
| LSTM | (None, 128) | 98,816 |
| Dropout (0.5) | (None, 128) | 0 |
| Dense (sigmoid) | (None, 1) | 129 |

---

## Preprocessing Pipeline

```
Raw Text → Lowercase → Remove punctuation → Tokenize
         → Pad sequences (maxlen=30) → Word embeddings
```

- Vocabulary size: 10,000 tokens
- Max sequence length: 30 tokens
- OOV token: ``
- Padding: post

---

## Repository Structure

```
clickbait-headline-detection/
├── notebook/
│   └── clickbait_detection.ipynb      # Full pipeline
├── results/
│   ├── figures/
│   │   ├── lstm_accuracy_curve.png
│   │   ├── lstm_loss_curve.png
│   │   ├── lstm_confusion_matrix.png
│   │   ├── lr_confusion_matrix.png
│   │   └── model_comparison_bar.png
│   └── classification_reports.txt     # Full sklearn output
├── report/
│   └── clickbait_detection_report.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup & Usage

```bash
# All experiments run in Google Colab — no local install required
# 1. Open notebook in Colab (badge above)
# 2. Mount Google Drive and place dataset at:
#    /content/drive/MyDrive/train1.csv

pip install tensorflow scikit-learn pandas numpy matplotlib
```

Dataset format expected: CSV with columns `headline` (str) and
`clickbait` (int, 0 or 1).

---

## Dataset

~32,000 headlines balanced across clickbait and non-clickbait classes.
80/10/10 stratified split into train, validation, and test sets.

---

## Tech Stack

`Python` · `TensorFlow/Keras` · `Scikit-learn` · `NumPy` · `Pandas` · `Matplotlib` · `Google Colab`

---

## Authors

Jana · Raghad · Samer · Mohammad — AI342 Deep Learning
