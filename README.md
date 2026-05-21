# clickbait-headline-detection
Deep learning and NLP-based clickbait headline detection using LSTM and Logistic Regression with TF-IDF baseline comparison.
Copy
# clickbait-headline-detection

Deep learning and NLP-based clickbait headline detection using LSTM and Logistic Regression with TF-IDF baseline comparison.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-blue?logo=scikit-learn)](https://scikit-learn.org)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

Clickbait headlines exploit psychological triggers to drive traffic, often at the cost of content quality.

This project implements and benchmarks two approaches to automated clickbait detection:

- **Baseline:** TF-IDF vectorization with Logistic Regression
- **Deep Learning:** LSTM with trainable embeddings

Both models are evaluated on the same train/validation/test split using:
- Accuracy
- Precision
- Recall
- F1-score

---

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression (baseline) | 96.19% | — | — | 96.16% |
| **LSTM** | **97.97%** | **98.30%** | **97.63%** | **97.96%** |

The LSTM outperformed the TF-IDF baseline by approximately **1.78 percentage points** in F1-score, demonstrating the importance of sequential context modeling in headline classification.

---

## Model Architecture

```text
Input → Embedding(10000, 64) → LSTM(128)
      → Dropout(0.5) → Dense(1, sigmoid)
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Embedding | (None, 30, 64) | 640,000 |
| LSTM | (None, 128) | 98,816 |
| Dropout (0.5) | (None, 128) | 0 |
| Dense (sigmoid) | (None, 1) | 129 |

---

## Preprocessing Pipeline

```text
Raw Text → Lowercase → Remove punctuation → Tokenize
         → Pad sequences → Embedding representation
```

### Configuration
- Vocabulary size: 10,000 tokens
- Maximum sequence length: 30
- OOV token: `<OOV>`
- Padding strategy: post-padding

---

## Repository Structure

```text
clickbait-headline-detection/
├── notebook/
│   └── clickbait_detection.ipynb
├── results/
│   ├── figures/
│   │   ├── lstm_accuracy_curve.png
│   │   ├── lstm_loss_curve.png
│   │   ├── lstm_confusion_matrix.png
│   │   ├── lr_confusion_matrix.png
│   │   └── model_comparison_bar.png
│   └── classification_reports.txt
├── report/
│   └── clickbait_detection_report.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Visualizations

The project includes:
- Accuracy Curves
- Loss Curves
- Confusion Matrices
- Model Comparison Charts
- Classification Reports

---

## Setup & Usage

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

Run the notebook in:
- Google Colab
- Jupyter Notebook

Dataset format expected:

| Column | Description |
|---|---|
| headline | News headline text |
| clickbait | Binary label (0 or 1) |

---

## Dataset

Approximately 32,000 headlines balanced across:
- clickbait
- non-clickbait

Dataset split:
- 80% training
- 10% validation
- 10% testing

using stratified sampling.

---

## Tech Stack

`Python` · `TensorFlow/Keras` · `Scikit-learn` · `NumPy` · `Pandas` · `Matplotlib` · `Google Colab`

---

## Future Improvements

Potential future extensions include:

- BERT / DistilBERT comparison
- Attention visualization
- Explainable AI methods
- Hyperparameter optimization
- Ensemble learning
- Transformer-based architectures
