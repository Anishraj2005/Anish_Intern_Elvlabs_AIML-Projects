# 🧠 Elevate Labs AI/ML Internship Projects

**End-to-End Machine Learning & Deep Learning Portfolio**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Computer%20Vision-MediaPipe-0097A7?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Web%20Apps-Streamlit-red?style=for-the-badge&logo=streamlit" />
</p>
  
---

## 📌 Overview

This repository contains two complete AI/ML engineering projects developed as part of the **Elevate Labs AI/ML Internship Program**.

Each project demonstrates an end-to-end pipeline — from data processing and model training to evaluation and real-time deployment.

The focus is on:

- Handling real-world imbalanced datasets
- Building robust machine learning & deep learning pipelines
- Deploying real-time AI systems
- Ensuring interpretability and usability

---

## 🚀 Projects

### 🛡️ Project 1 — Credit Card Fraud Detection

#### 🔍 Problem Statement

Detect fraudulent credit card transactions in a highly imbalanced dataset.

#### 🧠 Approach

- Isolation Forest (Unsupervised anomaly detection)
- Local Outlier Factor (LOF)
- XGBoost classifier with SMOTE
- Isotonic regression for calibration
- SHAP explainability

#### 📊 Results (Test Set)

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Isolation Forest | 0.21 | 0.27 | 0.23 | 0.95 |
| LOF | 0.38 | 0.77 | 0.51 | 0.94 |
| XGBoost | 0.90 | 0.83 | 0.86 | 0.98 |

#### 🎯 Highlights

- Handles extreme imbalance (578:1 ratio)
- SHAP-based explainability for each prediction
- Streamlit dashboard for real-time inference
- Strong performance with calibrated probabilities

📁 **Folder:** `Credit-Card-Fraud-Detection/`

---

### 🤟 Project 2 — Real-Time ASL Alphabet Recognition

#### 🔍 Problem Statement

Translate American Sign Language (A–Z) gestures into text using a webcam.

#### 🧠 Approach

- MediaPipe hand landmark detection (21 points)
- Normalised 63-feature representation
- 1D-CNN (TensorFlow) + MLP (scikit-learn)
- OpenCV real-time inference pipeline

#### 📊 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 99.8% |
| Misclassifications | 3 only |

#### 🎯 Highlights

- Real-time gesture → text system
- Debounce logic for stable predictions
- Sentence builder with spacing and correction
- Optional text-to-speech output

📁 **Folder:** `Real-Time-Sign-Language-Recognition/`

---

## 🧰 Tech Stack

- Python 3.10+
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- MediaPipe
- OpenCV
- SHAP
- Streamlit
- Matplotlib / Seaborn / Plotly
- NumPy / Pandas

---

## 📂 Repository Structure

```
Anish_Intern_Elvlabs_AIML-Projects/
│
├── Anish_Intern_Elvlabs_AIML-Project-1-Credit-Card-Fraud-Detection/
│   ├── README.md
│   └── ...
│
└── Anish_Intern_Elvlabs_AIML-Project-2-Rea-Time-Sign-Language-Recognition/
    ├── README.md
    └── ...
```

---

## ⚙️ How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2️⃣ Run Project 1 — Fraud Detection

```bash
cd Anish_Intern_Elvlabs_AIML-Project-1-Credit-Card-Fraud-Detection
pip install -r requirements.txt
jupyter notebook
```

Open: `Credit Card Fraud Detection.ipynb`

### 3️⃣ Run Project 2 — ASL Recognition

```bash
cd Anish_Intern_Elvlabs_AIML-Project-2-Rea-Time-Sign-Language-Recognition
pip install -r requirements.txt
jupyter notebook ASL_Recognition.ipynb
```

For live webcam inference:

```bash
python webcam_recognition.py
```

---

## 🧠 Key Learnings

- Accuracy is misleading in imbalanced datasets
- Precision-Recall is more important than accuracy in fraud detection
- Landmark-based CV is faster and more robust than image-based models
- Real-time ML requires smoothing and temporal logic
- Calibration improves real-world reliability of ML models

---

## 👤 Author

**Anish Raj**

- Internship: Elevate Labs AI/ML Program
- GitHub: [@Anishraj2005](https://github.com/Anishraj2005)

---

## 📄 License

This project is for educational and internship purposes.
Datasets belong to their respective providers (Kaggle).

---

⭐ If you like this project, consider giving it a star!
