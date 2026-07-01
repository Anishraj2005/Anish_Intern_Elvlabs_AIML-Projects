# 🛡️ Credit Card Fraud Detection
## Anomaly Detection & Supervised Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-red?style=for-the-badge&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?style=for-the-badge&logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-0.45.1-blueviolet?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-5.22.0-3F4F75?style=for-the-badge&logo=plotly)

---

## 📌 Project Overview

This project implements a **comprehensive Credit Card Fraud Detection system** that combines unsupervised anomaly detection with supervised classification on the Kaggle Credit Card Fraud dataset. The pipeline compares three models — **Isolation Forest**, **Local Outlier Factor (LOF)**, and a calibrated **XGBoost classifier** — evaluated across ROC-AUC, Precision-Recall, F1-Score, and confusion matrices. A fully interactive **Streamlit dashboard** provides real-time single-transaction predictions, batch CSV scoring, dynamic threshold control, and SHAP-based explainability for every prediction.

🎥 **Project Demo Video:** [▶ Demonstration and Explanation](https://drive.google.com/file/d/1wwKUNpRSgVg1bz68y_dnYbFz7_TqDF3E/view?usp=sharing)

> **Note:** The demonstration video linked above shows an earlier version of the Streamlit application. After recording the video, I identified and fixed the Threshold bar issue and updated the application. The source code in this repository reflects the latest version with the fix applied, although the video does not.

> 🏢 **Internship Project-1** | Elevate Labs AI/ML Program

---

## ✨ Features

- 🔍 **Unsupervised Anomaly Detection** — Isolation Forest & LOF trained exclusively on legitimate transactions, requiring no fraud labels at training time
- 🤖 **Supervised Classification** — XGBoost with SMOTE oversampling (inside cross-validation folds to prevent leakage) and Isotonic Calibration for well-calibrated fraud probabilities
- 🎯 **Hyperparameter Tuning** — RandomizedSearchCV over 30 configurations with 5-Fold Stratified Cross-Validation, optimising for F1-Score
- 📊 **Comprehensive Evaluation** — ROC curves, Precision-Recall curves, per-model confusion matrices, average precision scores, and a unified metrics comparison table
- 💡 **SHAP Explainability** — TreeExplainer with a 100-sample background provides per-prediction feature contribution scores, saved as `shap_explainer.joblib`
- 🖥️ **Streamlit Dashboard** — Dark-themed interactive UI with dynamic threshold slider, real-time classification, batch CSV upload with ground truth evaluation, and interactive Plotly charts
- 🧪 **Synthetic Test Generator** — `test_fraud.py` generates 40 realistic synthetic transactions (30 legitimate + 10 fraud) using feature distributions derived from the Kaggle dataset, with a `True_Label` column for instant dashboard evaluation

---

## 📂 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── Credit Card Fraud Detection.ipynb      # End-to-end ML pipeline (EDA → training → evaluation → export)
├── streamlit_app.py                       # Streamlit fraud detection dashboard
├── test_fraud.py                          # Synthetic test CSV generator (40 transactions, 10 fraud)
├── requirements.txt                       # Pinned Python dependencies
├── Credit_Card_Fraud_Detection_Report.pdf # Project Report
├── Credit Card Fraud Detection.pdf        # If the notebook is not rendering properly use the pdf instead
│
├── Screenshots/
│   ├── Picture 1.png
│   └── Picture 2.png
│
├── data/
│   ├── analysis.md                              # Detailed model analysis & false-negative deep-dive
│   ├── fraud_predictions_with_explanations.csv  # Predictions + SHAP contributions per transaction
│   └── test_transactions.csv                    # 40-row synthetic test set (30 features + True_Label)
│
└── Graphs/
    ├── class_distribution.png          # Class imbalance bar chart (log scale) + amount distribution
    ├── feature_correlation.png         # Pearson correlation of all features with the fraud label
    ├── cm_isolation_forest.png         # Isolation Forest confusion matrix (Purples)
    ├── cm_lof.png                      # LOF confusion matrix (Oranges)
    ├── cm_xgboost.png                  # XGBoost confusion matrix (Greens)
    ├── confusion_matrices_all.png      # All three confusion matrices side-by-side
    ├── roc_curves.png                  # ROC-AUC curves for all three models
    ├── pr_curves.png                   # Precision-Recall curves (primary metric for imbalanced data)
    ├── metrics_comparison.png          # Bar chart: Precision, Recall, F1, ROC-AUC per model
    ├── xgb_feature_importance.png      # Top-20 XGBoost feature importances (F-score)
    └── Information_of_the_Graphs.md    # Detailed explanation of every graph
```

---

## 📊 Dataset

| Property                | Value                                          |
|-------------------------|------------------------------------------------|
| **Source**              | Kaggle Credit Card Fraud Detection             |
| **Total Transactions**  | 284,807                                        |
| **Time Span**           | 2 days                                         |
| **Fraudulent**          | 492 (0.1727%)                                  |
| **Legitimate**          | 284,315 (99.83%)                               |
| **Imbalance Ratio**     | 578 : 1                                        |
| **Original Features**   | Time, Amount (raw), V1–V28 (PCA-anonymised)    |
| **Processed Features**  | scaled_amount, scaled_time, V1–V28 (30 total)  |
| **Missing Values**      | None                                           |

> ⚠️ The dataset (`creditcard.csv`) is not included due to size. Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root before running the notebook.

### Feature Correlation with Fraud Class

Top features **positively correlated** with fraud (push probability toward fraud):
`V17`, `V14`, `V12`, `V10`, `V16`

Top features **negatively correlated** with fraud (push probability toward legitimate):
`V26`, `V15`, `V25`, `V23`, `V22`

---

## 🛠️ Tech Stack

| Category             | Library / Framework               | Version   | Purpose                                                     |
|----------------------|-----------------------------------|-----------|-------------------------------------------------------------|
| **Language**         | Python                            | 3.10+     | Core development language                                   |
| **Data Handling**    | NumPy                             | 1.26.4    | Numerical arrays and random seed control                    |
|                      | Pandas                            | 2.2.2     | DataFrame operations, CSV I/O                               |
| **Visualisation**    | Matplotlib                        | 3.9.0     | Static plots — confusion matrices, ROC, PR curves           |
|                      | Seaborn                           | 0.13.2    | Heatmap styling for confusion matrices                      |
|                      | Plotly                            | 5.22.0    | Interactive charts in the Streamlit dashboard               |
| **Preprocessing**    | scikit-learn `RobustScaler`       | 1.5.0     | IQR-based scaling of Amount and Time (robust to outliers)   |
|                      | scikit-learn `train_test_split`   | 1.5.0     | Stratified 80/20 split preserving fraud ratio               |
| **Anomaly Detection**| scikit-learn `IsolationForest`    | 1.5.0     | Unsupervised baseline — isolation path length scoring       |
|                      | scikit-learn `LocalOutlierFactor` | 1.5.0     | Unsupervised baseline — local density deviation scoring     |
| **Classifier**       | XGBoost `XGBClassifier`           | 2.0.3     | Gradient-boosted trees with `binary:logistic` objective      |
| **Imbalance Handling**| imbalanced-learn `SMOTE`         | 0.12.3    | Synthetic minority oversampling inside CV folds only        |
|                      | imbalanced-learn `Pipeline`       | 0.12.3    | Ensures SMOTE is never applied to validation/test data      |
| **Tuning**           | scikit-learn `RandomizedSearchCV` | 1.5.0     | 30-iteration random search over hyperparameter space        |
|                      | scikit-learn `StratifiedKFold`    | 1.5.0     | 5-fold CV preserving fraud class ratio in every fold        |
| **Calibration**      | scikit-learn `IsotonicRegression` | 1.5.0     | Calibrates SMOTE-inflated probabilities to true prevalence  |
| **Explainability**   | SHAP `TreeExplainer`              | 0.45.1    | Feature contribution scores for individual predictions      |
| **Persistence**      | joblib                            | 1.4.2     | Serialises pipeline, calibrator, and SHAP explainer         |
| **Dashboard**        | Streamlit                         | 1.36.0    | Interactive web UI for real-time fraud prediction           |

---

## 🧠 Models & Methodology

### 1. 🌲 Isolation Forest (Unsupervised Baseline)

Isolation Forest builds an ensemble of random trees and measures how quickly each point is isolated from the rest. Anomalies such as fraudulent transactions are isolated with **fewer splits**, resulting in shorter average path lengths and lower anomaly scores.

**Training strategy:**
- Trained exclusively on the **legitimate (non-fraud) subset** of the training set (227,451 samples), with no fraud labels required
- `contamination` parameter set to the dataset's true fraud prevalence (~0.17%) to calibrate the decision threshold
- `n_estimators=200` for stable anomaly scores; `n_jobs=-1` for parallel execution
- Anomaly scores are normalised to [0, 1] (higher = more anomalous) for consistent ROC/PR plotting

**Limitation:** Isolation Forest struggles with dense, structured tabular data where fraud patterns overlap with legitimate ones, resulting in a high false-negative rate.

---

### 2. 📍 Local Outlier Factor — LOF (Unsupervised Baseline)

LOF compares the **local density** of each point against its *k* nearest neighbours. A point whose density is significantly lower than its neighbours is flagged as an outlier (potential fraud).

**Training strategy:**
- `novelty=True` mode enables `predict()` on unseen test data (proper train/test setup)
- Trained on a **10,000-sample subset** of legitimate training transactions for computational efficiency
- `n_neighbors=20`, `contamination` matched to dataset fraud prevalence
- LOF raw scores (more negative = more anomalous) normalised to [0, 1] for consistent plotting

**Advantage over Isolation Forest:** Captures local density anomalies more effectively on structured data, achieving significantly higher recall (77.55% vs 27.55%).

---

### 3. ⚡ XGBoost + SMOTE + Isotonic Calibration (Supervised — Best)

A full supervised pipeline using gradient boosting with explicit handling for class imbalance and probability calibration.

**Pipeline architecture:**
1. **SMOTE** (`sampling_strategy=0.5`, `k_neighbors=5`) — synthetically oversamples the minority (fraud) class in training data only, never applied to validation or test folds
2. **XGBClassifier** (`objective='binary:logistic'`, `eval_metric='aucpr'`) — optimises for PR-AUC, which is more informative than ROC-AUC on heavily imbalanced datasets

**Hyperparameter tuning:**
- `RandomizedSearchCV` with **30 random configurations** and **5-Fold Stratified Cross-Validation**, scoring on F1
- Search space covered: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha` (L1), `reg_lambda` (L2), `min_child_weight`, `smote__sampling_strategy`

**Best parameters found:**

| Parameter              | Best Value |
|------------------------|------------|
| `n_estimators`         | 200        |
| `max_depth`            | 6          |
| `learning_rate`        | 0.2        |
| `subsample`            | 0.9        |
| `colsample_bytree`     | 0.7        |
| `reg_alpha` (L1)       | 0.1        |
| `reg_lambda` (L2)      | 1.0        |
| `min_child_weight`     | 1          |
| `smote__sampling_strategy` | 0.5   |
| **Best CV F1-Score**   | **0.8099** |

**Isotonic Calibration:**
- SMOTE inflates raw predicted probabilities (calibrated for a 50/50 distribution, not the true 0.17% fraud rate)
- A 15% hold-out calibration split (stratified, no SMOTE applied) trains an `IsotonicRegression` model that maps raw probabilities back to true prevalence
- Sanity check: an all-zeros feature vector receives a fraud probability of ~4.84%, close to the dataset base rate of 0.1727% — demonstrating effective calibration

**SHAP Explainability:**
- `shap.TreeExplainer` fitted with a 100-sample random background from the training set
- Provides signed feature contribution scores for every individual prediction
- Both the explainer and background dataset are serialised to disk for use in the Streamlit dashboard

---

## 📈 Results

### Train / Test Split

| Split          | Samples    | Fraud %  |
|----------------|------------|----------|
| **Training**   | 227,845    | 0.1729%  |
| **Test**       | 56,962     | 0.1720%  |

### Full Model Comparison (Test Set — 56,962 transactions, 98 fraud)

| Model                | Precision | Recall   | F1-Score | ROC-AUC  | Avg Precision | TP  | FP  | FN  | TN     |
|----------------------|-----------|----------|----------|----------|---------------|-----|-----|-----|--------|
| Isolation Forest     | 0.2109    | 0.2755   | 0.2389   | 0.9523   | 0.1196        | 27  | 101 | 71  | 56,763 |
| LOF                  | 0.3800    | 0.7755   | 0.5101   | 0.9460   | 0.6335        | 76  | 124 | 22  | 56,740 |
| **XGBoost ✅**       | **0.9000**| **0.8265**| **0.8617**| **0.9767**| **0.8432** | **81**| **9**| **17**| **56,855** |

### XGBoost Confusion Matrix (56,962 test transactions)

|                       | Predicted Legitimate | Predicted Fraud |
|-----------------------|----------------------|-----------------|
| **Actual Legitimate** | 56,855 (TN)          | 9 (FP)          |
| **Actual Fraud**      | 17 (FN)              | 81 (TP)         |

- ✅ **Only 9 false positives** out of 56,864 legitimate transactions — minimal customer friction
- ⚡ **82.65% recall** — 81 out of 98 actual fraud cases caught
- 🔑 **Top fraud indicators** (by XGBoost feature importance): `V14`, `V10`, `V4`, `V17`, `V12`, `V8`, `V18`, `V13`, `V7`

---

## 🧪 Synthetic Test Data Generator

`test_fraud.py` generates a **40-transaction synthetic CSV** (30 legitimate + 10 fraud) using the feature distributions from the original Kaggle dataset. It is specifically designed for Streamlit dashboard validation.

**How it works:**
- Legitimate transactions are sampled from distributions centred near 0 (matching PCA component statistics for normal activity)
- Fraud transactions are sampled with significantly shifted means on the key discriminating features (e.g., V14 mean of −6.52, V3 mean of −7.50, V7 mean of −5.43)
- The output CSV includes a `True_Label` column (0 = Legitimate, 1 = Fraud)
- The file is shuffled before saving so fraud rows are not bunched together

**Generated statistics:**

| Feature | Legitimate Mean | Fraud Mean  | Separation   |
|---------|-----------------|-------------|--------------|
| V14     | ~+0.05          | ~−6.52      | Very strong  |
| V3      | ~+0.02          | ~−7.50      | Very strong  |
| V7      | ~−0.03          | ~−5.43      | Strong       |
| V12     | ~+0.01          | ~−5.11      | Strong       |
| V4      | ~−0.03          | ~+4.66      | Strong       |

**Usage:**
```bash
python test_fraud.py
```
This outputs `test_transactions.csv`. Upload it to the Streamlit dashboard's **Batch CSV** mode — the app auto-detects the `True_Label` column and renders a live confusion matrix, accuracy, precision, recall, and F1-Score, highlighting incorrect predictions in orange.

---

## 🖥️ Streamlit Dashboard

The dashboard is a **dark-themed interactive web application** (gradient background: `#0f0c29 → #302b63 → #24243e`) built with Streamlit and Plotly.

**Key features:**

**Single Transaction Mode**
- Input all 30 scaled feature values via sliders or number fields
- Get an instant fraud/legitimate verdict with calibrated probability displayed in a styled result box (animated pulse for fraud, static teal for legitimate)
- View a SHAP bar chart showing the top contributing features for that specific prediction

**Batch CSV Upload Mode**
- Upload any CSV with the 30 feature columns
- If a `True_Label` column is present, the app automatically evaluates model performance and displays a live confusion matrix, classification report, and row-level prediction table with fraud rows highlighted in orange
- Download the predictions as a CSV with an added `fraud_probability` column

**Dynamic Threshold Control**
- A sidebar slider (range 0.01–1.00, default 0.50) adjusts the classification threshold in real time without re-running the model
- Lowering the threshold increases recall at the cost of more false positives — useful for risk-sensitive deployments

**CalibratedWrapper Architecture**
The dashboard loads two separate joblib files (`xgboost_pipeline.joblib` and `isotonic_calibrator.joblib`) and wraps them in a `CalibratedWrapper` class that chains the raw XGBoost probability through the isotonic calibrator before making decisions, ensuring predictions reflect true fraud prevalence.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone -b Project-1-Credit-Card-Fraud-Detection https://github.com/Anishraj2005/Anish_Intern_Elvlabs_AIML.git
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

---

## 🚀 Usage

### Step 1 — Run the Jupyter Notebook
```bash
jupyter notebook "Credit Card Fraud Detection.ipynb"
```
Run all cells sequentially. The notebook will:
- Load and explore the raw dataset (284,807 transactions)
- Apply `RobustScaler` to `Amount` and `Time`, drop originals
- Perform a stratified 80/20 train/test split preserving the 0.17% fraud ratio
- Train Isolation Forest on 227,451 legitimate training transactions
- Train LOF on a 10,000-sample subset of legitimate transactions
- Run RandomizedSearchCV (30 iterations, 5-Fold CV) to find the best XGBoost + SMOTE pipeline
- Calibrate probabilities via Isotonic Regression on a 15% hold-out calibration split
- Generate and save all 10 evaluation graphs to `Graphs/`
- Serialise the pipeline, calibrator, feature names, and SHAP explainer to disk

### Step 2 — Generate synthetic test data (optional)
```bash
python test_fraud.py
```
Outputs `test_transactions.csv` with 40 rows (30 legitimate, 10 fraud) ready for dashboard upload.

### Step 3 — Launch the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501` in your browser. The dashboard requires `xgboost_pipeline.joblib`, `isotonic_calibrator.joblib`, `feature_names.json`, and `shap_explainer.joblib` to be present (generated by the notebook in Step 1).

---

## 📦 Dependencies

```
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.5.0
xgboost==2.0.3
imbalanced-learn==0.12.3
joblib==1.4.2
shap==0.45.1
plotly==5.22.0
streamlit==1.36.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📉 Key Visualisations

| File                          | Description                                                                  |
|-------------------------------|------------------------------------------------------------------------------|
| `class_distribution.png`      | Bar chart (log scale) showing 284,315 vs 492 transactions + amount distribution by class |
| `feature_correlation.png`     | Horizontal bar chart of Pearson correlation coefficients between each feature and the fraud label |
| `cm_isolation_forest.png`     | Confusion matrix for Isolation Forest (purple palette)                       |
| `cm_lof.png`                  | Confusion matrix for LOF (orange palette)                                    |
| `cm_xgboost.png`              | Confusion matrix for XGBoost (green palette)                                 |
| `confusion_matrices_all.png`  | All three confusion matrices side-by-side for direct visual comparison       |
| `roc_curves.png`              | ROC-AUC curves for all models vs the random classifier diagonal              |
| `pr_curves.png`               | Precision-Recall curves — most informative metric for imbalanced classification |
| `metrics_comparison.png`      | 4-panel bar chart comparing Precision, Recall, F1-Score, and ROC-AUC across models |
| `xgb_feature_importance.png`  | Top-20 XGBoost features ranked by F-score split importance                   |

---

## 🔑 Key Findings

- The dataset is **highly imbalanced** at 578:1 — raw accuracy exceeds 99% even for a model that predicts everything as legitimate, making it a completely unreliable evaluation metric
- **Precision-Recall AUC** is the most informative metric: XGBoost achieves 0.8432 vs LOF at 0.6335 and Isolation Forest at only 0.1196
- Unsupervised methods provide important baselines — Isolation Forest's ROC-AUC of 0.9523 suggests its ranking ability is strong, but its hard-label Precision (0.21) and Recall (0.28) confirm it is unsuitable for production use
- LOF substantially outperforms Isolation Forest in recall (0.7755 vs 0.2755), demonstrating that **local density anomaly detection** is more suited to this dataset's structure than isolation-based scoring
- **XGBoost with SMOTE and calibration** delivers the best balance: 90% precision with 82.65% recall at the default 0.5 threshold, with only 9 false positives across 56,864 legitimate transactions
- Features **V14, V10, V4, V17, and V12** dominate XGBoost's decision-making (confirmed by both feature importance and SHAP analysis), aligning with their strong Pearson correlations with the fraud label
- Isotonic calibration is essential: without it, SMOTE-trained models over-estimate fraud probabilities because they were optimised on a 50% fraud training distribution rather than the true 0.17% prevalence

---

## 👤 Author

**Anish Raj**
- Internship: Elevate Labs AI/ML Program
- GitHub: [@Anishraj2005](https://github.com/Anishraj2005)

---

## 📄 License

This project is for educational and internship purposes. The dataset is subject to [Kaggle's terms of use](https://www.kaggle.com/terms).

---

*⭐ If you found this project helpful, please give it a star!*
