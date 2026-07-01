# Information_of_the_Graphs.md

# Credit Card Fraud Detection – Graph Explanations

Dataset: **Kaggle Credit Card Fraud Detection Dataset**
- Total transactions: 284,807
- Fraudulent transactions: 492 (~0.17%)
- Legitimate transactions: 284,315
- Highly imbalanced binary classification problem

---

## 1(a). Transaction Class Distribution (`class_distribution.png`)

### Purpose
Shows the imbalance between legitimate and fraudulent transactions.

### Interpretation
- Legitimate transactions dominate the dataset.
- Fraud cases represent only a tiny fraction of all records.
- A logarithmic scale is used to make the fraud count visible.

### Key Insight
The extreme class imbalance means that accuracy alone is not a reliable evaluation metric.

---

## 1(b). Transaction Amount Distribution by Class (`class_distribution.png`)

### Purpose
Compares transaction amount distributions for legitimate and fraudulent transactions.

### Interpretation
- Most transactions occur at lower amounts.
- Fraudulent transactions are concentrated in lower and medium transaction ranges.
- Both classes show a right-skewed distribution.

### Key Insight
Transaction amount alone is insufficient for fraud detection and must be combined with other features.

---

## 2. Feature Correlation with Fraud Class (`feature_correlation.png`)

### Purpose
Displays Pearson correlation coefficients between features and the target class.

### Interpretation
- Positive values indicate features associated with fraud.
- Negative values indicate features associated with legitimate transactions.
- Features such as V14, V17, V12, and V10 often show strong relationships with fraud.

### Key Insight
Certain PCA-transformed features contain strong fraud-discriminating information.

---

## 3. Isolation Forest Confusion Matrix (`cm_isolation_forest.png`)

### Purpose
Evaluates Isolation Forest anomaly detection performance.

### Matrix Meaning
- Top-left: Correctly classified legitimate transactions (True Negatives).
- Top-right: Legitimate transactions incorrectly flagged as fraud (False Positives).
- Bottom-left: Fraud transactions missed (False Negatives).
- Bottom-right: Correctly detected frauds (True Positives).

### Key Insight
Isolation Forest detects fraud without labels but misses a significant portion of fraudulent transactions.

---

## 4. Local Outlier Factor (LOF) Confusion Matrix (`cm_lof.png`)

### Purpose
Evaluates Local Outlier Factor anomaly detection performance.

### Interpretation
- Better fraud detection than Isolation Forest.
- Produces more true positives.
- Still generates some false alarms.

### Key Insight
LOF captures local density anomalies and is more effective than Isolation Forest on this dataset.

---

## 5. XGBoost Confusion Matrix (`cm_xgboost.png`)

### Purpose
Evaluates the supervised XGBoost classifier.

### Interpretation
- Very high number of correctly classified legitimate transactions.
- High fraud detection rate.
- Very few false positives and false negatives.

### Key Insight
XGBoost substantially outperforms unsupervised anomaly detection methods.

---

## 6. Combined Confusion Matrices (`confusion_matrices_all.png`)

### Purpose
Provides side-by-side comparison of all three models.

### Interpretation
- Isolation Forest has the weakest fraud detection capability.
- LOF improves detection performance.
- XGBoost achieves the best balance between fraud detection and false alarms.

### Key Insight
Visual comparison clearly demonstrates the superiority of supervised learning for this problem.

---

## 7. ROC Curves (`roc_curves.png`)

### Purpose
Compares model discrimination ability across different classification thresholds.

### Interpretation
- Curves closer to the upper-left corner indicate better performance.
- The diagonal line represents random guessing.
- Higher AUC indicates stronger separation between fraud and legitimate transactions.

### Key Insight
XGBoost achieves the highest ROC-AUC, followed by LOF and Isolation Forest.

---

## 8. Precision–Recall Curves (`pr_curves.png`)

### Purpose
Evaluates performance on the highly imbalanced dataset.

### Interpretation
- Precision measures how many predicted frauds are actually fraud.
- Recall measures how many actual frauds are detected.
- Higher area under the curve indicates better fraud detection.

### Key Insight
Precision–Recall analysis is more informative than ROC analysis for highly imbalanced datasets, and XGBoost performs best.

---

## 9. Model Performance Comparison (`metrics_comparison.png`)

### Purpose
Compares Precision, Recall, F1-Score, and ROC-AUC for all models.

### Interpretation
- Isolation Forest shows the weakest overall performance.
- LOF improves recall and F1-score.
- XGBoost achieves the highest scores across nearly all metrics.

### Key Insight
XGBoost is the most effective model for credit card fraud detection in this project.

---

## 10. XGBoost Feature Importance (`xgb_feature_importance.png`)

### Purpose
Ranks the 20 most influential features used by XGBoost.

### Interpretation
- Features near the top contribute most to prediction decisions.
- V14, V10, V4, V17 and related PCA components are among the strongest predictors.
- Importance values represent relative contribution to model splits.

### Key Insight
Only a subset of features drives most fraud detection decisions, which helps explain model behavior.

---

# Overall Conclusion

### Model Ranking
1. XGBoost (Best)
2. Local Outlier Factor (LOF)
3. Isolation Forest

### Main Findings
- The dataset is extremely imbalanced.
- Precision–Recall metrics are critical for evaluation.
- Unsupervised anomaly detection provides useful baselines.
- Supervised learning (XGBoost) delivers significantly superior fraud detection performance.
- Features such as V14, V17, V12, and V10 contribute heavily to identifying fraudulent transactions.
