# Fraud Detection Model Analysis Report

## 1. Overview

This report analyzes the predictions made by a calibrated XGBoost fraud detection model on a test set of **40 credit card transactions**. The test set contains 10 actual fraudulent transactions (25% fraud rate, higher than the typical 0.17% prevalence – this is a deliberately enriched sample for evaluation). The model outputs a fraud probability and a binary prediction (threshold = 0.5), along with the top five features contributing to each prediction.

Data sources:
- `test_transactions.csv` – contains 30 scaled features (`scaled_amount`, `scaled_time`, V1–V28) and the true label.
- `fraud_predictions_with_explanations.csv` – adds model predictions, probabilities, and SHAP-based feature contributions.

---

## 2. Model Performance Summary

| Metric       | Value |
|--------------|-------|
| Accuracy     | 95.0% |
| Precision    | 100%  |
| Recall       | 80.0% |
| F1‑Score     | 88.9% |
| True Positives  | 8     |
| False Negatives | 2     |
| False Positives | 0     |
| True Negatives  | 30    |

**Confusion Matrix**

|                | Predicted Legitimate | Predicted Fraud |
|----------------|----------------------|-----------------|
| Actual Legitimate | 30                   | 0               |
| Actual Fraud      | 2                    | 8               |

- **No false positives** – every transaction flagged as fraud was indeed fraudulent. This is excellent for a fraud detection system, as it avoids annoying legitimate customers.
- **Recall of 80%** – the model caught 8 out of 10 fraud cases. The two missed frauds (false negatives) are analysed separately.

---

## 3. Fraud Probability Distribution

### For Actual Fraudulent Transactions (n=10)

| Fraud Probability | Count |
|-------------------|-------|
| 100%              | 2     |
| 91.67%            | 1     |
| 74.58%            | 1     |
| 57.14%            | 4     |
| 4.84%             | 1 (FN)|
| 0.43%             | 1 (FN)|

- Most detected frauds have probabilities well above the 0.5 threshold.
- The two false negatives have extremely low probabilities (<5%), indicating the model is confidently wrong. This suggests these fraud patterns may be very different from the training data or resemble legitimate behaviour.

### For Actual Legitimate Transactions (n=30)

All legitimate transactions received fraud probabilities below 0.5. The highest probability among them was 0.43 (still classified as legitimate). Most were below 0.2. This demonstrates excellent calibration – the model does not cry wolf.

---

## 4. Top Contributing Features

The model uses SHAP values to explain each prediction. Aggregating across all fraud cases, the most influential features that **increase** fraud risk are:

| Feature | Role in Fraud Detection |
|---------|-------------------------|
| **V14** | Most consistently increases risk (present in every fraud explanation) |
| **V4**  | Second strongest positive contributor |
| **V3**  | Frequently increases risk |
| **V12** | Often pushes probability upward |
| **V10** | Notable in high-confidence frauds |

For **legitimate transactions**, the top features that **decrease** fraud risk (i.e., push predictions toward “legitimate”) include:

- **V8** (often decreases risk)
- **V14** (when negative, it reduces risk)
- **V4**, **V1**, **V12** can act in either direction depending on their values.

**Example from a correctly detected fraud (True Label=1, Probability=100%):**
> • V14: increases risk by 7.689  
> • V4: increases risk by 4.085  
> • V12: increases risk by 2.967  
> • V3: increases risk by 2.738  
> • V10: increases risk by 2.506  

**Example from a false negative (True Label=1, Probability=4.84%):**
> • V14: increases risk by 4.341  
> • V4: increases risk by 3.983  
> • V12: increases risk by 2.751  
> • V3: increases risk by 2.724  
> • V10: increases risk by 2.453  

Even though the same features were pushing toward fraud, the final probability remained low. This suggests that other features (not shown in top 5) strongly countered the fraud signal, or the interaction effects were unusual.

---

## 5. False Negative Analysis

The two missed frauds (rows 15 and 40 in the dataset) have the following characteristics:

| Row | True Label | Predicted Probability | Prediction | Key features (top contrib.) |
|-----|------------|----------------------|------------|-----------------------------|
| 15  | 1          | 0.43%                | Legitimate | V4 (+3.197), V14 (+3.193), V12 (+2.667), V8 (+1.851), V28 (−1.808) |
| 40  | 1          | 4.84%                | Legitimate | V14 (+4.341), V4 (+3.983), V12 (+2.751), V3 (+2.724), V10 (+2.453) |

Despite strong positive contributions from typical fraud indicators (V14, V4, V12), the final probability remained below 5%. Possible reasons:
- **Strong counteracting features** – features not listed in the top five (e.g., V28 in row 15 decreased risk) may have overwhelmed the signal.
- **Unusual feature interactions** – the combination of values might be rare in the training set, causing the model to default to “legitimate.”
- **Calibration edge** – isotonic regression may map very low raw probabilities to even lower calibrated probabilities for such edge cases.

**Recommendation:** Investigate these two transactions further. If they represent a new fraud pattern, consider retraining with additional examples or adjusting the decision threshold (e.g., lowering it to 0.3 would catch row 40 but might increase false positives).

---

## 6. Conclusion

The XGBoost model with isotonic calibration performs exceptionally well on this test set:

- **Perfect precision** – zero false alarms, ideal for production where user friction must be minimised.
- **80% recall** – catches most frauds, but two evaded detection due to very low predicted probabilities.
- **Well-calibrated probabilities** – legitimate transactions rarely exceed 0.5, and detected frauds show high confidence.

The model’s primary fraud indicators are **V14, V4, V3, V12, and V10**. These should be monitored for drift over time.

**Next steps:**
- Analyse the two false negatives to understand if they are anomalies or require model updates.
- Consider a lower decision threshold (e.g., 0.3) if the business can tolerate a small number of false positives.
- Deploy with monitoring (SHAP explanations, probability distribution over time) to maintain performance.

---*Report generated from fraud_predictions_with_explanations.csv and test_transactions.csv*  