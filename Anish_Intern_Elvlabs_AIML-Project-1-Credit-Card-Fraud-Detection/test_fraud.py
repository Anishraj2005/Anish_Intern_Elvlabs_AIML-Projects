"""
test_fraud.py — Realistic Test CSV Generator
=============================================
Generates 40 synthetic transactions (30 legitimate + 10 fraud) using
feature distributions derived from the Kaggle creditcard.csv dataset.

The output CSV includes a True_Label column (0 = Legit, 1 = Fraud) so
you can upload it to the Streamlit app and immediately see:
  - Which rows are fraud vs legitimate (known ground truth)
  - Confusion matrix + accuracy / precision / recall / F1
  - Which specific transactions were correctly or incorrectly classified

Usage:
    python test_fraud.py

Output:
    test_transactions.csv   (upload this to the Streamlit app)
"""

import pandas as pd
import numpy as np

np.random.seed(42)

FEATURES = [
    'scaled_amount', 'scaled_time',
    'V1',  'V2',  'V3',  'V4',  'V5',  'V6',  'V7',  'V8',  'V9',  'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
]

# ── Feature distributions from Kaggle creditcard.csv ─────────────────────────
# Each entry: (mean, std) for that class.
# Legitimate: V features are centered near 0 (PCA components of normal transactions).
# Fraud: key features (V14, V12, V17, V11, V4, V10, V3, V7) are far from 0.

LEGIT_PARAMS = {
    'scaled_amount': ( 0.30, 1.50),
    'scaled_time':   ( 0.00, 1.20),
    'V1':  (-0.01, 1.50), 'V2':  ( 0.02, 1.40), 'V3':  ( 0.02, 1.50),
    'V4':  (-0.03, 1.30), 'V5':  (-0.04, 1.40), 'V6':  (-0.04, 1.30),
    'V7':  (-0.03, 1.30), 'V8':  ( 0.00, 1.20), 'V9':  ( 0.00, 1.00),
    'V10': (-0.01, 1.00), 'V11': ( 0.00, 0.99), 'V12': ( 0.01, 0.91),
    'V13': (-0.01, 0.90), 'V14': ( 0.05, 0.92), 'V15': ( 0.00, 0.90),
    'V16': (-0.01, 0.85), 'V17': ( 0.04, 1.00), 'V18': (-0.01, 0.85),
    'V19': ( 0.00, 0.80), 'V20': (-0.01, 0.75), 'V21': (-0.01, 0.70),
    'V22': ( 0.00, 0.70), 'V23': (-0.01, 0.65), 'V24': ( 0.00, 0.65),
    'V25': ( 0.00, 0.60), 'V26': ( 0.00, 0.55), 'V27': ( 0.00, 0.45),
    'V28': ( 0.00, 0.40),
}

FRAUD_PARAMS = {
    # Fraud amounts are often small (fraudsters test cards with minor charges)
    'scaled_amount': ( 0.10, 1.20),
    'scaled_time':   ( 0.00, 1.20),
    # Key discriminating features — significantly different from legitimate
    'V1':  (-3.50, 2.00), 'V2':  ( 2.50, 2.50), 'V3':  (-7.50, 5.00),
    'V4':  ( 4.66, 2.03), 'V5':  (-3.00, 2.50), 'V6':  (-1.50, 2.00),
    'V7':  (-5.43, 4.00), 'V8':  ( 0.50, 1.50), 'V9':  (-1.50, 2.00),
    'V10': (-3.55, 2.67), 'V11': ( 2.59, 2.01), 'V12': (-5.11, 3.05),
    'V13': (-0.10, 1.00), 'V14': (-6.52, 2.25), 'V15': ( 0.00, 1.00),
    'V16': (-2.62, 2.54), 'V17': (-4.13, 2.55), 'V18': (-1.50, 2.00),
    'V19': ( 0.50, 1.50), 'V20': ( 0.50, 1.50), 'V21': ( 0.50, 1.50),
    'V22': ( 0.20, 1.00), 'V23': (-0.20, 1.00), 'V24': ( 0.10, 0.80),
    'V25': ( 0.10, 0.80), 'V26': ( 0.10, 0.70), 'V27': ( 0.30, 1.00),
    'V28': ( 0.20, 0.80),
}

# Most important discriminating features (based on feature importance + dataset EDA)
IMPORTANT_FEATURES = ['V14', 'V12', 'V17', 'V11', 'V4', 'V10', 'V3', 'V7', 'V16']


def generate_rows(params, n):
    return [
        {feat: np.random.normal(mu, sigma)
         for feat, (mu, sigma) in params.items()}
        for _ in range(n)
    ]


# Generate transactions
legit_rows = generate_rows(LEGIT_PARAMS, 30)
fraud_rows = generate_rows(FRAUD_PARAMS, 10)

legit_df = pd.DataFrame(legit_rows, columns=FEATURES)
legit_df['True_Label'] = 0

fraud_df = pd.DataFrame(fraud_rows, columns=FEATURES)
fraud_df['True_Label'] = 1

# Combine and shuffle
df = pd.concat([legit_df, fraud_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv('test_transactions.csv', index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("test_transactions.csv created")
print("=" * 60)
print(f"  Total rows  : {len(df)}")
print(f"  Legitimate  : {(df['True_Label'] == 0).sum()}  (True_Label = 0)")
print(f"  Fraud       : {(df['True_Label'] == 1).sum()}  (True_Label = 1)")
print()

fraud_rows_idx = df[df['True_Label'] == 1].index.tolist()
print(f"Fraud row indices (0-based): {fraud_rows_idx}")
print()

# Show the key V14 values to illustrate the separation
print("V14 sample (most discriminating feature):")
print(f"  Legitimate mean : {df.loc[df['True_Label']==0, 'V14'].mean():+.2f}  (true dataset: ~+0.05)")
print(f"  Fraud mean      : {df.loc[df['True_Label']==1, 'V14'].mean():+.2f}  (true dataset: ~-6.52)")
print()

print("─" * 60)
print("How to use:")
print("  1. Run this script to (re)generate test_transactions.csv")
print("  2. Open the Streamlit app → Upload CSV mode")
print("  3. Upload test_transactions.csv")
print("  4. The app detects the True_Label column automatically")
print("     and shows: confusion matrix, accuracy, precision,")
print("     recall, F1, and which rows were wrong (orange)")
print("─" * 60)
print()
print("Reading the confusion matrix output in the app:")
print("  True Positive  (TP) — fraud correctly caught")
print("  True Negative  (TN) — legitimate correctly cleared")
print("  False Negative (FN) — fraud MISSED → most dangerous error")
print("  False Positive (FP) — legitimate wrongly flagged → annoying but safe")
