"""
Credit Card Fraud Detection — Streamlit Dashboard with SHAP Explanations
========================================================================
Features:
- Dynamic threshold slider (0.01–1.0) with manual input
- Real‑time classification updates (no model re‑run)
- SHAP‑based explanation for each prediction
- Batch CSV upload with feature‑importance column
- Modern dark‑theme UI with gradient backgrounds
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

# =============================================================================
#  CalibratedWrapper class 
# =============================================================================
class CalibratedWrapper:
    def __init__(self, model, iso):
        self.model = model
        self.iso = iso

    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        calib = np.clip(self.iso.predict(raw), 0.0, 1.0)
        return np.vstack([1 - calib, calib]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ── Page Configuration & Custom CSS ──────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff6a00, #ee0979);
        -webkit-background-clip: text;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #aaa;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .fraud-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 16px;
        padding: 25px;
        color: white;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        animation: pulse 1.5s infinite;
    }
    .legit-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 16px;
        padding: 25px;
        color: white;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 15px;
        margin: 5px;
        border-left: 4px solid #ff6a00;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.02); opacity: 0.95; }
        100% { transform: scale(1); opacity: 1; }
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff6a00, #ee0979);
        border: none;
        border-radius: 30px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(255,106,0,0.3);
    }
    .explanation-box {
        background: rgba(0,0,0,0.4);
        border-radius: 12px;
        padding: 15px;
        margin-top: 15px;
        border-left: 4px solid #ff6a00;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load All Artifacts (Model, Explainer, Features) ──────────────────────
@st.cache_resource
def load_artifacts():
    try:
        pipeline = joblib.load("xgboost_pipeline.joblib")
        iso = joblib.load("isotonic_calibrator.joblib")
        model = CalibratedWrapper(pipeline, iso)
        explainer = joblib.load("shap_explainer.joblib")
        background = joblib.load("shap_background.joblib")
        with open("feature_names.json") as f:
            features = json.load(f)
        return model, explainer, background, features
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {e}. Please run the training notebook first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load artifacts: {e}")
        st.stop()

MODEL, EXPLAINER, BACKGROUND, FEATURE_NAMES = load_artifacts()

# ── Session State Initialisation ─────────────────────────────────────────
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5
if "manual_proba" not in st.session_state:
    st.session_state.manual_proba = None
if "manual_shap" not in st.session_state:
    st.session_state.manual_shap = None
if "csv_probas" not in st.session_state:
    st.session_state.csv_probas = None
if "csv_uploaded_hash" not in st.session_state:
    st.session_state.csv_uploaded_hash = None

# ── Helper: SHAP Explanation for a Single Row ────────────────────────────
def get_shap_explanation(input_df):
    """
    Returns:
        base_value: expected model output (log‑odds) – converted to probability later
        contributions: Series of SHAP values per feature
        explanation_text: formatted string with top contributing features
    """
    # Extract the underlying XGBoost model from the pipeline
    xgb_model = MODEL.model.named_steps["xgb"]
    shap_values = EXPLAINER.shap_values(input_df)
    base_value = EXPLAINER.expected_value
    # For binary classification, shap_values shape = (n_samples, n_features)
    contributions = pd.Series(shap_values[0], index=FEATURE_NAMES)
    # Top absolute contributions
    top_n = 5
    top_contrib = contributions.abs().sort_values(ascending=False).head(top_n)
    explanation_lines = []
    for feat in top_contrib.index:
        sign = "increases" if contributions[feat] > 0 else "decreases"
        explanation_lines.append(f"• **{feat}**: {sign} risk by {abs(contributions[feat]):.3f}")
    explanation_text = "\n".join(explanation_lines)
    return base_value, contributions, explanation_text

def probability_from_logit(logit):
    """Convert log‑odds (SHAP base value + sum of contributions) to probability."""
    return 1 / (1 + np.exp(-logit))

# ── Sidebar: Threshold Control ───────────────────────────────────────────
def _on_slider_change():
    st.session_state.threshold = st.session_state.thresh_slider
    st.session_state.thresh_num = st.session_state.thresh_slider

def _on_num_change():
    st.session_state.threshold = st.session_state.thresh_num
    st.session_state.thresh_slider = st.session_state.thresh_num

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fraud.png", width=80)
    st.title("⚙️ Controls")
    st.markdown("### 🎛️ Decision Threshold")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.slider(
            "Threshold",
            min_value=0.01,
            max_value=1.0,
            value=st.session_state.threshold,
            step=0.01,
            format="%.2f",
            key="thresh_slider",
            on_change=_on_slider_change,
        )
    with col2:
        st.number_input(
            "",
            min_value=0.01,
            max_value=1.0,
            value=st.session_state.threshold,
            step=0.01,
            format="%.2f",
            key="thresh_num",
            on_change=_on_num_change,
        )

    thresh_color = "#2ecc71" if st.session_state.threshold <= 0.3 else \
                   "#f39c12" if st.session_state.threshold <= 0.7 else "#e74c3c"
    st.markdown(
        f"<div style='text-align:center; padding:5px; border-radius:10px; "
        f"background:{thresh_color}20; border-left:3px solid {thresh_color};'>"
        f"<span style='color:{thresh_color}; font-weight:bold;'>⚡ Threshold: {st.session_state.threshold:.2f}</span></div>",
        unsafe_allow_html=True
    )
    st.divider()
    st.markdown("### 📊 Model Performance")
    st.markdown("""
    - **Precision:** 0.900
    - **Recall:** 0.827
    - **F1-Score:** 0.862
    - **ROC-AUC:** 0.977
    """)
    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "**XGBoost + SMOTE** pipeline calibrated with isotonic regression. "
        "**SHAP** explains each prediction."
    )

# ── Main Header ─────────────────────────────────────────────────────────
st.markdown("<div class='main-header'>🛡️ Credit Card Fraud Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Built by Evil Legend</div>", unsafe_allow_html=True)

# =============================================================================
# MANUAL INPUT MODE
# =============================================================================
st.subheader("📝 Single Transaction Analysis")
with st.form("manual_form"):
    col1, col2, col3 = st.columns(3)
    input_vals = {}
    with col1:
        st.markdown("**Transaction Metadata**")
        input_vals["scaled_amount"] = st.number_input("scaled_amount", value=0.0, format="%.6f")
        input_vals["scaled_time"] = st.number_input("scaled_time", value=0.0, format="%.6f")
        st.markdown("**V1 – V10**")
        for i in range(1, 11):
            input_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.6f")
    with col2:
        st.markdown("**V11 – V20**")
        for i in range(11, 21):
            input_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.6f")
    with col3:
        st.markdown("**V21 – V28**")
        for i in range(21, 29):
            input_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.6f")

    submitted = st.form_submit_button("🔍 Analyze & Explain", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([input_vals])[FEATURE_NAMES]
    proba = MODEL.predict_proba(input_df)[0, 1]
    st.session_state.manual_proba = proba
    # Compute SHAP explanation (may take a second)
    base_val, shap_vals, expl_text = get_shap_explanation(input_df)
    st.session_state.manual_shap = (base_val, shap_vals, expl_text)

# Display manual result (reacts to threshold changes)
if st.session_state.manual_proba is not None:
    prob = st.session_state.manual_proba
    thresh = st.session_state.threshold
    is_fraud = prob >= thresh

    col_res, col_gauge = st.columns([1, 1])
    with col_res:
        if is_fraud:
            st.markdown(
                f"<div class='fraud-box'>🚨 FRAUD DETECTED<br>"
                f"<span style='font-size:1rem'>Probability: {prob*100:.1f}% ≥ {thresh*100:.0f}% threshold</span></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='legit-box'>✅ LEGITIMATE<br>"
                f"<span style='font-size:1rem'>Probability: {prob*100:.1f}% < {thresh*100:.0f}% threshold</span></div>",
                unsafe_allow_html=True
            )
        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud Probability", f"{prob*100:.2f}%")
        m2.metric("Decision Threshold", f"{thresh*100:.0f}%")
        m3.metric("Prediction", "FRAUD" if is_fraud else "LEGITIMATE")

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#764ba2"},
                "steps": [
                    {"range": [0, 30], "color": "#2ecc71"},
                    {"range": [30, 70], "color": "#f39c12"},
                    {"range": [70, 100], "color": "#e74c3c"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": thresh * 100}
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # SHAP explanation
    if st.session_state.manual_shap is not None:
        base_val, shap_vals, expl_text = st.session_state.manual_shap
        # Convert base value (log‑odds) to probability
        base_prob = probability_from_logit(base_val)
        with st.expander("🔍 Why this prediction? (SHAP Explanation)", expanded=True):
            st.markdown(
                f"<div class='explanation-box'>"
                f"<b>📊 Baseline fraud probability (average over background):</b> {base_prob:.2%}<br><br>"
                f"<b>⚡ Top features driving this prediction:</b><br>{expl_text}<br><br>"
                f"<b>💡 Interpretation:</b> Positive contributions increase fraud risk; negative contributions decrease it."
                f"</div>",
                unsafe_allow_html=True
            )

# =============================================================================
# BATCH CSV MODE
# =============================================================================
st.subheader("📂 Batch CSV Analysis")
st.markdown("Upload a CSV with the same feature columns. Results include fraud probability, prediction, and top‑contributing features.")
template_df = pd.DataFrame(columns=FEATURE_NAMES)
st.download_button(
    label="⬇️ Download CSV Template",
    data=template_df.to_csv(index=False),
    file_name="transaction_template.csv",
    mime="text/csv",
    use_container_width=True
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_upload")

if uploaded_file is not None:
    try:
        upload_df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in FEATURE_NAMES if c not in upload_df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            # Compute probabilities only once per file (using hash to detect changes)
            file_hash = hash(upload_df.values.tobytes())
            if st.session_state.csv_probas is None or st.session_state.csv_uploaded_hash != file_hash:
                with st.spinner("Running predictions..."):
                    probas = MODEL.predict_proba(upload_df[FEATURE_NAMES])[:, 1]
                    st.session_state.csv_probas = probas
                    st.session_state.csv_uploaded_hash = file_hash
            else:
                probas = st.session_state.csv_probas

            # Apply current threshold
            thresh = st.session_state.threshold
            preds = (probas >= thresh).astype(int)

            result_df = upload_df.copy()
            result_df["Fraud_Probability"] = (probas * 100).round(2)
            result_df["Prediction"] = np.where(preds == 1, "FRAUD", "LEGITIMATE")

            # Add SHAP explanations for the first 100 rows.
            # A single batched shap_values() call replaces the old per-row loop;
            # TreeExplainer is optimized for batch processing.
            with st.spinner("Generating explanations (top 100 rows)..."):
                limit = min(100, len(upload_df))
                batch_df = upload_df.iloc[:limit][FEATURE_NAMES]
                all_shap_vals = EXPLAINER.shap_values(batch_df)  # shape (limit, n_features)
                base_value = EXPLAINER.expected_value

                top_n = 5
                explanations = []
                for i in range(limit):
                    contribs = pd.Series(all_shap_vals[i], index=FEATURE_NAMES)
                    top_feats = contribs.abs().sort_values(ascending=False).head(top_n)
                    lines = []
                    for feat in top_feats.index:
                        sign = "increases" if contribs[feat] > 0 else "decreases"
                        lines.append(f"• {feat}: {sign} risk by {abs(contribs[feat]):.3f}")
                    explanations.append("; ".join(lines))

                # Rows beyond the limit get a placeholder
                explanations.extend(["(limit reached)"] * (len(upload_df) - limit))
            result_df["Top_Contributing_Features"] = explanations

            # Summary
            fraud_cnt = int(preds.sum())
            legit_cnt = len(preds) - fraud_cnt
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", f"{len(preds):,}")
            col2.metric("🚨 Fraud Detected", fraud_cnt, delta=f"{fraud_cnt/len(preds)*100:.2f}%", delta_color="inverse")
            col3.metric("✅ Legitimate", legit_cnt)
            col4.metric("Current Threshold", f"{thresh*100:.0f}%")

            # Preview
            st.dataframe(result_df.head(20), use_container_width=True)

            # Download full results
            st.download_button(
                label="⬇️ Download Full Results (with explanations)",
                data=result_df.to_csv(index=False),
                file_name="fraud_predictions_with_explanations.csv",
                mime="text/csv",
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")

st.divider()
st.caption("🛡️ Fraud Detection System | Explanations powered by SHAP | Threshold updates classification instantly")