import streamlit as st
import numpy as np
import pandas as pd
import joblib
import itertools

# -------------------------------
# Load model and feature columns
# -------------------------------
model = joblib.load("xgb_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# -------------------------------
# Feature functions
# -------------------------------
def compute_aac(seq):
    seq = seq.upper()
    length = len(seq)
    aac = {}
    for aa in AA_LIST:
        aac[f"AAC_{aa}"] = seq.count(aa) / length
    return aac

def compute_dpc(seq):
    seq = seq.upper()
    length = len(seq) - 1
    dpc = {}
    for a1 in AA_LIST:
        for a2 in AA_LIST:
            pair = a1 + a2
            dpc[f"DPC_{pair}"] = 0

    for i in range(len(seq) - 1):
        pair = seq[i:i+2]
        if pair in [a+b for a in AA_LIST for b in AA_LIST]:
            dpc[f"DPC_{pair}"] += 1

    if length > 0:
        for k in dpc:
            dpc[k] /= length

    return dpc

def extract_features(seq):
    aac = compute_aac(seq)
    dpc = compute_dpc(seq)
    feats = {**aac, **dpc}
    df = pd.DataFrame([feats])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Influenza Epitope Predictor", layout="centered")

st.title("🧬 Influenza T-cell Epitope Predictor")
st.write("XGBoost model using AAC + DPC features")

seq = st.text_input("Enter peptide sequence (e.g. GILGFVFTL):")

if st.button("Predict"):
    if len(seq) < 8:
        st.error("Sequence too short!")
    else:
        X = extract_features(seq)
        prob = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]

        st.subheader("🔬 Prediction Result")

        st.write(f"**Epitope Probability:** `{prob:.4f}`")

        if pred == 1:
            st.success("✅ Predicted as EPITOPE")
        else:
            st.warning("❌ Predicted as NON-EPITOPE")
