# pages/ML_Driven_Analytics.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import shap
from utils import (generate_instrument_health_data, train_instrument_model,
                   generate_multivariate_qc_data, train_anomaly_model,
                   generate_rca_data, train_rca_model)

st.set_page_config(page_title="ML-Driven Analytics", layout="wide")
st.title("🤖 ML-Driven Process Analytics")
st.markdown("### Proactive and predictive insights using advanced machine learning models.")

with st.expander("⚠️ Important Disclaimer & Regulatory Context"):
    st.warning("""
    The models on this page are for **investigational use only**. They are designed to provide insights, accelerate troubleshooting, and guide process improvement activities. They do not replace validated QC procedures, SPC rules, or formal CAPA investigations required by **21 CFR 820** and **ISO 13485**.
    """)

tab1, tab2, tab3 = st.tabs(["**Predictive Instrument Health**", "**Multivariate Anomaly Detection**", "**Automated Root Cause Insights**"])

with tab1:
    st.header("Predictive Instrument Health (e.g., HPLC System)")
    instrument_df = generate_instrument_health_data()
    model, X = train_instrument_model(instrument_df)
    instrument_df['Health Score'] = 1 - model.predict_proba(X)[:, 1]
    col1, col2 = st.columns([2,1])
    with col1:
        fig = px.line(instrument_df, x='Run ID', y='Health Score', title="Instrument Health Score Over Time", range_y=[0, 1])
        fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="Failure Threshold")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Current Health Score", f"{instrument_df['Health Score'].iloc[-1]:.2%}")
        st.metric("Predicted Runs to Failure", "5-10" if instrument_df['Health Score'].iloc[-1] < 0.7 else ">20")
    st.subheader("Model Explainability (SHAP Analysis)")
    explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(X)
    fig_shap, ax_shap = plt.subplots(figsize=(10, 4))
    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False); st.pyplot(fig_shap)

with tab2:
    st.header("Multivariate Anomaly Detection in QC Data")
    qc_df = generate_multivariate_qc_data()
    model, qc_df_encoded = train_anomaly_model(qc_df)
    qc_df['Anomaly'] = model.predict(qc_df_encoded[['Operator', 'Reagent Lot', 'Value']])
    fig_3d = px.scatter_3d(qc_df, x='Run', y='Value', z='Operator', color=qc_df['Anomaly'].astype(str),
        color_discrete_map={'1': 'blue', '-1': 'red'}, symbol='Reagent Lot', title="3D View of QC Data with ML-Detected Anomalies")
    fig_3d.update_traces(marker=dict(size=5), selector=dict(mode='markers')); st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.header("Automated Root Cause Insights")
    rca_df = generate_rca_data()
    model, X, y = train_rca_model(rca_df)
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Failure Investigation Decision Tree")
        fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
        plot_tree(model, feature_names=X.columns, class_names=y.unique(), filled=True, rounded=True, ax=ax_tree, fontsize=10)
        st.pyplot(fig_tree)
    with col2:
        st.subheader("Most Important Factors")
        st.dataframe(pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False))
        st.subheader("Simulate a New Failure")
        age = st.slider("Instrument Age (mo)", 1, 36, 10); reagent = st.slider("Reagent Age (days)", 1, 90, 80); exp = st.slider("Operator Experience (yr)", 1, 5, 2)
        st.error(f"**Predicted Root Cause:** {model.predict([[age, reagent, exp]])[0]}")
