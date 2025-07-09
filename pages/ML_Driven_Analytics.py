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
    st.markdown("Using a Random Forest model to predict instrument failure from sensor data *before* it occurs.")
    
    # This call now correctly unpacks the 3 values returned by the updated utils function
    instrument_df = generate_instrument_health_data()
    model, X, best_params = train_instrument_model(instrument_df)
    instrument_df['Health Score'] = 1 - model.predict_proba(X)[:, 1]

    col1, col2 = st.columns([2,1])
    with col1:
        fig = px.line(instrument_df, x='Run ID', y='Health Score', title="Instrument Health Score Over Time", range_y=[0, 1])
        fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="Failure Threshold")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Current Health Score", f"{instrument_df['Health Score'].iloc[-1]:.2%}")
        st.metric("Predicted Runs to Failure", "5-10" if instrument_df['Health Score'].iloc[-1] < 0.7 else ">20")
        with st.expander("Optimized Model Parameters"):
            st.json(best_params)

    st.subheader("Model Explainability (SHAP Analysis)")
    st.markdown("This plot shows the impact of each feature on the prediction of a failure. Features in red increase the likelihood of failure.")
    
    # --- THIS IS THE DEFINITIVE FIX FOR THE AssertionError ---
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # We know shap_values is a list of 2 arrays for our binary classifier.
    # We explicitly select the SHAP values for class 1 (Failure) to match the data shape.
    # This directly resolves the AssertionError.
    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    
    # We use the robust pattern of capturing the current figure and clearing it.
    st.pyplot(plt.gcf())
    plt.clf()
    # --- END OF FIX ---

with tab2:
    st.header("Multivariate Anomaly Detection in QC Data")
    st.markdown("Using an Isolation Forest to find subtle, multi-dimensional anomalies that univariate SPC charts miss.")
    
    qc_df = generate_multivariate_qc_data()
    model, qc_df_encoded = train_anomaly_model(qc_df)
    qc_df['Anomaly'] = model.predict(qc_df_encoded[['Operator', 'Reagent Lot', 'Value']])
    
    fig_3d = px.scatter_3d(
        qc_df, x='Run', y='Value', z='Operator', color=qc_df['Anomaly'].astype(str),
        color_discrete_map={'1': 'blue', '-1': 'red'}, symbol='Reagent Lot',
        title="3D View of QC Data with ML-Detected Anomalies"
    )
    fig_3d.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.header("Automated Root Cause Insights")
    st.markdown("Using a Decision Tree to provide a first-pass suggestion for the root cause of a process failure.")
    
    # This call now correctly unpacks the 4 values returned by the updated utils function
    rca_df = generate_rca_data()
    model, X, y, best_params_rca = train_rca_model(rca_df)
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Failure Investigation Decision Tree")
        fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
        plot_tree(model, feature_names=X.columns, class_names=sorted(y.unique()), filled=True, rounded=True, ax=ax_tree, fontsize=10)
        st.pyplot(fig_tree)
    with col2:
        st.subheader("Most Important Factors")
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
        st.dataframe(feature_importance)
        st.subheader("Simulate a New Failure")
        age = st.slider("Instrument Age (mo)", 1, 36, 10)
        reagent = st.slider("Reagent Age (days)", 1, 90, 80)
        exp = st.slider("Operator Experience (yr)", 1, 5, 2)
        prediction = model.predict([[age, reagent, exp]])
        st.error(f"**Predicted Root Cause:** {prediction[0]}")
        with st.expander("Optimized Model Parameters"):
            st.json(best_params_rca)
