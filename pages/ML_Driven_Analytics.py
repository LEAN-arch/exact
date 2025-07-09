# pages/ML_Driven_Analytics.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt  # This line will now work correctly
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
    
    # --- Model Training & Prediction ---
    instrument_df = generate_instrument_health_data()
    # Note: Assuming train_instrument_model from utils returns best_params as the third element
    model, X, best_params = train_instrument_model(instrument_df)
    instrument_df['Health Score'] = 1 - model.predict_proba(X)[:, 1]

    # --- Visualization ---
    col1, col2 = st.columns([2,1])
    with col1:
        fig = px.line(instrument_df, x='Run ID', y='Health Score', title="Instrument Health Score Over Time", range_y=[0, 1])
        fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="Failure Threshold")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Current Health Score", f"{instrument_df['Health Score'].iloc[-1]:.2%}")
        st.metric("Predicted Runs to Failure", "5-10" if instrument_df['Health Score'].iloc[-1] < 0.7 else ">20")
        st.info("A score below 50% indicates an imminent failure prediction.")

    st.subheader("Model Explainability (SHAP Analysis)")
    st.markdown("This plot shows *why* the model is making its prediction. Red features are pushing the score towards failure.")
    
    # SHAP Plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    fig_shap, ax_shap = plt.subplots(figsize=(10, 4))
    # Check if shap_values is a list (for binary classification)
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig_shap)

with tab2:
    st.header("Multivariate Anomaly Detection in QC Data")
    st.markdown("Using an Isolation Forest to find subtle, multi-dimensional anomalies that univariate SPC charts miss.")
    
    # --- Model Training & Prediction ---
    qc_df = generate_multivariate_qc_data()
    model, qc_df_encoded = train_anomaly_model(qc_df)
    qc_df['Anomaly'] = model.predict(qc_df_encoded[['Operator', 'Reagent Lot', 'Value']])
    
    # --- Visualization ---
    fig_3d = px.scatter_3d(
        qc_df, x='Run', y='Value', z='Operator', color=qc_df['Anomaly'].astype(str),
        color_discrete_map={'1': 'blue', '-1': 'red'},
        symbol='Reagent Lot',
        title="3D View of QC Data with ML-Detected Anomalies"
    )
    fig_3d.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    st.plotly_chart(fig_3d, use_container_width=True)

    with st.expander("SME Interpretation & Action Items"):
        st.markdown("""
        - **Blue Points (1)**: Data points the model considers 'normal'.
        - **Red Points (-1)**: Anomalies. Notice how the red point might not be an outlier on the 'Value' axis alone, but its combination of `Value` and `Operator` is unusual.
        - **Action**: When an anomaly is detected, investigate the specific combination of factors for that run. This is far more powerful than just looking at the value itself.
        """)

with tab3:
    st.header("Automated Root Cause Insights")
    st.markdown("Using a Decision Tree to provide a first-pass suggestion for the root cause of a process failure.")
    
    # --- Model Training & Visualization ---
    rca_df = generate_rca_data()
    # Note: Assuming train_rca_model from utils returns best_params as the fourth element
    model, X, y, best_params_rca = train_rca_model(rca_df)
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Failure Investigation Decision Tree")
        fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
        plot_tree(model, feature_names=X.columns, class_names=y.unique(), filled=True, rounded=True, ax=ax_tree, fontsize=10)
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

    with st.expander("SME Interpretation & Action Items"):
        st.markdown("""
        - **Decision Tree**: This chart visualizes the logic learned from historical data. Start at the top and follow the arrows based on the parameters of a new failure to see the likely cause.
        - **Feature Importance**: This shows which factors are most predictive of failure overall. 'Reagent Lot Age' is clearly the dominant factor in our historical data.
        - **Action**: Use the model's prediction as the *first hypothesis* in a formal CAPA investigation, drastically reducing troubleshooting time.
        """)
