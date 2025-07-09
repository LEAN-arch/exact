# pages/ML_Driven_Analytics.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
from utils import (generate_instrument_health_data, train_instrument_model,
                   generate_golden_batch_data, generate_live_qc_data, train_autoencoder_model,
                   generate_rca_data, train_rca_model)

st.set_page_config(page_title="ML-Driven Analytics", layout="wide")
st.title("🤖 ML-Driven Process Analytics")
st.markdown("### Advanced analytics for proactive process control and deep insights.")

with st.expander("⚠️ Important Disclaimer & Regulatory Context"):
    st.warning("""
    The models on this page are for **investigational use only**. They are designed to provide insights, accelerate troubleshooting, and guide process improvement activities. They do not replace validated QC procedures, SPC rules, or formal CAPA investigations required by **21 CFR 820** and **ISO 13485**.
    """)

tab1, tab2, tab3 = st.tabs(["**Predictive Instrument Health**", "**Multivariate Anomaly Detection**", "**Automated Root Cause Insights**"])

with tab1:
    st.header("Predictive Instrument Health with LightGBM & SHAP")
    
    instrument_df = generate_instrument_health_data()
    model, X = train_instrument_model(instrument_df)
    instrument_df['Health Score'] = 1 - model.predict_proba(X)[:, 1]

    col1, col2 = st.columns([2,1.2])
    with col1:
        st.markdown("**Instrument Health Score Over Time**")
        fig = px.line(instrument_df, x='Run ID', y='Health Score', title="Health Score (1 - Prob. of Failure)", range_y=[0, 1])
        fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="Failure Threshold")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Current Health Score", f"{instrument_df['Health Score'].iloc[-1]:.2%}")
        st.metric("Predicted Runs to Failure", "3-8" if instrument_df['Health Score'].iloc[-1] < 0.7 else ">20")
    
    st.subheader("Actionable Insight: Why is the latest run's score what it is?")
    st.markdown("This **SHAP Force Plot** provides an intuitive explanation for the most recent prediction. Features in red push the prediction towards failure, while features in blue push it towards health.")
    
    # --- THIS IS THE DEFINITIVE FIX FOR THE IndexError ---
    with st.spinner("Generating SHAP Force Plot..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        last_instance_idx = X.shape[0] - 1

        # Handle the explainer.expected_value being a scalar or a list
        # For class 1 (Failure), the base value is the second element if it's a list,
        # or the single scalar value itself.
        if isinstance(explainer.expected_value, list):
            base_value = explainer.expected_value[1]
        else:
            base_value = explainer.expected_value

        # Use the most robust plotting pattern
        shap.force_plot(
            base_value,
            shap_values[1][last_instance_idx,:],
            X.iloc[last_instance_idx,:],
            matplotlib=True,
            show=False,
            text_rotation=10
        )
        st.pyplot(plt.gcf(), bbox_inches='tight')
        plt.clf() # Clear the figure to prevent state leakage
    # --- END OF FIX ---


with tab2:
    st.header("Multivariate Anomaly Detection with an Autoencoder")
    st.markdown("A neural network is trained on a 'golden batch' of normal data. It learns to reconstruct normal patterns. Anomalies are detected when the **Reconstruction Error** is high.")

    # 1. Train model on normal data
    golden_df = generate_golden_batch_data()
    autoencoder, scaler = train_autoencoder_model(golden_df)

    # 2. Apply to live data (which includes anomalies)
    live_df = generate_live_qc_data(golden_df)
    X_live_scaled = scaler.transform(live_df.drop('Run', axis=1))
    X_live_pred = autoencoder.predict(X_live_scaled)
    
    # 3. Calculate reconstruction error
    mae_loss = np.mean(np.abs(X_live_pred - X_live_scaled), axis=1)
    live_df['Reconstruction Error'] = mae_loss
    
    # 4. Set a dynamic threshold
    X_train_pred = autoencoder.predict(scaler.transform(golden_df))
    train_mae_loss = np.mean(np.abs(X_train_pred - scaler.transform(golden_df)), axis=1)
    threshold = np.mean(train_mae_loss) + 3 * np.std(train_mae_loss)

    # 5. Visualize
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=live_df['Run'], y=live_df['Reconstruction Error'], name='Reconstruction Error'))
    fig.add_hline(y=threshold, line_dash="dot", line_color="red", annotation_text="Anomaly Threshold")
    fig.update_layout(title="Live Process Monitoring via Reconstruction Error",
                      xaxis_title="Run Number", yaxis_title="Mean Absolute Error")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("SME Interpretation & Action Items"):
        st.markdown("""
        - **Why this is better:** Unlike univariate SPC, this method captures the *inter-relationships* between all monitored variables. A run is anomalous if the *pattern* is wrong, even if each individual variable is within its own limits.
        - **Below Threshold:** The process is behaving like the "golden batch" and is considered normal.
        - **Above Threshold:** The process has deviated significantly from normal operation. This requires immediate investigation. Notice how the model flags both the gradual drift and the sudden spike.
        """)

with tab3:
    st.header("Automated Root Cause Insights with a Random Forest")
    st.markdown("A robust Random Forest model suggests the most likely root cause of a failure, while we can still inspect a single tree for interpretability.")
    
    rca_df = generate_rca_data()
    model, X, y = train_rca_model(rca_df)
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Example Decision Logic (from one tree in the forest)")
        fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
        plot_tree(model.estimators_[5], feature_names=X.columns, class_names=sorted(y.unique()), filled=True, rounded=True, ax=ax_tree, fontsize=10)
        st.pyplot(fig_tree)
    with col2:
        st.subheader("Most Important Factors (from the full forest)")
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
        st.dataframe(feature_importance)
        st.subheader("Simulate a New Failure")
        age = st.slider("Instrument Age (mo)", 1, 36, 10); reagent = st.slider("Reagent Age (days)", 1, 90, 80); exp = st.slider("Operator Experience (yr)", 1, 5, 2)
        prediction = model.predict([[age, reagent, exp]])
        st.error(f"**Predicted Root Cause:** {prediction[0]}")
