```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
import logging
from utils import (generate_instrument_health_data, train_instrument_model,
                   generate_golden_batch_data, generate_live_qc_data, train_autoencoder_model,
                   generate_rca_data, train_rca_model)

# Set Matplotlib backend for non-interactive rendering
plt.switch_backend('Agg')

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    
    try:
        instrument_df = generate_instrument_health_data()
        if instrument_df.empty or 'Run ID' not in instrument_df.columns:
            raise ValueError("Instrument data is empty or missing 'Run ID' column")
        model, X = train_instrument_model(instrument_df)
        if X.empty or not X.select_dtypes(include=[np.number]).columns.tolist():
            raise ValueError("Feature data (X) is empty or contains no numeric columns")
        # Replace spaces in feature names with underscores
        X.columns = [col.replace(' ', '_') for col in X.columns]
        instrument_df['Health Score'] = 1 - model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"Error loading instrument health data: {e}")
        logger.error(f"Instrument health data error: {e}")
        st.stop()

    col1, col2 = st.columns([2, 1])
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
    
    @st.cache_resource
    def get_shap_explainer(_model):
        return shap.TreeExplainer(_model)
    
    try:
        with st.spinner("Generating SHAP Force Plot..."):
            explainer = get_shap_explainer(model)
            shap_values = explainer.shap_values(X)
            last_instance_idx = X.shape[0] - 1

            # Debug output
            st.write(f"Debug: X shape: {X.shape}, SHAP values shape: {np.array(shap_values).shape}, Last instance index: {last_instance_idx}, Model classes: {model.classes_}")

            # Handle SHAP values for single-class or multi-class output
            if isinstance(shap_values, list):
                if len(shap_values) != len(model.classes_):
                    raise ValueError(f"SHAP values length ({len(shap_values)}) does not match model classes ({len(model.classes_)})")
                failure_class_idx = list(model.classes_).index(1) if 1 in model.classes_ else 1
                base_value = explainer.expected_value[failure_class_idx]
                shap_vals = shap_values[failure_class_idx][last_instance_idx, :]
            else:
                base_value = explainer.expected_value
                shap_vals = shap_values[last_instance_idx, :]

            if shap_vals.size == 0 or X.iloc[last_instance_idx, :].empty:
                raise ValueError("SHAP values or input data for last instance is empty")

            # HTML-based force plot
            shap.initjs()
            force_plot = shap.force_plot(
                base_value,
                shap_vals,
                X.iloc[last_instance_idx, :],
                feature_names=X.columns,
                show=False
            )
            shap.save_html("force_plot.html", force_plot)
            with open("force_plot.html", "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=400)

            # Alternative: Bar plot
            st.markdown("**Alternative: SHAP Bar Plot** (shown for additional context)")
            fig_bar = plt.figure(figsize=(10, 6))
            shap.bar_plot(shap_vals, feature_names=X.columns, max_display=10)
            st.pyplot(fig_bar)
            plt.close(fig_bar)
    except Exception as e:
        st.error(f"Error generating SHAP plot: {e}")
        logger.error(f"SHAP plot error: {e}")

with tab2:
    st.header("Multivariate Anomaly Detection with an Autoencoder")
    st.markdown("A neural network is trained on a 'golden batch' of normal data. It learns to reconstruct normal patterns. Anomalies are detected when the **Reconstruction Error** is high.")

    try:
        golden_df = generate_golden_batch_data()
        if golden_df.empty or 'Run' not in golden_df.columns:
            raise ValueError("Golden batch data is empty or missing 'Run' column")
        autoencoder, scaler = train_autoencoder_model(golden_df)

        live_df = generate_live_qc_data(golden_df)
        if live_df.empty or 'Run' not in live_df.columns:
            raise ValueError("Live QC data is empty or missing 'Run' column")
        X_live_scaled = scaler.transform(live_df.drop('Run', axis=1))
        X_live_pred = autoencoder.predict(X_live_scaled)
        
        mae_loss = np.mean(np.abs(X_live_pred - X_live_scaled), axis=1)
        live_df['Reconstruction Error'] = mae_loss
        
        X_train_scaled = scaler.transform(golden_df.drop('Run', axis=1))
        X_train_pred = autoencoder.predict(X_train_scaled)
        train_mae_loss = np.mean(np.abs(X_train_pred - X_train_scaled), axis=1)
        threshold = np.mean(train_mae_loss) + 3 * np.std(train_mae_loss)

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
    except Exception as e:
        st.error(f"Error in anomaly detection: {e}")
        logger.error(f"Anomaly detection error: {e}")

with tab3:
    st.header("Automated Root Cause Insights with a Random Forest")
    st.markdown("A robust Random Forest model suggests the most likely root cause of a failure, while we can still inspect a single tree for interpretability.")
    
    try:
        rca_df = generate_rca_data()
        if rca_df.empty:
            raise ValueError("RCA data is empty")
        model, X, y = train_rca_model(rca_df)
        if X.empty or y.empty:
            raise ValueError("Feature data (X) or target (y) is empty")
        # Replace spaces in feature names with underscores
        X.columns = [col.replace(' ', '_') for col in X.columns]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Example Decision Logic (from one tree in the forest)")
            fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
            plot_tree(model.estimators_[5], feature_names=X.columns, class_names=sorted(y.unique()), filled=True, rounded=True, ax=ax_tree, fontsize=10)
            st.pyplot(fig_tree)
            plt.close(fig_tree)
        with col2:
            st.subheader("Most Important Factors (from the full forest)")
            feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
            st.dataframe(feature_importance)
            st.subheader("Simulate a New Failure")
            age = st.slider("Instrument Age (mo)", 1, 36, 10)
            reagent = st.slider("Reagent Age (days)", 1, 90, 80)
            exp = st.slider("Operator Experience (yr)", 1, 5, 2)
            input_data = pd.DataFrame([[age, reagent, exp]], columns=X.columns)
            prediction = model.predict(input_data)
            st.write(f"**Predicted Root Cause:** {prediction[0]}")
    except Exception as e:
        st.error(f"Error in root cause analysis: {e}")
        logger.error(f"Root cause analysis error: {e}")
```
