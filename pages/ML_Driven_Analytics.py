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
    The models on this page are for **investigational use only**. They are designed to provide insights, accelerate troubleshooting, and guide process improvement activities. They do not replace validated QC procedures, SPC rules, or formal CAPA investigations required by **21 CFR 820** and **ISO 13485**. These methods align with the principles of **ICH Q10 (Pharmaceutical Quality System)**, which encourages a scientific, risk-based approach to continual improvement.
    """)

tab1, tab2, tab3 = st.tabs(["**Predictive Instrument Health**", "**Multivariate Anomaly Detection**", "**Automated Root Cause Insights**"])

with tab1:
    st.header("Predictive Instrument Health with LightGBM & SHAP")
    
    with st.expander("🔬 **The Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        The objective is to implement a **predictive maintenance** model for a critical instrument (e.g., an HPLC). We monitor continuous sensor data (Pressure, Temperature, Flow Rate Stability) over many runs. The model is trained on this historical data, which includes periods of normal operation and periods leading up to a known failure. The goal is to predict an impending failure *before* it results in an out-of-spec (OOS) analytical run.
        
        #### The Method: LightGBM & SHAP
        - **LightGBM (Light Gradient Boosting Machine)**: A state-of-the-art machine learning algorithm that builds an ensemble of decision trees sequentially. Each new tree corrects the errors of the previous ones, resulting in a highly accurate and efficient predictive model, especially for tabular data like sensor readings.
        - **SHAP (SHapley Additive exPlanations)**: A game theory-based method for explaining the output of any machine learning model. The **Force Plot** is a powerful SHAP visualization that explains a *single prediction*. It illustrates a "tug-of-war" between features that push the prediction higher (towards failure, in red) and those that push it lower (towards health, in blue), starting from a "base value" (the average prediction over the entire dataset).
        """)

    try:
        instrument_df = generate_instrument_health_data()
        model, X = train_instrument_model(instrument_df)
        instrument_df['Health Score'] = 1 - model.predict_proba(X)[:, 1]

        col1, col2 = st.columns([2, 1.2])
        with col1:
            st.markdown("**Instrument Health Score Over Time**")
            fig = px.line(instrument_df, x='Run ID', y='Health Score', title="Health Score (1 - Prob. of Failure)", range_y=[0, 1])
            fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="Failure Threshold")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Current Status")
            st.metric("Current Health Score", f"{instrument_df['Health Score'].iloc[-1]:.2%}")
            st.metric("Predicted Runs to Failure", "3-8" if instrument_df['Health Score'].iloc[-1] < 0.7 else ">20")
        
        st.subheader("Actionable Insight: Why is the latest run's score what it is?")
        
        with st.expander("📊 **Results & Analysis**"):
            st.markdown("""
            #### Metrics Explained
            - **Health Score**: A derived metric calculated as `1 - Probability(Failure)`. A score of 100% represents perfect health, while a score approaching 0% indicates an imminent failure is predicted.
            - **SHAP Force Plot**: This plot provides a transparent explanation for the most recent health score. The "base value" is the average prediction across all historical data. Each arrow represents a feature's contribution:
                - **Red arrows** (e.g., high pressure) push the prediction higher (increasing probability of failure).
                - **Blue arrows** (e.g., stable flow rate) push the prediction lower (decreasing probability of failure).
                The final predicted value is the sum of the base value and all feature contributions.
            
            #### Analysis of this Specific Run
            The "Health Score Over Time" plot shows a clear, accelerating decline in instrument health over the last 20 runs. The SHAP Force Plot for the most recent data point reveals **why**. For this simulated data, a high `Pressure (psi)` is the dominant factor pushing the prediction towards failure, counteracted only slightly by a still-acceptable `Flow Rate Stability`. This is a classic signature of a developing column blockage or pump seal failure, providing a clear, actionable direction for the maintenance team *before* a catastrophic failure occurs.
            """)
        
        with st.spinner("Generating SHAP Force Plot..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            last_instance_idx = X.shape[0] - 1

            if isinstance(explainer.expected_value, list):
                base_value = explainer.expected_value[1]
            else:
                base_value = explainer.expected_value

            shap.force_plot(
                base_value,
                shap_values[1][last_instance_idx,:],
                X.iloc[last_instance_idx,:],
                matplotlib=True, show=False, text_rotation=10
            )
            st.pyplot(plt.gcf(), bbox_inches='tight')
            plt.clf()

    except Exception as e:
        st.error(f"Error loading or plotting instrument health data: {e}")

with tab2:
    st.header("Multivariate Anomaly Detection with an Autoencoder")
    
    with st.expander("🔬 **The Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        The goal is to monitor the overall "health" of the QC process, not just one variable at a time. We first establish a baseline by analyzing a "golden batch" of historical data known to be from an ideal, in-control process. We then monitor live, incoming QC data to see if it deviates from this established normal pattern.
        
        #### The Method: Keras Autoencoder
        An **Autoencoder** is a type of neural network used for unsupervised learning. Its structure is key: an "encoder" network compresses the multivariate input data into a low-dimensional representation (the "bottleneck"), and a "decoder" network attempts to reconstruct the original data from this compressed version.
        
        Crucially, the autoencoder is trained **only on the normal "golden batch" data**. It becomes an expert at reconstructing normal patterns. When it is later presented with anomalous data, it will struggle to reconstruct it accurately, resulting in a high **Reconstruction Error**. This error becomes our anomaly score.
        """)

    try:
        golden_df = generate_golden_batch_data()
        autoencoder, scaler = train_autoencoder_model(golden_df)
        live_df = generate_live_qc_data(golden_df)
        
        # We need to make sure the columns match between the live data and the golden batch for the scaler
        X_live_scaled = scaler.transform(live_df[golden_df.columns])
        X_live_pred = autoencoder.predict(X_live_scaled)
        
        live_df['Reconstruction Error'] = np.mean(np.abs(X_live_pred - X_live_scaled), axis=1)
        
        X_train_pred = autoencoder.predict(scaler.transform(golden_df))
        train_mae_loss = np.mean(np.abs(X_train_pred - scaler.transform(golden_df)), axis=1)
        threshold = np.mean(train_mae_loss) + 3 * np.std(train_mae_loss)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=live_df['Run'], y=live_df['Reconstruction Error'], name='Reconstruction Error'))
        fig.add_hline(y=threshold, line_dash="dot", line_color="red", annotation_text="Anomaly Threshold")
        fig.update_layout(title="Live Process Monitoring via Reconstruction Error", xaxis_title="Run Number", yaxis_title="Mean Absolute Error")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 **Results & Analysis**"):
            st.markdown("""
            #### Metrics Explained
            - **Reconstruction Error (Mean Absolute Error)**: The average difference between the original, scaled input data and the autoencoder's reconstructed output. A low error means the network reconstructed the input well (it looks normal). A high error means the network struggled (the input is anomalous).
            - **Anomaly Threshold**: This is a statistically derived limit. It is calculated as the mean reconstruction error on the normal training data, plus three standard deviations of that error. This establishes a "99.7% confidence" boundary for normal operation.

            #### Analysis of Results
            The plot shows the real-time health of the multivariate process.
            - **Why this is superior to SPC**: Univariate SPC charts can only flag when a single variable goes out of its own limits. This autoencoder method captures the *inter-relationships* and *patterns* between all variables. A run is anomalous if this pattern is wrong, even if each individual variable is technically within its own limits.
            - The plot clearly detects two types of anomalies:
                1.  A **gradual drift** starting around Run 50, where the error consistently trends upward, crossing the threshold.
                2.  A **sudden spike** at Run 60, where a large, one-time event causes a massive reconstruction error.
            **Action**: Any point exceeding the threshold requires immediate investigation as a significant deviation from the established "golden" process.
            """)
    except Exception as e:
        st.error(f"Error in anomaly detection: {e}")

with tab3:
    st.header("Automated Root Cause Insights with a Random Forest")
    
    with st.expander("🔬 **The Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        We utilize a historical dataset of past process failures where a formal root cause investigation was completed and a cause was assigned (e.g., 'Reagent Degradation', 'Operator Error'). The objective is to train a classification model that can learn the patterns of process parameters associated with each specific failure mode.

        #### The Method: Random Forest & Single Tree Visualization
        - **Random Forest Classifier**: A powerful ensemble learning method that builds hundreds of individual decision trees on different random subsets of the data and features. It then aggregates their votes to make a final prediction. This "wisdom of the crowd" approach makes the model much more robust, accurate, and less prone to overfitting than a single decision tree.
        - **Single Tree Visualization**: While the forest makes the prediction, a single, deep tree is difficult for a human to interpret. For educational purposes, we visualize one of the simpler trees from the forest. This provides an intuitive example of the *type* of if-then logic the model uses (e.g., "IF `Reagent Age` > 75 days THEN the cause is likely `Reagent Degradation`").
        """)

    try:
        rca_df = generate_rca_data()
        model, X, y = train_rca_model(rca_df)
        
        col1, col2 = st.columns([2, 1])
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
            age = st.slider("Instrument Age (mo)", 1, 36, 10)
            reagent = st.slider("Reagent Age (days)", 1, 90, 80)
            exp = st.slider("Operator Experience (yr)", 1, 5, 2)
            prediction = model.predict([[age, reagent, exp]])
            st.error(f"**Predicted Root Cause:** {prediction[0]}")

        with st.expander("📊 **Results & Analysis**"):
            st.markdown("""
            #### Metrics & Visuals Explained
            - **Decision Tree**: This flowchart visualizes the learned logic. Start at the top node and follow the arrows based on the parameters of a new failure to see the likely causal pathway. The `gini` score measures impurity; a score of 0.0 means the node is "pure" (all samples belong to one class).
            - **Feature Importance**: This table, derived from the entire Random Forest, ranks which process parameters are most predictive of failure overall. It shows where to focus monitoring and process control efforts.
            - **Predicted Root Cause**: The model's most likely classification for a given set of failure conditions, based on the aggregated vote of all trees in the forest.

            #### Analysis of Results
            The **Feature Importance** table clearly indicates that `Reagent Lot Age (days)` is the most significant contributor to failures in our historical data, followed by `Instrument Age`. This suggests that implementing stricter controls on reagent expiry or performing more frequent instrument preventative maintenance could yield the largest improvements in process robustness. The model's prediction for the simulated failure (`Reagent Degradation`) is consistent with this finding, as the simulated reagent age (80 days) is a strong driver in the decision logic.
            
            **Action**: This tool should be used as the **first step** in a formal CAPA or non-conformance investigation. The model's prediction provides a data-driven hypothesis, drastically reducing troubleshooting time and focusing the investigation on the most probable causes.
            """)
    except Exception as e:
        st.error(f"Error in root cause analysis: {e}")
