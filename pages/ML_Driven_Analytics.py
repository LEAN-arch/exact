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
                   train_rca_model, generate_rca_data) # generate_rca_data is now fixed in memory to return RAW data

st.set_page_config(page_title="ML-Driven Analytics | Exact Sciences", layout="wide")
st.title("🤖 ML-Driven Process & Instrument Analytics")
st.markdown("### Applying advanced analytics for proactive process control, predictive maintenance, and accelerated troubleshooting.")

with st.expander("⚠️ Important Disclaimer & Regulatory Context"):
    st.warning("""
    **FOR INVESTIGATIONAL USE ONLY.**

    The models on this page are powerful analytical tools intended to **guide and accelerate investigations**, not to replace validated QC procedures. They serve as a mechanism for:
    - **Proactive Process Monitoring:** Identifying subtle trends before they breach SPC limits.
    - **Accelerating Troubleshooting:** Providing data-driven hypotheses for root cause analysis.
    - **Process Understanding:** Uncovering complex interactions between process parameters.

    These models are **not validated for GxP decision-making** (e.g., lot release). Their use aligns with the principles of **ICH Q10 (Pharmaceutical Quality System)** and **FDA's Process Analytical Technology (PAT) guidance**, which encourage a scientific, risk-based approach to process understanding and continual improvement.
    """)

tab1, tab2 = st.tabs(["**Predictive Instrument Health (NGS Sequencer)**", "**Automated Root Cause Insights (Cologuard® Assay)**"])

with tab1:
    st.header("Predictive Instrument Health with LightGBM & SHAP")

    with st.expander("🔬 **The Goal & Method**"):
        st.markdown("""
        #### The Goal: Predictive Maintenance for a NovaSeq Instrument
        The objective is to move from reactive maintenance (fixing the sequencer after it fails a run) to **predictive maintenance**. We monitor continuous sensor data from the instrument (e.g., laser power, flow cell temperature, pump pressure) over many runs. The model is trained on this historical data, which includes periods of normal operation and periods leading up to known failures. The goal is to predict an impending failure *before* it results in a costly, out-of-spec (OOS) sequencing run, saving time, reagents, and precious samples.

        #### The Method: LightGBM & SHAP
        - **LightGBM (Light Gradient Boosting Machine)**: A state-of-the-art machine learning algorithm that builds an "ensemble" of decision trees. It is highly effective for the tabular, time-series sensor data we collect from our instruments. It learns the complex, non-linear patterns that often precede a failure.
        - **SHAP (SHapley Additive exPlanations)**: A cutting-edge technique from game theory used to explain the output of any machine learning model. For each prediction, SHAP assigns an "importance value" to each feature, showing how much that feature contributed to pushing the model's prediction away from the baseline. This transforms a "black box" prediction into a transparent, actionable insight.
        """)

    try:
        instrument_df = generate_instrument_health_data()
        model, X = train_instrument_model(instrument_df)
        instrument_df['Health Score'] = 1 - model.predict_proba(X)[:, 1]

        col1, col2 = st.columns([2, 1.2])
        with col1:
            st.markdown("**NGS Sequencer Health Score Over Time**")
            fig = px.line(instrument_df, x='Run_ID', y='Health Score', title="Health Score (1 - Probability of Failure)", range_y=[0, 1])
            fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="Action Threshold")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Current Instrument Status")
            st.metric("Current Health Score", f"{instrument_df['Health Score'].iloc[-1]:.2%}")
            st.metric("Predicted Runs to Failure", "<5" if instrument_df['Health Score'].iloc[-1] < 0.7 else ">10")
            st.info("A score below 50% triggers a work order for the maintenance team.")

        st.subheader("SHAP Analysis: Why is the latest run's score what it is?")
        st.markdown("The SHAP Force Plot below provides a transparent explanation for the most recent health score. It visualizes a 'tug-of-war' between features pushing the model towards predicting a failure (red) and features pushing it towards predicting normal operation (blue).")

        with st.spinner("Generating SHAP Force Plot..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            last_instance_idx = X.shape[0] - 1

            shap_values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

            # --- FIX: Explicitly capture the plot object from SHAP and pass it to st.pyplot ---
            force_plot = shap.force_plot(
                base_value,
                shap_values_for_plot[last_instance_idx,:],
                X.iloc[last_instance_idx,:],
                matplotlib=True,
                show=False # Important: Do not let SHAP try to show the plot itself
            )
            st.pyplot(force_plot, bbox_inches='tight')
            plt.clf()
            # --- END OF FIX ---

        with st.expander("📊 **Results & Analysis**"):
            st.markdown("""
            #### Interpreting the Plots
            - **Health Score Trend**: The line chart shows a clear, accelerating decline in instrument health over the last 15 runs. This provides a crucial early warning.
            - **SHAP Force Plot**: This plot provides the actionable "why."
                - The **base value** is the average prediction across all historical data.
                - **Red arrows** (e.g., low `Laser_A_Power`) are features pushing the prediction higher (increasing probability of failure). Their size indicates the magnitude of their impact.
                - **Blue arrows** (e.g., normal `Pump_Pressure_psi`) push the prediction lower (decreasing probability of failure).
                - The **final predicted value (f(x))** is the health score, which is the sum of the base value and all feature contributions.

            #### Analysis of this Specific Run
            The SHAP plot reveals that the sharp drop in the health score is being driven almost entirely by the **degrading `Laser_A_Power`**. The other parameters are still within their normal operating ranges and are trying to "pull" the prediction back to a healthy state. This is a classic signature of a laser nearing its end-of-life.

            **Actionable Insight:** This is not a generic "instrument failing" alarm. This is a specific, data-driven hypothesis: "The 'A' laser is the likely point of failure." The maintenance team can now proactively schedule the replacement of this specific component, preventing an OOS run and minimizing instrument downtime.
            """)

    except Exception as e:
        st.error(f"Error loading or plotting instrument health data: {e}")


with tab2:
    st.header("Automated Root Cause Insights with a Random Forest")

    with st.expander("🔬 **The Goal & Method**"):
        st.markdown("""
        #### The Goal: Accelerate Root Cause Investigation for Cologuard® Assay Failures
        When a Cologuard® manufacturing batch fails QC, a time-consuming investigation is required to determine the root cause. We can accelerate this process by training a model on a historical dataset of past failures where a formal root cause was assigned (e.g., 'Reagent Degradation', 'Liquid Handler Error'). The model learns the patterns of process parameters associated with each specific failure mode.

        #### The Method: Random Forest Classifier
        - **Random Forest**: A powerful ensemble learning method that builds hundreds of individual decision trees. By aggregating the "votes" from all trees, it produces a highly accurate and robust prediction. It is particularly good at handling a mix of numerical and categorical data, like we have in our process data (reagent age, instrument IDs, operator IDs).
        - **Feature Importance**: A key output of the Random Forest model. It calculates which process parameters were, on average, the most influential in making correct predictions across the entire forest. This tells us where the biggest sources of process variability are.
        """)

    try:
        # --- FIX: Separate data generation from encoding ---
        # 1. Generate the raw data with original categorical columns. The function in utils.py is now corrected in memory.
        rca_df_raw = generate_rca_data()

        # 2. Create the encoded dataframe for model training
        rca_df_encoded = pd.get_dummies(rca_df_raw, columns=['Instrument ID', 'Operator ID'], drop_first=True)
        model, X_encoded, y = train_rca_model(rca_df_encoded)
        # --- END OF FIX ---

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Top Drivers of Historical Failures")
            st.markdown("Feature importance from the full Random Forest model.")
            feature_importance = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
            
            fig_imp = px.bar(feature_importance.head(10), x='Importance', y='Feature', orientation='h', title="Top 10 Most Predictive Features")
            fig_imp.update_yaxes(categoryorder="total ascending", title=None)
            st.plotly_chart(fig_imp, use_container_width=True)

        with col2:
            st.subheader("Simulate a New Failure for RCA")
            st.markdown("Enter the parameters for a new failed run to get a predicted root cause.")
            
            # Use the raw dataframe for user-friendly selectors
            reagent = st.slider("Reagent Lot Age (days)", 1, 120, 95, key="rca_reagent")
            instrument = st.selectbox("Instrument ID", rca_df_raw['Instrument ID'].unique())
            operator = st.selectbox("Operator ID", rca_df_raw['Operator ID'].unique())

            # --- FIX: Correctly encode the user's input for prediction ---
            input_data = pd.DataFrame([[reagent, instrument, operator]], columns=['Reagent Lot Age (days)', 'Instrument ID', 'Operator ID'])
            input_encoded = pd.get_dummies(input_data).reindex(columns=X_encoded.columns, fill_value=0)
            # --- END OF FIX ---
            
            prediction = model.predict(input_encoded)
            prediction_proba = model.predict_proba(input_encoded)
            
            st.error(f"**Predicted Root Cause:** {prediction[0]}")
            
            proba_df = pd.DataFrame(prediction_proba, columns=model.classes_).T
            proba_df.columns = ['Probability']
            proba_df = proba_df.sort_values('Probability', ascending=False)
            st.write("Prediction Confidence:")
            st.dataframe(proba_df.style.format({'Probability': '{:.1%}'.format}))


        with st.expander("📊 **Results & Analysis**"):
            st.markdown("""
            #### Interpreting the Plots & Predictions
            - **Feature Importance**: This chart is invaluable for long-term process improvement. It clearly shows that `Reagent Lot Age` and issues related to a specific instrument (`Instrument ID_HML-02`) are the most significant systemic causes of failures in our historical data. This provides a data-driven justification to focus preventive efforts, such as tightening controls on reagent expiry or scheduling a deep-dive investigation and potential overhaul of the HML-02 liquid handler.
            - **Predicted Root Cause**: For a new failure, the model provides an immediate, data-driven hypothesis. Instead of starting a broad, unfocused investigation, the team can immediately prioritize checking the age and storage conditions of the reagent lot used in the failed run.

            **Actionable Insight:** This tool should be used as the **first step** in a formal non-conformance investigation. The model's prediction, along with the confidence scores, drastically narrows the search space, focusing the investigation on the most probable causes and significantly reducing the time to resolution. This allows us to get production back online faster and implement more effective corrective actions.
            """)
    except Exception as e:
        st.error(f"Error in root cause analysis module: {e}")
