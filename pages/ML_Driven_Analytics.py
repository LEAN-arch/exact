# pages/ML_Driven_Analytics.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import (train_instrument_model, generate_instrument_health_data,
                   train_rca_model, generate_rca_data)

st.set_page_config(
    page_title="ML-Driven Analytics | Exact Sciences",
    layout="wide"
)

st.title("🔬 ML-Driven Analytics & Predictive QC")
st.markdown("### Investigational tools for proactive monitoring and root cause analysis of the NGS workflow.")

with st.expander("🌐 Regulatory Context & The Future of Process Monitoring"):
    st.markdown("""
    This page showcases advanced, investigational analytics that align with modern regulatory expectations for **Continual Process Verification (CPV)**. While traditional SPC charts (as seen on the 'QC Performance' page) are essential for demonstrating a state of control, these machine learning (ML) tools represent the next frontier in process understanding.

    - **Advanced Statistical Techniques (21 CFR 820.250)**: ML models are a powerful extension of the statistical techniques required by the QSR. They can analyze complex, non-linear interactions between multiple process parameters that simpler methods cannot.
    - **Risk Management & Pro-activity (ISO 14971)**: The primary benefit of these tools is a shift from reactive to **proactive risk management**. The predictive instrument failure model, for example, aims to identify the risk of an out-of-spec (OOS) event *before* it occurs, allowing for preventative maintenance and saving costly NGS runs.
    - **CAPA & Root Cause Analysis (21 CFR 820.100)**: The RCA model provides a data-driven approach to OOS investigations. By identifying the most probable causes of failure from historical data, it helps focus and accelerate the CAPA process, leading to more effective corrective actions.
    - **Validation of Automated Processes (21 CFR 820.70(i))**: The validation of these ML models themselves is a critical activity. The "Experiment & Methodology" sections on this page allude to the data splitting, training, and testing processes that would be formally documented in a software validation plan for these tools.
    """)


# --- Data Generation ---
# These functions from utils.py generate data for these advanced use cases
instrument_df = generate_instrument_health_data()
instrument_model, X_inst = train_instrument_model(instrument_df)

rca_df = generate_rca_data()
rca_model, rca_encoded_columns, y_rca = train_rca_model(rca_df)


# --- Page Tabs based on NGS QC Tiers ---
tab1, tab2, tab3 = st.tabs([
    "**QC Tier 1: Predictive Instrument Health**",
    "**QC Tier 2/3: Root Cause Analysis (RCA)**",
    "**The NGS QC Funnel (Tiers 0-3)**"
])


with tab1:
    st.header("QC Tier 1: Predictive Instrument Health (NGS Sequencer)")
    st.caption("Using instrument telemetry to predict run failures before they happen.")

    with st.expander("🔬 The Experiment & Methodology"):
        st.markdown("""
        #### Purpose: Validate Sequencing Run Integrity (Proactively)
        This tool analyzes continuous sensor data from our NGS sequencers (e.g., laser power, pump pressure, temperature) to predict the likelihood of an imminent run failure. This moves beyond simple pass/fail on metrics like **%Q30** or **Cluster Density** and into a predictive maintenance paradigm.

        #### The Experiment
        - **Data**: A historical dataset of instrument telemetry from hundreds of previous sequencing runs, with each run labeled as 'Success' or 'Failure'.
        - **Model**: A **LightGBM Classifier**, a powerful gradient-boosting framework, is trained on this data. It learns the subtle patterns and drifts in sensor readings that precede a failure.

        #### The Method & Significance
        The model continuously analyzes real-time data from active instruments.
        - **Visualization**: The line charts below show the telemetry leading up to a simulated failure event. The model's feature importance chart tells us *which* sensors were most predictive.
        - **Significance**: By flagging an instrument for maintenance *before* it fails, we can prevent the loss of valuable reagents, samples, and weeks of work. This is a direct application of data science to mitigate operational risk and increase lab efficiency.
        """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Live Instrument Telemetry (Simulated)")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('Laser A Power', 'Flow Cell Temp (°C)', 'Pump Pressure (psi)'))
        fig.add_trace(go.Scatter(x=instrument_df['Run_ID'], y=instrument_df['Laser_A_Power'], name='Laser Power'), row=1, col=1)
        fig.add_trace(go.Scatter(x=instrument_df['Run_ID'], y=instrument_df['Flow_Cell_Temp_C'], name='Flow Cell Temp'), row=2, col=1)
        fig.add_trace(go.Scatter(x=instrument_df['Run_ID'], y=instrument_df['Pump_Pressure_psi'], name='Pump Pressure'), row=3, col=1)
        fig.add_vrect(x0=92, x1=100, fillcolor="red", opacity=0.15, line_width=0, annotation_text="Failure Zone", annotation_position="top left", row='all', col=1)
        fig.update_layout(height=500, showlegend=False, title_text="Sensor Drift Leading to Failure")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Failure Prediction Model")
        # Get prediction for the last data point
        last_data_point = instrument_df.drop(['Failure', 'Run_ID'], axis=1).tail(1)
        pred_proba = instrument_model.predict_proba(last_data_point)[0][1]
        st.metric("Predicted Failure Probability (Last Run)", f"{pred_proba:.1%}",
                  delta="High Risk - Maintenance Recommended", delta_color="inverse")

        st.subheader("Predictive Feature Importance")
        feature_imp = pd.DataFrame(sorted(zip(instrument_model.feature_importances_, X_inst.columns)), columns=['Value','Feature'])
        fig_imp = px.bar(feature_imp, x="Value", y="Feature", orientation='h', title="Top Predictors of Instrument Failure")
        st.plotly_chart(fig_imp, use_container_width=True)

with tab2:
    st.header("QC Tier 2 & 3: AI-Powered Root Cause Analysis (RCA)")
    st.caption("Classifying the cause of failed Cologuard® assay runs to accelerate CAPA investigations.")

    with st.expander("🔬 The Experiment & Methodology"):
        st.markdown("""
        #### Purpose: Identify Root Cause for Sample/Variant-Level Failures
        When a sample fails QC (e.g., low **mapping rate**, poor **coverage uniformity**) or a batch fails controls (e.g., a known variant is not detected), an investigation is required. This tool uses ML to suggest the most likely root cause based on the run's context.

        #### The Experiment
        - **Data**: A curated dataset of past failed runs, where each failure has been assigned a root cause by an expert (e.g., 'Reagent Degradation', 'Operator Error', 'Instrument Malfunction').
        - **Model**: A **RandomForest Classifier** is trained to associate input parameters (Reagent Lot Age, Operator ID, Instrument ID) with a final root cause category.

        #### The Method & Significance
        - **Interactive RCA**: The user inputs the parameters of a newly failed run. The trained model predicts the probability of each potential root cause.
        - **Pareto Analysis**: The chart below shows the historical distribution of root causes, identifying systemic problems.
        - **Significance**: This tool dramatically accelerates OOS/CAPA investigations. Instead of a wide-ranging, trial-and-error approach, the investigation can immediately focus on the most probable cause, saving time and resources. It also helps identify systemic issues, such as a specific instrument being associated with a high rate of failures.
        """)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Interactive RCA Predictor")
        op_id = st.selectbox("Operator ID", rca_df['Operator ID'].unique())
        inst_id = st.selectbox("Instrument ID", rca_df['Instrument ID'].unique())
        reagent_age = st.slider("Reagent Lot Age (days)", 1, 150, 95)
        if st.button("Predict Root Cause"):
            # CORRECTED: The prediction input must be encoded in the exact same way as the training data.
            input_data = {
                'Reagent Lot Age (days)': [reagent_age],
                'Operator ID': [op_id],
                'Instrument ID': [inst_id]
            }
            input_df = pd.DataFrame(input_data)

            # One-hot encode the input using the same logic as training
            input_encoded = pd.get_dummies(input_df, columns=['Operator ID', 'Instrument ID'], prefix=['Operator ID', 'Instrument ID'])

            # Align columns with the training data: add missing columns and fill with 0
            input_aligned = input_encoded.reindex(columns=rca_encoded_columns, fill_value=0)

            # Now predict with the correctly formatted data
            prediction = rca_model.predict(input_aligned)[0]
            pred_proba = rca_model.predict_proba(input_aligned)

            st.success(f"**Most Likely Cause:** {prediction}")

            # Display probabilities
            proba_df = pd.DataFrame(pred_proba, columns=rca_model.classes_).T.reset_index()
            proba_df.columns = ['Cause', 'Probability']
            proba_df = proba_df.sort_values('Probability', ascending=False)
            st.dataframe(proba_df, hide_index=True)


    with col2:
        st.subheader("Historical Root Cause Distribution (Pareto)")
        rca_counts = rca_df['Root Cause'].value_counts().reset_index()
        rca_counts.columns = ['Root Cause', 'Count']
        fig = px.bar(rca_counts, x='Root Cause', y='Count', color='Root Cause', title="Pareto of Historical Failure Causes")
        fig.update_layout(xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("The NGS QC Funnel (Tiers 0-3 Overview)")
    st.caption("A holistic view of the entire NGS workflow, from sample receipt to final result.")
    
    with st.expander("🔬 The Framework & Methodology"):
        st.markdown("""
        #### Purpose: Holistic Process Yield & Bottleneck Analysis
        This visualization provides a "bird's-eye view" of the entire NGS QC process, tracking a cohort of samples as they pass through each quality gate. This helps identify which stage of the process is responsible for the most sample loss, highlighting major bottlenecks.

        #### The Tiers
        - **Tier 0: Pre-Analytical QC (Wet Lab)**: Ensures input quality. Metrics: DNA quantity, RIN score.
        - **Tier 1: Run-Level QC (Instrument & Raw Data)**: Validates sequencing run integrity. Metrics: %Q30, cluster density.
        - **Tier 2: Sample-Level QC (Mapping & Coverage)**: Confirms sample data meets quality thresholds. Metrics: Mapping rate, coverage.
        - **Tier 3: Variant-Level QC (Result Integrity)**: Validates reportable results. Metrics: VAF, control concordance.
        
        #### Significance
        By visualizing the process as a funnel, we can instantly see where our process is least efficient. For example, a large drop-off between Tier 1 and Tier 2 might indicate a systemic problem with our library preparation protocol, leading to poor mapping rates, rather than an issue with the sequencer itself. This allows leadership to focus process improvement efforts where they will have the greatest impact.
        """)

    # Mock data for the funnel chart
    funnel_data = {
        'Number': [100, 95, 85, 81, 78],
        'Stage': [
            'Tier 0: Samples Received',
            'Tier 0: Pass Pre-Analytical QC',
            'Tier 1: Pass Run-Level QC',
            'Tier 2: Pass Sample-Level QC',
            'Tier 3: Pass Variant QC (Reportable)'
        ]
    }
    df_funnel = pd.DataFrame(funnel_data)

    fig = go.Figure(go.Funnel(
        y = df_funnel['Stage'],
        x = df_funnel['Number'],
        textposition = "inside",
        textinfo = "value+percent initial",
        opacity = 0.8, marker = {"color": ["#1A3A6D", "#0072B2", "#00B0F0", "#90D8F4", "#2ca02c"],
        "line": {"width": [4, 3, 2, 1, 0], "color": ["white"]}},
        connector = {"line": {"color": "royalblue", "dash": "dot", "width": 3}}
    ))

    fig.update_layout(
        title="NGS QC Funnel: Batch Yield by Tier",
        margin=dict(l=50, r=50, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)
