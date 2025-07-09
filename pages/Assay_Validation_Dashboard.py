# pages/Assay_Validation_Dashboard.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_linearity_data, generate_precision_data, generate_msa_data
import statsmodels.api as sm
import numpy as np
import pandas as pd

st.set_page_config(page_title="Assay Validation Dashboard", layout="wide")

st.title("📈 Assay & Method Validation Dashboard")
st.markdown("### In-depth statistical analysis of method performance characteristics.")

with st.expander("🌐 Regulatory Context & Legend"):
    st.markdown("""
    Test Method Validation (TMV) is a critical activity to ensure that analytical methods are suitable for their intended purpose. The data presented on this page provides the objective evidence required for compliance.
    - **Overall Mandate**: **21 CFR 820.72(a)** requires that all test equipment is "capable of producing valid results."
    - **Software Validation**: For methods with software, this contributes to **21 CFR 820.70(i)**.
    - **Best Practices**: The analyses follow principles from **CLSI Guidelines** (e.g., EP05, EP06, EP17).
    """)

linearity_df = generate_linearity_data()
precision_df = generate_precision_data()
msa_data = generate_msa_data()

tab1, tab2, tab3 = st.tabs(["**Linearity & Accuracy**", "**Measurement System Analysis (MSA)**", "**Precision (Repeatability & Reproducibility)**"])

with tab1:
    st.header("Assay Linearity and Bias Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_lin = px.scatter(linearity_df, x='Expected Concentration', y='Observed Signal', trendline='ols', title="Assay Linearity with Residuals Plot")
        st.plotly_chart(fig_lin, use_container_width=True)
        model = sm.OLS(linearity_df['Observed Signal'], sm.add_constant(linearity_df['Expected Concentration'])).fit()
        linearity_df['Residuals'] = model.resid
        fig_res = px.scatter(linearity_df, x='Expected Concentration', y='Residuals', title="Residuals Plot")
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)
    with col2:
        st.subheader("Linearity Statistics")
        st.metric("R-squared (R²)", f"{model.rsquared:.4f}")
        st.metric("Slope", f"{model.params[1]:.3f}")
        st.metric("Y-Intercept", f"{model.params[0]:.3f}")

with tab2:
    st.header("Measurement System Analysis (Gage R&R)")
    total_var = msa_data['part_var'] + msa_data['repeatability_var'] + msa_data['reproducibility_var']
    gage_rr_var = msa_data['repeatability_var'] + msa_data['reproducibility_var']
    pct_gage_rr = (gage_rr_var / total_var) * 100
    ndc = int(1.41 * (np.sqrt(msa_data['part_var']) / np.sqrt(gage_rr_var)))
    
    col1, col2 = st.columns(2)
    with col1:
        # --- THIS IS THE CORRECTED SECTION ---
        # We restructure the data to use the more robust 'path' argument for the sunburst chart.
        # Each row represents a leaf node, and columns define the hierarchy.
        data_for_sunburst = {
            'level_1': ['Total Variation', 'Total Variation'],
            'level_2': ['Part-to-Part', 'Gage R&R'],
            'level_3': [None, 'Repeatability'],
            'level_4': [None, 'Reproducibility'],
            'values': [msa_data['part_var'], msa_data['repeatability_var']] # Add a placeholder for reproducibility
        }
        
        # A small hack is needed to show both repeatability and reproducibility under Gage R&R
        # We create a dataframe with two rows for the Gage R&R path.
        df_sunburst = pd.DataFrame({
            'Path': [
                ['Total Variation', 'Part-to-Part'],
                ['Total Variation', 'Gage R&R', 'Repeatability'],
                ['Total Variation', 'Gage R&R', 'Reproducibility']
            ],
            'Values': [
                msa_data['part_var'],
                msa_data['repeatability_var'],
                msa_data['reproducibility_var']
            ]
        })

        fig = px.sunburst(
            df_sunburst,
            path='Path',
            values='Values',
            title="Hierarchical Sources of Variation"
        )
        fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Key MSA Metrics")
        st.metric("Gage R&R (% Study Var)", f"{pct_gage_rr:.2f}%")
        st.metric("Number of Distinct Categories (ndc)", f"{ndc}")

with tab3:
    st.header("CLSI EP05-A3 Style Precision Analysis")
    fig = px.box(precision_df, x="Day", y="Value", color="Operator", title="Precision: Repeatability and Between-Operator/Day Variability", points="all")
    st.plotly_chart(fig, use_container_width=True)
    repeatability_cv = precision_df.groupby(['Day', 'Operator'])['Value'].std().mean() / precision_df['Value'].mean() * 100
    total_precision_cv = precision_df['Value'].std() / precision_df['Value'].mean() * 100
    col1, col2 = st.columns(2)
    col1.metric("Within-Run Precision (Repeatability)", f"{repeatability_cv:.2f}%")
    col2.metric("Total Precision (Reproducibility)", f"{total_precision_cv:.2f}%")
