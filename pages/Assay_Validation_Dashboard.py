# pages/Assay_Validation_Dashboard.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_linearity_data, generate_precision_data, generate_msa_data, generate_specificity_data
from scipy.stats import ttest_ind
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

# --- Data Generation ---
linearity_df = generate_linearity_data()
precision_df = generate_precision_data()
msa_data = generate_msa_data()

# --- Page Tabs ---
tab1, tab2, tab3 = st.tabs(["**Linearity & Accuracy**", "**Measurement System Analysis (MSA)**", "**Precision (Repeatability & Reproducibility)**", "**Specificity & Interference**"])

with tab1:
    st.header("Assay Linearity and Bias Analysis")
    
    with st.expander("🔬 **Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        To evaluate **linearity**, a series of samples with known concentrations of the target analyte, spanning the expected reportable range of the assay, are prepared and tested. This experiment aims to demonstrate that the measured signal is directly proportional to the analyte concentration. **Accuracy** is assessed by how close the measured values are to the true, expected values.
        
        #### The Method
        - **Ordinary Least Squares (OLS) Regression**: A linear model is fitted to the data to describe the relationship between the Expected Concentration (X) and the Observed Signal (Y).
        - **Residual Analysis**: The residuals (the differences between the observed values and the values predicted by the model) are plotted against the expected concentrations. This helps to visually assess the appropriateness of the linear model.
        """)
        
    col1, col2 = st.columns([2, 1])
    with col1:
        model = sm.OLS(linearity_df['Observed Signal'], sm.add_constant(linearity_df['Expected Concentration'])).fit()
        linearity_df['Residuals'] = model.resid
        
        fig_lin = px.scatter(linearity_df, x='Expected Concentration', y='Observed Signal', trendline='ols',
                             title="Linearity: Observed Signal vs. Expected Concentration",
                             labels={'Expected Concentration': 'Expected Concentration (units/mL)', 'Observed Signal': 'Instrument Signal (AU)'})
        st.plotly_chart(fig_lin, use_container_width=True)
        
        fig_res = px.scatter(linearity_df, x='Expected Concentration', y='Residuals', title="Residuals Plot")
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)
        
    with col2:
        st.subheader("Linearity Metrics")
        r_squared = model.rsquared
        slope = model.params.iloc[1]
        intercept = model.params.iloc[0]
        
        st.metric("R-squared (R²)", f"{r_squared:.4f}")
        st.metric("Slope", f"{slope:.3f}")
        st.metric("Y-Intercept", f"{intercept:.3f}")

        with st.expander("📊 **Results & Analysis**"):
            st.markdown(f"""
            #### Metrics Explained
            - **R-squared (R²)**: The coefficient of determination. It represents the proportion of the variance in the observed signal that is predictable from the analyte concentration. A value of 1.0 indicates a perfect linear fit.
            - **Slope**: Describes the change in signal per unit of concentration. An ideal slope is 1.0, indicating a one-to-one response.
            - **Y-Intercept**: The predicted signal when the concentration is zero. An ideal intercept is 0, indicating no signal in the absence of the analyte.

            #### Analysis of Results
            - The **R² value of {r_squared:.4f}** is extremely high, suggesting a strong linear relationship and excellent model fit. *Acceptance Criterion: Typically > 0.995*.
            - The **Slope of {slope:.3f}** indicates a slight proportional bias (the signal changes slightly less than expected per unit).
            - The **Y-Intercept of {intercept:.3f}** suggests a minimal constant bias or background signal.
            - The **Residuals Plot** shows a subtle but clear pattern: the residuals are slightly positive at low and high concentrations and negative in the middle. This "frown" shape indicates a minor non-linear effect (e.g., saturation at high concentrations), which is confirmed by the simulated data's non-linear factor. While the R² is high, this pattern warrants investigation to ensure it doesn't impact clinical accuracy at the extremes of the range.
            """)

with tab2:
    st.header("Measurement System Analysis (Gage R&R)")

    with st.expander("🔬 **Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        A **Gage Repeatability & Reproducibility (Gage R&R)** study is performed to understand the sources of variation within the measurement system itself. A set of representative parts (samples) are measured multiple times by different operators over different days. This allows for the partitioning of the total observed variation into its constituent components.

        #### The Method
        - **Analysis of Variance (ANOVA)**: A statistical method used to partition the total observed variability into components attributable to the parts, the operators (reproducibility), and the measurement instrument itself (repeatability).
        - **Calculation of Key Metrics**: From the ANOVA results, key metrics like % Contribution, % Study Var, and the Number of Distinct Categories (ndc) are calculated.
        """)

    total_var = msa_data['part_var'] + msa_data['repeatability_var'] + msa_data['reproducibility_var']
    gage_rr_var = msa_data['repeatability_var'] + msa_data['reproducibility_var']
    pct_gage_rr = (gage_rr_var / total_var) * 100
    ndc = int(1.41 * (np.sqrt(msa_data['part_var']) / np.sqrt(gage_rr_var)))
    
    col1, col2 = st.columns(2)
    with col1:
        df_sunburst = pd.DataFrame([
            dict(level1="Total Variation", level2="Part-to-Part", values=msa_data['part_var']),
            dict(level1="Total Variation", level2="Gage R&R", level3="Repeatability", values=msa_data['repeatability_var']),
            dict(level1="Total Variation", level2="Gage R&R", level3="Reproducibility", values=msa_data['reproducibility_var']),
        ])
        fig = px.sunburst(df_sunburst, path=['level1', 'level2', 'level3'], values='values', title="Hierarchical Sources of Variation")
        fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Key MSA Metrics")
        st.metric("Gage R&R (% Study Var)", f"{pct_gage_rr:.2f}%")
        st.metric("Number of Distinct Categories (ndc)", f"{ndc}")

        with st.expander("📊 **Results & Analysis**"):
            st.markdown(f"""
            #### Metrics Explained
            - **Part-to-Part Variation**: The real variation between the different samples being measured. This should be the largest source of variation.
            - **Repeatability**: Variation from the measurement instrument itself when measuring the same part repeatedly under the same conditions.
            - **Reproducibility**: Variation from different operators or conditions (e.g., days) when measuring the same part.
            - **Gage R&R (% Study Var)**: The percentage of total variation that is due to the measurement system (repeatability + reproducibility). A low value is desirable.
            - **ndc**: An indicator of the measurement system's resolution. It represents how many distinct groups of parts the system can reliably distinguish. A higher number is better.
            
            #### Analysis of Results
            - The **sunburst plot** clearly shows that Part-to-Part variation is the dominant source, which is ideal. The measurement system's contribution (Gage R&R) is small.
            - The **Gage R&R % of {pct_gage_rr:.2f}%** is well below the common acceptance threshold of 10%, indicating an **excellent** measurement system with very little noise.
            - The **ndc of {ndc}** is significantly greater than the minimum requirement of 5, confirming that the assay can robustly differentiate between samples with different analyte levels. The measurement system is deemed suitable for its intended purpose.
            """)

with tab3:
    st.header("CLSI EP05-A3 Style Precision Analysis")

    with st.expander("🔬 **Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        To evaluate **precision**, a small number of samples (often QC materials) are tested in replicates, by multiple operators, over multiple days. This nested design allows for the quantification of different components of random error.
        - **Repeatability (Within-Run Precision)**: The variation observed when the same sample is measured multiple times by the same operator on the same instrument run.
        - **Reproducibility (Total Precision)**: The variation observed when the same sample is measured across different runs, different days, and by different operators. It encompasses all potential sources of random error in the system.

        #### The Method
        - **Box Plots**: Used to visually compare the distribution of results across the different experimental conditions (days and operators).
        - **Coefficient of Variation (%CV)**: A standardized measure of dispersion (standard deviation divided by the mean). It is used to quantify and compare precision across different conditions and is the primary metric for acceptance.
        """)
        
    fig = px.box(precision_df, x="Day", y="Value", color="Operator", title="Precision: Distribution of Results by Day and Operator", points="all")
    st.plotly_chart(fig, use_container_width=True)
    
    repeatability_cv = precision_df.groupby(['Day', 'Operator'])['Value'].std().mean() / precision_df['Value'].mean() * 100
    total_precision_cv = precision_df['Value'].std() / precision_df['Value'].mean() * 100
    
    col1, col2 = st.columns(2)
    col1.metric("Within-Run Precision (%CV)", f"{repeatability_cv:.2f}%")
    col2.metric("Total Precision (%CV)", f"{total_precision_cv:.2f}%")
    with st.expander("📊 **Results & Analysis**"):
        st.markdown(f"""
    with st.expander("📊 **Results & Analysis**"):
            st.markdown(f"""
            #### Metrics Explained
            - **Within-Run Precision (%CV)**: Quantifies the tightest possible precision of the assay under ideal conditions.
            - **Total Precision (%CV)**: Quantifies the expected real-world precision of the assay, accounting for all sources of variability. This is the more critical metric for product claims.
    
            #### Analysis of Results
            - The **Box Plot** provides a visual assessment. We can see that Operator 2's results are consistently higher than Operator 1's, suggesting a minor systematic bias between operators. Additionally, the size of the boxes (interquartile range) appears slightly larger on Day 3, indicating potentially higher variability on that day.
            - The **Within-Run %CV of {repeatability_cv:.2f}%** and **Total %CV of {total_precision_cv:.2f}%** would be compared against the pre-defined acceptance criteria in the validation plan. For a typical quantitative assay, these values would likely be considered acceptable. The small difference between the two CVs indicates that the between-day and between-operator components of variance are minimal, despite the visual shift.
            """)""")
with tab4:
    st.header("Assay Specificity & Interference Analysis")
    with st.expander("🔬 **The Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        **Specificity** is the ability of an assay to measure the target analyte exclusively, without signal from other components. This experiment tests the assay's response to blank samples, samples containing only potentially interfering substances, the target analyte alone, and the target analyte "spiked" with the interferents.
        #### The Method
        - **Box Plots**: Used to visually compare the signal distributions of the different sample types.
        - **Two-Sample t-test**: A statistical test used to determine if there is a significant difference between the means of two independent groups. We compare the "Target Only" group to the "Target + Interferents" group.
        """)
    
    specificity_df = generate_specificity_data()
    fig = px.box(specificity_df, x="Sample Type", y="Signal", color="Sample Type", title="Assay Response to Potential Interferents")
    st.plotly_chart(fig, use_container_width=True)

    target_only = specificity_df[specificity_df['Sample Type'] == 'Target Only']['Signal']
    target_with_interferents = specificity_df[specificity_df['Sample Type'] == 'Target + Interferents']['Signal']
    ttest_res = ttest_ind(target_only, target_with_interferents)

    st.subheader("Statistical Interference Test")
    st.metric("T-test P-value (Target vs Target + Interferents)", f"{ttest_res.pvalue:.3f}")
    

    with st.expander("📊 **Results & Analysis**"):
        st.markdown(f"""
        #### Metrics Explained
        - **Within-Run Precision (%CV)**: Quantifies the tightest possible precision of the assay under ideal conditions.
        - **Total Precision (%CV)**: Quantifies the expected real-world precision of the assay, accounting for all sources of variability. This is the more critical metric for product claims.

        #### Analysis of Results
        - The **Box Plot** provides a visual assessment. We can see that Operator 2's results are consistently higher than Operator 1's, suggesting a minor systematic bias between operators. Additionally, the size of the boxes (interquartile range) appears slightly larger on Day 3, indicating potentially higher variability on that day.
        - The **Within-Run %CV of {repeatability_cv:.2f}%** and **Total %CV of {total_precision_cv:.2f}%** would be compared against the pre-defined acceptance criteria in the validation plan. For a typical quantitative assay, these values would likely be considered acceptable. The small difference between the two CVs indicates that the between-day and between-operator components of variance are minimal, despite the visual shift.
        """)
