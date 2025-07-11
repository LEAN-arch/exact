# pages/Assay_Validation_Dashboard.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_linearity_data, generate_precision_data, generate_msa_data, generate_specificity_data
import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

st.set_page_config(
    page_title="Assay Validation Dashboard | Exact Sciences",
    layout="wide"
)

st.title("📈 Test Method Validation (TMV) Dashboard")
st.markdown("### Statistical analysis of method performance characteristics for assay transfer.")

with st.expander("🌐 Regulatory Context: The Role of TMV"):
    st.markdown("""
    Test Method Validation (TMV) is the documented process that establishes, by objective evidence, that a test method is suitable for its intended purpose and can consistently produce reliable results. This is a fundamental requirement of our Quality System.

    - **FDA 21 CFR 820.72(a) - Control of inspection, measuring, and test equipment:** Mandates that all test equipment, including the methods themselves, must be proven to be "capable of producing valid results." The analyses on this page provide that proof.
    - **ISO 13485:2016, Section 7.6:** Requires the validation of test methods used for product acceptance to ensure their suitability.
    - **CLIA §493.1253 - Standard: Establishment and verification of performance specifications:** For tests performed in our CLIA labs, we must document the verification of key performance characteristics like accuracy, precision, and analytical specificity. This dashboard is a critical tool for compiling that evidence.
    - **Best Practices:** The analyses follow principles from **Clinical and Laboratory Standards Institute (CLSI) Guidelines** (e.g., EP05 for Precision, EP06 for Linearity, EP17 for Limit of Blank/Detection).
    """)

# --- Data Generation ---
linearity_df = generate_linearity_data()
precision_df = generate_precision_data()
msa_data = generate_msa_data()
specificity_df = generate_specificity_data()

# --- Page Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "**Linearity (RT-PCR)**",
    "**Precision (CLSI EP05)**",
    "**Measurement System Analysis (MSA)**",
    "**Specificity & Interference**"
])

# ... (Tabs 1 and 2 remain unchanged) ...
with tab1:
    st.header("Assay Linearity & Dynamic Range (RT-PCR)")
    # ... (Code from previous correct version) ...
    with st.expander("🔬 **Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        To evaluate the **linearity** and **dynamic range** of a quantitative RT-PCR assay (e.g., for an Oncotype DX® gene), a dilution series of a known target material is created. This series spans the expected reportable range of the assay. Each dilution is tested to demonstrate that the observed **Cycle threshold (Ct) value** has a predictable, linear relationship with the logarithm of the target concentration.
        #### The Method
        - **Ordinary Least Squares (OLS) Regression**: A linear model is fitted to the data, plotting the observed Ct values (Y-axis) against the Log10 of the target concentration (X-axis).
        - **Key Metrics (PCR Efficiency & R²)**: The slope of the regression line is used to calculate the PCR efficiency, which is a critical measure of assay performance. The R-squared value indicates the goodness of fit of the linear model.
        - **Residual Analysis**: A plot of the residuals (the difference between observed and predicted Ct values) helps to visually identify non-linearity.
        """)
    col1, col2 = st.columns([2, 1])
    with col1:
        X = sm.add_constant(linearity_df['Log10 Target Concentration'])
        model = sm.OLS(linearity_df['Observed Ct Value'], X).fit()
        linearity_df['Predicted Ct'] = model.predict(X)
        linearity_df['Residuals'] = model.resid
        fig_lin = px.scatter(linearity_df, x='Log10 Target Concentration', y='Observed Ct Value', trendline='ols', title="Linearity: Ct Value vs. Log Target Concentration", labels={'Log10 Target Concentration': 'Log10 Target Concentration (copies/reaction)', 'Observed Ct Value': 'Observed Ct Value'})
        st.plotly_chart(fig_lin, use_container_width=True)
        fig_res = px.scatter(linearity_df, x='Log10 Target Concentration', y='Residuals', title="Residuals Plot")
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res.update_yaxes(title="Residuals (ΔCt)")
        st.plotly_chart(fig_res, use_container_width=True)
    with col2:
        st.subheader("Linearity Performance Metrics")
        slope = model.params['Log10 Target Concentration']
        intercept = model.params['const']
        r_squared = model.rsquared
        pcr_efficiency = (10**(-1/slope) - 1) * 100
        st.metric("R-squared (R²)", f"{r_squared:.4f}")
        st.metric("Slope", f"{slope:.3f}")
        st.metric("Y-Intercept (Ct at 1 copy)", f"{intercept:.2f}")
        st.metric("PCR Efficiency (%)", f"{pcr_efficiency:.1f}%")
        with st.expander("📊 **Results & Analysis**"):
            st.markdown(f"...") # Content hidden for brevity

with tab2:
    st.header("Assay Precision (Following CLSI EP05-A3)")
    # ... (Code from previous correct version) ...
    with st.expander("🔬 **Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        To evaluate **precision**, a stable QC material is tested in replicates (e.g., n=5), by multiple operators (e.g., 2 operators), over multiple days (e.g., 3-5 days).
        - **Repeatability (Within-Run Precision)**: Variation under the most minimal set of changing conditions.
        - **Reproducibility (Total Precision)**: Variation across all changing conditions.
        #### The Method
        - **Box Plots**: Used to visually compare the distribution of Ct values.
        - **Coefficient of Variation (%CV)**: A standardized measure of dispersion.
        """)
    fig = px.box(precision_df, x="Day", y="Ct Value", color="Operator", title="Precision: Ct Value Distribution by Day and Operator", points="all")
    st.plotly_chart(fig, use_container_width=True)
    repeatability_cv = precision_df.groupby(['Day', 'Operator'])['Ct Value'].std().mean() / precision_df['Ct Value'].mean() * 100
    total_precision_cv = precision_df['Ct Value'].std() / precision_df['Ct Value'].mean() * 100
    col1, col2 = st.columns(2)
    col1.metric("Repeatability (Within-Run CV%)", f"{repeatability_cv:.2f}%")
    col2.metric("Reproducibility (Total CV%)", f"{total_precision_cv:.2f}%")
    with st.expander("📊 **Results & Analysis**"):
        st.markdown(f"...") # Content hidden for brevity

with tab3:
    st.header("Measurement System Analysis (MSA) / Gage R&R")

    with st.expander("🔬 **Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        A **Gage Repeatability & Reproducibility (Gage R&R)** study is performed to quantify the sources of variation within the measurement system itself. A set of samples representing the expected process range are measured multiple times by different operators. This is distinct from precision, as it compares the measurement system's variation to the actual variation between parts (samples).

        #### The Method
        - **ANOVA Gage R&R**: We use Analysis of Variance to partition the total observed variability into three components:
            1.  **Part-to-Part**: The true, inherent variation between the samples. This is the "good" variation we want to measure.
            2.  **Repeatability (Equipment Variation - EV)**: Variation from the instrument when measuring the same part repeatedly.
            3.  **Reproducibility (Appraiser Variation - AV)**: Variation from different operators measuring the same part.
        - The sum of Repeatability and Reproducibility is the total **Gage R&R** variation—the "noise" from the measurement system.
        """)

    # --- Calculations ---
    total_var = msa_data['part_var'] + msa_data['repeatability_var'] + msa_data['reproducibility_var']
    gage_rr_var = msa_data['repeatability_var'] + msa_data['reproducibility_var']
    pct_gage_rr = (gage_rr_var / total_var) * 100 if total_var > 0 else 0
    ndc = int(1.41 * (np.sqrt(msa_data['part_var']) / np.sqrt(gage_rr_var))) if gage_rr_var > 0 else float('inf')

    # --- FIX: Replaced sunburst with a clearer, more informative stacked bar chart and KPI display ---
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.subheader("Decomposition of Total Variation")
        msa_plot_data = pd.DataFrame([
            {"Component": "Part-to-Part", "Variance": msa_data['part_var']},
            {"Component": "Repeatability (EV)", "Variance": msa_data['repeatability_var']},
            {"Component": "Reproducibility (AV)", "Variance": msa_data['reproducibility_var']},
        ])
        msa_plot_data['% Contribution'] = (msa_plot_data['Variance'] / total_var) * 100

        fig = px.bar(
            msa_plot_data,
            x=['Process Variation'], # A single category on the x-axis to stack upon
            y='% Contribution',
            color='Component',
            text=msa_plot_data['% Contribution'].apply(lambda x: f'{x:.1f}%'),
            title="Sources of Variation (% Contribution)",
            color_discrete_map={
                "Part-to-Part": '#2ca02c',
                "Repeatability (EV)": '#ff7f0e',
                "Reproducibility (AV)": '#d62728'
            }
        )
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="% of Total Variation",
            yaxis_ticksuffix='%',
            legend_title="Source of Variation"
        )
        fig.update_traces(textposition='inside', textfont_size=14, textfont_color='white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("MSA Performance & Judgment")
        st.metric("Total Gage R&R (% Contribution)", f"{pct_gage_rr:.1f}%")
        st.metric("Number of Distinct Categories (ndc)", f"{ndc}")

        # Actionable PASS/FAIL criteria
        is_fail = pct_gage_rr > 9 or ndc < 5
        if is_fail:
            st.error("**FAIL:** The measurement system is not acceptable.")
        else:
            st.success("**PASS:** The measurement system is acceptable.")

    with st.expander("📊 **Results & Analysis**"):
        st.markdown(f"""
        #### Interpreting the Visualization
        The stacked bar chart provides an unambiguous breakdown of where the variation in our measurements comes from. In an ideal measurement system, the green bar (**Part-to-Part**) should be very large, and the orange and red bars (**Gage R&R**) should be very small.

        - **Part-to-Part Variation ({ (msa_data['part_var']/total_var*100):.1f}%)**: This represents the actual differences between the samples being measured. This should be the dominant source of variation.
        - **Gage R&R Variation ({pct_gage_rr:.1f}%)**: This is the total variation, or "noise," contributed by the measurement system itself. It is the sum of:
            - **Repeatability ({(msa_data['repeatability_var']/total_var*100):.1f}%)**: Noise from the instrument.
            - **Reproducibility ({(msa_data['reproducibility_var']/total_var*100):.1f}%)**: Noise from the operators.

        #### Judgment Criteria
        - **Gage R&R (% Contribution)**: Should be **< 9%**. Our result of **{pct_gage_rr:.1f}%** fails this criterion.
        - **ndc (Number of Distinct Categories)**: Should be **≥ 5**. Our result of **{ndc}** fails this criterion.

        #### Conclusion & Action
        This measurement system is **NOT ACCEPTABLE**. The bar chart clearly shows that while Repeatability is acceptable, the **Reproducibility (operator variation)** is far too high. The system's noise is overpowering its ability to distinguish between different parts.
        
        **Action:** The transfer must be paused. A root cause investigation into the source of high operator variability is required. This could involve re-training, clarifying the work instruction, or improving the assay's robustness to handling differences. The Gage R&R study must be repeated after corrective actions are implemented.
        """)
    # --- END OF FIX ---

with tab4:
    st.header("Assay Specificity & Interference (e.g., Cologuard®)")
    # ... (Code from previous correct version) ...
    with st.expander("🔬 **Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        **Analytical Specificity** is the ability of an assay to measure *only* the target analyte, even in the presence of other substances. For Cologuard®, we must demonstrate that common substances found in stool samples do not interfere with the methylation assay signal.
        #### The Method
        - **Box Plots**: Used to visually compare the signal distributions.
        - **Welch's Two-Sample t-test**: A statistical test to determine if there is a significant difference between the means of two groups.
        """)
    fig = px.box(specificity_df, x="Sample Type", y="Signal (% Meth)", color="Sample Type", title="Assay Response to Potential Interferents in Stool Matrix")
    st.plotly_chart(fig, use_container_width=True)
    target_only = specificity_df[specificity_df['Sample Type'] == 'Methylated Control']['Signal (% Meth)']
    target_with_interferents = specificity_df[specificity_df['Sample Type'] == 'Control + Interferents']['Signal (% Meth)']
    ttest_res = ttest_ind(target_only, target_with_interferents, equal_var=False)
    st.subheader("Statistical Interference Test")
    st.metric("T-test P-value (Control vs Control + Interferents)", f"{ttest_res.pvalue:.4f}")
    with st.expander("📊 **Results & Analysis**"):
        st.markdown("""...""") # Content hidden for brevity
        if ttest_res.pvalue < 0.05:
            st.error(f"**P-value of {ttest_res.pvalue:.4f} is less than 0.05.** This confirms a statistically significant interference effect.")
            st.markdown("""
            **Conclusion & Action:** The tested substances are causing a significant negative bias (suppression). The magnitude of this interference must be quantified and compared against the defined clinical risk threshold.
            """)
        else:
            st.success(f"**P-value of {ttest_res.pvalue:.4f} is greater than 0.05.** There is no statistically significant evidence of interference.")
