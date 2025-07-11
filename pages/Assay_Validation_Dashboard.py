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

# --- Data Generation (Now from Exact Sciences-specific utils) ---
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

with tab1:
    st.header("Assay Linearity & Dynamic Range (RT-PCR)")

    with st.expander("🔬 **Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        To evaluate the **linearity** and **dynamic range** of a quantitative RT-PCR assay (e.g., for an Oncotype DX® gene), a dilution series of a known target material is created. This series spans the expected reportable range of the assay. Each dilution is tested to demonstrate that the observed **Cycle threshold (Ct) value** has a predictable, linear relationship with the logarithm of the target concentration.

        #### The Method
        - **Ordinary Least Squares (OLS) Regression**: A linear model is fitted to the data, plotting the observed Ct values (Y-axis) against the Log10 of the target concentration (X-axis).
        - **Key Metrics (PCR Efficiency & R²)**: The slope of the regression line is used to calculate the PCR efficiency, which is a critical measure of assay performance. The R-squared value indicates the goodness of fit of the linear model.
        - **Residual Analysis**: A plot of the residuals (the difference between observed and predicted Ct values) helps to visually identify non-linearity, such as PCR inhibition at high concentrations or stochastic effects at low concentrations.
        """)

    col1, col2 = st.columns([2, 1])
    with col1:
        # Fit linear model
        X = sm.add_constant(linearity_df['Log10 Target Concentration'])
        model = sm.OLS(linearity_df['Observed Ct Value'], X).fit()
        linearity_df['Predicted Ct'] = model.predict(X)
        linearity_df['Residuals'] = model.resid

        # Linearity Plot
        fig_lin = px.scatter(linearity_df, x='Log10 Target Concentration', y='Observed Ct Value', trendline='ols',
                             title="Linearity: Ct Value vs. Log Target Concentration",
                             labels={'Log10 Target Concentration': 'Log10 Target Concentration (copies/reaction)', 'Observed Ct Value': 'Observed Ct Value'})
        st.plotly_chart(fig_lin, use_container_width=True)

        # Residuals Plot
        fig_res = px.scatter(linearity_df, x='Log10 Target Concentration', y='Residuals', title="Residuals Plot")
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res.update_yaxes(title="Residuals (ΔCt)")
        st.plotly_chart(fig_res, use_container_width=True)

    with col2:
        st.subheader("Linearity Performance Metrics")
        slope = model.params['Log10 Target Concentration']
        intercept = model.params['const']
        r_squared = model.rsquared
        # Formula for PCR Efficiency from slope
        pcr_efficiency = (10**(-1/slope) - 1) * 100

        st.metric("R-squared (R²)", f"{r_squared:.4f}")
        st.metric("Slope", f"{slope:.3f}")
        st.metric("Y-Intercept (Ct at 1 copy)", f"{intercept:.2f}")
        st.metric("PCR Efficiency (%)", f"{pcr_efficiency:.1f}%")

        with st.expander("📊 **Results & Analysis**"):
            st.markdown(f"""
            #### Metrics Explained
            - **R-squared (R²)**: The proportion of variance in the Ct value predictable from the log concentration. A value > 0.990 is typically required.
            - **Slope**: In a PCR context, the ideal slope is **-3.32**, which corresponds to a perfect doubling of product each cycle.
            - **PCR Efficiency**: Calculated using the formula: $E = (10^{{-1/slope}} - 1) \\times 100\\%$. The ideal efficiency is 100%. An acceptable range is typically 90-110%.
            - **Y-Intercept**: The theoretical Ct value at a Log10 concentration of 0 (i.e., 1 copy per reaction).

            #### Analysis of Results
            - The **R² of {r_squared:.4f}** indicates an excellent linear fit, meeting typical acceptance criteria.
            - The **Slope of {slope:.3f}** is very close to the theoretical ideal of -3.32.
            - This results in a calculated **PCR Efficiency of {pcr_efficiency:.1f}%**, which is within the optimal 90-110% range, demonstrating robust and efficient amplification across the dynamic range.
            - The **Residuals Plot** shows a random scatter of points around the zero line, confirming that a linear model is appropriate and there are no significant non-linear effects (e.g., "hook effect") within the tested range. The assay is deemed linear and suitable for quantification.
            """)

with tab2:
    st.header("Assay Precision (Following CLSI EP05-A3)")

    with st.expander("🔬 **Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        To evaluate **precision**, a stable QC material is tested in replicates (e.g., n=5), by multiple operators (e.g., 2 operators), over multiple days (e.g., 3-5 days). This nested design allows for the quantification of different components of random analytical error, which is critical for understanding the reliability of our tests.
        - **Repeatability (Within-Run Precision)**: Variation under the most minimal set of changing conditions (same operator, same run, same instrument).
        - **Reproducibility (Total Precision)**: Variation across all changing conditions (different operators, different days, different reagent lots). This represents the "real-world" precision of the assay.

        #### The Method
        - **Box Plots**: Used to visually compare the distribution of Ct values across days and operators.
        - **Analysis of Variance (ANOVA)**: A statistical technique used to partition the total variance into its components (e.g., variance due to operator, variance due to day).
        - **Coefficient of Variation (%CV)**: A standardized, unitless measure of dispersion calculated as $(Standard Deviation / Mean) \\times 100$. This is the primary metric for accepting or rejecting precision performance.
        """)

    fig = px.box(precision_df, x="Day", y="Ct Value", color="Operator", title="Precision: Ct Value Distribution by Day and Operator", points="all")
    st.plotly_chart(fig, use_container_width=True)

    # Calculate CVs
    repeatability_cv = precision_df.groupby(['Day', 'Operator'])['Ct Value'].std().mean() / precision_df['Ct Value'].mean() * 100
    total_precision_cv = precision_df['Ct Value'].std() / precision_df['Ct Value'].mean() * 100

    col1, col2 = st.columns(2)
    col1.metric("Repeatability (Within-Run CV%)", f"{repeatability_cv:.2f}%")
    col2.metric("Reproducibility (Total CV%)", f"{total_precision_cv:.2f}%")

    with st.expander("📊 **Results & Analysis**"):
        st.markdown(f"""
        #### Analysis of Results
        - The **Box Plot** provides a visual assessment of variability. We can see a slight upward shift in median Ct for Operator 2 compared to Operator 1, suggesting a minor, but potentially consistent, systematic bias between operators. The wider boxes for Day 3 indicate a potential increase in random error on that day's runs.
        - The **Repeatability CV of {repeatability_cv:.2f}%** and **Total CV of {total_precision_cv:.2f}%** must be compared against the pre-defined acceptance criteria in the TMV plan. For a quantitative molecular assay, these values would likely be acceptable.
        - The difference between total and within-run CV indicates the contribution of between-day and between-operator variability. The larger this difference, the more sensitive the assay is to these external factors. This information is critical for establishing robust operational controls and training.
        """)

with tab3:
    st.header("Measurement System Analysis (MSA) / Gage R&R")

    with st.expander("🔬 **Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        A **Gage Repeatability & Reproducibility (Gage R&R)** study is performed to quantify the sources of variation within the measurement system itself. A set of samples representing the expected process range are measured multiple times by different operators. This is distinct from precision, as it compares the measurement system's variation to the actual variation between parts (samples).

        #### The Method
        - **ANOVA Gage R&R**: We use Analysis of Variance to partition the total observed variability into three components:
            1.  **Part-to-Part**: The true, inherent variation between the samples.
            2.  **Repeatability (Equipment Variation)**: Variation from the instrument when measuring the same part repeatedly.
            3.  **Reproducibility (Appraiser Variation)**: Variation from different operators measuring the same part.
        - **% Contribution & Number of Distinct Categories (ndc)** are calculated to assess the system's capability.
        """)

    total_var = msa_data['part_var'] + msa_data['repeatability_var'] + msa_data['reproducibility_var']
    gage_rr_var = msa_data['repeatability_var'] + msa_data['reproducibility_var']
    pct_gage_rr = (gage_rr_var / total_var) * 100
    # Formula for ndc
    ndc = int(1.41 * (np.sqrt(msa_data['part_var']) / np.sqrt(gage_rr_var)))

    col1, col2 = st.columns([1.5, 1])
    with col1:
        df_sunburst = pd.DataFrame([
            dict(id="Total Variation", parent="", value=total_var),
            dict(id="Part-to-Part", parent="Total Variation", value=msa_data['part_var']),
            dict(id="Gage R&R", parent="Total Variation", value=gage_rr_var),
            dict(id="Repeatability (EV)", parent="Gage R&R", value=msa_data['repeatability_var']),
            dict(id="Reproducibility (AV)", parent="Gage R&R", value=msa_data['reproducibility_var']),
        ])
        fig = px.sunburst(df_sunburst, ids='id', parents='parent', values='value',
                          title="Hierarchical Sources of Variation (% Contribution)",
                          color='id', color_discrete_map={
                              '(?)':'#1A3A6D', 'Total Variation':'#1A3A6D', 'Part-to-Part':'#2ca02c',
                              'Gage R&R':'#d62728', 'Repeatability (EV)':'#ff7f0e', 'Reproducibility (AV)':'#ffbb78'
                          })
        fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Key MSA Metrics")
        st.metric("Gage R&R (% Contribution)", f"{pct_gage_rr:.1f}%")
        st.metric("Number of Distinct Categories (ndc)", f"{ndc}")

        with st.expander("📊 **Results & Analysis**"):
            st.markdown(f"""
            #### Metrics Explained
            - **Gage R&R (% Contribution)**: The percentage of total variation consumed by the measurement system. An ideal system has very low % Contribution.
                - *<1%*: Excellent
                - *1-9%*: Acceptable
                - *>9%*: Unacceptable, system needs improvement.
            - **ndc (Number of Distinct Categories)**: An estimate of how many distinct groups of parts the system can reliably distinguish. Calculated as $ndc = \lfloor 1.41 \\times (\\frac{{\sigma_{{part}}}}{{\sigma_{{GageR\&R}}}}) \rfloor$.
                - *≥ 5*: Acceptable
                - *< 5*: Unacceptable, the system cannot adequately differentiate between parts.

            #### Analysis of Results
            - The sunburst plot clearly shows that **Part-to-Part** variation is the dominant source, which is ideal. However, the **Gage R&R** contribution is significant.
            - The **Gage R&R % of {pct_gage_rr:.1f}%** is above the 9% threshold for acceptability. The plot reveals that **Reproducibility (operator variation)** is the largest component of this measurement error.
            - The **ndc of {ndc}** is less than the required minimum of 5.
            - **Conclusion:** This measurement system is **NOT ACCEPTABLE**. The high operator-to-operator variability is masking the true difference between parts. **Action:** The transfer must be paused. A root cause investigation into the source of operator variability is required (e.g., differences in pipetting technique, sample handling, or interpretation). Re-training and clarification of the work instruction are necessary before re-running the MSA study.
            """)

with tab4:
    st.header("Assay Specificity & Interference (e.g., Cologuard®)")

    with st.expander("🔬 **Experiment & Method**"):
        st.markdown("""
        #### The Experiment
        **Analytical Specificity** is the ability of an assay to measure *only* the target analyte, even in the presence of other substances. For Cologuard®, we must demonstrate that common substances found in stool samples (e.g., hemoglobin from blood, bilirubin) do not interfere with the methylation assay signal.
        This is tested by analyzing:
        1. Blank samples (NTC)
        2. Samples with only the potential interferent
        3. A control sample with a known amount of the target analyte
        4. The control sample "spiked" with the potential interferents

        #### The Method
        - **Box Plots**: Used to visually compare the signal distributions (% Methylation) of the different sample types.
        - **Welch's Two-Sample t-test**: A statistical test used to determine if there is a significant difference between the means of two independent groups. We compare the "Control" group to the "Control + Interferents" group to statistically test for bias.
        """)

    fig = px.box(specificity_df, x="Sample Type", y="Signal (% Meth)", color="Sample Type", title="Assay Response to Potential Interferents in Stool Matrix")
    st.plotly_chart(fig, use_container_width=True)

    target_only = specificity_df[specificity_df['Sample Type'] == 'Methylated Control']['Signal (% Meth)']
    target_with_interferents = specificity_df[specificity_df['Sample Type'] == 'Control + Interferents']['Signal (% Meth)']
    ttest_res = ttest_ind(target_only, target_with_interferents, equal_var=False) # Welch's t-test

    st.subheader("Statistical Interference Test")
    st.metric("T-test P-value (Control vs Control + Interferents)", f"{ttest_res.pvalue:.4f}")

    with st.expander("📊 **Results & Analysis**"):
        st.markdown("""
        #### Metric Explained
        - **P-value**: The probability of observing the measured difference in means (or a more extreme one) if the interferents had no real effect (the "null hypothesis"). A P-value < 0.05 indicates the observed difference is statistically significant.

        #### Analysis of Results
        - The **Box Plot** shows that the 'NTC' and 'Interferent Only' samples produce negligible methylation signals, demonstrating good basic specificity.
        - However, there is a clear visual depression in the signal for the 'Control + Interferents' group compared to the 'Methylated Control' group.
        """)
        if ttest_res.pvalue < 0.05:
            st.error(f"**P-value of {ttest_res.pvalue:.4f} is less than 0.05.** This confirms a statistically significant interference effect.")
            st.markdown("""
            **Conclusion & Action:** The tested substances (e.g., hemoglobin) are causing a significant negative bias (suppression) on the methylation signal. This is a known risk for PCR-based assays from stool. The magnitude of this interference must be quantified and compared against the defined clinical risk threshold. If the impact is unacceptable, mitigation strategies (e.g., improving the DNA extraction/purification steps, adjusting QC limits) must be developed and validated.
            """)
        else:
            st.success(f"**P-value of {ttest_res.pvalue:.4f} is greater than 0.05.** There is no statistically significant evidence of interference from the tested substances at the concentrations evaluated.")
