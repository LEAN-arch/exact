# pages/QC_Performance_Analytics.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import generate_spc_data, generate_lot_data, detect_westgard_rules, calculate_cpk
from scipy.stats import f_oneway, norm

st.set_page_config(
    page_title="QC Performance Analytics | Exact Sciences",
    layout="wide"
)

st.title("📊 QC Performance & Process Capability")
st.markdown("### Monitoring transferred method health with advanced statistical process control and capability analysis.")

with st.expander("🌐 Regulatory Context: Maintaining a State of Control"):
    st.markdown("""
    This dashboard provides the tools for ongoing process monitoring to ensure our transferred methods remain in a state of statistical control and are capable of meeting specifications. This is a continuous activity throughout the product lifecycle.

    - **Statistical Techniques (21 CFR 820.250)**: Mandates the use of valid statistical techniques for establishing and maintaining process control. The Levey-Jennings charts, ANOVA, and Cpk analyses on this page are direct implementations of this requirement.
    - **Lot Acceptance (21 CFR 820.80)**: The lot-to-lot comparison provides objective evidence to accept or reject incoming lots of critical reagents, a key component of our device acceptance activities.
    - **CAPA (21 CFR 820.100)**: Out-of-control events detected by SPC charts or failing lot qualifications are primary inputs into the Corrective and Preventive Action (CAPA) system for investigation.
    - **CLIA §493.1281-1282**: Requires that QC procedures are established to monitor the accuracy and precision of the complete analytic process. Levey-Jennings charts are the standard method for fulfilling this requirement in clinical laboratories.
    """)

# --- Data Generation ---
spc_df = generate_spc_data()
lot_df = generate_lot_data()

# --- SPC Section ---
st.header("RT-PCR Control Monitoring (Levey-Jennings & Westgard Rules)")
st.caption("Monitoring the ongoing performance of an Oncotype DX® assay positive control.")

with st.expander("🔬 **The Method & Interpretation for Ct Values**"):
    st.markdown("""
    #### The Method: Levey-Jennings Chart with Westgard Rules
    A **Levey-Jennings Chart** is a specialized control chart used to plot Quality Control (QC) data over time. For our RT-PCR assays, we plot the **Cycle threshold (Ct) value** of a consistent positive control material. The chart displays the mean and control limits (typically at ±1, ±2, and ±3 standard deviations) derived from historical data.

    **Westgard Rules** are a set of multi-rule QC procedures that provide objective criteria to determine if an analytical run is in-control or out-of-control. This improves error detection while reducing false alarms.

    **Crucial Interpretation for Ct Values:** Remember that for Ct values, a **lower number is better** (indicating more efficient amplification), and a **higher number is worse**. Therefore, a downward shift is a positive process change (improvement), while an upward shift is a negative process change (degradation).
    """)

mean = spc_df['Ct Value'].mean()
std = spc_df['Ct Value'].std()

fig = go.Figure()
# Add control limits
fig.add_hline(y=mean, line_dash="solid", line_color="green", annotation_text="Mean", annotation_font_color="green")
for i, dash in zip([1, 2, 3], ["dot", "dash", "longdash"]):
    fig.add_hline(y=mean + i * std, line_dash=dash, line_color="orange", opacity=0.7, annotation_text=f"+{i}SD (Worse)")
    fig.add_hline(y=mean - i * std, line_dash=dash, line_color="orange", opacity=0.7, annotation_text=f"-{i}SD (Better)")

fig.add_trace(go.Scatter(x=spc_df['Run'], y=spc_df['Ct Value'], mode='lines+markers', name='Positive Control Ct', marker_color='#1f77b4'))

# Highlight Westgard Violations
violations = detect_westgard_rules(spc_df, value_col='Ct Value')
if not violations.empty:
    fig.add_trace(go.Scatter(x=violations['Run'], y=violations['Value'], mode='markers',
                             marker=dict(color='red', size=14, symbol='x-thin', line=dict(width=2)),
                             name='Rule Violation', hovertemplate='<b>Run %{x}</b><br>Ct: %{y:.2f}<br>Rule: %{text}<extra></extra>',
                             text=violations['Rule']))

fig.update_layout(title='Levey-Jennings Chart for Oncotype DX® Positive Control (Ct Value)', xaxis_title='Run Number', yaxis_title='Ct Value', yaxis_autorange='reversed')
st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 **Results & Analysis**"):
    if not violations.empty:
        st.error("Out-of-control state detected! An investigation must be initiated per **21 CFR 820.100**.")
        st.dataframe(violations, use_container_width=True)
        st.markdown("""
        **Interpretation of Violations:**
        - **10_x Rule Violation (Runs 15-24)**: Ten consecutive points are on the same side of the mean (in this case, below it). This signals a **sustained systematic shift** in the process, indicating it is now running more efficiently. While this may seem positive, any un-investigated shift is a risk.
        - **1_3s Rule Violation (Run 25)**: One point has exceeded the +3 SD limit (a higher Ct value). This is a critical **run rejection** rule, indicating a significant, negative deviation, possibly due to a large random error (e.g., pipetting error, inhibitor).

        **Action Required:** The `1_3s` violation requires immediate rejection of the run and associated patient samples. The `10_x` trend, although seemingly an "improvement," requires a formal investigation to determine the root cause of the shift (e.g., a new reagent lot, a change in instrument performance). The process mean and limits may need to be recalculated after the cause is identified and controlled.
        """)
    else:
        st.success("Process is in a state of statistical control. All QC results are within established limits and no Westgard rule violations were detected.")

st.divider()

# --- Lot-to-Lot Section ---
st.header("Reagent Lot-to-Lot Qualification (NGS Library Prep)")
st.caption("Comparing the performance of a new OncoExTra® library prep kit lot against established lots.")

with st.expander("🔬 **The Experiment & Method**"):
    st.markdown("""
    #### The Experiment
    To ensure the consistency of our NGS assays, each new lot of a critical reagent (like a library preparation kit) must be formally qualified against the current in-use lot. A standard DNA control is run through the library preparation process using both the new and reference lots, and a key performance metric (e.g., **Final Library Yield**) is compared.

    #### The Method: ANOVA
    - **Box Plots**: Used to visually compare the distribution (median, spread, and outliers) of library yields from each reagent lot.
    - **Analysis of Variance (ANOVA)**: A statistical test used to determine if there are any statistically significant differences between the means of two or more groups. The test calculates a **P-value**, which is the probability of observing the data if there were truly no difference between the lots. A low P-value suggests a real difference exists.
    """)

col1, col2 = st.columns([2, 1])
with col1:
    fig_box = px.box(lot_df, x='Lot ID', y='Library Yield (ng)', color='Lot ID',
                     title='Comparison of Library Prep Kit Lot Performance', points='all')
    st.plotly_chart(fig_box, use_container_width=True)
with col2:
    st.subheader("ANOVA Test for Lot Differences")
    lot_groups = [group['Library Yield (ng)'].values for name, group in lot_df.groupby('Lot ID')]
    f_stat, p_value = f_oneway(*lot_groups)
    st.metric("ANOVA P-value", f"{p_value:.4f}")

    with st.expander("📊 **Results & Analysis**"):
        if p_value < 0.05:
            st.error(f"**P-value of {p_value:.4f} is less than 0.05.** This provides strong statistical evidence that at least one reagent lot performs significantly different from the others.")
            st.markdown("""
            **Conclusion & Action:** The new reagent lot ("Lot D") demonstrates a statistically significant lower library yield. This lot **fails the incoming qualification criteria** and must be rejected and quarantined. An investigation should be initiated with the supplier to determine the root cause of the poor performance.
            """)
        else:
            st.success(f"**P-value of {p_value:.4f} is greater than 0.05.** There is no statistically significant difference detected between the reagent lots.")
            st.markdown("""
            **Conclusion & Action:** The new reagent lot performs equivalently to the reference lots. It passes the incoming acceptance criteria and can be released for use in production.
            """)

st.divider()

# --- Process Capability Section ---
st.header("Process Capability Analysis (Cpk) for Cologuard® Reagent")
st.caption("Assessing the ability of a critical reagent manufacturing step to meet its specifications.")

with st.expander("🔬 **The Method & Metrics**"):
    st.markdown("""
    #### The Method
    **Process Capability** analysis determines how well a process, in a state of statistical control, is able to meet its specification limits. The **Process Capability Index (Cpk)** is the standard metric used to quantify this.
    The formula for Cpk is:
    """)
    st.latex(r''' C_{pk} = \min\left(\frac{USL - \mu}{3\sigma}, \frac{\mu - LSL}{3\sigma}\right) ''')
    st.markdown(r"""
    Where:
    - $USL$ is the Upper Specification Limit
    - $LSL$ is the Lower Specification Limit
    - $\mu$ is the process mean
    - $\sigma$ is the process standard deviation

    #### The Metrics
    - **Cpk**: A higher Cpk value indicates a more capable process. A common target is **Cpk ≥ 1.33**, which corresponds to a process that is well-centered and has a low probability of producing out-of-spec material.
    - **PPM (Parts Per Million)**: The expected number of defective units per million produced, a direct measure of business and quality impact.
    """)

lsl, usl = 24.0, 26.0
cpk_value = calculate_cpk(spc_df['Ct Value'], usl, lsl)
mean_val, std_dev = spc_df['Ct Value'].mean(), spc_df['Ct Value'].std()
z_usl = (usl - mean_val) / std_dev; z_lsl = (lsl - mean_val) / std_dev
ppm = (norm.sf(z_usl) + norm.cdf(z_lsl)) * 1_000_000

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Capability Metrics")
    st.metric("Lower Spec Limit (LSL)", f"{lsl}")
    st.metric("Upper Spec Limit (USL)", f"{usl}")
    st.metric("Process Capability (Cpk)", f"{cpk_value:.2f}")
    st.metric("Expected OOS Rate (PPM)", f"{ppm:,.0f}")
    if cpk_value < 1.0: st.error("Process is NOT CAPABLE.")
    elif cpk_value < 1.33: st.warning("Process is MARGINALLY CAPABLE.")
    else: st.success("Process is CAPABLE.")

with col2:
    st.subheader("Process Distribution vs. Specification Limits")
    fig_hist = px.histogram(spc_df, x="Ct Value", nbins=20, histnorm='probability density', title="Process Capability Histogram")
    x_range = np.linspace(spc_df['Ct Value'].min(), spc_df['Ct Value'].max(), 200)
    fig_hist.add_trace(go.Scatter(x=x_range, y=norm.pdf(x_range, mean_val, std_dev), mode='lines', name='Normal Fit', line=dict(color='firebrick')))
    fig_hist.add_vline(x=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
    fig_hist.add_vline(x=usl, line_dash="dash", line_color="red", annotation_text="USL")
    fig_hist.add_vline(x=mean_val, line_dash="dot", line_color="green", annotation_text="Mean")
    st.plotly_chart(fig_hist, use_container_width=True)
