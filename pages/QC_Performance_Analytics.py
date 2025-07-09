# pages/QC_Performance_Analytics.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import generate_spc_data, generate_lot_data, detect_westgard_rules, calculate_cpk
from scipy.stats import f_oneway, norm

st.set_page_config(page_title="QC Performance Analytics", layout="wide")
st.title("📊 QC Performance & Analytics Dashboard")
st.markdown("### Monitoring transferred method health with advanced statistical process control.")

with st.expander("🌐 Regulatory Context & Legend"):
    st.markdown("""
    This dashboard provides tools for ongoing process monitoring to ensure methods remain in a state of control.
    - **Statistical Process Control**: The charts meet requirements for statistical techniques under **21 CFR 820.250**.
    - **Investigations & CAPA**: Out-of-control events trigger investigations under **21 CFR 820.100**.
    - **Lot Acceptance**: The lot comparison supports incoming acceptance activities per **21 CFR 820.80**.
    """)

spc_df = generate_spc_data()
lot_df = generate_lot_data()

# --- SPC Section ---
st.header("Statistical Process Control (SPC) with Westgard Rule Analysis")

with st.expander("🔬 **The Experiment & Method**"):
    st.markdown("""
    #### The Experiment
    To monitor the ongoing performance and stability of a validated assay, a stable Quality Control (QC) material is tested with every batch of production samples. The results of the QC material are plotted over time to detect any shifts, trends, or increased variability in the process.

    #### The Method: Levey-Jennings Chart & Westgard Rules
    - **Levey-Jennings Chart**: A graphical chart used to plot QC data over time. It displays the mean and control limits (typically at ±1, ±2, and ±3 standard deviations (SD) from the mean) derived from historical data.
    - **Westgard Rules**: A set of multi-rule QC procedures used to evaluate whether an analytical run is in-control or out-of-control. By combining several rules, this method improves the detection of both random and systematic errors while reducing false rejection rates. Common rules include:
        - **1_3s**: One point exceeds the ±3 SD limit. Detects large random errors.
        - **2_2s**: Two consecutive points exceed the same ±2 SD limit. Detects systematic error (bias).
        - **4_1s**: Four consecutive points exceed the same ±1 SD limit. Detects small, sustained systematic error.
    """)

mean = spc_df['Value'].mean()
std = spc_df['Value'].std()

fig = go.Figure()
fig.add_hline(y=mean, line_dash="solid", line_color="green", annotation_text="Mean")
for i, dash in zip([1, 2, 3], ["dot", "dash", "longdash"]):
    fig.add_hline(y=mean + i * std, line_dash=dash, line_color="orange", opacity=0.7, annotation_text=f"+{i}SD")
    fig.add_hline(y=mean - i * std, line_dash=dash, line_color="orange", opacity=0.7, annotation_text=f"-{i}SD")
    
fig.add_trace(go.Scatter(x=spc_df['Run'], y=spc_df['Value'], mode='lines+markers', name='QC Value', marker_color='#1f77b4'))
violations = detect_westgard_rules(spc_df)
if not violations.empty:
    fig.add_trace(go.Scatter(x=violations['Run'], y=violations['Value'], mode='markers', marker=dict(color='red', size=14, symbol='x-thin', line=dict(width=2)), name='Rule Violation', hovertext=violations['Rule']))
fig.update_layout(title='Control Chart for QC Material Performance', xaxis_title='Run Number', yaxis_title='Performance Value')
st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 **Results & Analysis**"):
    st.markdown("""
    #### Analysis of the Control Chart
    The Levey-Jennings chart provides a visual timeline of the process's stability. Points should be randomly distributed around the mean and within the ±3 SD control limits. The application of Westgard rules provides objective criteria for action.
    """)
    if not violations.empty:
        st.error("Out-of-control state detected! An investigation must be initiated per **21 CFR 820.100**.")
        st.dataframe(violations, use_container_width=True)
        st.markdown("""
        **Interpretation of Violations:**
        - The **1_2s warning** at Run 10 indicates a point to watch closely.
        - The **2_2s violation** at Run 11 confirms a systematic shift has occurred, as two consecutive points are on the same side of the mean beyond 2 SD.
        - The **1_3s violation** at Run 15 is a critical "run rejection" rule, indicating a significant deviation, possibly due to a large random error.
        - The **4_1s violation** ending at Run 23 indicates a sustained, small systematic bias in the process.
        
        **Action Required:** The presence of multiple rule violations, especially rejection rules like 1_3s and 2_2s, indicates that the process is not in a state of statistical control. All patient results associated with these runs should be held pending a full investigation into potential causes (e.g., reagent issues, instrument malfunction, calibration drift).
        """)
    else:
        st.success("Process is in a state of statistical control. All QC results are within established limits and no Westgard rule violations were detected.")

st.divider()

# --- Lot-to-Lot Section ---
st.header("Reagent Lot-to-Lot Performance with Statistical Significance")

with st.expander("🔬 **The Experiment & Method**"):
    st.markdown("""
    #### The Experiment
    To ensure the consistency of manufacturing, each new lot of a critical reagent must be tested and compared against the current, in-use lot. This is a crucial part of the **Incoming Acceptance** process. Data is collected from multiple runs of both the new and reference lots.

    #### The Method: ANOVA
    - **Box Plots**: Used to visually compare the distribution (median, spread, and range) of results from each reagent lot.
    - **Analysis of Variance (ANOVA)**: A powerful statistical test used to determine whether there are any statistically significant differences between the means of two or more independent groups. The test calculates a **P-value**, which is the probability of observing the data if there were truly no difference between the lots.
    """)

col1, col2 = st.columns([2, 1])
with col1:
    fig_box = px.box(lot_df, x='Lot ID', y='Performance Metric', color='Lot ID', title='Comparison of Reagent Lot Performance', points='all')
    st.plotly_chart(fig_box, use_container_width=True)
with col2:
    st.subheader("ANOVA Test for Lot Differences")
    lot_groups = [group['Performance Metric'].values for name, group in lot_df.groupby('Lot ID')]
    f_stat, p_value = f_oneway(*lot_groups)
    st.metric("ANOVA P-value", f"{p_value:.4f}")

    with st.expander("📊 **Results & Analysis**"):
        st.markdown("""
        #### Metric Explained
        - **P-value**: The probability that the observed differences in lot means occurred by random chance alone. A small P-value (typically < 0.05) suggests that the observed difference is "real" and statistically significant.
        
        #### Analysis of Results
        - The **Box Plot** visually suggests that "Lot C (New)" has a higher median value compared to the other lots. The other lots appear to be centered around a similar value.
        """)
        if p_value < 0.05:
            st.error(f"**P-value of {p_value:.4f} is less than 0.05.** This provides strong statistical evidence that at least one reagent lot performs significantly different from the others.")
            st.markdown("""
            **Conclusion & Action:** The new reagent lot ("Lot C") has a statistically significant positive bias. This lot fails the lot-to-lot comparison criteria and should **not be released** for use in production. An investigation into the manufacturing of Lot C is warranted to determine the root cause of the shift.
            """)
        else:
            st.success(f"**P-value of {p_value:.4f} is greater than 0.05.** There is no statistically significant difference detected between the reagent lots.")
            st.markdown("""
            **Conclusion & Action:** The new reagent lot performs equivalently to the reference lots. It passes the incoming acceptance criteria and can be released for use in production.
            """)

st.divider()

# --- Process Capability Section ---
st.header("Process Capability Analysis (Cpk)")
with st.expander("🔬 **The Method & Metrics**"):
    st.markdown("""
    #### The Method
    **Process Capability** analysis determines how well a process, in a state of statistical control, is able to meet its specification limits. The **Process Capability Index (Cpk)** is a standard metric used to quantify this. The analysis is visualized by plotting the process distribution against the specification limits.
    
    #### The Metrics
    - **Cpk**: Measures how close you are to your target and how consistent you are to around your average performance. It quantifies the "room" between your process distribution and the nearest specification limit. A higher Cpk value indicates a more capable process.
        - **Cpk > 1.33**: Generally considered capable for most processes (often aligned with Six Sigma goals).
        - **1.0 < Cpk < 1.33**: Marginally capable; may require tighter control.
        - **Cpk < 1.0**: Not capable; the process is producing, or is at high risk of producing, out-of-spec results.
    - **PPM (Parts Per Million)**: The expected number of defective parts per million units produced, calculated from the process distribution and specification limits. This translates the statistical Cpk value into a direct business and quality impact.
    """)

# Use the same data as the SPC chart for this example
lsl = 95 # Lower Specification Limit
usl = 105 # Upper Specification Limit

# --- Layout for Capability Analysis ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Capability Metrics")
    cpk_value = calculate_cpk(spc_df['Value'], usl, lsl)
    
    # Calculate PPM
    mean = spc_df['Value'].mean()
    std_dev = spc_df['Value'].std()
    z_usl = (usl - mean) / std_dev
    z_lsl = (lsl - mean) / std_dev
    prob_above_usl = 1 - norm.cdf(z_usl)
    prob_below_lsl = norm.cdf(z_lsl)
    ppm = (prob_above_usl + prob_below_lsl) * 1_000_000

    st.metric("Lower Spec Limit (LSL)", f"{lsl}")
    st.metric("Upper Spec Limit (USL)", f"{usl}")
    st.metric("Process Capability (Cpk)", f"{cpk_value:.2f}")
    st.metric("Expected Defective Rate (PPM)", f"{ppm:,.0f}")

    if cpk_value < 1.0:
        st.error("Process is NOT CAPABLE of meeting specifications.")
    elif cpk_value < 1.33:
        st.warning("Process is MARGINALLY CAPABLE. Improvements to reduce variability or re-center the mean are recommended.")
    else:
        st.success("Process is CAPABLE of meeting specifications.")

with col2:
    st.subheader("Process Distribution vs. Specification Limits")
    # Create the histogram
    fig_hist = px.histogram(spc_df, x="Value", nbins=20, histnorm='probability density', marginal="rug",
                            title="Process Capability Histogram")
    
    # Add the smoothed KDE curve
    # We need to manually create the KDE data to overlay it
    from statsmodels.nonparametric.kde import KDEUnivariate
    kde = KDEUnivariate(spc_df['Value'].values)
    kde.fit()
    x_kde = np.linspace(spc_df['Value'].min(), spc_df['Value'].max(), 100)
    y_kde = kde.evaluate(x_kde)
    fig_hist.add_trace(go.Scatter(x=x_kde, y=y_kde, mode='lines', name='KDE', line=dict(color='firebrick')))
    
    # Add vertical lines for LSL, USL, and Mean
    fig_hist.add_vline(x=lsl, line_width=2, line_dash="dash", line_color="red", annotation_text="LSL")
    fig_hist.add_vline(x=usl, line_width=2, line_dash="dash", line_color="red", annotation_text="USL")
    fig_hist.add_vline(x=mean, line_width=2, line_dash="dot", line_color="green", annotation_text="Mean")

    # Add shaded regions for out-of-spec areas
    fig_hist.add_vrect(x0=spc_df['Value'].min(), x1=lsl, fillcolor="red", opacity=0.1, line_width=0, annotation_text="OOS", annotation_position="top left")
    fig_hist.add_vrect(x0=usl, x1=spc_df['Value'].max(), fillcolor="red", opacity=0.1, line_width=0)
    
    fig_hist.update_layout(yaxis_title="Density")
    st.plotly_chart(fig_hist, use_container_width=True)

with st.expander("📊 **Results & Analysis**"):
    st.markdown(f"""
    #### Analysis of the Capability Plot
    This plot provides a clear visual assessment of how the process distribution ("voice of the process") fits within the specification limits ("voice of the customer").
    - The bulk of the process data, represented by the histogram bars and the red KDE curve, is centered at a mean of **{mean:.2f}**.
    - We can see that the process is not perfectly centered between the LSL ({lsl}) and USL ({usl}). The mean is closer to the USL.
    - The "tail" of the process distribution on the right side extends beyond the USL, which is visually confirmed by the shaded red "OOS" (Out of Spec) region. This is the direct cause of the predicted defective PPM.
    - The **Cpk of {cpk_value:.2f}** quantifies this relationship. Since the value is less than 1.0, it confirms the process is not capable. The primary reason is that the process is not centered, and its variability (spread) is too wide relative to the specification range.
    
    **Action Required:** To improve this process, two strategies should be considered:
    1.  **Re-center the Process**: Investigate and correct the systematic bias that is pushing the mean towards the USL.
    2.  **Reduce Variation**: Implement process improvements (e.g., using Six Sigma or DOE methodologies) to reduce the standard deviation, which would make the distribution "taller and narrower," pulling the tails away from the specification limits.
    """)
