# pages/Software_V_V_Dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import (generate_v_model_data, generate_traceability_data,
                   generate_defect_trend_data, generate_defect_category_data)

st.set_page_config(
    page_title="Software V&V Dashboard | Exact Sciences",
    layout="wide"
)

st.title("🖥️ Software Verification & Validation (V&V) Dashboard")
st.markdown("### Tracking the development lifecycle of regulated QC software and bioinformatics pipelines.")

with st.expander("🌐 Regulatory Context: Validating Software as a Medical Device (SaMD)"):
    st.markdown("""
    The software we develop to analyze QC data or process patient data (e.g., Oncotype DX® Recurrence Score® calculation, OncoExTra® variant calling) is considered Software as a Medical Device (SaMD) or part of a medical device. As such, its development must follow a rigorous, controlled process.

    - **IEC 62304**: This is the international standard for the **Software Development Lifecycle (SDLC)** for medical device software. This dashboard provides tools to manage and document our adherence to this standard's processes, from requirements gathering to testing and release. The V-Model is a classic representation of this lifecycle.
    - **FDA 21 CFR 820.30 (Design Controls)**: Software is a component of our devices and is subject to the same design control requirements as hardware, including design planning, inputs, outputs, review, verification, validation, and transfer.
    - **FDA 21 CFR Part 11 (Electronic Records; Electronic Signatures)**: This regulation is critical. It sets the requirements for ensuring that electronic records and signatures are trustworthy, reliable, and equivalent to paper records. Our QC software must have features like secure audit trails, access controls, and the ability to generate accurate copies, all of which must be validated.
    - **Requirements Traceability**: A core tenet of validated software. The traceability matrix on this page provides the objective evidence that all user needs have been translated into specifications and that every specification has been formally tested. This is a primary focus during regulatory audits.
    """)

# --- 1. KPIs for Oncotype DX® QC Software v2.1 Project ---
req_data = generate_traceability_data()
req_df = pd.DataFrame(req_data)
pass_count = (req_df['Test Status'] == 'Pass').sum()
fail_count = (req_df['Test Status'] == 'Fail').sum()
total_reqs = len(req_df)

st.header("1. Software Quality KPIs (Example Project: Oncotype DX® QC v2.1)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Requirements Coverage", f"{(total_reqs / total_reqs) * 100:.0f}%", help="Percentage of requirements with linked test cases.")
col2.metric("Test Case Pass Rate", f"{pass_count / total_reqs * 100:.0f}%", help="Percentage of requirements whose tests have passed.")
col3.metric("Requirements with Failures", f"{fail_count}", delta=f"{fail_count} Blocking", delta_color="inverse", help="Count of requirements with failing test cases. These are release blockers.")
col4.metric("21 CFR Part 11 Tests", "1/1 Fail", help="Status of tests specifically for Part 11 compliance.")


st.divider()

# --- 2. V-Model and Traceability ---
st.header("2. Development Lifecycle & Requirements Traceability")
col1, col2 = st.columns([1.2, 1.8])

with col1:
    st.subheader("SDLC V-Model")
    with st.expander("🔬 **The V-Model Explained**"):
        st.markdown("""
        The **V-Model** illustrates how testing activities (Verification & Validation) are logically linked to each phase of development.
        - The left side represents **specification and design**, moving from high-level user needs down to detailed module design.
        - The right side represents the **testing and integration** phases, moving up from component testing to full system acceptance.
        - Each level on the right directly tests the artifacts produced at the corresponding level on the left, ensuring nothing is missed.
        """)
    v_model_df = generate_v_model_data()
    fig_v = go.Figure(go.Scatter(x=v_model_df['x'], y=v_model_df['y'], mode='lines+markers+text',
                                 text=v_model_df['text'], textposition="top center",
                                 line=dict(width=3), marker=dict(size=15)))
    fig_v.add_annotation(x=4.5, y=2.5, text="<b>Verification & Validation</b>", showarrow=False, font_size=16)
    fig_v.update_layout(title="Software Development V-Model (IEC 62304)", showlegend=False,
                        xaxis_visible=False, yaxis_visible=False, height=550)
    st.plotly_chart(fig_v, use_container_width=True)

with col2:
    st.subheader("Requirements Traceability Matrix (RTM)")
    with st.expander("🔬 **Purpose & Analysis**"):
        st.markdown("""
        The **RTM** is arguably the most critical document in a software validation package. It provides the auditable, objective evidence that every single user requirement has a corresponding functional specification and, most importantly, has been fully tested.
        - **Analysis**: This matrix provides an immediate view of validation readiness. We can see a critical **release blocker**: the requirement for **21 CFR Part 11 compliance (URS-03) has a failing test case**. This must be resolved and re-tested before the software can be released. The PDF report generation is also still in progress.
        """)
    def style_status(val):
        if val == 'Pass': return 'background-color: #28a745; color: white;'
        if val == 'Fail': return 'background-color: #dc3545; color: white;'
        if val == 'In Progress': return 'background-color: #ffc107; color: black;'
        return ''
    # Updated from .applymap to .map to resolve deprecation warning
    st.dataframe(req_df.style.map(style_status, subset=['Test Status']), use_container_width=True, height=550, hide_index=True)

st.divider()

# --- 3. Defect Analysis ---
st.header("3. Defect Management & Analysis")
st.caption("Tracking and analyzing software anomalies (bugs) to ensure quality and prioritize fixes.")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Defect Open vs. Close Trend (Burnup Chart)")
    with st.expander("🔬 **Purpose & Analysis**"):
        st.markdown("""
        A **Defect Burnup Chart** tracks the total number of defects found (red line) against the total number of defects fixed (green line). The gap between the lines represents the current backlog of open defects. The goal is for the green line to meet the red line, indicating all known defects are resolved.
        - **Analysis**: This chart shows a stable rate of defect discovery, with the development team keeping pace with fixes. The sharp increase in closed defects near the end indicates a focused "bug bash" period, which is common before a release candidate is built. The narrowing gap is a positive sign of project stabilization.
        """)
    defect_trend_df = generate_defect_trend_data()
    fig_burnup = go.Figure()
    fig_burnup.add_trace(go.Scatter(x=defect_trend_df['Date'], y=defect_trend_df['Opened'], mode='lines', name='Total Defects Opened', line=dict(color='red', width=3)))
    fig_burnup.add_trace(go.Scatter(x=defect_trend_df['Date'], y=defect_trend_df['Closed'], mode='lines', name='Total Defects Closed', line=dict(color='green', width=3), fill='tozeroy', fillcolor='rgba(40,167,69,0.2)'))
    fig_burnup.update_layout(title="Defect Burnup Chart", yaxis_title="Cumulative Count of Defects", legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig_burnup, use_container_width=True)

with col2:
    st.subheader("Defect Pareto Analysis (Bioinformatics Pipeline)")
    with st.expander("🔬 **Purpose & Analysis**"):
        st.markdown("""
        The **Pareto Chart** helps identify the "vital few" areas that cause the majority of software defects. This allows us to focus our limited testing and code review resources for maximum impact.
        - **Analysis**: This analysis is a powerful tool for a lead scientist. It clearly shows that the **Bioinformatics Algorithm** itself is the single largest source of defects. This is a data-driven insight telling us that our highest priority for quality improvement should be on peer-reviewing the algorithm logic and adding more rigorous, targeted unit tests to the calculation engine, rather than focusing on lower-impact areas like the UI.
        """)
    pareto_df = generate_defect_category_data()
    pareto_df['Cumulative %'] = (pareto_df['Count'].cumsum() / pareto_df['Count'].sum()) * 100

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(go.Bar(x=pareto_df['Category'], y=pareto_df['Count'], name='Defect Count'), secondary_y=False)
    fig_pareto.add_trace(go.Scatter(x=pareto_df['Category'], y=pareto_df['Cumulative %'], name='Cumulative %', line=dict(color='red')), secondary_y=True)
    fig_pareto.update_layout(title_text="Pareto Chart of Defect Categories", legend=dict(x=0.6, y=0.9))
    fig_pareto.update_yaxes(title_text="<b>Count</b>", secondary_y=False)
    fig_pareto.update_yaxes(title_text="<b>Cumulative Percentage (%)</b>", secondary_y=True, range=[0,101])
    st.plotly_chart(fig_pareto, use_container_width=True)
