# pages/Software_V_V_Dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import (generate_v_model_data, generate_traceability_data,
                   generate_defect_trend_data, generate_defect_category_data)

st.set_page_config(page_title="Software V&V Dashboard", layout="wide")
st.title("🖥️ Software Verification & Validation (V&V) Dashboard")
st.markdown("### Tracking the lifecycle of QC software development and validation.")

with st.expander("🌐 Regulatory Context & Legend (IEC 62304)"):
    st.markdown("""
    This dashboard provides tools to manage the Software Development Lifecycle (SDLC) in alignment with **IEC 62304**, the international standard for medical device software, and supports compliance with **21 CFR 820**.
    - **V-Model**: A visual representation of the SDLC, emphasizing the relationship between development phases and their corresponding testing (V&V) activities.
    - **Requirements Traceability**: A core tenet of validated software. This matrix provides objective evidence that all user needs have been translated into specifications and that every specification has been formally tested.
    - **Defect Management**: Demonstrates a controlled process for identifying, evaluating, and resolving software anomalies (bugs), a key requirement for maintaining software quality.
    """)

# --- 1. KPIs ---
req_data = generate_traceability_data()
req_df = pd.DataFrame(req_data)
test_pass_count = (req_df['Test Status'] == 'Pass').sum()
total_tests = len(req_df)
open_critical_defects = 2 # Hardcoded for KPI example

st.header("1. Software Quality KPIs")
col1, col2, col3 = st.columns(3)
col1.metric("Requirements Test Coverage", f"{total_tests / len(req_df) * 100:.0f}%", help="Percentage of defined requirements that have at least one associated test case.")
col2.metric("Test Case Pass Rate", f"{test_pass_count / total_tests * 100:.0f}%", help="Percentage of all test cases that have passed successfully.")
col3.metric("Open Critical/High Defects", f"{open_critical_defects}", delta=open_critical_defects, delta_color="inverse", help="Count of unresolved software defects with a high or critical severity rating.")

st.divider()

# --- 2. V-Model and Traceability ---
st.header("2. Development Lifecycle & Traceability")
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("SDLC V-Model")
    with st.expander("🔬 **The Method Explained**"):
        st.markdown("""
        The **V-Model** is a conceptual framework for the Software Development Lifecycle (SDLC) that is strongly preferred in regulated industries. It illustrates how testing activities are logically linked to development activities.
        - The left side of the 'V' represents the **specification and design** phases, moving from high-level user needs down to detailed module design.
        - The right side represents the **testing and integration** phases (Verification & Validation).
        - Each level on the right side directly **verifies or validates** the corresponding level on the left. For example, Unit Testing verifies the Module Design, while Acceptance Testing validates the User Requirements. This structure ensures that testing is planned in parallel with development and that every requirement is tested.
        """)
    v_model_df = generate_v_model_data()
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=v_model_df['x'], y=v_model_df['y'], mode='lines', line=dict(color='royalblue', width=3)))
    fig_v.add_trace(go.Scatter(
        x=v_model_df['x'], y=v_model_df['y'], mode='markers+text',
        marker=dict(color='aliceblue', size=80, symbol='square', line=dict(color='royalblue', width=2)),
        text=v_model_df['text'], textposition="middle center", textfont=dict(size=12)
    ))
    fig_v.add_annotation(x=2.5, y=3.5, text="<b>Validation<br>(Are we building the right product?)</b>", showarrow=False, font=dict(color='grey'))
    fig_v.add_annotation(x=6.5, y=3.5, text="<b>Verification<br>(Are we building the product right?)</b>", showarrow=False, font=dict(color='grey'))
    fig_v.update_layout(showlegend=False, height=500, margin=dict(l=20, r=20, b=20, t=20))
    fig_v.update_xaxes(visible=False); fig_v.update_yaxes(visible=False)
    st.plotly_chart(fig_v, use_container_width=True)

with col2:
    st.subheader("Requirements Traceability Matrix")
    with st.expander("🔬 **The Method & Metrics Explained**"):
        st.markdown("""
        #### The Method
        A **Requirements Traceability Matrix (RTM)** is a document that traces the lineage of each requirement throughout the development lifecycle. It provides a many-to-many relationship mapping between user needs, functional specifications, design documents, and the test cases that verify them.
        
        #### The Metrics
        - **Test Status**: The current outcome of the associated test case. 'Pass' indicates the requirement has been successfully verified. 'Fail' indicates a defect was found. 'In Progress' means testing has not been completed.
        - **Requirements Coverage (KPI)**: A direct measure of traceability. A 100% coverage rate is a prerequisite for software release, ensuring no requirement has been overlooked.
        
        #### Analysis of Results
        This matrix provides an auditable, at-a-glance view of validation readiness. In this example, we can see that user need **URS-03** has a failing test case, which is a blocker for release. Furthermore, user need **URS-04** is still in progress. These two items represent the critical path to completing the software validation.
        """)
    def style_status(val):
        if val == 'Pass': return 'background-color: #28a745; color: white;'
        if val == 'Fail': return 'background-color: #dc3545; color: white;'
        if val == 'In Progress': return 'background-color: #ffc107; color: black;'
        return ''
    st.dataframe(req_df.style.applymap(style_status, subset=['Test Status']), use_container_width=True, height=500)

st.divider()

# --- 3. Defect Analysis ---
st.header("3. Defect Management & Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Defect Open vs. Close Trend")
    with st.expander("🔬 **The Method & Metrics Explained**"):
        st.markdown("""
        #### The Method
        A **Defect Burnup Chart** is a visual tool for tracking progress over time. It plots the cumulative number of defects opened (the "scope") against the cumulative number of defects closed (the "progress").
        
        #### The Metrics
        - **Total Defects Opened (Red Line)**: Represents the total number of bugs found to date. A steep, continuous rise can indicate systemic quality issues.
        - **Total Defects Closed (Green Line)**: Represents the total number of bugs fixed, verified, and closed.
        - **The Gap**: The vertical distance between the two lines represents the number of **open defects** at any point in time. The goal is for the green line to meet the red line.
        
        #### Analysis of Results
        This chart tells a story about the project's stability and the team's efficiency. In this simulation, we see a steady rate of defect discovery early on. Around the 20-day mark, the rate of defect closure (slope of the green line) increases sharply, indicating a focused effort to resolve bugs, likely in preparation for a release. The gap between the lines is narrowing, which is a positive sign of progress towards stabilization.
        """)
    defect_trend_df = generate_defect_trend_data()
    fig_burnup = go.Figure()
    fig_burnup.add_trace(go.Scatter(x=defect_trend_df['Date'], y=defect_trend_df['Opened'], mode='lines', name='Total Defects Opened', line=dict(color='red', width=3)))
    fig_burnup.add_trace(go.Scatter(x=defect_trend_df['Date'], y=defect_trend_df['Closed'], mode='lines', name='Total Defects Closed', line=dict(color='green', width=3), fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'))
    fig_burnup.update_layout(title="Defect Burnup Chart", yaxis_title="Cumulative Count of Defects", legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig_burnup, use_container_width=True)

with col2:
    st.subheader("Defect Pareto Analysis")
    with st.expander("🔬 **The Method & Metrics Explained**"):
        st.markdown("""
        #### The Method
        A **Pareto Chart** is a type of chart that contains both bars and a line graph. It is based on the **Pareto Principle (or 80/20 rule)**, which states that for many events, roughly 80% of the effects come from 20% of the causes. In software, this means a small number of modules or categories often contain the majority of the defects.
        
        #### The Metrics
        - **Defect Count (Bars)**: The absolute number of defects found in each category, sorted from highest to lowest.
        - **Cumulative Percentage (Red Line)**: The cumulative sum of the percentage of total defects for each category.
        
        #### Analysis of Results
        This chart is a powerful tool for prioritizing quality improvement efforts. The analysis clearly shows that the **UI/UX** and **Calculation Engine** categories account for the vast majority of all defects (approximately 70%, based on the cumulative line). This is a strong, data-driven insight. Instead of spreading testing resources thinly across all modules, the team should focus their efforts—such as code reviews, adding more unit tests, or targeted exploratory testing—on these two "vital few" areas to achieve the greatest impact on overall software quality.
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
