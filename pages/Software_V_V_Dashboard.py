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

col1, col2, col3 = st.columns(3)
col1.metric("Requirements Test Coverage", f"{total_tests / len(req_df) * 100:.0f}%")
col2.metric("Test Case Pass Rate", f"{test_pass_count / total_tests * 100:.0f}%")
col3.metric("Open Critical/High Defects", f"{open_critical_defects}", delta=open_critical_defects, delta_color="inverse")

st.divider()

# --- 2. V-Model and Traceability ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("SDLC V-Model")
    v_model_df = generate_v_model_data()
    fig_v = go.Figure()
    # Add connecting lines
    fig_v.add_trace(go.Scatter(x=v_model_df['x'], y=v_model_df['y'], mode='lines', line=dict(color='royalblue', width=3)))
    # Add boxes and text
    fig_v.add_trace(go.Scatter(
        x=v_model_df['x'], y=v_model_df['y'], mode='markers+text',
        marker=dict(color='aliceblue', size=80, symbol='square', line=dict(color='royalblue', width=2)),
        text=v_model_df['text'], textposition="middle center", textfont=dict(size=12)
    ))
    # Add V&V labels
    fig_v.add_annotation(x=2.5, y=3.5, text="<b>Validation<br>(Are we building the right product?)</b>", showarrow=False, font=dict(color='grey'))
    fig_v.add_annotation(x=6.5, y=3.5, text="<b>Verification<br>(Are we building the product right?)</b>", showarrow=False, font=dict(color='grey'))
    fig_v.update_layout(showlegend=False, height=500, margin=dict(l=20, r=20, b=20, t=20))
    fig_v.update_xaxes(visible=False); fig_v.update_yaxes(visible=False)
    st.plotly_chart(fig_v, use_container_width=True)

with col2:
    st.subheader("Requirements Traceability Matrix")
    def style_status(val):
        if val == 'Pass': return 'background-color: #28a745; color: white;'
        if val == 'Fail': return 'background-color: #dc3545; color: white;'
        if val == 'In Progress': return 'background-color: #ffc107; color: black;'
        return ''
    st.dataframe(req_df.style.applymap(style_status, subset=['Test Status']), use_container_width=True, height=500)

st.divider()

# --- 3. Defect Analysis ---
st.header("Defect Management & Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Defect Open vs. Close Trend")
    defect_trend_df = generate_defect_trend_data()
    fig_burnup = go.Figure()
    fig_burnup.add_trace(go.Scatter(x=defect_trend_df['Date'], y=defect_trend_df['Opened'], mode='lines', name='Total Defects Opened', line=dict(color='red', width=3)))
    fig_burnup.add_trace(go.Scatter(x=defect_trend_df['Date'], y=defect_trend_df['Closed'], mode='lines', name='Total Defects Closed', line=dict(color='green', width=3), fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'))
    fig_burnup.update_layout(title="Defect Burnup Chart", yaxis_title="Cumulative Count of Defects", legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig_burnup, use_container_width=True)

with col2:
    st.subheader("Defect Pareto Analysis")
    pareto_df = generate_defect_category_data()
    pareto_df['Cumulative %'] = (pareto_df['Count'].cumsum() / pareto_df['Count'].sum()) * 100

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    # Bar chart for counts
    fig_pareto.add_trace(go.Bar(x=pareto_df['Category'], y=pareto_df['Count'], name='Defect Count'), secondary_y=False)
    # Line chart for cumulative percentage
    fig_pareto.add_trace(go.Scatter(x=pareto_df['Category'], y=pareto_df['Cumulative %'], name='Cumulative %', line=dict(color='red')), secondary_y=True)

    fig_pareto.update_layout(title_text="Pareto Chart of Defect Categories")
    fig_pareto.update_yaxes(title_text="<b>Count</b>", secondary_y=False)
    fig_pareto.update_yaxes(title_text="<b>Cumulative Percentage (%)</b>", secondary_y=True, range=[0,101])
    st.plotly_chart(fig_pareto, use_container_width=True)
