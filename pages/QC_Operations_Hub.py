# pages/QC_Operations_Hub.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_instrument_schedule_data, generate_training_data_for_heatmap

st.set_page_config(page_title="QC Operations Hub", layout="wide")
st.title("🔧 QC Operations Hub")
st.markdown("### Managing the operational readiness of equipment and personnel.")

with st.expander("🌐 Regulatory Context & Legend"):
    st.markdown("""
    This dashboard directly supports the operational management requirements of a regulated laboratory.
    - **Control of Test Equipment (21 CFR 820.72)**: The instrument schedule provides objective evidence of maintenance (PM), calibration status, and suitability for use (availability vs. OOS).
    - **Personnel & Training (21 CFR 820.25)**: The training competency heatmap and readiness charts demonstrate that personnel have the necessary training and skills to perform their assigned responsibilities correctly.
    """)

# --- 1. KPIs ---
st.header("1. Operational KPIs")
col1, col2, col3 = st.columns(3)
col1.metric("Overall Instrument Uptime", "92%", help="Percentage of scheduled time that instruments are available for use (not OOS or in unscheduled maintenance).")
col2.metric("Overdue PM/Calibrations", "1", delta="1", delta_color="inverse", help="Number of instruments currently overdue for scheduled preventative maintenance or calibration.")
col3.metric("Critical Assay Readiness", "75%", help="Lowest readiness percentage across all critical assays, based on certified personnel.")

st.divider()

# --- 2. Instrument & Equipment Management ---
st.header("2. Instrument & Equipment Management")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Instrument Schedule Timeline")
    with st.expander("🔬 **Plot Purpose & Analysis**"):
        st.markdown("""
        #### Purpose
        This Gantt chart provides a dynamic, forward-looking view of instrument utilization. It replaces a static list with an interactive schedule, allowing for efficient resource planning and identification of potential bottlenecks.
        #### Analysis
        The timeline clearly shows that **PCR-01** is currently blocked for overdue preventative maintenance, and **PCR-02** is non-operational (OOS). This indicates an immediate risk to any PCR-based workflow. Conversely, HPLC-02 is available today but scheduled for use tomorrow, allowing for proactive planning of other experiments.
        """)
    schedule_df = generate_instrument_schedule_data()
    fig_schedule = px.timeline(
        schedule_df,
        x_start="Start", x_end="Finish", y="Instrument", color="Status",
        title="Weekly Instrument Schedule & Status",
        hover_name="Details",
        color_discrete_map={
            'In Use': 'royalblue',
            'Scheduled': 'lightgrey',
            'PM Due': 'orange',
            'Out of Spec': 'red'
        }
    )
    fig_schedule.add_vline(x=pd.Timestamp.now(), line_width=2, line_dash="dash", line_color="green", annotation_text="Now")
    st.plotly_chart(fig_schedule, use_container_width=True)

with col2:
    st.subheader("Equipment Action Items")
    with st.expander("🔬 **Table Purpose**"):
        st.markdown("""
        This table is filtered to show only those instruments that require immediate attention, creating a clear, prioritized worklist for the lab manager or responsible scientist.
        """)
    action_items_df = schedule_df[schedule_df['Status'].isin(['PM Due', 'Out of Spec'])]
    st.dataframe(action_items_df[['Instrument', 'Status', 'Details']], use_container_width=True, hide_index=True)

st.divider()

# --- 3. Personnel & Training Readiness ---
st.header("3. Personnel & Training Readiness")
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Training Competency Heatmap")
    with st.expander("🔬 **Plot Purpose & Analysis**"):
        st.markdown("""
        #### Purpose
        A heatmap provides an instant visual summary of the lab's skill matrix. It is far more effective than a table of checkmarks for identifying gaps and single points of failure.
        #### Analysis
        The heatmap immediately reveals that **Peter Jones** is a potential risk, as he is not trained on any of the advanced assays (PCR or NGS). It also shows that **John Doe** is the only analyst currently in training for the critical NGS Lib Prep method, highlighting a potential future bottleneck if Jane Smith or Susan Chen are unavailable.
        """)
    training_df = generate_training_data_for_heatmap()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=training_df.values,
        x=training_df.columns,
        y=training_df.index,
        colorscale=[[0, 'lightgrey'], [0.5, '#ffc107'], [1, '#28a745']], # Not Trained, In Training, Certified
        colorbar=dict(tickvals=[0, 1, 2], ticktext=['Not Trained', 'In Training', 'Certified'])
    ))
    fig_heatmap.update_layout(title="Lab Analyst Competency Matrix")
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col2:
    st.subheader("Assay Readiness Level")
    with st.expander("🔬 **Plot Purpose & Analysis**"):
        st.markdown("""
        #### Purpose
        This bar chart aggregates the heatmap data into a high-level management metric. It answers the question: "For any given assay, what percentage of our team is fully certified to run it?"
        #### Analysis
        While readiness for HPLC and Safety is 100% of the core team, the readiness for **NGS Lib Prep** is only 50%. This is a significant risk. If the two certified analysts are unavailable, the entire NGS workflow stops. This data justifies the need for cross-training Peter Jones and completing John Doe's training as a high priority.
        """)
    readiness = (training_df == 2).mean() * 100
    readiness_df = readiness.reset_index()
    readiness_df.columns = ['Assay', 'Readiness Pct']
    
    fig_bar = px.bar(
        readiness_df,
        x='Assay', y='Readiness Pct', color='Assay',
        title='Percent of Analysts Certified per Assay',
        range_y=[0, 100]
    )
    fig_bar.add_hline(y=75, line_width=2, line_dash="dash", line_color="red", annotation_text="Readiness Target")
    st.plotly_chart(fig_bar, use_container_width=True)
