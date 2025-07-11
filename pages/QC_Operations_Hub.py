# pages/QC_Operations_Hub.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_instrument_schedule_data, generate_training_data_for_heatmap, generate_reagent_lot_status_data

st.set_page_config(
    page_title="QC Operations Hub | Exact Sciences",
    layout="wide"
)

st.title("🔧 QC Operations & Readiness Hub")
st.markdown("### Managing the operational readiness of equipment, personnel, and critical reagents.")

with st.expander("🌐 Regulatory Context: Ensuring a State of Control"):
    st.markdown("""
    This dashboard provides objective evidence that the QC laboratory is maintained in a state of control, which is a foundational requirement of our Quality System.

    - **Control of Test Equipment (21 CFR 820.72)**: The instrument schedule provides a real-time log of maintenance (PM), calibration status, and suitability for use (availability vs. OOS), demonstrating that our equipment is fit for its intended purpose.
    - **Personnel & Training (21 CFR 820.25)**: The training competency heatmap and readiness charts provide documented evidence that personnel have the necessary education, background, training, and experience to correctly perform their assigned responsibilities.
    - **Device Acceptance / Incoming QC (21 CFR 820.80)**: The reagent lot management table is a critical tool for tracking the status and acceptance of incoming materials and reagents that are used in production.
    - **CLIA §493.1200 - Condition: Laboratory Director Responsibilities**: While a broader regulation, this dashboard provides the tools a Lab Director or their designee (such as a Staff Scientist) needs to ensure the lab has the resources, qualified personnel, and equipment to provide quality testing services.
    """)

# --- 1. KPIs ---
st.header("1. Operational Readiness KPIs")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Instrument Uptime", "91%", help="Percentage of scheduled time that critical instruments are available for use.")
col2.metric("Overdue PM/Calibrations", "1", delta="1", delta_color="inverse", help="Count of instruments currently overdue for scheduled maintenance.")
col3.metric("Critical Reagent Lots (<30% remaining)", "2", help="Number of in-use critical lots with low inventory.")
col4.metric("NGS Assay Readiness", "50%", help="Lowest readiness percentage across all critical assays, based on certified personnel.")

st.divider()

# --- 2. Instrument & Equipment Management ---
st.header("2. Instrument & Equipment Management")
st.caption("Real-time status of critical high-complexity instruments like NGS sequencers, liquid handlers, and PCR machines.")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Weekly Instrument Schedule & Status")
    schedule_df = generate_instrument_schedule_data()
    fig_schedule = px.timeline(
        schedule_df,
        x_start="Start", x_end="Finish", y="Instrument", color="Status",
        title="Weekly Instrument Schedule & Status",
        hover_name="Details",
        color_discrete_map={
            'In Use': '#1f77b4',
            'Scheduled': '#aec7e8',
            'Available': '#2ca02c',
            'PM Due': '#ff7f0e',
            'Out of Spec': '#d62728'
        }
    )

    today = pd.Timestamp.now()
    fig_schedule.add_shape(type="line", x0=today, y0=-0.5, x1=today, y1=len(schedule_df['Instrument'].unique()) - 0.5, line=dict(color="Red", width=2, dash="dash"))
    fig_schedule.add_annotation(x=today, y=1.05, yref='paper', text="Now", showarrow=False, font=dict(color="red", size=14))
    fig_schedule.update_yaxes(categoryorder="array", categoryarray=schedule_df['Instrument'].unique()[::-1]) # Keep order stable
    st.plotly_chart(fig_schedule, use_container_width=True)

with col2:
    st.subheader("Equipment Action Board")
    st.info("Filtered list of instruments requiring immediate attention.")
    action_items_df = schedule_df[schedule_df['Status'].isin(['PM Due', 'Out of Spec'])]
    st.dataframe(action_items_df[['Instrument', 'Status', 'Details']], use_container_width=True, hide_index=True)

    with st.expander("🔬 **Analysis**"):
        st.markdown("""
        The timeline clearly flags an immediate risk to our PCR and NGS workflows.
        - **QuantStudio-01** is overdue for preventative maintenance, making it unavailable and jeopardizing Oncotype DX® testing capacity.
        - **NovaSeq-02** is Out of Spec, halting all OncoExTra® sequencing on that instrument. An investigation (OOS-451) is underway.
        This provides a clear, prioritized worklist for the lab manager and engineering team.
        """)

st.divider()

# --- NEW: Reagent Lot Management ---
st.header("3. Reagent & Consumable Lot Management")
st.caption("Tracking the status, expiry, and quantity of critical lots to ensure process consistency and prevent downtime.")
reagent_df = generate_reagent_lot_status_data()

def style_reagent_status(df):
    style = pd.DataFrame('', index=df.index, columns=df.columns)
    style.loc[df['Status'] == 'Expired', :] = 'background-color: #d62728; color: white'
    style.loc[df['Status'] == 'On Hold', :] = 'background-color: #ff7f0e; color: white'
    style.loc[df['Quantity Remaining (%)'] < 30, 'Quantity Remaining (%)'] = 'background-color: #ffbb78'
    style.loc[pd.to_datetime(df['Expiry Date']) < pd.to_datetime(date.today() + timedelta(days=30)), 'Expiry Date'] = 'background-color: #ffbb78'
    return style

st.dataframe(reagent_df.style.apply(style_reagent_status, axis=None), use_container_width=True, hide_index=True)

with st.expander("🔬 **Analysis & Action**"):
    st.markdown("""
    This table provides critical visibility into a major source of process variability and lab downtime.
    - **Action Item 1 (Expired)**: Lot `CG-EB-2350` is expired and must be removed from inventory immediately to prevent accidental use.
    - **Action Item 2 (On Hold)**: Lot `OEX-LPK-2406` is on hold pending an investigation. This creates a supply risk for the OncoExTra assay, as there is only one other qualified lot (`OEX-LPK-2405`). The investigation must be prioritized.
    - **Proactive Action (Low Inventory)**: Lot `ODX-PK-2399` is below the 30% re-order threshold. A new order should be placed to ensure continuity of Oncotype DX testing.
    """)

st.divider()

# --- 4. Personnel & Training Readiness ---
st.header("4. Personnel & Training Readiness")
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("QC Analyst Competency Matrix")
    training_df = generate_training_data_for_heatmap()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=training_df.values,
        x=training_df.columns,
        y=training_df.index,
        colorscale=[[0, '#d62728'], [0.5, '#ff7f0e'], [1, '#2ca02c']],
        colorbar=dict(tickvals=[0, 1, 2], ticktext=['Not Trained', 'In Training', 'Certified'])
    ))
    fig_heatmap.update_layout(title="Lab Analyst Competency by Assay/System", yaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col2:
    st.subheader("Assay Team Readiness Level")
    readiness = (training_df == 2).mean() * 100
    readiness_df = readiness.reset_index()
    readiness_df.columns = ['Assay', 'Readiness Pct']

    fig_bar = px.bar(
        readiness_df,
        x='Readiness Pct', y='Assay', orientation='h', color='Assay',
        title='Percent of Team Certified per Assay',
        range_x=[0, 100]
    )
    fig_bar.add_vline(x=75, line_width=2, line_dash="dash", line_color="red", annotation_text="Target")
    fig_bar.update_layout(xaxis_title="Certification Rate (%)", yaxis_title=None)
    st.plotly_chart(fig_bar, use_container_width=True)

with st.expander("🔬 **Analysis & Action**"):
    st.markdown("""
    This analysis identifies personnel-related risks to our testing capacity.
    - **Single Point of Failure**: The heatmap reveals that the **Staff Scientist is the only person certified on the OncoExTra Bioinformatics software (SW-301)**. This is a critical risk and a bottleneck for data analysis. Cross-training Jane Smith must be a top priority.
    - **NGS Bottleneck**: Only two people are certified on the complex **OncoExTra NGS Lib Prep** method. If one is unavailable, our NGS capacity is severely limited. Completing John Doe's training is crucial.
    - **Training Plan**: This data provides the justification needed to develop a formal cross-training plan to mitigate these risks and improve overall lab flexibility and robustness.
    """)
