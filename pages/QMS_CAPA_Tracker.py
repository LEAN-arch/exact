# pages/QMS_CAPA_Tracker.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import logging
from utils import generate_capa_source_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="QMS & CAPA Tracker", layout="wide")
st.title("📋 QMS & CAPA Tracker")
st.markdown("### Managing formal Quality System documents and corrective/preventive actions.")

with st.expander("🌐 Regulatory Context & Legend (21 CFR 820)"):
    st.markdown("""
    This dashboard provides tools to manage and oversee critical components of a Quality Management System (QMS) as required by **21 CFR 820** and **ISO 13485**.
    - **CAPA Management**: Directly supports the requirements of **21 CFR 820.100 (Corrective and preventive action)** by providing a system to track the investigation, action, verification, and closure of quality issues. The Pareto chart helps in analyzing data to identify existing and potential causes of nonconforming product.
    - **Document Control**: The document review timeline supports the requirements of **21 CFR 820.40 (Document controls)**, which mandates procedures for the review and approval of all controlled documents.
    """)

# --- 1. KPIs ---
st.header("1. Quality System KPIs")
try:
    total_open_capas = 5
    overdue_items = 2  # 1 CAPA + 1 Doc Review
    avg_cycle_time_days = 45

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Open CAPAs", f"{total_open_capas}")
    col2.metric("Overdue Items (CAPA & Docs)", f"{overdue_items}", delta=overdue_items, delta_color="inverse")
    col3.metric("Avg. CAPA Cycle Time (Days)", f"{avg_cycle_time_days}")
except Exception as e:
    st.error(f"Error displaying KPIs: {e}")
    logger.error(f"KPI error: {e}")
    st.stop()

st.divider()

# --- 2. CAPA Management ---
st.header("2. CAPA Management")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Open CAPA Aging & Status")
    with st.expander("🔬 **The Method & Analysis**"):
        st.markdown("""
        #### The Method: Gantt-style Waterfall Chart
        This chart visualizes the lifecycle of each open CAPA. The length of the bar represents the time elapsed since the CAPA was opened. The color indicates the current phase of the CAPA process. This provides an at-a-glance view of both workload and urgency.
        
        #### Analysis of Results
        This visualization immediately highlights priorities. **CAPA-00125** is colored red because its due date has passed, making it the highest priority item for follow-up. We can also see that the majority of the current workload is in the investigation and implementation phases, which is typical for an active quality system.
        """)
    try:
        capa_data = {
            'CAPA ID': ['CAPA-00123', 'CAPA-00125', 'CAPA-00128', 'CAPA-00129', 'CAPA-00130'],
            'Phase': ['Root Cause Investigation', 'Effectiveness Check', 'Implementation', 'Containment', 'Root Cause Investigation'],
            'Opened Date': [date.today() - timedelta(days=20), date.today() - timedelta(days=65), date.today() - timedelta(days=5), date.today() - timedelta(days=10), date.today() - timedelta(days=15)],
            'Due Date': [date.today() + timedelta(days=10), date.today() - timedelta(days=5), date.today() + timedelta(days=25), date.today() + timedelta(days=20), date.today() + timedelta(days=15)]
        }
        capa_df = pd.DataFrame(capa_data)
        
        if capa_df.empty:
            raise ValueError("CAPA data is empty")
        capa_df['Opened Date'] = pd.to_datetime(capa_df['Opened Date'])
        capa_df['Due Date'] = pd.to_datetime(capa_df['Due Date'])
        
        capa_df['Status'] = capa_df['Due Date'].apply(lambda x: 'Overdue' if x < pd.Timestamp.now() else 'On Time')
        
        fig_gantt = px.timeline(
            capa_df, x_start="Opened Date", x_end="Due Date", y="CAPA ID", color="Phase",
            title="Open CAPA Lifecycle & Due Dates",
            custom_data=['Status']
        )
        fig_gantt.update_traces(
            hovertemplate="<b>CAPA ID</b>: %{y}<br><b>Phase</b>: %{color}<br><b>Opened</b>: %{base|%Y-%m-%d}<br><b>Due</b>: %{x|%Y-%m-%d}<br><b>Status</b>: %{customdata[0]}<extra></extra>"
        )
        fig_gantt.add_vline(x=pd.Timestamp.now(), line_width=2, line_dash="dash", line_color="red", annotation_text="Today")
        st.plotly_chart(fig_gantt, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering CAPA Gantt chart: {e}")
        logger.error(f"CAPA Gantt chart error: {e}")

with col2:
    st.subheader("CAPA Source Pareto Analysis")
    with st.expander("🔬 **The Method & Analysis**"):
        st.markdown("""
        #### The Method: Pareto Chart
        This chart helps identify the "vital few" sources that cause the majority of quality issues, based on the **Pareto Principle (80/20 rule)**. It displays the absolute count of CAPAs from each source (bars) and the cumulative percentage of total CAPAs (line).
        
        #### Analysis of Results
        This analysis is critical for **preventive action**. It clearly shows that **Out of Specification (OOS) events** and **Internal Audits** are the two biggest drivers of corrective actions, together accounting for over 60% of all CAPAs. This provides a data-driven justification to focus preventive efforts, such as process improvement projects to reduce OOS rates or additional training in areas identified by audits.
        """)
    try:
        pareto_df = generate_capa_source_data()
        if pareto_df.empty or 'Source' not in pareto_df.columns or 'Count' not in pareto_df.columns:
            raise ValueError("Pareto data is empty or missing required columns")
        pareto_df['Cumulative %'] = (pareto_df['Count'].cumsum() / pareto_df['Count'].sum()) * 100

        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pareto.add_trace(go.Bar(x=pareto_df['Source'], y=pareto_df['Count'], name='CAPA Count'), secondary_y=False)
        fig_pareto.add_trace(go.Scatter(x=pareto_df['Source'], y=pareto_df['Cumulative %'], name='Cumulative %', line=dict(color='red')), secondary_y=True)
        
        fig_pareto.update_layout(title_text="Pareto Chart of CAPA Sources", legend=dict(x=0.6, y=0.9))
        fig_pareto.update_yaxes(title_text="<b>Count</b>", secondary_y=False)
        fig_pareto.update_yaxes(title_text="<b>Cumulative Percentage (%)</b>", secondary_y=True, range=[0, 101])
        st.plotly_chart(fig_pareto, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering Pareto chart: {e}")
        logger.error(f"Pareto chart error: {e}")

st.divider()

# --- 3. Document Review Cycle ---
st.header("3. Document Review & Management")
with st.expander("🔬 **The Method & Analysis**"):
    st.markdown("""
    #### The Method: Document Timeline
    This Gantt chart provides a forward-looking view of the document lifecycle. Each bar represents a controlled document, with its start date being the last review and its end date being the next scheduled review. This visual format is more intuitive for workload planning than a simple table.
    
    #### Analysis of Results
    The timeline immediately draws attention to **WI-101-A**, which is now overdue for its periodic review, making it a compliance priority. It also allows for proactive planning, showing that a major SOP review for **SOP-QC-001** is due in approximately 45 days, allowing the owner to schedule the necessary time and resources in advance.
    """)
try:
    doc_data = {
        'Document ID': ['SOP-QC-001', 'TM-101', 'WI-101-A'],
        'Title': ['General Lab Safety', 'HPLC Method for Product X', 'Instrument Startup Procedure'],
        'Last Review': [date.today() - timedelta(days=320), date.today() - timedelta(days=275), date.today() - timedelta(days=375)],
        'Next Review Due': [date.today() + timedelta(days=45), date.today() + timedelta(days=90), date.today() - timedelta(days=10)]
    }
    doc_df = pd.DataFrame(doc_data)
    
    if doc_df.empty:
        raise ValueError("Document data is empty")
    doc_df['Last Review'] = pd.to_datetime(doc_df['Last Review'])
    doc_df['Next Review Due'] = pd.to_datetime(doc_df['Next Review Due'])

    fig_docs = px.timeline(
        doc_df, x_start="Last Review", x_end="Next Review Due", y="Document ID", color="Title",
        title="Document Periodic Review Timeline"
    )
    fig_docs.add_vline(x=pd.Timestamp.now(), line_width=2, line_dash="dash", line_color="red", annotation_text="Today")
    st.plotly_chart(fig_docs, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering document timeline: {e}")
    logger.error(f"Document timeline error: {e}")
