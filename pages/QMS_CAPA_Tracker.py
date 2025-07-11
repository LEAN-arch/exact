# pages/QMS_CAPA_Tracker.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils import generate_capa_source_data

st.set_page_config(
    page_title="QMS & CAPA Tracker | Exact Sciences",
    layout="wide"
)

st.title("📋 QMS & Investigation Tracker")
st.markdown("### Managing formal CAPAs, document controls, and active non-conformance investigations.")

with st.expander("🌐 Regulatory Context: The Foundation of Quality Assurance"):
    st.markdown("""
    This dashboard provides tools to manage and oversee critical components of our Quality Management System (QMS) as required by **21 CFR 820** and **ISO 13485**. As a Staff Scientist, leading and documenting investigations is a core responsibility.

    - **Corrective and Preventive Action (CAPA) (21 CFR 820.100)**: This regulation requires a formal system for investigating and correcting quality issues to prevent their recurrence. This page provides tools to track CAPA lifecycles and analyze their sources, fulfilling the mandate to "analyze...data to identify existing and potential causes of nonconforming product."
    - **Nonconformance (21 CFR 820.90)**: This regulation governs the control of product that does not meet specifications. The OOS tracker on this page is our primary tool for managing the identification, documentation, evaluation, and disposition of these events.
    - **Document Control (21 CFR 820.40)**: Requires procedures for the review and approval of all controlled documents. The document review timeline helps manage this process proactively.
    """)

# --- 1. KPIs ---
st.header("1. Quality System Health KPIs")
total_open_capas = 4
open_oos_investigations = 3
overdue_items = 2 # 1 CAPA + 1 Doc Review
avg_cycle_time_days = 52

col1, col2, col3, col4 = st.columns(4)
col1.metric("Open CAPAs", f"{total_open_capas}")
col2.metric("Active OOS Investigations", f"{open_oos_investigations}")
col3.metric("Overdue Items (All)", f"{overdue_items}", delta=f"{overdue_items} Overdue", delta_color="inverse")
col4.metric("Avg. CAPA Cycle Time (Days)", f"{avg_cycle_time_days}")

st.divider()

# --- 2. Active Non-Conformance & OOS Investigation Tracker ---
st.header("2. Active Non-Conformance & OOS Investigation Tracker")
st.caption("Real-time status of technical investigations led by Operations scientists *before* they escalate to a full CAPA.")
try:
    oos_data = {
        'OOS ID': ['OOS-24-101', 'OOS-24-102', 'OOS-24-103'],
        'Product/Assay': ['OncoExTra®', 'Cologuard®', 'Oncotype DX®'],
        'Description': ['Positive control failed library yield spec in NGS run.', 'Reagent manufacturing lot failed final purity spec.', 'Positive control Ct value shifted >2 SD on Levey-Jennings.'],
        'Lead Investigator': ['S. Scientist', 'J. Doe', 'S. Scientist'],
        'Investigation Phase': ['Hypothesis Testing', 'Data Analysis', 'Impact Assessment'],
        'Age (Days)': [5, 12, 2],
        'Escalate to CAPA?': ['Pending', 'Yes', 'No (Monitor Trend)']
    }
    oos_df = pd.DataFrame(oos_data)
    st.dataframe(oos_df, use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"Error rendering OOS Investigation tracker: {e}")

st.divider()

# --- 3. CAPA Management ---
st.header("3. Formal CAPA Management")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Open CAPA Aging & Status")
    with st.expander("🔬 **Purpose & Analysis**"):
        st.markdown("""
        This chart visualizes the lifecycle of each formal CAPA, providing an at-a-glance view of workload and priorities. A CAPA is typically initiated when a non-conformance is found to be systemic or poses a significant product risk.
        """)
    try:
        capa_data = {
            'CAPA ID': ['CAPA-24-015', 'CAPA-24-018', 'CAPA-24-019', 'CAPA-24-020'],
            'Title': ['NGS sequence quality degradation on NovaSeq-02', 'Bioinformatics pipeline calculation error at LIMS interface', 'Recurring OOS for Cologuard reagent lot purity', 'Supplier non-conformance for critical plasticware'],
            'Phase': ['Root Cause Investigation', 'Effectiveness Check', 'Implementation', 'Containment'],
            'Opened Date': [pd.to_datetime(date.today() - timedelta(days=20)), pd.to_datetime(date.today() - timedelta(days=65)), pd.to_datetime(date.today() - timedelta(days=5)), pd.to_datetime(date.today() - timedelta(days=10))],
            'Due Date': [pd.to_datetime(date.today() + timedelta(days=10)), pd.to_datetime(date.today() - timedelta(days=5)), pd.to_datetime(date.today() + timedelta(days=25)), pd.to_datetime(date.today() + timedelta(days=20))]
        }
        capa_df = pd.DataFrame(capa_data)
        fig_gantt = px.timeline(
            capa_df, x_start="Opened Date", x_end="Due Date", y="CAPA ID", color="Phase",
            title="Open CAPA Lifecycle & Due Dates", hover_name="Title"
        )
        today = pd.Timestamp.now()
        fig_gantt.add_shape(type="line", x0=today, y0=-0.5, x1=today, y1=len(capa_df['CAPA ID'])-0.5, line=dict(color="Red", width=2, dash="dash"))
        fig_gantt.add_annotation(x=today, y=1.05, yref='paper', text="Today", showarrow=False, font=dict(color="red"))
        st.plotly_chart(fig_gantt, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering CAPA Gantt chart: {e}")

with col2:
    st.subheader("CAPA Source Pareto Analysis")
    with st.expander("🔬 **Purpose & Analysis**"):
        st.markdown("""
        This chart identifies the "vital few" sources causing the majority of our quality issues, based on the **Pareto Principle (80/20 rule)**. This is critical for driving **preventive** actions.
        """)
    try:
        pareto_df = generate_capa_source_data()
        pareto_df['Cumulative %'] = (pareto_df['Count'].cumsum() / pareto_df['Count'].sum()) * 100
        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pareto.add_trace(go.Bar(x=pareto_df['Source'], y=pareto_df['Count'], name='CAPA Count'), secondary_y=False)
        fig_pareto.add_trace(go.Scatter(x=pareto_df['Source'], y=pareto_df['Cumulative %'], name='Cumulative %', line=dict(color='red')), secondary_y=True)
        fig_pareto.update_layout(title_text="Pareto Chart of CAPA Sources")
        fig_pareto.update_yaxes(title_text="<b>Count</b>", secondary_y=False)
        fig_pareto.update_yaxes(title_text="<b>Cumulative Percentage (%)</b>", secondary_y=True, range=[0, 101])
        st.plotly_chart(fig_pareto, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering Pareto chart: {e}")

st.divider()

# --- 4. Document Review Cycle ---
st.header("4. Document Periodic Review Management")
st.caption("Proactively managing the review cycle for controlled documents like SOPs and Test Methods.")
try:
    doc_data = {
        'Document ID': ['SOP-QC-001', 'TM-ODX-001', 'WI-CG-101-A', 'SOP-LAB-005'],
        'Title': ['General QC Laboratory Practices', 'Oncotype DX RT-PCR Test Method', 'Cologuard Reagent Prep Work Instruction', 'Handling of Out-of-Specification Results'],
        'Owner': ['QC Manager', 'S. Scientist', 'J. Doe', 'QA'],
        'Last Review': [pd.to_datetime(date.today() - timedelta(days=320)), pd.to_datetime(date.today() - timedelta(days=275)), pd.to_datetime(date.today() - timedelta(days=375)), pd.to_datetime(date.today() - timedelta(days=700))],
        'Next Review Due': [pd.to_datetime(date.today() + timedelta(days=45)), pd.to_datetime(date.today() + timedelta(days=90)), pd.to_datetime(date.today() - timedelta(days=10)), pd.to_datetime(date.today() + timedelta(days=30))]
    }
    doc_df = pd.DataFrame(doc_data)
    fig_docs = px.timeline(
        doc_df, x_start="Last Review", x_end="Next Review Due", y="Document ID", color="Owner",
        title="Document Periodic Review Timeline", hover_name="Title"
    )
    today = pd.Timestamp.now()
    fig_docs.add_shape(type="line", x0=today, y0=-0.5, x1=today, y1=len(doc_df['Document ID'])-0.5, line=dict(color="Red", width=2, dash="dash"))
    fig_docs.add_annotation(x=today, y=1.05, yref='paper', text="Today", showarrow=False, font=dict(color="red"))
    st.plotly_chart(fig_docs, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering document timeline: {e}")
