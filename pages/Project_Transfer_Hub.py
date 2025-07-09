# pages/Project_Transfer_Hub.py
import streamlit as st
import pandas as pd
from utils import generate_risk_data

st.set_page_config(page_title="Project Transfer Hub", layout="wide")
st.title("🚚 Project Transfer Hub")
st.markdown("### A tactical view of project tasks, documentation, and risk mitigation.")

with st.expander("🌐 Regulatory Context & Legend"):
    st.markdown("""
    This view provides evidence that design activities were conducted according to plan and that all outputs are properly documented for the **Design History File (DHF)** and **Device Master Record (DMR)**.
    - **Design Transfer**: Visualizing this process supports **21 CFR 820.30(h)**.
    - **Document Control**: The tracker is critical for compiling the DHF (**21 CFR 820.30(j)**) and DMR (**21 CFR 820.181**).
    """)

tasks = {
    'Design': [('QC-LIMS-Script-A', 'Finalize Software Specs', 'S. Scientist', '2024-12-15')],
    'Development': [('RT-PCR-Assay-Y', 'Execute Precision Study', 'J. Doe', '2024-12-01')],
    'Validation': [('NGS-Assay-X', 'Complete Final Report', 'QA Team', '2024-11-30')],
    'Monitoring': [('HPLC-Method-Z', 'Quarterly Performance Review', 'QC Team', '2025-01-15')]
}
docs_data = {
    'Document ID': ['TP-001', 'VP-001', 'VR-001', 'TP-002', 'WI-123'],
    'Document Name': ['NGS-Assay-X Transfer Plan', 'NGS-Assay-X TMV Protocol', 'NGS-Assay-X TMV Report', 'RT-PCR-Assay-Y Transfer Plan', 'HPLC-Method-Z Work Instruction'],
    'Status': ['Approved', 'Approved', 'In Review', 'Draft', 'Approved'], 'Progress': [100, 100, 75, 25, 100], 'Link': ['#', '#', '#', '#', '#']
}
docs_df = pd.DataFrame(docs_data); risk_df = generate_risk_data()

st.header("Project Kanban Board")
cols = st.columns(4); column_map = {'Design': cols[0], 'Development': cols[1], 'Validation': cols[2], 'Monitoring': cols[3]}
for stage, col in column_map.items():
    with col:
        st.subheader(stage)
        for task in tasks.get(stage, []):
            st.markdown(f"""
            <div style="border-left: 5px solid #1f77b4; padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
                <p style="font-weight: bold; margin: 0;">{task[0]}</p>
                <p style="font-size: 0.9em; margin: 0;">{task[1]}</p><small>Owner: {task[2]} | Due: {task[3]}</small>
            </div>""", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.header("Document Control Tracker")
    st.data_editor(docs_df, column_config={"Progress": st.column_config.ProgressColumn("Status", format="%d%%", min_value=0, max_value=100), "Link": st.column_config.LinkColumn("Open Document")},
        hide_index=True, use_container_width=True)
with col2:
    st.header("Risk Mitigation Plan")
    st.dataframe(risk_df[['Risk ID', 'Description', 'Owner', 'Project', 'Risk_Score']], use_container_width=True, hide_index=True,
        column_config={"Risk_Score": st.column_config.NumberColumn("Score", format="%d 🔥")})
