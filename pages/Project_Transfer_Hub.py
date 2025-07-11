# pages/Project_Transfer_Hub.py
import streamlit as st
import pandas as pd
from utils import generate_risk_data

st.set_page_config(
    page_title="Project Transfer Hub | Exact Sciences",
    layout="wide"
)

st.title("🚚 Assay & Software Transfer Hub")
st.markdown("### A tactical view of project tasks, documentation, risk mitigation, and training for successful transfer to Operations.")

with st.expander("🌐 Regulatory Context: Design Transfer & The DHF"):
    st.markdown("""
    This hub provides a centralized, auditable view of the entire design transfer process, ensuring all activities are controlled, documented, and executed according to plan. This is critical for compliance with our Quality System.

    - **Design Transfer (21 CFR 820.30(h))**: This regulation requires that "each manufacturer shall establish and maintain procedures to ensure that the device design is correctly translated into production specifications." This hub is a key tool for managing that translation.
    - **Design History File (DHF) (21 CFR 820.30(j))**: The DHF is a compilation of records which describes the design history of a finished device. The document tracker and sign-off matrix on this page are essential for compiling and demonstrating the completeness of the DHF.
    - **Device Master Record (DMR) (21 CFR 820.181)**: The outputs of this transfer process, such as approved Work Instructions (WIs) and final specifications, will become part of the DMR, which contains all the information and specifications required to produce the device.
    - **Personnel (21 CFR 820.25)**: The training checklist provides objective evidence that personnel have the necessary training and skills to perform the newly transferred test methods correctly.
    """)

# --- Data Generation ---
tasks = {
    'Design': [
        ('OncoExTra® Lib Prep Automation', 'Finalize User Requirements', 'S. Scientist', 'Due: in 10 days'),
        ('Cologuard® v3.0 QC Software', 'Define software architecture', 'Bioinformatics', 'Due: in 25 days')
    ],
    'Development': [
        ('Oncotype DX® QC Pipeline v2.1', 'Develop Python analysis script', 'Systems Dev', 'Due: in 5 days'),
        ('Cologuard® Reagent Qualification (HPLC)', 'Develop and characterize method', 'J. Doe', 'Due: in 15 days')
    ],
    'Validation': [
        ('Oncotype DX® QC Pipeline v2.1', 'Execute Software V&V Protocol', 'S. Scientist / QE', 'Due: in 30 days'),
        ('Cologuard® Reagent Qualification (HPLC)', 'Execute TMV Protocol', 'J. Doe / QC', 'Due: in 45 days')
    ],
    'Transfer & Training': [
        ('Oncotype DX® QC Pipeline v2.1', 'Train QC analysts on new software', 'S. Scientist', 'Due: in 50 days'),
        ('Cologuard® Reagent Qualification (HPLC)', 'Finalize Work Instruction', 'J. Doe', 'Due: in 60 days')
    ]
}
docs_data = {
    'Document ID': ['TP-OEX-001', 'VP-ODX-SW-002', 'VR-ODX-SW-002', 'WI-CG-HPLC-101'],
    'Document Name': [
        'OncoExTra Lib Prep Automation Transfer Plan',
        'Oncotype DX QC Pipeline v2.1 V&V Protocol',
        'Oncotype DX QC Pipeline v2.1 V&V Report',
        'Work Instruction for Cologuard Reagent Qualification by HPLC'
    ],
    'Project': ['OncoExTra Lib Prep', 'Oncotype DX QC', 'Oncotype DX QC', 'Cologuard HPLC'],
    'Status': ['Approved', 'In Review', 'Draft', 'Draft'],
    'Progress': [100, 80, 25, 15],
    'Link': ['#', '#', '#', '#']
}
docs_df = pd.DataFrame(docs_data)
risk_df = generate_risk_data()

st.header("Project Phase Kanban Board")
st.caption("Tracking high-level tasks through the transfer lifecycle.")
cols = st.columns(len(tasks))
column_map = dict(zip(tasks.keys(), cols))

for stage, col in column_map.items():
    with col:
        st.subheader(stage)
        for task in tasks.get(stage, []):
            st.info(f"**{task[0]}**\n\n{task[1]}\n\n*Owner: {task[2]} | {task[3]}*")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.header("DHF Document Control Tracker")
    st.caption("Status of critical validation and transfer documents.")
    st.data_editor(
        docs_df,
        column_config={
            "Progress": st.column_config.ProgressColumn("Status", format="%d%%", min_value=0, max_value=100),
            "Link": st.column_config.LinkColumn("Open Document", disabled=True)
        },
        hide_index=True,
        use_container_width=True
    )
with col2:
    st.header("Active Risk Mitigation Plan")
    st.caption("Top project and product risks from the ISO 14971 risk analysis.")
    st.dataframe(
        risk_df[['Risk ID', 'Description', 'Owner', 'Risk_Score']],
        use_container_width=True,
        hide_index=True,
        column_config={"Risk_Score": st.column_config.NumberColumn("Score", format="%d 🔥")}
    )

st.divider()

# --- NEW: Training and Knowledge Transfer Checklist ---
st.header("Training & Knowledge Transfer Checklist")
st.caption("Ensuring QC team readiness and documenting knowledge transfer as per 21 CFR 820.25.")

training_data = {
    'Transfer Item': [
        "**Oncotype DX® QC Pipeline v2.1**", "Oncotype DX® QC Pipeline v2.1", "Oncotype DX® QC Pipeline v2.1",
        "**Cologuard® HPLC Method**", "Cologuard® HPLC Method", "Cologuard® HPLC Method", "Cologuard® HPLC Method"
    ],
    'Knowledge Area': [
        'Software: User Interface & Operation',
        'Software: Interpretation of Results & Flags',
        'Troubleshooting: Common Error Codes',
        'Assay: Theory & Principles',
        'Assay: Execution of Work Instruction',
        'Instrumentation: HPLC Startup & Shutdown',
        'Troubleshooting: Common Chromatographic Issues'
    ],
    'Lead Trainer': ['S. Scientist', 'S. Scientist', 'S. Scientist', 'J. Doe', 'J. Doe', 'QC Sr. Analyst', 'J. Doe'],
    'Status': ['Complete', 'Complete', 'In Progress', 'Complete', 'In Progress', 'Not Started', 'Not Started'],
    'Completion Date': ['2024-11-15', '2024-11-15', '-', '2024-11-20', '-', '-', '-']
}
training_df = pd.DataFrame(training_data)

st.data_editor(
    training_df,
    column_config={
        "Status": st.column_config.SelectboxColumn(
            "Status",
            options=["Not Started", "In Progress", "Complete"],
            required=True,
        )
    },
    use_container_width=True,
    hide_index=True
)

st.divider()

st.header("Stakeholder Sign-off Matrix")
st.caption("Tracking key document approvals across functional teams to ensure cross-functional alignment before release.")
signoff_data = {
    'Document': [
        "User Requirements Specification (URS)",
        "Software V&V Plan",
        "Test Method Validation (TMV) Plan",
        "TMV / V&V Final Report",
        "Transfer to Operations Plan"
    ],
    'R&D/Dev': ['✔️ Approved', '✔️ Approved', '✔️ Approved', 'In Review', 'N/A'],
    'Bioinformatics': ['✔️ Approved', '✔️ Approved', 'N/A', 'In Review', '✔️ Approved'],
    'Systems Dev': ['✔️ Approved', '✔️ Approved', 'N/A', 'In Review', '✔️ Approved'],
    'Quality Engineering': ['✔️ Approved', '✔️ Approved', '✔️ Approved', 'In Review', 'In Review'],
    'QC Operations': ['✔️ Approved', 'In Review', '✔️ Approved', 'Pending', 'In Review'],
    'MSAT': ['N/A', 'N/A', '✔️ Approved', 'Pending', '✔️ Approved']
}
signoff_df = pd.DataFrame(signoff_data)

def style_signoff(val):
    if '✔️' in str(val):
        return 'color: green; font-weight: bold;'
    elif 'Review' in str(val):
        return 'color: orange;'
    elif 'Pending' in str(val):
        return 'color: red;'
    return ''

# Updated from .applymap to .map to resolve deprecation warning
st.dataframe(signoff_df.style.map(style_signoff), use_container_width=True)
