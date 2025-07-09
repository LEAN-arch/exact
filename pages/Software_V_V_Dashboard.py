# pages/Software_V_V_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Software V&V Dashboard", layout="wide")
st.title("🖥️ Software Verification & Validation (V&V) Dashboard")
st.markdown("### Tracking the lifecycle of QC software development and validation.")

# --- V-Model Visualization ---
st.header("SDLC V-Model Progress")
st.image("https://i.imgur.com/OfiL1v5.png", caption="The V-Model illustrates how testing activities (Verification) correspond to each development stage (Validation).")

# --- Requirements Traceability Matrix ---
st.header("Requirements Traceability Matrix")
req_data = {
    'User Need (URS)': ['URS-01: Must calculate final concentration', 'URS-02: Must flag failing samples', 'URS-03: Must generate a PDF report'],
    'Functional Spec (FRS)': ['FRS-01.1', 'FRS-02.1', 'FRS-03.1, FRS-03.2'],
    'Test Case ID': ['TC-001, TC-002', 'TC-003', 'TC-004'],
    'Test Status': ['Pass', 'Pass', 'Fail']
}
req_df = pd.DataFrame(req_data)
st.dataframe(req_df, use_container_width=True)

st.divider()

# --- Software Defect Tracker ---
st.header("Open Software Defect Tracker")
bug_data = {
    'Bug ID': ['BUG-045', 'BUG-048'],
    'Description': ['PDF report fails to generate if sample name has special characters.', 'Concentration calculation uses incorrect dilution factor for re-tests.'],
    'Severity': ['High', 'Critical'],
    'Assigned To': ['Systems Dev', 'S. Scientist'],
    'Status': ['In Progress', 'Open']
}
bug_df = pd.DataFrame(bug_data)
st.dataframe(bug_df, use_container_width=True)
