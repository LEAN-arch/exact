# pages/QMS_CAPA_Tracker.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta

st.set_page_config(page_title="QMS & CAPA Tracker", layout="wide")
st.title("📋 QMS & CAPA Tracker")
st.markdown("### Managing formal Quality System documents and corrective/preventive actions.")

# --- CAPA Tracker ---
st.header("My Open CAPAs")
capa_data = {
    'CAPA ID': ['CAPA-00123', 'CAPA-00125', 'CAPA-00128'],
    'Source': ['Internal Audit', 'OOS: Lot 12345', 'Customer Complaint'],
    'Phase': ['Root Cause Investigation', 'Effectiveness Check', 'Implementation of Actions'],
    'Owner': ['S. Scientist', 'J. Doe', 'S. Scientist'],
    'Due Date': [date.today() + timedelta(days=14), date.today() - timedelta(days=5), date.today() + timedelta(days=30)]
}
capa_df = pd.DataFrame(capa_data)
st.dataframe(capa_df, use_container_width=True)

st.divider()

# --- Document Review Cycle ---
st.header("My Documents Due for Periodic Review")
doc_data = {
    'Document ID': ['SOP-QC-001', 'TM-101', 'WI-101-A'],
    'Title': ['General Lab Safety', 'HPLC Method for Product X', 'Instrument Startup Procedure'],
    'Current Version': [4.0, 2.0, 1.0],
    'Next Review Due': [date.today() + timedelta(days=45), date.today() + timedelta(days=90), date.today() - timedelta(days=10)]
}
doc_df = pd.DataFrame(doc_data)
st.dataframe(doc_df, use_container_width=True)
