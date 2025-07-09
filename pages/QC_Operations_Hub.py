# pages/QC_Operations_Hub.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta

st.set_page_config(page_title="QC Operations Hub", layout="wide")
st.title("🔧 QC Operations Hub")
st.markdown("### Managing the operational readiness of equipment and personnel.")

# --- Instrument & Equipment Dashboard ---
st.header("Instrument & Equipment Dashboard")
inst_data = {
    'Instrument ID': ['HPLC-01', 'HPLC-02', 'PCR-01', 'PCR-02', 'Hamilton-01'],
    'Type': ['HPLC', 'HPLC', 'RT-PCR', 'RT-PCR', 'Liquid Handler'],
    'Status': ['In Use', 'Available', 'PM Due', 'OOS', 'Available'],
    'Calibration Due': [date.today() + timedelta(days=90), date.today() + timedelta(days=120), date.today() - timedelta(days=2), date.today() + timedelta(days=30), date.today() + timedelta(days=180)],
    'Assigned To': ['J. Doe', '', 'Maintenance', 'S. Scientist', '']
}
inst_df = pd.DataFrame(inst_data)

def style_status(val):
    color = 'grey'
    if val == 'Available': color = 'green'
    elif val == 'In Use': color = 'orange'
    elif val == 'PM Due' or val == 'OOS': color = 'red'
    return f'background-color: {color}; color: white;'

st.dataframe(inst_df.style.applymap(style_status, subset=['Status']), use_container_width=True)

st.divider()

# --- Training & Certification Matrix ---
st.header("Training & Certification Matrix")
training_data = {
    'Analyst': ['John Doe', 'Jane Smith', 'Peter Jones', 'Susan Chen'],
    'TM-101 (HPLC)': ['✔️', '✔️', '✔️', 'In Training'],
    'TM-201 (PCR)': ['✔️', '✔️', 'Not Trained', '✔️'],
    'TM-202 (NGS Lib Prep)': ['In Training', '✔️', 'Not Trained', '✔️'],
    'SOP-001 (Safety)': ['✔️', '✔️', '✔️', '✔️']
}
training_df = pd.DataFrame(training_data).set_index('Analyst')
st.dataframe(training_df, use_container_width=True)
