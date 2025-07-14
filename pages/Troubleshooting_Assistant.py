# pages/Troubleshooting_Assistant.py

import streamlit as st
from utils import mock_get_sop, mock_get_reagent_info, mock_get_instrument_log, troubleshoot_experiment

st.set_page_config(page_title="Troubleshooting Assistant | Exact Sciences", layout="wide")

st.title("🧑‍🔬 Experimental Troubleshooting Assistant")
st.markdown("### Diagnose failed experiments by cross-referencing your protocol with internal data sources.")

st.info("""
**How to use this tool:** Enter the details of your failed experiment below. The assistant will check your protocol against official SOPs, look up information on the specific reagent lot you used, check instrument logs, and provide a ranked list of the most likely causes of failure.
""")

# --- User Input Section ---
st.header("1. Describe Your Failed Experiment")

with st.form("troubleshooting_form"):
    st.subheader("User-Submitted Protocol")
    c1, c2 = st.columns(2)
    with c1:
        experiment_type = st.selectbox("Experiment Type", ["Western Blot"])
        antibody_lot = st.text_input("Antibody Lot #", "ABC-123")
        gel_type = st.text_input("Gel Type", "10% Tris-Glycine")
    with c2:
        incubation = st.text_input("Antibody Incubation", "1 hour at Room Temp")
        transfer_method = st.text_input("Transfer Method", "Semi-dry transfer for 30 mins")
        instrument_id = st.text_input("Instrument ID (e.g., Blotter #)", "Blotter #2")

    submitted = st.form_submit_button("🔬 Diagnose Experiment")

# --- AI Analysis Section ---
if submitted:
    st.header("2. AI-Powered Troubleshooting Report")
    with st.spinner("Analyzing protocol and cross-referencing internal data..."):
        # --- Simulate gathering context ---
        user_protocol = {
            "Experiment": experiment_type, "Antibody Lot": antibody_lot,
            "Incubation": incubation, "Gel": gel_type,
            "Transfer": transfer_method, "Instrument": instrument_id
        }
        
        # --- Simulate AI analysis call ---
        troubleshooting_report = troubleshoot_experiment(user_protocol)

        # --- Display Results ---
        st.subheader("Analysis Summary")
        st.write("The following potential root causes have been identified by comparing your submitted protocol against internal system data. They are ranked from most to least likely.")

        for item in troubleshooting_report:
            with st.container(border=True):
                st.metric(f"Rank #{item['Rank']}: {item['Most Likely Root Cause']}", "")
                st.markdown(f"**Evidence:** {item['Evidence']}")
                st.success(f"**Recommended Corrective Action:** {item['Corrective Action']}")
