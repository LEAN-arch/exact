# pages/Troubleshooting_Assistant.py

import streamlit as st
import pandas as pd
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
    st.subheader("Protocol & Run Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        product_line = st.selectbox("Product Line / Assay", ["OncoExTra®", "Cologuard®", "Oncotype DX®"])
        library_kit_lot = st.text_input("Library Prep Kit Lot #", "LPK-23-9981")
    with c2:
        sequencer_id = st.selectbox("Sequencer ID", ["NovaSeq-01", "NovaSeq-02", "NextSeq-550"])
        dna_input = st.number_input("DNA Input (ng)", value=50)
    with c3:
        cleanup_method = st.selectbox("Bead Cleanup Method", ["Standard SPRI", "Double-sided SPRI"])
        
    failure_description = st.text_area(
        "Describe the observed failure",
        "Run failed sequencing QC. FastQC report shows very low average Q30 scores (<80%) and a large peak in the adapter content module."
    )

    submitted = st.form_submit_button("🔬 Diagnose Experiment")

# --- AI Analysis Section ---
if submitted:
    st.header("2. AI-Powered Troubleshooting Report")
    with st.spinner("Analyzing protocol and cross-referencing internal data..."):
        # --- Simulate gathering context and AI analysis ---
        user_protocol = {"DNA Input": dna_input, "Cleanup Method": cleanup_method, "Library Kit Lot": library_kit_lot}
        report, comparison_data = troubleshoot_experiment(user_protocol)

        # --- Display Visual Deviation Dashboard ---
        st.subheader("Deviation Dashboard: User Protocol vs. System of Record")
        st.write("This dashboard highlights critical deviations from validated procedures and known reagent issues.")
        cols = st.columns(len(comparison_data))
        for i, item in enumerate(comparison_data):
            with cols[i]:
                if item['Deviation']:
                    st.error(f"**{item['Parameter']}**", icon="❗️")
                    st.write(f"**Your Protocol:** {item['User Protocol']}")
                    st.write(f"**SOP/System Data:** {item['SOP Requirement']}")
                else:
                    st.success(f"**{item['Parameter']}**", icon="✅")
                    st.write(f"**Your Protocol:** {item['User Protocol']}")
                    st.write(f"**SOP/System Data:** {item['SOP Requirement']}")

        st.subheader("Ranked Root Cause Analysis")
        st.write("Based on the deviations above, here is a ranked analysis of likely root causes for your failed experiment:")

        for item in report:
            with st.container(border=True):
                st.metric(f"Rank #{item['Rank']}: {item['Most Likely Root Cause']}", "")
                st.markdown(f"**Evidence:** {item['Evidence']}")
                st.success(f"**Recommended Corrective Action:** {item['Corrective Action']}")
        
        with st.expander("🔬 **Methodology & Significance**"):
            st.markdown("""
            #### Methodology: Differential Diagnosis via Data Integration
            This tool performs an automated differential diagnosis for failed experiments. Its power comes from integrating and comparing data from three distinct internal sources:
            1.  **Your Submitted Protocol**: The specific steps you took for the failed run.
            2.  **The System of Record (SOPs)**: The validated, official procedure for this assay from our document control system.
            3.  **Real-time System State**: Data from other hubs, including known issues with specific reagent lots (from the Reagent Hub) and the maintenance/error status of the instrument used (from the Operations Hub).
            
            The AI cross-references these sources to identify deviations and known issues, then ranks them based on their likely impact on the observed failure mode (e.g., high adapter-dimer content).

            #### The Deviation Dashboard: Visualizing the Gap
            The "Deviation Dashboard" is the key visualization. It provides an immediate, at-a-glance summary of where your protocol diverged from the validated state or known good conditions. Red "❗️" cards instantly draw the scientist's attention to the most critical discrepancies, while green "✅" cards confirm which parts of the protocol were likely performed correctly, saving time by ruling out potential causes.

            #### Significance of Results: Accelerating OOS Investigations
            The significance of this tool is a dramatic reduction in the time and resources required to resolve an Out-of-Specification (OOS) or non-conforming laboratory result. Instead of a traditional, unfocused investigation that might involve randomly re-running experiments with one variable changed at a time, this tool provides an **immediate, evidence-based, and prioritized action plan**.

            For this specific failure (low Q30, high adapter content), the analysis points directly to a known problematic reagent lot and two significant deviations in the library preparation protocol. A scientist can now proceed with high confidence by taking the recommended corrective actions, potentially resolving a multi-day investigation in a matter of hours. This directly translates to increased R&D velocity and reduced operational downtime.
            """)
