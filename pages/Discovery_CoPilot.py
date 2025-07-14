# pages/Discovery_CoPilot.py

import streamlit as st
import pandas as pd
from utils import mock_uniprot_api, mock_pubmed_api, generate_hypothesis

st.set_page_config(page_title="Discovery Co-Pilot | Exact Sciences", layout="wide")

st.title("🧬 Discovery Co-Pilot")
st.markdown("### Synthesizing internal data with public knowledge to generate novel therapeutic hypotheses.")

st.info("""
**How to use this tool:** Enter the basic details of your discovery (e.g., a hit compound and its target). The Co-Pilot will automatically:
1.  Enrich your target information using public databases like UniProt.
2.  Scan recent literature from PubMed for relevant context.
3.  Synthesize all data to propose high-impact, testable therapeutic hypotheses for your compound.
""")

# --- User Input Section ---
st.header("1. Enter Your Discovery Details")

with st.form(key='discovery_form'):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        compound = st.text_input("Compound Name", "Cmpd-42")
    with col2:
        target_name = st.text_input("Target Name", "Target Kinase Z")
    with col3:
        target_id = st.text_input("Target UniProt ID", "P12345")
    with col4:
        activity = st.number_input("Activity (IC50, nM)", value=15.0, format="%.1f")

    submitted = st.form_submit_button("🚀 Generate Hypotheses")

# --- AI Analysis Section ---
if submitted:
    st.header("2. AI-Generated Therapeutic Hypotheses")

    with st.spinner("Synthesizing internal and external knowledge..."):
        # --- Simulate API Calls ---
        uniprot_data = mock_uniprot_api(target_id)
        pubmed_data = mock_pubmed_api(target_name)
        internal_data = {
            "Compound": compound,
            "Target": target_name,
            "Activity": f"Potent Inhibitor (IC50 = {activity} nM)"
        }
        external_data = {
            "UniProt Data": uniprot_data,
            "Recent PubMed Titles": pubmed_data
        }

        # --- Simulate LLM Call ---
        hypotheses = generate_hypothesis(internal_data, external_data)

        # --- Display Results ---
        st.subheader("Synthesized Context")
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"""
            **Internal Data:**
            - **Compound:** {internal_data['Compound']}
            - **Target:** {internal_data['Target']}
            - **Activity:** {internal_data['Activity']}
            """)
        with c2:
            st.warning(f"""
            **External Data (Simulated):**
            - **UniProt Function:** {uniprot_data.get('Function', 'N/A')}
            - **PubMed Hit 1:** {pubmed_data[0]}
            - **PubMed Hit 2:** {pubmed_data[1]}
            """)

        st.subheader("Proposed Research Avenues")
        st.table(pd.DataFrame(hypotheses))
