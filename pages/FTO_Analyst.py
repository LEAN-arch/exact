# pages/FTO_Analyst.py

import streamlit as st
import pandas as pd
from utils import mock_patent_api, analyze_fto

st.set_page_config(page_title="FTO Analyst | Exact Sciences", layout="wide")

st.title("⚖️ \"Freedom to Operate\" (FTO) AI Analyst")
st.markdown("### A first-pass analysis of the patent landscape for novel discoveries.")

st.warning("""
**Disclaimer:** This tool is for informational and strategic planning purposes only and does **not** constitute legal advice. A formal FTO analysis must be conducted by qualified patent counsel.
""")

# --- User Input ---
st.header("1. Describe Your Invention")
invention_description = st.text_area(
    "Provide a detailed description of your molecule, method, or technology.",
    height=200,
    value="A hetero-bifunctional PROTAC molecule comprising a known BTK-binding moiety, a novel E3 ligase-binding moiety targeting the RNF114 ligase, and a polyethylene glycol (PEG) linker of 5-8 units in length."
)

if st.button("Analyze FTO Landscape"):
    if not invention_description:
        st.error("Please provide a description of your invention.")
    else:
        st.header("2. AI-Powered FTO Analysis")
        with st.spinner("Scanning patent databases and analyzing claims..."):
            # --- Simulate API calls and AI Analysis ---
            # In a real app, the invention_description would be parsed for keywords
            competitor_patents = mock_patent_api(competitors=['Arvinas', 'Kymera', 'Novartis'])
            fto_results = analyze_fto(invention_description)
            fto_df = pd.DataFrame(fto_results)

            # --- Display Results ---
            st.subheader("Prior Art Context (Simulated)")
            st.dataframe(pd.DataFrame(competitor_patents), hide_index=True, use_container_width=True)

            st.subheader("Novelty & Risk Assessment")

            def style_risk(val):
                color = 'red' if val == 'High' else 'orange' if val == 'Medium' else 'green'
                return f'color: {color}; font-weight: bold;'

            st.dataframe(
                fto_df.style.map(style_risk, subset=['Risk Level']),
                hide_index=True,
                use_container_width=True
            )
