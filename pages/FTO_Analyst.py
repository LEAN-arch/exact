# pages/FTO_Analyst.py

import streamlit as st
import pandas as pd
import plotly.express as px
from utils import mock_patent_api, analyze_fto

st.set_page_config(page_title="FTO Analyst | Exact Sciences", layout="wide")

def _plot_fto_analysis_chart(df):
    """Creates an expert bar chart to visualize FTO analysis."""
    df['Color'] = df['Risk Level'].map({'High': '#D32F2F', 'Medium': '#F57C00', 'Low': '#388E3C'})
    fig = px.bar(
        df,
        x='Novelty Score',
        y='Aspect of Invention',
        orientation='h',
        color='Risk Level',
        text='Risk Level',
        title='Freedom-to-Operate: Novelty & IP Density',
        color_discrete_map={'High': '#D32F2F', 'Medium': '#F57C00', 'Low': '#388E3C'},
        labels={'Novelty Score': 'Novelty / White Space Score (Higher is Better)', 'Aspect of Invention': ''}
    )
    fig.update_layout(
        xaxis=dict(range=[0, 10]),
        yaxis={'categoryorder': 'total ascending'},
        legend_title_text='IP Risk Level'
    )
    fig.update_traces(textposition='inside', textfont=dict(color='white', size=14, family="Helvetica, bold"))
    return fig

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
            competitor_patents = mock_patent_api(competitors=['Arvinas', 'Kymera', 'Genentech'])
            fto_results = analyze_fto(invention_description)
            fto_df = pd.DataFrame(fto_results)

            # --- Display Visual & Tabular Results ---
            st.plotly_chart(_plot_fto_analysis_chart(fto_df), use_container_width=True)

            with st.expander("🔬 **Methodology & Significance**"):
                st.markdown("""
                #### Methodology: Deconstruction & Risk Scoring
                The FTO Analyst deconstructs the invention into its core conceptual components (e.g., scaffold, target, E3 ligase, linker). Each component is then compared against a simulated database of key competitor patent claims. The AI, acting as a patent analyst, assesses two key metrics for each component:
                1.  **Risk Level**: The likelihood that a component infringes on a broad, existing patent claim. This is a qualitative assessment (High, Medium, Low).
                2.  **Novelty Score**: A quantitative score from 1 (Not Novel) to 10 (Highly Novel) representing the degree of "white space" or opportunity for new intellectual property.

                #### The Visualization: The FTO Heatmap
                The bar chart above provides an instant strategic overview.
                - **Length of the Bar**: Represents the Novelty Score. Longer bars indicate greater "white space" and a stronger basis for a patent claim.
                - **Color of the Bar**: Represents the IP Risk Level. Green indicates a low-risk area, while red indicates a high-risk area with dense, established IP from competitors.

                #### Significance of Results: Guiding Patent & R&D Strategy
                This analysis is a critical early-stage gatekeeping tool. Its significance lies in its ability to guide strategy and efficiently allocate resources. The results clearly indicate that the **E3 Ligase Binder (RNF114)** is the most valuable and defensible aspect of this invention, possessing both a high novelty score and a low-risk profile. Conversely, the general scaffold and the choice of BTK as a target are in very crowded IP spaces.

                This insight allows the project team to:
                1.  **Focus Patent Strategy**: Draft claims that are narrowly focused on the use of the novel RNF114 ligase, rather than making broad, indefensible claims about PROTACs in general.
                2.  **Guide Further R&D**: Prioritize experiments that characterize the unique benefits of using RNF114, further strengthening the patent application's non-obviousness argument.
                3.  **Mitigate Risk**: Flag the BTK-binding moiety as a high-risk component that requires immediate, detailed legal review to ensure it does not infringe on existing chemical matter patents.
                """)

            with st.expander("Detailed Analysis & Recommendations"):
                # CORRECTED LINE: Removed the non-existent 'Recommendation' column.
                st.dataframe(fto_df[['Aspect of Invention', 'Analysis']], hide_index=True, use_container_width=True)
            
            with st.expander("Cited Prior Art (Simulated)"):
                 st.dataframe(pd.DataFrame(competitor_patents), hide_index=True, use_container_width=True)
