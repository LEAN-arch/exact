# pages/Discovery_CoPilot.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import mock_uniprot_api, mock_pubmed_api, generate_hypothesis_data

st.set_page_config(page_title="Discovery Co-Pilot | Exact Sciences", layout="wide")

def _plot_hypothesis_graph(graph_data):
    """Creates an expert-level Plotly network graph to visualize the hypothesis landscape."""
    fig = go.Figure()
    
    # Add Edges
    for edge in graph_data['edges']:
        fig.add_trace(go.Scatter(
            x=[graph_data['nodes'][edge[0]][0], graph_data['nodes'][edge[1]][0]],
            y=[graph_data['nodes'][edge[0]][1], graph_data['nodes'][edge[1]][1]],
            mode='lines', line=dict(width=2, color='rgba(0,0,0,0.2)')
        ))
        
    # Add Nodes
    for node, (x, y) in graph_data['nodes'].items():
        is_target = node == list(graph_data['nodes'].keys())[0]
        is_pathway = "Pathway" in node
        fig.add_trace(go.Scatter(
            x=[x], y=[y], text=[node.replace(" ", "<br>")], mode='markers+text',
            textposition='middle center',
            marker=dict(
                symbol='diamond' if is_target else 'square' if is_pathway else 'circle',
                color='#1A3A6D' if is_target else '#0072B2' if is_pathway else '#00B0F0',
                size=40 if is_target else 36,
                line=dict(width=2, color='white')
            ),
            textfont=dict(size=12, color='white')
        ))

    # Add Edge Annotations
    for ann in graph_data['annotations']:
        start_node = graph_data['nodes'][ann['source']]
        end_node = graph_data['nodes'][ann['target']]
        fig.add_annotation(
            x=(start_node[0] + end_node[0]) / 2, y=(start_node[1] + end_node[1]) / 2,
            text=f"<b>{ann['text']}</b>", showarrow=True, arrowhead=2, ax=start_node[0], ay=start_node[1],
            axref='x', ayref='y', xanchor='center', yanchor='middle',
            font=dict(color='#D32F2F', size=11)
        )

    fig.update_layout(
        title="Knowledge Graph: Target to Therapeutic Hypothesis",
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 7]), yaxis=dict(visible=False, range=[1, 5]),
        plot_bgcolor='rgba(240, 242, 246, 0.95)',
        margin=dict(t=50, b=0, l=0, r=0)
    )
    return fig

st.title("🧬 Discovery Co-Pilot")
st.markdown("### Synthesizing internal screening data with public knowledge to generate novel therapeutic hypotheses.")

st.info("""
**How to use this tool:** Enter the basic details of your discovery (e.g., a hit compound and its target). The Co-Pilot will automatically:
1.  Enrich your target information using public databases like UniProt.
2.  Scan recent literature from PubMed for relevant context (e.g., resistance pathways).
3.  Synthesize all data to propose high-impact, testable therapeutic hypotheses relevant to our diagnostic portfolio.
""")

# --- User Input Section ---
st.header("1. Enter Your Discovery Details")

with st.form(key='discovery_form'):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        compound = st.text_input("Compound Name", "Cmpd-X")
    with col2:
        target_name = st.text_input("Target Gene Symbol", "KRAS")
    with col3:
        target_id = st.text_input("Target UniProt ID", "P01116")
    with col4:
        activity = st.number_input("Activity (IC50, nM)", value=25.0, format="%.1f")
    
    mutation = st.text_input("Specific Mutation (if any)", "G12C", help="e.g., G12C, V600E")

    submitted = st.form_submit_button("🚀 Generate Hypotheses")

# --- AI Analysis Section ---
if submitted:
    st.header("2. AI-Powered Analysis & Proposed Research Avenues")

    with st.spinner("Synthesizing internal and external knowledge..."):
        # --- Simulate API Calls & AI analysis ---
        uniprot_data = mock_uniprot_api(target_id)
        pubmed_query = f"{target_name} {mutation}"
        pubmed_data = mock_pubmed_api(pubmed_query)
        internal_data = {"Compound": compound, "Target": f"{target_name} ({mutation})", "Activity": f"Potent Inhibitor (IC50 = {activity} nM)"}
        external_data = {"UniProt Data": uniprot_data, "Recent PubMed Titles": pubmed_data}
        hypotheses, graph_data = generate_hypothesis_data(internal_data, external_data)

        # --- Display Visual & Tabular Results ---
        col1, col2 = st.columns([1.5, 2])
        with col1:
            st.markdown("#### Knowledge Graph")
            st.plotly_chart(_plot_hypothesis_graph(graph_data), use_container_width=True)
            
        with col2:
            st.markdown("#### Proposed Research Avenues")
            st.dataframe(pd.DataFrame(hypotheses), hide_index=True, use_container_width=True)

        with st.expander("🔬 **Methodology & Significance**"):
            st.markdown("""
            #### Methodology: Knowledge Triangulation
            The Co-Pilot employs a "knowledge triangulation" strategy. It synthesizes three distinct data sources to generate its recommendations:
            1.  **Internal Data:** Your proprietary, high-confidence screening results (e.g., compound, target, potency).
            2.  **Biological Context Data:** Structured information from public databases (e.g., UniProt) to understand the target's function and pathway associations.
            3.  **Recent Scientific Literature:** Unstructured data from sources like PubMed to identify the most current trends, resistance mechanisms, and unmet needs related to the target pathway.

            The AI acts as the synthesizer, constructing a prompt that provides all this context and then asks for specific, actionable outputs.

            #### Experimental Design: From Data to Decision
            The "experiment" here is the generation of a high-quality recommendation. The **Knowledge Graph** visualization is a key output, designed to make the AI's reasoning transparent. It visually maps the connection from our internal target (`KRAS G12C`) to its core biological pathway (`MAPK/ERK`) and then to the various disease contexts identified in the literature. The annotations on the connecting lines provide the specific rationale (e.g., "Resistance to Osimertinib," "Cologuard® Dx Link") that forms the basis of each hypothesis.

            #### Significance of Results: De-risking R&D
            The primary value of this tool is to **de-risk and accelerate the next phase of research**. Instead of a broad, unfocused screening campaign, the Co-Pilot provides a ranked list of the most promising, data-backed strategies. Each hypothesis is paired with a specific, cost-effective "Next Experiment" designed for rapid validation or invalidation. This allows the R&D team to make faster, more confident decisions, efficiently allocating resources to the projects with the highest probability of clinical and commercial success.
            """)
