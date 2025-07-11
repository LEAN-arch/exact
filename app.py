# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import generate_project_data, generate_risk_data
from datetime import date

# --- Page Configuration ---
st.set_page_config(
    page_title="Staff Scientist Command Center | Exact Sciences",
    page_icon="🔬",
    layout="wide"
)

# --- Data Loading ---
# These functions are now adapted in utils.py to generate Exact Sciences-specific data
projects_df = generate_project_data()
risks_df = generate_risk_data()

# --- Page Title and Header ---
st.title("🔬 Staff Scientist Command Center | Exact Sciences")
st.markdown("### A strategic dashboard for leading QC software and assay transfer for Cologuard®, Oncotype DX®, and NGS platforms.")

# --- KPIs ---
st.header("Executive Summary: Assay Transfer & Risk Posture")
total_projects = len(projects_df)
active_projects = projects_df[projects_df['Overall Status'] != 'Complete'].shape[0]
vv_phase_projects = projects_df[projects_df['Current Phase'] == 'Validation'].shape[0]
high_risk_score_items = risks_df[risks_df['Risk_Score'] >= 8].shape[0] # Adjusted threshold for higher impact

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Assay Transfers", f"{active_projects}")
col2.metric("Assays in V&V Phase", f"{vv_phase_projects}")
col3.metric("High-Impact Risks (>8)", f"{high_risk_score_items}", delta=f"{high_risk_score_items} High", delta_color="inverse")

# Calculate average duration for non-completed projects
active_projects_df = projects_df[projects_df['Overall Status'] != 'Complete']
if not active_projects_df.empty:
    avg_duration = (pd.to_datetime(active_projects_df['Due Date']) - pd.to_datetime(active_projects_df['Start Date'])).dt.days.mean()
    col4.metric("Avg. Transfer Duration (Days)", f"{avg_duration:.1f}")
else:
    col4.metric("Avg. Transfer Duration (Days)", "N/A")


st.divider()

# --- Main Content Area ---
col1, col2 = st.columns((2, 1.2))

with col1:
    st.header("Assay Transfer Portfolio: Gantt Chart")
    st.caption("Visualizing the execution of design transfer activities for all active and planned QC methods.")
    fig = px.timeline(
        projects_df,
        x_start="Start Date",
        x_end="Due Date",
        y="Project/Assay",
        color="Current Phase",
        title="Project Timelines by Phase",
        hover_name="Project/Assay",
        hover_data={
            "Project Lead": True,
            "Overall Status": True,
            "Product Line": True,  # Added custom data for more context
            "Start Date": "|%B %d, %Y",
            "Due Date": "|%B %d, %Y",
        },
        color_discrete_map={ # Custom colors for better visual distinction
            'Design': '#1f77b4',
            'Development': '#ff7f0e',
            'Validation': '#2ca02c',
            'Monitoring': '#9467bd',
            'On Hold': '#d62728'
        }
    )
    fig.update_yaxes(categoryorder="total ascending", title=None)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Project Risk Matrix (ISO 14971)")
    st.caption("Prioritizing risks to product quality and project timelines based on severity and probability.")
    fig_risk = px.scatter(
        risks_df, x="Prob_Score", y="Impact_Score", size="Risk_Score", color="Risk_Score",
        color_continuous_scale=px.colors.sequential.Reds, hover_name="Description",
        hover_data=["Project", "Owner", "Mitigation"], size_max=40, title="Impact vs. Probability"
    )
    fig_risk.update_layout(
        xaxis=dict(tickvals=[1, 2, 3, 4, 5], ticktext=['Very Low', 'Low', 'Medium', 'High', 'Very High'], title='Probability of Occurrence'),
        yaxis=dict(tickvals=[1, 2, 3, 4, 5], ticktext=['Negligible', 'Minor', 'Moderate', 'Serious', 'Critical'], title='Impact on Product/Project'),
        coloraxis_showscale=False
    )
    # Add risk zones for clarity
    fig_risk.add_shape(type="rect", xref="x", yref="y", x0=3.5, y0=3.5, x1=5.5, y1=5.5, fillcolor="rgba(255, 0, 0, 0.2)", layer="below", line_width=0)
    fig_risk.add_annotation(x=4.5, y=4.5, text="High Risk Zone", showarrow=False, font=dict(color="red", size=14, family="Arial, bold"))
    st.plotly_chart(fig_risk, use_container_width=True)

st.header("Assay Transfer Portfolio: Details")
st.dataframe(projects_df, use_container_width=True, hide_index=True)

# --- REGULATORY LEGEND ---
st.divider()
with st.expander("🌐 Regulatory Context & Dashboard Purpose"):
    st.markdown("""
    As a Staff Scientist in Operations, my primary responsibility is to ensure that the analytical and software methods used to release our products are robust, reliable, and compliant. This command center is an essential tool for managing that responsibility, providing objective evidence for several key regulatory frameworks that govern our work at Exact Sciences.

    #### **How This Dashboard Supports Compliance:**

    - **Design Controls & Transfer (21 CFR 820.30):**
        - The **Gantt Chart** and **Portfolio Details** directly support **Design and Development Planning (820.30(b))** by providing a clear, dynamic plan for all assay transfer activities.
        - The **Project Transfer Hub** page provides a centralized location for tracking tasks and documentation, which is crucial for a successful **Design Transfer (820.30(h))** and for compiling the **Design History File (DHF) (820.30(j))**.

    - **Risk Management (ISO 14971 & 21 CFR 820.30(g)):**
        - The **Risk Matrix** is a direct implementation of risk management principles. It allows us to identify, evaluate, and prioritize risks to product quality and project success, ensuring we focus mitigation efforts where they are needed most. This is a continuous process throughout the product lifecycle.

    - **Production & Process Controls (21 CFR 820.70):**
        - **Test Method Validation (TMV) (820.72):** The **Assay Validation Dashboard** provides the statistical evidence (linearity, precision, MSA, etc.) to prove that our test methods are suitable for their intended purpose.
        - **Software Validation (820.70(i) & IEC 62304):** The **Software V&V Dashboard** provides a framework for managing the Software Development Lifecycle (SDLC) of our QC analysis software and bioinformatics pipelines, from requirements traceability to defect management. This is critical for our Oncotype DX® and OncoExTra® analysis pipelines.
        - **Personnel (820.25):** The **QC Operations Hub** includes a training matrix to ensure all personnel are adequately trained to perform their assigned QC testing responsibilities, a fundamental requirement for operational readiness.

    - **Statistical Techniques (21 CFR 820.250) & Process Monitoring:**
        - The **QC Performance Analytics** page provides tools like SPC/Levey-Jennings charts and Cpk analysis to establish and maintain a state of statistical control for our transferred processes. The **ML-Driven Analytics** page provides advanced, investigational tools that align with modern expectations for continual process verification.

    - **CAPA & Quality Management System (21 CFR 820.100 & ISO 13485):**
        - The **QMS & CAPA Tracker** provides oversight for formal corrective and preventive actions. The analysis of OOS events and other non-conformances on this page helps us identify and address systemic issues, fulfilling a core requirement of the CAPA subsystem.

    - **Clinical Laboratory Improvement Amendments (CLIA):**
        - While the QSR applies to manufacturing, our work directly impacts the CLIA-certified labs that run our tests. The robust validation and transfer of assays documented here is foundational to meeting CLIA requirements for establishing and verifying the performance specifications of high-complexity laboratory-developed tests (LDTs) and IVDs.
    """)
