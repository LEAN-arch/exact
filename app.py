# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import generate_project_data, generate_risk_data
from datetime import date

# --- Page Configuration ---
st.set_page_config(page_title="Staff Scientist Command Center", page_icon="🔬", layout="wide")

# --- Data Loading ---
projects_df = generate_project_data()
risks_df = generate_risk_data()

# --- Page Title and Header ---
st.title("🔬 Staff Scientist Command Center")
st.markdown("### A strategic dashboard for leading QC software and assay transfer initiatives.")

# --- KPIs ---
st.header("Executive Summary KPIs")
total_projects = len(projects_df)
complete_projects = projects_df[projects_df['Overall Status'] == 'Complete'].shape[0]
on_time_completion_rate = (complete_projects / total_projects) * 100 if total_projects > 0 else 0
high_risk_score_items = risks_df[risks_df['Risk_Score'] >= 6].shape[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Projects", f"{total_projects - complete_projects}")
col2.metric("High-Score Risks (>6)", f"{high_risk_score_items}", delta=high_risk_score_items, delta_color="inverse")
col3.metric("On-Time Completion %", f"{on_time_completion_rate:.1f}%", "Target: >90%")

# --- THIS IS THE CORRECTED LINE ---
# We first convert the Timedelta series to numeric days using .dt.days, then calculate the mean.
avg_duration = (projects_df['Due Date'] - projects_df['Start Date']).dt.days.mean()
col4.metric("Avg. Project Duration", f"{avg_duration:.1f} days")


st.divider()

# --- Main Content Area ---
col1, col2 = st.columns((2, 1.2))

with col1:
    st.header("Project Portfolio Gantt Chart")
    st.caption("Visualizing the execution of design transfer activities for all projects.")
    fig = px.timeline(
        projects_df,
        x_start="Start Date",
        x_end="Due Date",
        y="Project/Assay",
        color="Current Phase",
        title="Project Timelines by Phase",
        hover_name="Project/Assay",
        hover_data=["Project Lead", "Overall Status"]
    )
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Project Risk Matrix")
    st.caption("Prioritizing risks based on severity and probability of occurrence.")
    fig_risk = px.scatter(
        risks_df, x="Prob_Score", y="Impact_Score", size="Risk_Score", color="Risk_Score",
        color_continuous_scale=px.colors.sequential.Reds, hover_name="Description",
        hover_data=["Project", "Owner"], size_max=40, title="Impact vs. Probability"
    )
    fig_risk.update_layout(
        xaxis=dict(tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High'], title='Probability'),
        yaxis=dict(tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High'], title='Impact'),
        coloraxis_showscale=False
    )
    fig_risk.add_annotation(x=2.5, y=2.5, text="High Risk", showarrow=False, font=dict(color="red", size=14, family="Arial, bold"))
    st.plotly_chart(fig_risk, use_container_width=True)
    
st.header("Portfolio Details")
st.dataframe(projects_df, use_container_width=True)

# --- REGULATORY LEGEND ---
st.divider()
with st.expander("🌐 Regulatory Context & Legend"):
    st.markdown("""
    This dashboard provides oversight required by quality management systems to ensure projects are controlled, risks are managed, and progress is visible to leadership.

    - **Project & Timeline Management**: Supports the overall management of **Design Controls** as required by:
        - **21 CFR 820.30(b)**: *Design and Development Planning* - Establishes a framework for plans that identify, describe, and assign activities.
        - **ISO 13485:2016, Section 7.3.2**: *Design and Development Planning* - The organization shall plan and control the design and development of product.

    - **Risk Matrix**: Directly supports the implementation of a **Risk Management** process throughout the product lifecycle as mandated by:
        - **ISO 14971:2019**: *Medical devices — Application of risk management*. The visualization helps in identifying and prioritizing risks for control measures.
        - **21 CFR 820.30(g)**: *Design Risk Analysis* - Requires the identification and evaluation of risks, and ensuring they are mitigated.
        
    - **Advanced Process Analytics (See ML-Driven Page)**: This dashboard includes an additional page with Machine Learning models. These serve as **investigational and process improvement tools**, aligning with the principles of **ICH Q10 (Pharmaceutical Quality System)**, which encourages a scientific, risk-based approach to continual improvement throughout the product lifecycle. These models supplement, but do not replace, the validated methods required by 21 CFR 820 and ISO 13485.
    """)
