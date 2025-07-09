# pages/QC_Performance_Analytics.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import generate_spc_data, generate_lot_data, detect_westgard_rules
from scipy.stats import f_oneway

st.set_page_config(page_title="QC Performance Analytics", layout="wide")
st.title("📊 QC Performance & Analytics Dashboard")
st.markdown("### Monitoring transferred method health with advanced statistical process control.")

with st.expander("🌐 Regulatory Context & Legend"):
    st.markdown("""
    This dashboard provides tools for ongoing process monitoring to ensure methods remain in a state of control.
    - **Statistical Process Control**: The charts meet requirements for statistical techniques under **21 CFR 820.250**.
    - **Investigations & CAPA**: Out-of-control events trigger investigations under **21 CFR 820.100**.
    - **Lot Acceptance**: The lot comparison supports incoming acceptance activities per **21 CFR 820.80**.
    """)

spc_df = generate_spc_data(); lot_df = generate_lot_data()

st.header("Levey-Jennings Chart with Westgard Rule Analysis")
mean = spc_df['Value'].mean(); std = spc_df['Value'].std()
fig = go.Figure()
for i, dash in zip([1, 2, 3], ["dot", "dash", "longdash"]):
    fig.add_hline(y=mean + i * std, line_dash=dash, line_color="orange", opacity=0.7)
    fig.add_hline(y=mean - i * std, line_dash=dash, line_color="orange", opacity=0.7)
fig.add_hline(y=mean, line_dash="solid", line_color="green", annotation_text="Mean")
fig.add_trace(go.Scatter(x=spc_df['Run'], y=spc_df['Value'], mode='lines+markers', name='QC Value', marker_color='#1f77b4'))
violations = detect_westgard_rules(spc_df)
if not violations.empty:
    fig.add_trace(go.Scatter(x=violations['Run'], y=violations['Value'], mode='markers', marker=dict(color='red', size=14, symbol='x-thin', line=dict(width=2)), name='Rule Violation', hovertext=violations['Rule']))
fig.update_layout(title='Control Chart for QC Material Performance', xaxis_title='Run Number', yaxis_title='Performance Value')
st.plotly_chart(fig, use_container_width=True)
if not violations.empty:
    st.error("Out-of-control state detected! An investigation must be initiated per **21 CFR 820.100**.")
    st.dataframe(violations, use_container_width=True)

st.divider()

st.header("Reagent Lot-to-Lot Performance with Statistical Significance")
col1, col2 = st.columns([2, 1])
with col1:
    fig_box = px.box(lot_df, x='Lot ID', y='Performance Metric', color='Lot ID', title='Comparison of Reagent Lot Performance', points='all')
    st.plotly_chart(fig_box, use_container_width=True)
with col2:
    st.subheader("ANOVA Test for Lot Differences")
    lot_groups = [group['Performance Metric'].values for name, group in lot_df.groupby('Lot ID')]
    f_stat, p_value = f_oneway(*lot_groups)
    st.metric("ANOVA P-value", f"{p_value:.4f}")
    if p_value < 0.05: st.error("P < 0.05: Statistically significant difference detected between lots.")
    else: st.success("P >= 0.05: No significant difference detected.")
