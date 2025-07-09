# pages/Process_Optimization_DOE.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import generate_doe_data, fit_rsm_model_and_optimize

st.set_page_config(page_title="Process Optimization (DOE/RSM)", layout="wide")
st.title("🧪 Process Optimization: DOE & Response Surface Methodology")

with st.expander("🌐 Regulatory Context & Quality by Design (QbD)"):
    st.markdown("""
    This page embodies the principles of **Quality by Design (QbD)**, a concept heavily promoted by regulators and detailed in guidelines like **ICH Q8**.
    - **Design of Experiments (DOE)**: A structured, efficient method for planning experiments to determine the relationship between factors affecting a process and the output of that process.
    - **Response Surface Methodology (RSM)**: A collection of statistical and mathematical techniques used to model and analyze problems in which a response of interest is influenced by several variables. The objective is to optimize this response.
    By using DOE/RSM, we proactively build quality into the process by defining a **Design Space**—the multidimensional combination of input variables that have been demonstrated to provide assurance of quality.
    """)

# --- 1. Experimental Design & Data ---
st.header("1. Experimental Design & Data")
with st.expander("🔬 **The Experiment & Method**"):
    st.markdown("""
    #### The Experiment
    The goal is to efficiently map the relationship between critical process parameters (**factors**) and a key quality attribute (**response**). Here, we study the effect of **Temperature** and **pH** on the final process **Yield**.
    #### The Method: Central Composite Design (CCD)
    A **CCD** is used for building a second-order (quadratic) model. It consists of:
    1.  **Factorial Points**: The "corners" of the design space to estimate linear and interaction effects.
    2.  **Axial (Star) Points**: Points along the axes to estimate curvature (e.g., T², pH²).
    3.  **Center Points**: Replicates at the center to estimate pure experimental error.
    """)
doe_df = generate_doe_data()
st.dataframe(doe_df, use_container_width=True)


# --- 2. Fit RSM Model and Find Optimum ---
st.header("2. Response Surface Model & Optimization")
with st.expander("🔬 **The Method & Metrics**"):
    st.markdown("""
    #### The Method: Response Surface Modeling & Optimization
    A second-order polynomial model is fit to the DOE data: `Y = β₀ + β₁X₁ + β₂X₂ + β₁₁X₁² + β₂₂X₂² + β₁₂X₁X₂`. A numerical, gradient-based algorithm (**L-BFGS-B**) is then used to find the combination of Temperature and pH that maximizes the predicted Yield from this model.
    """)
model, poly_transformer, opt_settings, max_yield = fit_rsm_model_and_optimize(doe_df)
st.subheader("Predicted Process Optimum")
col1, col2, col3 = st.columns(3)
col1.metric("Optimal Temperature", f"{opt_settings[0]:.2f} °C")
col2.metric("Optimal pH", f"{opt_settings[1]:.2f}")
col3.metric("Predicted Maximum Yield", f"{max_yield:.2f} %")


# --- 3. Visualize the Design Space ---
st.header("3. Visualize the Design Space & Proven Acceptable Range (PAR)")
with st.expander("📊 **Results & Analysis**"):
    st.markdown("""
    #### Analysis of the Response Surface
    The plots below visualize the complete mathematical model of our process, creating a "map" of the **Design Space**. This is far more powerful than single-point results, as it allows us to understand the process's robustness and predict its behavior at any point within the studied range.
    - **Experimental Points (Black Dots)**: These show where we have actual data, giving us confidence in the model's predictions within this region.
    - **The Optimum (Red 'X')**: This marker shows the precise coordinates for maximum yield found by the optimizer.
    - **Proven Acceptable Range (PAR)**: The highlighted green area on the contour plot represents the PAR. This is the operating range where the process is predicted to consistently meet the defined quality specification (e.g., Yield > 80%). Operating within this PAR provides high assurance of product quality. The elliptical shape of the PAR is a direct result of the significant interaction between Temperature and pH.
    #### Conclusion
    The analysis has successfully modeled the process and identified a predicted optimum. More importantly, it has defined a robust Design Space and PAR. The next step is to perform **verification runs** at the optimum and at the edges of the PAR to confirm the model's predictions.
    """)

# --- Interactive Controls ---
c1, c2 = st.columns(2)
yield_spec = c1.slider(
    "Define Yield Specification (Lower Limit)",
    min_value=int(doe_df['Yield (%)'].min()),
    max_value=int(max_yield),
    value=80,
    help="Define the minimum acceptable yield. The green area on the contour plot will show the Proven Acceptable Range (PAR) that meets this spec."
)

# --- Data Grid for Plotting ---
temp_range = np.linspace(doe_df['Temperature (°C)'].min(), doe_df['Temperature (°C)'].max(), 50)
ph_range = np.linspace(doe_df['pH'].min(), doe_df['pH'].max(), 50)
temp_grid, ph_grid = np.meshgrid(temp_range, ph_range)
grid_points = np.c_[temp_grid.ravel(), ph_grid.ravel()]
grid_df = pd.DataFrame(grid_points, columns=['Temperature (°C)', 'pH'])
X_grid_poly = poly_transformer.transform(grid_df)
yield_pred = model.predict(X_grid_poly).reshape(temp_grid.shape)

# --- Combined 3D Surface and 2D Contour Plot ---
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'contour'}]],
    subplot_titles=('3D Response Surface', '2D Contour with Proven Acceptable Range')
)

# 3D Surface Plot
fig.add_trace(go.Surface(
    z=yield_pred, x=temp_range, y=ph_range, colorscale='RdBu', showscale=False,
    cmin=np.min(yield_pred), cmax=np.max(yield_pred)
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=doe_df['Temperature (°C)'], y=doe_df['pH'], z=doe_df['Yield (%)'],
    mode='markers', marker=dict(size=5, color='black'), name='DOE Points'
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=[opt_settings[0]], y=[opt_settings[1]], z=[max_yield],
    mode='markers', marker=dict(size=10, color='yellow', symbol='cross'), name='Optimum'
), row=1, col=1)

# 2D Contour Plot
fig.add_trace(go.Contour(
    z=yield_pred, x=temp_range, y=ph_range, colorscale='RdBu', showscale=False,
    contours=dict(coloring='lines', showlabels=True), line=dict(width=1)
), row=1, col=2)
fig.add_trace(go.Contour(
    z=yield_pred, x=temp_range, y=ph_range, showscale=False,
    contours_coloring='lines', line_width=0,
    contours=dict(start=yield_spec, end=max_yield, showlabels=False),
    fillcolor='rgba(0, 255, 0, 0.3)',
), row=1, col=2)
fig.add_trace(go.Scatter(
    x=doe_df['Temperature (°C)'], y=doe_df['pH'],
    mode='markers', marker=dict(color='black'), name='DOE Points', showlegend=False
), row=1, col=2)
fig.add_trace(go.Scatter(
    x=[opt_settings[0]], y=[opt_settings[1]], mode='markers',
    marker=dict(size=12, color='red', symbol='cross'), name='Optimum', showlegend=False
), row=1, col=2)

# --- THIS IS THE CORRECTED SECTION ---
# Update layout properties for the entire figure and for specific axes/scenes
fig.update_layout(
    height=700, 
    title_text="Process Design Space Visualization", 
    margin=dict(l=40, r=40, b=40, t=90),
    # Update the 3D scene directly by its name 'scene'
    scene = dict(
        xaxis_title='Temperature (°C)',
        yaxis_title='pH',
        zaxis_title='Yield (%)'
    ),
    # Update the 2D axes by their names 'xaxis' and 'yaxis' (for the second subplot)
    xaxis2_title="Temperature (°C)",
    yaxis2_title="pH"
)

st.plotly_chart(fig, use_container_width=True)
