# pages/Process_Optimization_DOE.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

# --- 1. Generate and Display DOE Data ---
st.header("1. Experimental Design & Data")
st.markdown("Data is simulated from a **Central Composite Design (CCD)**, an efficient design for fitting a second-order (quadratic) model.")
doe_df = generate_doe_data()
st.dataframe(doe_df)

# --- 2. Fit RSM Model and Find Optimum ---
st.header("2. Response Surface Model & Optimization")
st.markdown("A quadratic model is fit to the data. Then, a gradient-based optimizer finds the settings for Temperature and pH that are predicted to maximize the Yield.")

model, poly_transformer, opt_settings, max_yield = fit_rsm_model_and_optimize(doe_df)

col1, col2, col3 = st.columns(3)
col1.metric("Optimal Temperature", f"{opt_settings[0]:.2f} °C")
col2.metric("Optimal pH", f"{opt_settings[1]:.2f}")
col3.metric("Predicted Maximum Yield", f"{max_yield:.2f} %")

# --- 3. Visualize the Design Space ---
st.header("3. Visualize the Design Space")
st.markdown("The 3D Surface Plot and 2D Contour Plot visualize the entire modeled relationship between factors and the response. The red dot marks the calculated optimum.")

# Create a grid of data points to predict over
temp_range = np.linspace(doe_df['Temperature (°C)'].min(), doe_df['Temperature (°C)'].max(), 30)
ph_range = np.linspace(doe_df['pH'].min(), doe_df['pH'].max(), 30)
temp_grid, ph_grid = np.meshgrid(temp_range, ph_range)

# Predict yield for each point on the grid
grid_points = np.c_[temp_grid.ravel(), ph_grid.ravel()]
grid_df = pd.DataFrame(grid_points, columns=['Temperature (°C)', 'pH'])
X_grid_poly = poly_transformer.transform(grid_df)
yield_pred = model.predict(X_grid_poly).reshape(temp_grid.shape)

# Create the plots
col1, col2 = st.columns(2)
with col1:
    # 3D Surface Plot
    fig3d = go.Figure(data=[go.Surface(z=yield_pred, x=temp_range, y=ph_range, colorscale='Viridis')])
    fig3d.update_layout(
        title='3D Response Surface',
        scene=dict(
            xaxis_title='Temperature (°C)',
            yaxis_title='pH',
            zaxis_title='Yield (%)'
        ),
        height=600
    )
    st.plotly_chart(fig3d, use_container_width=True)

with col2:
    # 2D Contour Plot
    fig2d = go.Figure(data=go.Contour(
        z=yield_pred,
        x=temp_range,
        y=ph_range,
        colorscale='Viridis',
        contours=dict(coloring='heatmap', showlabels=True)
    ))
    # Add the optimum point
    fig2d.add_trace(go.Scatter(
        x=[opt_settings[0]],
        y=[opt_settings[1]],
        mode='markers',
        marker=dict(size=12, color='red', symbol='x'),
        name='Optimum'
    ))
    fig2d.update_layout(
        title='2D Contour Plot with Optimum',
        xaxis_title='Temperature (°C)',
        yaxis_title='pH',
        height=600
    )
    st.plotly_chart(fig2d, use_container_width=True)
