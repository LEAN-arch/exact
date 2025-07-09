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

# --- 1. Experimental Design & Data ---
st.header("1. Experimental Design & Data")

with st.expander("🔬 **The Experiment & Method**"):
    st.markdown("""
    #### The Experiment
    The goal of this experiment is to efficiently map the relationship between critical process parameters (**factors**) and a key quality attribute (**response**). In this case, we are studying the effect of **Temperature** and **pH** on the final process **Yield**.
    
    #### The Method: Central Composite Design (CCD)
    Instead of testing one factor at a time, we use a **Central Composite Design (CCD)**, which is a highly efficient design for building a second-order (quadratic) model. A CCD consists of three types of experimental points:
    1.  **Factorial Points**: The "corners" of the design space, used to estimate the linear effects of each factor and their interactions.
    2.  **Axial (Star) Points**: Points located along the axes outside the factorial space. These are crucial for estimating the curvature of the response surface (i.e., the quadratic terms like T² and pH²).
    3.  **Center Points**: Replicates at the center of the design space, used to estimate pure experimental error and check for model lack of fit.
    
    The table below shows the exact experimental conditions (the "runs") that would be performed in the lab according to this design.
    """)
doe_df = generate_doe_data()
st.dataframe(doe_df, use_container_width=True)


# --- 2. Fit RSM Model and Find Optimum ---
st.header("2. Response Surface Model & Optimization")

with st.expander("🔬 **The Method & Metrics**"):
    st.markdown("""
    #### The Method: Response Surface Modeling
    The data from the DOE is used to fit a mathematical model that approximates the true relationship between the factors and the response. For RSM, a second-order polynomial model is typically used:  
    `Y = β₀ + β₁X₁ + β₂X₂ + β₁₁X₁² + β₂₂X₂² + β₁₂X₁X₂`  
    Where `Y` is the Yield, `X₁` is Temperature, `X₂` is pH, and the `β` terms are the model coefficients determined by multiple linear regression.
    
    #### The Method: Numerical Optimization
    Once the model is built, the goal is to find the combination of `(X₁, X₂)` that maximizes `Y`. This is achieved by finding the point where the partial derivatives of the response surface equation are zero. A numerical, gradient-based algorithm (**L-BFGS-B**) is used to efficiently solve this optimization problem and find the precise optimal settings.
    """)

model, poly_transformer, opt_settings, max_yield = fit_rsm_model_and_optimize(doe_df)

col1, col2, col3 = st.columns(3)
col1.metric("Optimal Temperature", f"{opt_settings[0]:.2f} °C")
col2.metric("Optimal pH", f"{opt_settings[1]:.2f}")
col3.metric("Predicted Maximum Yield", f"{max_yield:.2f} %")


# --- 3. Visualize the Design Space ---
st.header("3. Visualize the Design Space")

with st.expander("📊 **Results & Analysis**"):
    st.markdown("""
    #### Analysis of the Response Surface
    The plots below visualize the complete mathematical model of our process, creating a "map" of the **Design Space**. This is far more powerful than single-point results, as it allows us to understand the process's robustness and predict its behavior at any point within the studied range.
    
    - **3D Surface Plot**: This provides an intuitive, topographical view of the response surface. The "peak" of the surface represents the region of maximum yield. This plot helps visualize the overall shape of the process response.
    
    - **2D Contour Plot**: This can be thought of as a "top-down" map of the 3D surface. Each colored band or line represents a region of constant yield. The shape of the contours is highly informative: elliptical contours, as seen here, indicate a significant **interaction** between Temperature and pH. This means the effect of Temperature on Yield depends on the current level of pH, and vice-versa. The center of the "hottest" region (highest yield) is the optimum.
    
    - **The Optimum Point (Red 'X')**: This marker shows the precise coordinates for Temperature and pH found by the numerical optimizer. Its location on the peak of the 3D surface and in the center of the highest-yield contour provides visual confirmation of the optimization result.
    
    #### Conclusion
    The analysis has successfully modeled the process and identified a predicted optimum at **{opt_settings[0]:.2f} °C** and a **pH of {opt_settings[1]:.2f}**, which is expected to yield approximately **{max_yield:.2f}%**. The next critical step would be to perform several **verification runs** in the lab at these exact settings to confirm the model's prediction.
    """)

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
    fig3d = go.Figure(data=[go.Surface(z=yield_pred, x=temp_range, y=ph_range, colorscale='Viridis', cmin=np.min(yield_pred), cmax=np.max(yield_pred))])
    # Add optimum point to 3D plot
    fig3d.add_trace(go.Scatter3d(
        x=[opt_settings[0]], y=[opt_settings[1]], z=[max_yield],
        mode='markers', marker=dict(size=8, color='red', symbol='x'), name='Optimum'
    ))
    fig3d.update_layout(title='3D Response Surface', scene=dict(xaxis_title='Temperature (°C)', yaxis_title='pH', zaxis_title='Yield (%)'), height=600)
    st.plotly_chart(fig3d, use_container_width=True)

with col2:
    # 2D Contour Plot
    fig2d = go.Figure(data=go.Contour(z=yield_pred, x=temp_range, y=ph_range, colorscale='Viridis', contours=dict(coloring='heatmap', showlabels=True)))
    # Add the optimum point
    fig2d.add_trace(go.Scatter(x=[opt_settings[0]], y=[opt_settings[1]], mode='markers', marker=dict(size=12, color='red', symbol='x'), name='Optimum'))
    fig2d.update_layout(title='2D Contour Plot with Optimum', xaxis_title='Temperature (°C)', yaxis_title='pH', height=600)
    st.plotly_chart(fig2d, use_container_width=True)
