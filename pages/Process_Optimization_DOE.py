# pages/Process_Optimization_DOE.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import generate_doe_data, fit_rsm_model_and_optimize

st.set_page_config(
    page_title="Process Optimization (DOE/RSM) | Exact Sciences",
    layout="wide"
)

st.title("🧪 Process Optimization: Design of Experiments (DOE)")
st.markdown("### Using Response Surface Methodology (RSM) to define a robust Design Space for a critical RT-PCR assay.")

with st.expander("🌐 Regulatory Context & Quality by Design (QbD)"):
    st.markdown("""
    This page embodies the principles of **Quality by Design (QbD)**, a systematic approach to development that begins with predefined objectives and emphasizes product and process understanding and process control, based on sound science and quality risk management. This is a core expectation from regulatory bodies, as detailed in guidelines like **ICH Q8 (Pharmaceutical Development)**.

    - **Design of Experiments (DOE)**: An efficient, structured methodology for determining the relationship between factors affecting a process and the output of that process.
    - **Response Surface Methodology (RSM)**: A collection of statistical and mathematical techniques used to model and analyze problems in which a response of interest is influenced by several variables, with the objective of optimizing this response.
    - **Design Space**: The multidimensional combination and interaction of input variables (e.g., material attributes, process parameters) that have been demonstrated to provide assurance of quality. Working within the Design Space is not considered a change. Its definition is a key output of this work.
    - **Proven Acceptable Range (PAR)**: The characterization of a process parameter for which operation within that range, while keeping other parameters constant, will result in producing a material meeting relevant quality criteria.
    """)

# --- 1. Experimental Design & Data ---
st.header("1. Experimental Design: RT-PCR Annealing Optimization")
with st.expander("🔬 **The Experiment & Method**"):
    st.markdown("""
    #### The Goal
    To efficiently map the relationship between two critical process parameters (**factors**) of an RT-PCR assay and a key quality attribute (**response**). Here, we study the effect of **Annealing Temperature** and **Primer Concentration** on the final **PCR Efficiency**. The goal is to find the combination of factors that maximizes efficiency while understanding the process's robustness to small variations.

    #### The Method: Central Composite Design (CCD)
    A **Central Composite Design** is a highly efficient experimental design for building a second-order (quadratic) model needed for RSM. It consists of:
    1.  **Factorial Points (2²)**: The "corners" of the design space (e.g., Low/High Temp & Low/High Primer Conc.) to estimate linear and interaction effects.
    2.  **Axial (Star) Points (2k)**: Points along the axes, outside the factorial range, to allow for the estimation of curvature (e.g., Temp², Primer²).
    3.  **Center Points (n)**: Replicates at the center of the design to estimate pure experimental error and check for model fit.
    """)
doe_df = generate_doe_data()
st.dataframe(doe_df, use_container_width=True)


# --- 2. Fit RSM Model and Find Optimum ---
st.header("2. Response Surface Model & Optimization Results")
with st.expander("🔬 **The Method & Mathematical Model**"):
    st.markdown("""
    #### The Method: Response Surface Modeling
    A second-order polynomial model is fitted to the DOE data to describe the relationship between the factors and the response. The general form of the model is:
    """)
    st.latex(r'''
    Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_{11} X_1^2 + \beta_{22} X_2^2 + \beta_{12} X_1 X_2
    ''')
    st.markdown(r"""
    Where:
    - $Y$ is the predicted response (PCR Efficiency)
    - $X_1, X_2$ are the coded values of the factors (Temperature, Primer Concentration)
    - $\beta_0$ is the intercept term
    - $\beta_1, \beta_2$ are the linear effect coefficients
    - $\beta_{11}, \beta_{22}$ are the quadratic (curvature) effect coefficients
    - $\beta_{12}$ is the interaction effect coefficient

    A numerical, gradient-based algorithm (**L-BFGS-B**) is then used to find the combination of Temperature and Primer Concentration that maximizes the predicted PCR Efficiency based on this model.
    """)
model, poly_transformer, opt_settings, max_response = fit_rsm_model_and_optimize(doe_df)
st.subheader("Predicted Process Optimum")
col1, col2, col3 = st.columns(3)
col1.metric("Optimal Annealing Temp", f"{opt_settings[0]:.1f} °C")
col2.metric("Optimal Primer Conc.", f"{opt_settings[1]:.1f} nM")
col3.metric("Predicted Max Efficiency", f"{max_response:.1f} %")


# --- 3. Visualize the Design Space ---
st.header("3. Visualize the Design Space & Proven Acceptable Range (PAR)")

# --- Interactive Controls ---
c1, c2 = st.columns(2)
efficiency_spec = c1.slider(
    "Define Minimum Acceptable Efficiency (%)",
    min_value=int(doe_df['PCR Efficiency (%)'].min()),
    max_value=int(max_response),
    value=90,
    help="Define the minimum acceptable PCR efficiency. The green area on the contour plot will show the Proven Acceptable Range (PAR) that meets this specification."
)

# --- Data Grid for Plotting ---
temp_range = np.linspace(doe_df['Annealing Temp (°C)'].min(), doe_df['Annealing Temp (°C)'].max(), 50)
primer_range = np.linspace(doe_df['Primer Conc. (nM)'].min(), doe_df['Primer Conc. (nM)'].max(), 50)
temp_grid, primer_grid = np.meshgrid(temp_range, primer_range)
grid_points = np.c_[temp_grid.ravel(), primer_grid.ravel()]
grid_df = pd.DataFrame(grid_points, columns=['Annealing Temp (°C)', 'Primer Conc. (nM)'])
X_grid_poly = poly_transformer.transform(grid_df)
response_pred = model.predict(X_grid_poly).reshape(temp_grid.shape)

# --- Combined 3D Surface and 2D Contour Plot ---
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'contour'}]],
    subplot_titles=('3D Response Surface', '2D Contour Map with PAR')
)

# 3D Surface Plot
fig.add_trace(go.Surface(
    z=response_pred, x=temp_range, y=primer_range, colorscale='Viridis', showscale=False,
    cmin=np.min(response_pred), cmax=np.max(response_pred),
    name='Response Surface'
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=doe_df['Annealing Temp (°C)'], y=doe_df['Primer Conc. (nM)'], z=doe_df['PCR Efficiency (%)'],
    mode='markers', marker=dict(size=5, color='black', symbol='diamond'), name='DOE Points'
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=[opt_settings[0]], y=[opt_settings[1]], z=[max_response],
    mode='markers', marker=dict(size=10, color='red', symbol='cross'), name='Predicted Optimum'
), row=1, col=1)

# 2D Contour Plot
contour_coloring = 'lines' if efficiency_spec > 85 else 'heatmap'
fig.add_trace(go.Contour(
    z=response_pred, x=temp_range, y=primer_range, colorscale='Viridis', showscale=False,
    contours=dict(coloring=contour_coloring, showlabels=True), line=dict(width=1)
), row=1, col=2)
# Highlight the Proven Acceptable Range (PAR)
fig.add_trace(go.Contour(
    z=response_pred, x=temp_range, y=primer_range, showscale=False,
    contours_coloring='lines', line_width=0,
    contours=dict(start=efficiency_spec, end=max_response, showlabels=False),
    fillcolor='rgba(40, 167, 69, 0.4)',
    name='PAR'
), row=1, col=2)
fig.add_trace(go.Scatter(
    x=doe_df['Annealing Temp (°C)'], y=doe_df['Primer Conc. (nM)'],
    mode='markers', marker=dict(color='black', symbol='diamond'), name='DOE Points', showlegend=False
), row=1, col=2)
fig.add_trace(go.Scatter(
    x=[opt_settings[0]], y=[opt_settings[1]], mode='markers',
    marker=dict(size=12, color='red', symbol='cross'), name='Optimum', showlegend=False
), row=1, col=2)

fig.update_layout(
    height=700,
    title_text="RT-PCR Process Design Space Visualization",
    margin=dict(l=40, r=40, b=40, t=90),
    scene=dict(
        xaxis_title='Annealing Temp (°C)',
        yaxis_title='Primer Conc. (nM)',
        zaxis_title='PCR Efficiency (%)'
    ),
    xaxis2=dict(title='Annealing Temp (°C)'),
    yaxis2=dict(title='Primer Conc. (nM)')
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 **Results & Analysis: Defining the Design Space**"):
    st.markdown(f"""
    #### Analysis of the Response Surface
    The plots above visualize the complete mathematical model of our RT-PCR annealing step, effectively creating a "map" of the **Design Space**. This is far more powerful than one-factor-at-a-time (OFAT) experiments, as it allows us to understand the process's robustness and predict its behavior at any point within the studied range.

    - **The Optimum (Red Cross)**: The model predicts that maximum efficiency ({max_response:.1f}%) is achieved at **{opt_settings[0]:.1f} °C** and **{opt_settings[1]:.1f} nM**. This becomes our target operating condition.
    - **Interaction Effects**: The elliptical shape of the contour lines indicates a significant interaction between Temperature and Primer Concentration. At lower temperatures, the process is more sensitive to changes in primer concentration, while at higher temperatures, it is more robust. This is a critical insight that OFAT experiments would miss.
    - **Proven Acceptable Range (PAR)**: The green shaded area on the contour plot represents the PAR for achieving an efficiency of at least **{efficiency_spec}%**. This range is not a simple box; it's a scientifically-defined boundary. Operating anywhere within this PAR provides a high degree of assurance that the assay will meet its quality specification.

    #### Conclusion & Next Steps
    This DOE has successfully modeled the process and identified a robust operating window.
    1.  **Set Target & Normal Operating Range (NOR)**: The target will be set at the predicted optimum. The NOR will be a tighter range set well within the PAR to account for normal operational variability.
    2.  **Verification**: The next step is to perform **verification runs** at the target condition and, critically, at the *edges* of the proposed PAR to confirm the model's predictions.
    3.  **Documentation**: This entire analysis, including the data and model, will be a key component of the **Design History File (DHF)** and will be referenced in the Test Method Validation report.
    """)
