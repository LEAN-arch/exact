# pages/ML_Driven_Analytics.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt
from utils import (generate_instrument_health_data, train_instrument_model,
                   generate_multivariate_qc_data, train_anomaly_model,
                   generate_rca_data, train_rca_model)

st.set_page_config(page_title="ML-Driven Analytics", layout="wide")
st.title("🤖 ML-Driven Process Analytics")
st.markdown("### Proactive and predictive insights using model-agnostic explainability.")

# ... (Disclaimer expander remains the same) ...

tab1, tab2, tab3 = st.tabs(["**Predictive Instrument Health**", "**Multivariate Anomaly Detection**", "**Automated Root Cause Insights**"])

with tab1:
    st.header("Predictive Instrument Health (e.g., HPLC System)")
    
    instrument_df = generate_instrument_health_data()
    model, X, best_params = train_instrument_model(instrument_df)
    instrument_df['Health Score'] = 1 - model.predict_proba(X)[:, 1]

    # ... (Health score plot and metrics remain the same) ...
    col1, col2 = st.columns([2,1])
    # ...

    st.subheader("Model Explainability: Which Features Matter Most?")
    
    # --- NEW: Permutation Feature Importance ---
    st.markdown("**Permutation Feature Importance** measures how much the model's accuracy decreases when a feature's values are randomly shuffled. A larger drop means the feature is more important.")
    
    perm_importance = permutation_importance(model, X, y=instrument_df['Failure'], n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by="Importance", ascending=True)

    fig_perm = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance (Permutation Method)')
    st.plotly_chart(fig_perm, use_container_width=True)

    # --- NEW: Partial Dependence Contour Plot ---
    st.markdown("**Partial Dependence Plot (PDP)** shows how the model's prediction changes as we vary one or two features, holding all others constant. This 2D plot shows the interaction between the two most important features.")
    
    top_two_features = importance_df.sort_values(by="Importance", ascending=False).head(2)['Feature'].tolist()
    
    # Using scikit-learn's display function to generate the plot data
    fig_pdp, ax_pdp = plt.subplots(figsize=(8, 6))
    pdp_display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=top_two_features, # Features to plot
        kind="average", # The default, shows average effect
        ax=ax_pdp
    )
    ax_pdp.set_title(f'Partial Dependence of Failure Prediction on\n{top_two_features[0]} and {top_two_features[1]}')
    st.pyplot(fig_pdp)


# ... (The rest of the file for tabs 2 and 3 can remain the same) ...

with tab2:
    st.header("Multivariate Anomaly Detection in QC Data")
    st.markdown("Using an Isolation Forest to find subtle, multi-dimensional anomalies that univariate SPC charts miss.")
    
    qc_df = generate_multivariate_qc_data()
    model, qc_df_encoded = train_anomaly_model(qc_df)
    qc_df['Anomaly'] = model.predict(qc_df_encoded[['Operator', 'Reagent Lot', 'Value']])
    
    fig_3d = px.scatter_3d(
        qc_df, x='Run', y='Value', z='Operator', color=qc_df['Anomaly'].astype(str),
        color_discrete_map={'1': 'blue', '-1': 'red'}, symbol='Reagent Lot',
        title="3D View of QC Data with ML-Detected Anomalies"
    )
    fig_3d.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.header("Automated Root Cause Insights")
    st.markdown("Using a Decision Tree to provide a first-pass suggestion for the root cause of a process failure.")
    
    # This call now correctly unpacks the 4 values returned by the updated utils function
    rca_df = generate_rca_data()
    model, X, y, best_params_rca = train_rca_model(rca_df)
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Failure Investigation Decision Tree")
        fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
        plot_tree(model, feature_names=X.columns, class_names=sorted(y.unique()), filled=True, rounded=True, ax=ax_tree, fontsize=10)
        st.pyplot(fig_tree)
    with col2:
        st.subheader("Most Important Factors")
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
        st.dataframe(feature_importance)
        st.subheader("Simulate a New Failure")
        age = st.slider("Instrument Age (mo)", 1, 36, 10)
        reagent = st.slider("Reagent Age (days)", 1, 90, 80)
        exp = st.slider("Operator Experience (yr)", 1, 5, 2)
        prediction = model.predict([[age, reagent, exp]])
        st.error(f"**Predicted Root Cause:** {prediction[0]}")
        with st.expander("Optimized Model Parameters"):
            st.json(best_params_rca)
