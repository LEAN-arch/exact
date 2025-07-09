# utils.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date, timedelta
from scipy import stats

# --- MODIFIED: Added GridSearchCV for model optimization ---
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import shap

# --- Custom Plotly Template for Elegance ---
scientist_template = {
    "layout": {
        "font": {"family": "sans-serif", "size": 12, "color": "#333"},
        "title": {"font": {"family": "sans-serif", "size": 18, "color": "#111"}},
        "plot_bgcolor": "#f0f2f6",
        "paper_bgcolor": "#ffffff",
        "colorway": px.colors.qualitative.D3,
        "xaxis": {"gridcolor": "#ddd", "linecolor": "#666", "zerolinecolor": "#ddd"},
        "yaxis": {"gridcolor": "#ddd", "linecolor": "#666", "zerolinecolor": "#ddd"},
    }
}
pio.templates["scientist"] = scientist_template
pio.templates.default = "scientist"

# --- Core Data Generation Functions ---
def generate_project_data():
    """MODIFIED: Ensures date columns have the correct datetime64 dtype to prevent errors."""
    data = {
        'Project/Assay': ['NGS-Assay-X', 'RT-PCR-Assay-Y', 'HPLC-Method-Z', 'QC-LIMS-Script-A'],
        'Project Lead': ['S. Scientist', 'S. Scientist', 'J. Doe', 'S. Scientist'],
        'Current Phase': ['Validation', 'Development', 'Monitoring', 'Design'], 'Completion %': [75, 40, 100, 15],
        'Overall Status': ['On Track', 'At Risk', 'Complete', 'On Track'],
        'Start Date': [date.today() - timedelta(days=45), date.today() - timedelta(days=20), date.today() - timedelta(days=90), date.today() - timedelta(days=5)],
        'Due Date': [date.today() + timedelta(days=14), date.today() + timedelta(days=25), date.today() - timedelta(days=30), date.today() + timedelta(days=40)],
    }
    df = pd.DataFrame(data)

    # --- THIS IS THE CRUCIAL FIX ---
    # Explicitly convert the date columns to pandas datetime objects.
    # This ensures that subsequent operations (like subtraction) result in a
    # Series with the correct timedelta64 dtype, which supports the .dt accessor.
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['Due Date'] = pd.to_datetime(df['Due Date'])

    return df

def generate_risk_data():
    data = {
        'Risk ID': ['R-01', 'R-02', 'R-03', 'R-04', 'R-05'],
        'Project': ['RT-PCR-Assay-Y', 'NGS-Assay-X', 'NGS-Assay-X', 'QC-LIMS-Script-A', 'RT-PCR-Assay-Y'],
        'Description': ['Key reagent supplier has long lead time.', 'Unexpected noise in baseline signal.', 'Bioinformatics pipeline validation requires more compute time.', 'Cloud environment has intermittent latency.', 'Instrument availability conflict with R&D.'],
        'Impact': ['High', 'High', 'Medium', 'Low', 'Medium'], 'Probability': ['Medium', 'Low', 'High', 'Medium', 'High'],
        'Owner': ['S. Scientist', 'J. Doe', 'S. Scientist', 'IT', 'Ops Manager'],
    }
    df = pd.DataFrame(data); impact_map = {'Low': 1, 'Medium': 2, 'High': 3}; prob_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Impact_Score'] = df['Impact'].map(impact_map); df['Prob_Score'] = df['Probability'].map(prob_map)
    df['Risk_Score'] = df['Impact_Score'] * df['Prob_Score']
    return df.sort_values(by='Risk_Score', ascending=False)

def generate_linearity_data():
    expected = np.array([10, 50, 100, 250, 500, 750, 1000])
    non_linear_factor = 1 - (expected / 4000)
    observed = (expected * np.random.normal(1.02, 0.03, expected.shape) + np.random.normal(0, 5, expected.shape)) * non_linear_factor
    return pd.DataFrame({'Expected Concentration': expected, 'Observed Signal': observed})

def generate_precision_data():
    days = ['Day 1', 'Day 2', 'Day 3']; operators = ['Op 1', 'Op 2']; data = []
    for day in days:
        for op in operators:
            mean = 101 if op == 'Op 2' else 99; stdev = 1.8 if day == 'Day 3' else 1.2
            values = np.random.normal(loc=mean, scale=stdev, size=5)
            for val in values: data.append({'Day': day, 'Operator': op, 'Value': val})
    return pd.DataFrame(data)

def generate_msa_data(): return {'part_var': 95.2, 'repeatability_var': 2.8, 'reproducibility_var': 2.0}
def generate_spc_data():
    np.random.seed(42); data = np.random.normal(loc=100, scale=1.5, size=30); data[10] = 104; data[11] = 103.5; data[15] = 105.1; data[20:24] = [102.5, 102.8, 103.1, 103.5]
    return pd.DataFrame({'Value': data, 'Run': range(1, 31)})
def generate_lot_data():
    np.random.seed(0); lots = ['Lot A', 'Lot B', 'Lot C (New)', 'Lot D']; data = []
    for lot in lots:
        mean = 104 if lot == 'Lot C (New)' else 100; values = np.random.normal(loc=mean, scale=1.5, size=20)
        for val in values: data.append({'Lot ID': lot, 'Performance Metric': val})
    return pd.DataFrame(data)

def detect_westgard_rules(df, value_col='Value'):
    mean = df[value_col].mean(); std = df[value_col].std(); violations = []
    for i in range(len(df)):
        val = df.loc[i, value_col]
        if val > mean + 3 * std or val < mean - 3 * std: violations.append({'Run': df.loc[i, 'Run'], 'Value': val, 'Rule': '1_3s'})
        elif val > mean + 2 * std or val < mean - 2 * std:
            if i > 0 and (df.loc[i-1, value_col] > mean + 2 * std or df.loc[i-1, value_col] < mean - 2 * std):
                 violations.append({'Run': df.loc[i, 'Run'], 'Value': val, 'Rule': '2_2s'})
            else: violations.append({'Run': df.loc[i, 'Run'], 'Value': val, 'Rule': '1_2s (Warning)'})
        if i >= 3:
            last_4 = df.loc[i-3:i, value_col]
            if all(last_4 > mean + std) or all(last_4 < mean - std):
                violations.append({'Run': df.loc[i, 'Run'], 'Value': val, 'Rule': '4_1s'})
    return pd.DataFrame(violations).drop_duplicates(subset=['Run'])

# --- NEW: Specificity/Interference Data Generation ---
def generate_specificity_data():
    """Generates data for an assay specificity and interference study."""
    np.random.seed(33); data = []
    data.extend([{'Sample Type': 'Blank', 'Signal': v} for v in np.random.normal(10, 2, 10)])
    data.extend([{'Sample Type': 'Target Only', 'Signal': v} for v in np.random.normal(200, 15, 10)])
    data.extend([{'Sample Type': 'Interferent A', 'Signal': v} for v in np.random.normal(12, 3, 10)])
    data.extend([{'Sample Type': 'Interferent B', 'Signal': v} for v in np.random.normal(15, 4, 10)])
    data.extend([{'Sample Type': 'Target + Interferents', 'Signal': v} for v in np.random.normal(205, 16, 10)])
    return pd.DataFrame(data)


# --- MODIFIED & IMPROVED ML Data and Model Functions ---

def generate_instrument_health_data():
    """MODIFIED: Generates more realistic multivariate time-series data with non-linear drift."""
    np.random.seed(101); runs = 100
    base_pressure = 1500 + 15 * np.sin(np.linspace(0, 3 * np.pi, runs)); pressure = base_pressure + np.random.normal(0, 5, runs)
    base_temp = 35 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, runs)); temperature = base_temp + np.random.normal(0, 0.1, runs)
    drift = np.exp(np.linspace(0, 2.5, 20)); pressure[80:] += drift * 2; temperature[80:] += drift / 10
    flow_rate_stability = np.random.normal(0.99, 0.005, runs); flow_rate_stability[85:] -= np.logspace(-3, -1.5, 15)
    df = pd.DataFrame({'Run ID': range(runs), 'Pressure (psi)': pressure, 'Column Temp (°C)': temperature, 'Flow Rate Stability': flow_rate_stability})
    df['Failure'] = 0; df.loc[90:, 'Failure'] = 1
    return df

def generate_multivariate_qc_data():
    np.random.seed(42); data = []
    for i in range(100):
        op = 'Operator A' if i % 2 == 0 else 'Operator B'; lot = 'Lot X' if i < 50 else 'Lot Y'
        val = np.random.normal(100, 2)
        if op == 'Operator B': val += 0.5
        if lot == 'Lot Y': val += 0.7
        data.append({'Run': i, 'Operator': op, 'Reagent Lot': lot, 'Value': val})
    df = pd.DataFrame(data); df.loc[80, 'Value'] -= 3; df.loc[80, 'Operator'] = 'Operator B'
    return df

def generate_rca_data():
    np.random.seed(0); n_samples = 200; instrument_age = np.random.randint(1, 36, n_samples)
    reagent_lot_age = np.random.randint(1, 90, n_samples); operator_experience = np.random.randint(1, 5, n_samples); causes = []
    for i in range(n_samples):
        if reagent_lot_age[i] > 75 or (reagent_lot_age[i] > 60 and np.random.rand() > 0.3): causes.append('Reagent Degradation')
        elif instrument_age[i] > 30 or (instrument_age[i] > 20 and np.random.rand() > 0.5): causes.append('Instrument Drift')
        elif operator_experience[i] == 1 and np.random.rand() > 0.4: causes.append('Operator Error')
        else: causes.append('No Fault Found')
    return pd.DataFrame({'Instrument Age (mo)': instrument_age, 'Reagent Lot Age (days)': reagent_lot_age, 'Operator Experience (yr)': operator_experience, 'Root Cause': causes})


def train_instrument_model(df):
    """MODIFIED: Trains a RandomForest model using GridSearchCV for hyperparameter optimization."""
    X = df[['Pressure (psi)', 'Column Temp (°C)', 'Flow Rate Stability']]; y = df['Failure']
    param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2, 4]}
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    return best_model, X, grid_search.best_params_

def train_anomaly_model(df):
    le_op = LabelEncoder(); le_lot = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['Operator'] = le_op.fit_transform(df_encoded['Operator'])
    df_encoded['Reagent Lot'] = le_lot.fit_transform(df_encoded['Reagent Lot'])
    model = IsolationForest(contamination=0.02, random_state=42); model.fit(df_encoded[['Operator', 'Reagent Lot', 'Value']])
    return model, df_encoded

def train_rca_model(df):
    """MODIFIED: Trains a DecisionTree model using GridSearchCV."""
    X = df.drop('Root Cause', axis=1); y = df['Root Cause']
    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 4, 5, 6], 'min_samples_split': [2, 5, 10]}
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    return best_model, X, y, grid_search.best_params_
