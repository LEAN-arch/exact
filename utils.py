# utils.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date, timedelta
from scipy import stats
from scipy.optimize import minimize

# --- MODIFIED: Updated ML library imports ---
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

def generate_doe_data():
    """Generates data simulating a Central Composite Design (CCD)."""
    np.random.seed(42)
    temp_levels = [-1.414, -1, -1, 1, 1, -1.414, 1.414, 0, 0, 0, 0, 0]
    ph_levels   = [0, -1, 1, -1, 1, 0, 0, -1.414, 1.414, 0, 0, 0]
    temp_real = np.array(temp_levels) * 10 + 60
    ph_real   = np.array(ph_levels) * 0.5 + 7.5
    true_yield = 80 + (5*np.array(temp_levels)) + (3*np.array(ph_levels)) - (6*np.array(temp_levels)**2) - (4*np.array(ph_levels)**2) + (2*np.array(temp_levels)*np.array(ph_levels))
    measured_yield = true_yield + np.random.normal(0, 1.5, len(temp_real))
    return pd.DataFrame({'Temperature (°C)': temp_real, 'pH': ph_real, 'Yield (%)': measured_yield})

def fit_rsm_model_and_optimize(df):
    """Fits a quadratic response surface model and finds the settings for maximum yield."""
    X = df[['Temperature (°C)', 'pH']]; y = df['Yield (%)']
    poly = PolynomialFeatures(degree=2, include_bias=False); X_poly = poly.fit_transform(X)
    model = LinearRegression(); model.fit(X_poly, y)
    def neg_yield(params):
        temp, ph = params
        x_in = pd.DataFrame([[temp, ph]], columns=['Temperature (°C)', 'pH'])
        x_in_poly = poly.transform(x_in)
        return -model.predict(x_in_poly)[0]
    initial_guess = [X['Temperature (°C)'].mean(), X['pH'].mean()]
    bounds = [(X['Temperature (°C)'].min(), X['Temperature (°C)'].max()), (X['pH'].min(), X['pH'].max())]
    result = minimize(neg_yield, initial_guess, method='L-BFGS-B', bounds=bounds)
    opt_settings = result.x; max_yield = -result.fun
    return model, poly, opt_settings, max_yield

def generate_instrument_health_data():
    """Generates more realistic multivariate time-series data with non-linear drift."""
    np.random.seed(101); runs = 100
    base_pressure = 1500 + 15 * np.sin(np.linspace(0, 3 * np.pi, runs)); pressure = base_pressure + np.random.normal(0, 5, runs)
    base_temp = 35 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, runs)); temperature = base_temp + np.random.normal(0, 0.1, runs)
    drift = np.exp(np.linspace(0, 2.5, 20)); pressure[80:] += drift * 2; temperature[80:] += drift / 10
    flow_rate_stability = np.random.normal(0.99, 0.005, runs); flow_rate_stability[85:] -= np.logspace(-3, -1.5, 15)
    
    # BEST PRACTICE FIX: Using underscores to prevent LightGBM warning
    df = pd.DataFrame({
        'Run_ID': range(runs), 
        'Pressure_psi': pressure, 
        'Column_Temp_C': temperature, 
        'Flow_Rate_Stability': flow_rate_stability
    })
    df['Failure'] = 0; df.loc[90:, 'Failure'] = 1
    return df

def generate_golden_batch_data():
    """Generates a 'golden batch' of normal multivariate QC data for autoencoder training."""
    np.random.seed(123)
    data = []
    for i in range(200):
        op_noise = np.random.normal(0, 0.1); lot_noise = np.random.normal(0, 0.2); val = 100 + op_noise + lot_noise + np.random.normal(0, 1.5)
        sensor1 = 5.0 + (val - 100) * 0.1 + np.random.normal(0, 0.05); sensor2 = 22.5 - (val - 100) * 0.05 + np.random.normal(0, 0.1)
        data.append({'Value': val, 'Sensor 1': sensor1, 'Sensor 2': sensor2})
    return pd.DataFrame(data)

def generate_live_qc_data(golden_df):
    """Generates live data including anomalies, based on the golden batch distribution."""
    live_data = []
    for i in range(50):
        val = np.random.normal(100, 1.5); sensor1 = 5.0 + (val - 100) * 0.1 + np.random.normal(0, 0.05)
        sensor2 = 22.5 - (val - 100) * 0.05 + np.random.normal(0, 0.1)
        live_data.append({'Value': val, 'Sensor 1': sensor1, 'Sensor 2': sensor2, 'Run': i})
    for i in range(50, 60):
        val = 100 + (i-50)*0.2; sensor1 = 5.0 + (val-100)*0.1 + (i-50)*0.05; sensor2 = 22.5 - (val-100)*0.05
        live_data.append({'Value': val, 'Sensor 1': sensor1, 'Sensor 2': sensor2, 'Run': i})
    live_data.append({'Value': 110, 'Sensor 1': 4.8, 'Sensor 2': 25.0, 'Run': 60})
    return pd.DataFrame(live_data)

def generate_rca_data():
    np.random.seed(0); n_samples = 200; instrument_age = np.random.randint(1, 36, n_samples)
    reagent_lot_age = np.random.randint(1, 90, n_samples); operator_experience = np.random.randint(1, 5, n_samples); causes = []
    for i in range(n_samples):
        if reagent_lot_age[i] > 75 or (reagent_lot_age[i] > 60 and np.random.rand() > 0.3): causes.append('Reagent Degradation')
        elif instrument_age[i] > 30 or (instrument_age[i] > 20 and np.random.rand() > 0.5): causes.append('Instrument Drift')
        elif operator_experience[i] == 1 and np.random.rand() > 0.4: causes.append('Operator Error')
        else: causes.append('No Fault Found')
    return pd.DataFrame({'Instrument Age (mo)': instrument_age, 'Reagent Lot Age (days)': reagent_lot_age, 'Operator Experience (yr)': operator_experience, 'Root Cause': causes})

def train_autoencoder_model(golden_df):
    """Trains a Keras Autoencoder on normal data only."""
    scaler = StandardScaler(); X_train = scaler.fit_transform(golden_df)
    n_features = X_train.shape[1]
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(n_features, activation='relu'),
        layers.Dense(2, activation='relu', name="bottleneck"),
        layers.Dense(n_features, activation='relu'),
        layers.Dense(n_features, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mae')
    model.fit(X_train, X_train, epochs=50, batch_size=16, validation_split=0.1, verbose=0)
    return model, scaler

def train_instrument_model(df):
    """UPGRADED: Trains a LightGBM model to predict instrument failure."""
    # Using the new column names with underscores
    X = df[['Pressure_psi', 'Column_Temp_C', 'Flow_Rate_Stability']]; y = df['Failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # BEST PRACTICE: Suppress verbose output in logs
    model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
    model.fit(X_train, y_train)
    return model, X

def train_rca_model(df):
    """UPGRADED: Trains a robust RandomForest model for Root Cause Analysis."""
    X = df.drop('Root Cause', axis=1); y = df['Root Cause']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, X, y
