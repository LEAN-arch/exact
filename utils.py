# utils.py
import pandas as pd
import numpy as np
import plotly.io as pio
from datetime import date, timedelta
from scipy import stats

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

# --- Advanced Data Generation Functions ---
def generate_project_data():
    data = {
        'Project/Assay': ['NGS-Assay-X', 'RT-PCR-Assay-Y', 'HPLC-Method-Z', 'QC-LIMS-Script-A'],
        'Project Lead': ['S. Scientist', 'S. Scientist', 'J. Doe', 'S. Scientist'],
        'Current Phase': ['Validation', 'Development', 'Monitoring', 'Design'],
        'Completion %': [75, 40, 100, 15],
        'Overall Status': ['On Track', 'At Risk', 'Complete', 'On Track'],
        'Start Date': [date.today() - timedelta(days=45), date.today() - timedelta(days=20), date.today() - timedelta(days=90), date.today() - timedelta(days=5)],
        'Due Date': [date.today() + timedelta(days=14), date.today() + timedelta(days=25), date.today() - timedelta(days=30), date.today() + timedelta(days=40)],
    }
    return pd.DataFrame(data)

def generate_risk_data():
    data = {
        'Risk ID': ['R-01', 'R-02', 'R-03', 'R-04', 'R-05'],
        'Project': ['RT-PCR-Assay-Y', 'NGS-Assay-X', 'NGS-Assay-X', 'QC-LIMS-Script-A', 'RT-PCR-Assay-Y'],
        'Description': ['Key reagent supplier has long lead time.', 'Unexpected noise in baseline signal.', 'Bioinformatics pipeline validation requires more compute time.', 'Cloud environment has intermittent latency.', 'Instrument availability conflict with R&D.'],
        'Impact': ['High', 'High', 'Medium', 'Low', 'Medium'],
        'Probability': ['Medium', 'Low', 'High', 'Medium', 'High'],
        'Owner': ['S. Scientist', 'J. Doe', 'S. Scientist', 'IT', 'Ops Manager'],
    }
    df = pd.DataFrame(data)
    # Calculate Risk Score for prioritization
    impact_map = {'Low': 1, 'Medium': 2, 'High': 3}
    prob_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Impact_Score'] = df['Impact'].map(impact_map)
    df['Prob_Score'] = df['Probability'].map(prob_map)
    df['Risk_Score'] = df['Impact_Score'] * df['Prob_Score']
    return df.sort_values(by='Risk_Score', ascending=False)

def generate_linearity_data():
    expected = np.array([10, 50, 100, 250, 500, 750, 1000])
    # Introduce slight non-linearity at the high end
    non_linear_factor = 1 - (expected / 4000)
    observed = (expected * np.random.normal(1.02, 0.03, expected.shape) + np.random.normal(0, 5, expected.shape)) * non_linear_factor
    return pd.DataFrame({'Expected Concentration': expected, 'Observed Signal': observed})

def generate_precision_data():
    days = ['Day 1', 'Day 2', 'Day 3']
    operators = ['Op 1', 'Op 2']
    data = []
    for day in days:
        for op in operators:
            # Introduce a slight bias for Operator 2
            mean = 101 if op == 'Op 2' else 99
            # Introduce slightly more variance on Day 3
            stdev = 1.8 if day == 'Day 3' else 1.2
            values = np.random.normal(loc=mean, scale=stdev, size=5)
            for val in values:
                data.append({'Day': day, 'Operator': op, 'Value': val})
    return pd.DataFrame(data)

def generate_msa_data():
    return {
        'part_var': 95.2,
        'repeatability_var': 2.8,
        'reproducibility_var': 2.0
    }

def generate_spc_data():
    np.random.seed(42)
    data = np.random.normal(loc=100, scale=1.5, size=30)
    # Westgard Rule Violations
    data[10] = 104  # 1_2s warning
    data[11] = 103.5 # 2_2s violation
    data[15] = 105.1 # 1_3s violation
    data[20:24] = [102.5, 102.8, 103.1, 103.5] # 4_1s violation
    return pd.DataFrame({'Value': data, 'Run': range(1, 31)})

def generate_lot_data():
    np.random.seed(0)
    lots = ['Lot A', 'Lot B', 'Lot C (New)', 'Lot D']
    data = []
    for lot in lots:
        # Introduce a statistically significant bias in Lot C
        mean = 104 if lot == 'Lot C (New)' else 100
        values = np.random.normal(loc=mean, scale=1.5, size=20)
        for val in values:
            data.append({'Lot ID': lot, 'Performance Metric': val})
    return pd.DataFrame(data)

def detect_westgard_rules(df, value_col='Value'):
    """Detects common Westgard rule violations in a DataFrame."""
    mean = df[value_col].mean()
    std = df[value_col].std()
    violations = []
    
    for i in range(len(df)):
        val = df.loc[i, value_col]
        if val > mean + 3 * std or val < mean - 3 * std:
            violations.append({'Run': df.loc[i, 'Run'], 'Value': val, 'Rule': '1_3s'})
        elif val > mean + 2 * std or val < mean - 2 * std:
            # Check for 2_2s
            if i > 0 and (df.loc[i-1, value_col] > mean + 2 * std or df.loc[i-1, value_col] < mean - 2 * std):
                 violations.append({'Run': df.loc[i, 'Run'], 'Value': val, 'Rule': '2_2s'})
            else: # Just a warning
                 violations.append({'Run': df.loc[i, 'Run'], 'Value': val, 'Rule': '1_2s (Warning)'})
        # Check for 4_1s
        if i >= 3:
            last_4 = df.loc[i-3:i, value_col]
            if all(last_4 > mean + std) or all(last_4 < mean - std):
                violations.append({'Run': df.loc[i, 'Run'], 'Value': val, 'Rule': '4_1s'})
    
    return pd.DataFrame(violations).drop_duplicates(subset=['Run'])
    # utils.py
# (Keep all previous functions: generate_project_data, generate_risk_data, etc.)

# --- NEW ML-SPECIFIC LIBRARIES ---
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap

# --- NEW ML DATA GENERATION ---

def generate_instrument_health_data():
    """Generates multivariate time-series data for an instrument (e.g., HPLC)."""
    np.random.seed(101)
    runs = 100
    pressure = np.linspace(1500, 1550, runs) + np.random.normal(0, 5, runs)
    temperature = np.linspace(35, 35.5, runs) + np.random.normal(0, 0.1, runs)
    # Introduce a drift leading to failure
    pressure[80:] += np.linspace(0, 50, 20)
    temperature[80:] += np.linspace(0, 1.5, 20)
    flow_rate_stability = np.random.normal(0.99, 0.005, runs)
    flow_rate_stability[85:] -= np.linspace(0, 0.05, 15)
    
    df = pd.DataFrame({
        'Run ID': range(runs),
        'Pressure (psi)': pressure,
        'Column Temp (°C)': temperature,
        'Flow Rate Stability': flow_rate_stability
    })
    # Failure is defined as the last 10 runs
    df['Failure'] = 0
    df.loc[90:, 'Failure'] = 1
    return df

def generate_multivariate_qc_data():
    """Generates QC data with multiple correlated factors."""
    np.random.seed(42)
    data = []
    for i in range(100):
        op = 'Operator A' if i % 2 == 0 else 'Operator B'
        lot = 'Lot X' if i < 50 else 'Lot Y'
        val = np.random.normal(100, 2)
        # Introduce subtle shifts
        if op == 'Operator B': val += 0.5
        if lot == 'Lot Y': val += 0.7
        data.append({'Run': i, 'Operator': op, 'Reagent Lot': lot, 'Value': val})
    df = pd.DataFrame(data)
    # Add a clear anomaly that univariate rules might miss
    df.loc[80, 'Value'] -= 3
    df.loc[80, 'Operator'] = 'Operator B' # The combination is anomalous
    return df

def generate_rca_data():
    """Generates historical failure data for root cause analysis."""
    np.random.seed(0)
    n_samples = 200
    instrument_age = np.random.randint(1, 36, n_samples)
    reagent_lot_age = np.random.randint(1, 90, n_samples)
    operator_experience = np.random.randint(1, 5, n_samples)
    causes = []
    for i in range(n_samples):
        if reagent_lot_age[i] > 75 or (reagent_lot_age[i] > 60 and np.random.rand() > 0.3):
            causes.append('Reagent Degradation')
        elif instrument_age[i] > 30 or (instrument_age[i] > 20 and np.random.rand() > 0.5):
            causes.append('Instrument Drift')
        elif operator_experience[i] == 1 and np.random.rand() > 0.4:
            causes.append('Operator Error')
        else:
            causes.append('No Fault Found')
    return pd.DataFrame({
        'Instrument Age (mo)': instrument_age,
        'Reagent Lot Age (days)': reagent_lot_age,
        'Operator Experience (yr)': operator_experience,
        'Root Cause': causes
    })

# --- NEW ML MODEL TRAINING FUNCTIONS ---

def train_instrument_model(df):
    """Trains a RandomForest model to predict instrument failure."""
    X = df[['Pressure (psi)', 'Column Temp (°C)', 'Flow Rate Stability']]
    y = df['Failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X

def train_anomaly_model(df):
    """Trains an IsolationForest model for anomaly detection."""
    le_op = LabelEncoder()
    le_lot = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['Operator'] = le_op.fit_transform(df_encoded['Operator'])
    df_encoded['Reagent Lot'] = le_lot.fit_transform(df_encoded['Reagent Lot'])
    model = IsolationForest(contamination=0.02, random_state=42) # Expect 2% anomalies
    model.fit(df_encoded[['Operator', 'Reagent Lot', 'Value']])
    return model, df_encoded

def train_rca_model(df):
    """Trains a DecisionTree model for root cause analysis."""
    X = df.drop('Root Cause', axis=1)
    y = df['Root Cause']
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    return model, X, y
