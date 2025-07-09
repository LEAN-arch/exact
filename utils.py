# utils.py

import pandas as pd
import numpy as np
import plotly.express as px
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
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# (Custom Plotly Template and all non-ML functions remain the same)
# ... (template code, generate_project_data, generate_risk_data, etc.) ...
def generate_project_data(): #...
def generate_risk_data(): #...
def generate_linearity_data(): #...
def generate_precision_data(): #...
def generate_msa_data(): #...
def generate_spc_data(): #...
def generate_lot_data(): #...
def detect_westgard_rules(df, value_col='Value'): #...
def generate_specificity_data(): #...
def generate_doe_data(): #...
def fit_rsm_model_and_optimize(df): #...
def generate_instrument_health_data(): #...
def generate_multivariate_qc_data(): #...
def generate_rca_data(): #...

# --- NEW: Autoencoder Data Generation ---
def generate_golden_batch_data():
    """Generates a 'golden batch' of normal multivariate QC data for autoencoder training."""
    np.random.seed(123)
    data = []
    for i in range(200): # Larger batch for stable training
        op_noise = np.random.normal(0, 0.1)
        lot_noise = np.random.normal(0, 0.2)
        val = 100 + op_noise + lot_noise + np.random.normal(0, 1.5)
        sensor1 = 5.0 + (val - 100) * 0.1 + np.random.normal(0, 0.05)
        sensor2 = 22.5 - (val - 100) * 0.05 + np.random.normal(0, 0.1)
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
        val = 100 + (i-50)*0.2; sensor1 = 5.0 + (val-100)*0.1 + (i-50)*0.05
        sensor2 = 22.5 - (val-100)*0.05
        live_data.append({'Value': val, 'Sensor 1': sensor1, 'Sensor 2': sensor2, 'Run': i})
        
    live_data.append({'Value': 110, 'Sensor 1': 4.8, 'Sensor 2': 25.0, 'Run': 60})
    return pd.DataFrame(live_data)


# --- REINVENTED & UPGRADED ML Model Functions ---

def train_autoencoder_model(golden_df):
    """Trains a Keras Autoencoder on normal data only."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(golden_df)
    n_features = X_train.shape[1]
    
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(n_features, activation='relu'),
        layers.Dense(2, activation='relu', name="bottleneck"), # Bottleneck layer
        layers.Dense(n_features, activation='relu'),
        layers.Dense(n_features, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mae')
    model.fit(X_train, X_train, epochs=50, batch_size=16, validation_split=0.1, verbose=0)
    
    return model, scaler

def train_instrument_model(df):
    """UPGRADED: Trains a LightGBM model to predict instrument failure."""
    X = df[['Pressure (psi)', 'Column Temp (°C)', 'Flow Rate Stability']]
    y = df['Failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X

def train_rca_model(df):
    """UPGRADED: Trains a robust RandomForest model for Root Cause Analysis."""
    X = df.drop('Root Cause', axis=1)
    y = df['Root Cause']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, X, y
