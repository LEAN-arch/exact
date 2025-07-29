# utils.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date, timedelta
from scipy import stats
from scipy.optimize import minimize

# --- ML/Advanced Analytics Library Imports ---
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

# --- Custom Plotly Template for Exact Sciences ---
exact_sciences_template = {
    "layout": {
        "font": {"family": "Helvetica, Arial, sans-serif", "size": 12, "color": "#333333"},
        "title": {"font": {"family": "Helvetica, Arial, sans-serif", "size": 18, "color": "#1A3A6D"}, "x": 0.05},
        "plot_bgcolor": "#FFFFFF",
        "paper_bgcolor": "#FFFFFF",
        "colorway": px.colors.qualitative.Plotly,
        "xaxis": {"gridcolor": "#E5E5E5", "linecolor": "#B0B0B0", "zerolinecolor": "#E5E5E5", "title_font": {"size": 14}},
        "yaxis": {"gridcolor": "#E5E5E5", "linecolor": "#B0B0B0", "zerolinecolor": "#E5E5E5", "title_font": {"size": 14}},
        "legend": {"bgcolor": "rgba(255,255,255,0.85)", "bordercolor": "#CCCCCC", "borderwidth": 1}
    }
}
pio.templates["exact_sciences"] = exact_sciences_template
pio.templates.default = "exact_sciences"

# === CORE DATA GENERATION (Adapted for Exact Sciences) ===
def generate_project_data():
    """Generates project data specific to Exact Sciences' product pipeline and transfer activities."""
    data = {
        'Project/Assay': [
            'Oncotype DX® QC Software Pipeline v2.1',
            'Cologuard® Reagent Qualification (HPLC)',
            'OncoExTra® Library Prep Automation Script',
            'Riskguard™ Bioinformatics Pipeline Update',
            'Cologuard® Methylation Assay Control Monitoring'
        ],
        'Product Line': ['Oncotype DX', 'Cologuard', 'OncoExTra', 'Riskguard', 'Cologuard'],
        'Project Lead': ['S. Scientist', 'J. Doe', 'S. Scientist', 'A. Turing', 'S. Scientist'],
        'Current Phase': ['Validation', 'Development', 'Design', 'On Hold', 'Monitoring'],
        'Completion %': [85, 50, 20, 95, 100],
        'Overall Status': ['On Track', 'At Risk', 'On Track', 'On Hold', 'Complete'],
        'Start Date': [date.today() - timedelta(days=60), date.today() - timedelta(days=30), date.today() - timedelta(days=10), date.today() - timedelta(days=120), date.today() - timedelta(days=180)],
        'Due Date': [date.today() + timedelta(days=20), date.today() + timedelta(days=45), date.today() + timedelta(days=60), date.today() + timedelta(days=15), date.today() - timedelta(days=90)],
    }
    df = pd.DataFrame(data)
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['Due Date'] = pd.to_datetime(df['Due Date'])
    return df

def generate_risk_data():
    """Generates risk data relevant to molecular diagnostics and software transfer."""
    data = {
        'Risk ID': ['R-ODX-01', 'R-CG-01', 'R-OEX-01', 'R-SW-01', 'R-SUP-01'],
        'Project': [
            'Oncotype DX® QC Software Pipeline v2.1',
            'Cologuard® Methylation Assay Control Monitoring',
            'OncoExTra® Library Prep Automation Script',
            'Riskguard™ Bioinformatics Pipeline Update',
            'Cologuard® Reagent Qualification (HPLC)'
        ],
        'Description': [
            'Algorithm change may impact Recurrence Score® reproducibility at clinical cutoffs.',
            'New bisulfite conversion kit lot shows minor shift in performance.',
            'Variability in plasticware affects automated liquid handling accuracy.',
            'Cloud computing environment upgrade may break legacy code dependencies.',
            'Single-source supplier for a critical HPLC column faces production delays.'
        ],
        'Impact': ['Critical', 'Moderate', 'Serious', 'Serious', 'High'],
        'Probability': ['Low', 'High', 'Medium', 'Medium', 'Medium'],
        'Owner': ['S. Scientist', 'QC Ops', 'MSAT', 'IT / Bioinformatics', 'Supply Chain'],
        'Mitigation': ['Extensive validation with clinical samples.', 'Perform guard-banding study.', 'Qualify secondary vendor.', 'Create containerized environment.', 'Qualify second source column.']
    }
    df = pd.DataFrame(data)
    impact_map = {'Negligible': 1, 'Minor': 2, 'Moderate': 3, 'Serious': 4, 'Critical': 5}
    prob_map = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5}
    df['Impact_Score'] = df['Impact'].map(impact_map).fillna(df['Impact'].map({'High': 4}))
    df['Prob_Score'] = df['Probability'].map(prob_map).fillna(df['Probability'].map({'Medium': 3, 'High': 4}))
    df['Risk_Score'] = df['Impact_Score'] * df['Prob_Score']
    return df.sort_values(by='Risk_Score', ascending=False)

def generate_linearity_data():
    """Generates linearity data for a quantitative RT-PCR assay (e.g., Oncotype DX® target gene)."""
    expected_log = np.array([2, 3, 4, 5, 6, 7])
    ct_values = 38 - 3.3 * (expected_log - (expected_log**2 / 20))
    observed_ct = ct_values + np.random.normal(0, 0.15, expected_log.shape)
    return pd.DataFrame({'Log10 Target Concentration': expected_log, 'Observed Ct Value': observed_ct})

def generate_precision_data():
    """Generates precision data (Ct values) for an Oncotype DX® QC sample."""
    days = ['Day 1', 'Day 2', 'Day 3']; operators = ['Op 1', 'Op 2']; data = []
    for day in days:
        for op in operators:
            mean_ct = 24.6 if op == 'Op 1' else 24.75
            stdev_ct = 0.25 if day == 'Day 3' else 0.15
            values = np.random.normal(loc=mean_ct, scale=stdev_ct, size=5)
            for val in values: data.append({'Day': day, 'Operator': op, 'Ct Value': val})
    return pd.DataFrame(data)

def generate_msa_data():
    """Generates MSA data for a Cologuard® methylation assay signal."""
    return {'part_var': 85.0, 'repeatability_var': 5.0, 'reproducibility_var': 10.0}

def generate_specificity_data():
    """Generates specificity data for a Cologuard® methylation marker, testing for interference."""
    np.random.seed(33); data = []
    data.extend([{'Sample Type': 'NTC (No Template Control)', 'Signal (% Meth)': v} for v in np.random.uniform(0.1, 0.5, 10)])
    data.extend([{'Sample Type': 'Methylated Control', 'Signal (% Meth)': v} for v in np.random.normal(95, 2, 10)])
    data.extend([{'Sample Type': 'Interferent (Hemoglobin)', 'Signal (% Meth)': v} for v in np.random.uniform(0.2, 0.8, 10)])
    data.extend([{'Sample Type': 'Interferent (Bilirubin)', 'Signal (% Meth)': v} for v in np.random.uniform(0.2, 0.7, 10)])
    data.extend([{'Sample Type': 'Control + Interferents', 'Signal (% Meth)': v} for v in np.random.normal(88, 3, 10)])
    return pd.DataFrame(data)

def generate_spc_data():
    """Generates Levey-Jennings data (Ct values) for an Oncotype DX® positive control."""
    np.random.seed(42);
    mean_ct = 25.0; std_ct = 0.2
    data = np.random.normal(loc=mean_ct, scale=std_ct, size=30)
    data[15:20] = data[15:20] - 0.3
    data[25] = mean_ct + 3.5 * std_ct
    return pd.DataFrame({'Ct Value': data, 'Run': range(1, 31)})

def detect_westgard_rules(df, value_col='Ct Value'):
    """Detects Westgard rules, adapted for Ct values where lower is better."""
    mean = df[value_col].mean(); std = df[value_col].std(); violations = []
    for i in range(len(df)):
        val = df.loc[i, value_col]
        if val > mean + 3 * std or val < mean - 3 * std: violations.append({'Run': df.loc[i, 'Run'], 'Value': val, 'Rule': '1_3s'})
        if i >= 1:
            last_2 = df.loc[i-1:i, value_col]
            if all(last_2 > mean + 2 * std) or all(last_2 < mean - 2 * std):
                violations.append({'Run': df.loc[i, 'Run'], 'Value': val, 'Rule': '2_2s'})
        if i >= 3:
            last_4 = df.loc[i-3:i, value_col]
            if all(last_4 > mean + std) or all(last_4 < mean - std):
                violations.append({'Run': df.loc[i, 'Run'], 'Value': val, 'Rule': '4_1s'})
        if i >= 9:
            last_10 = df.loc[i-9:i, value_col]
            if all(last_10 > mean) or all(last_10 < mean):
                violations.append({'Run': df.loc[i, 'Run'], 'Value': val, 'Rule': '10_x'})
    return pd.DataFrame(violations).drop_duplicates(subset=['Run', 'Rule'])

def generate_lot_data():
    """Generates lot-to-lot data for an OncoExTra® library prep kit (yield in ng)."""
    np.random.seed(0); lots = ['Lot A (Ref)', 'Lot B', 'Lot C', 'Lot D (New)']; data = []
    for lot in lots:
        mean_yield = 75 if lot == 'Lot D (New)' else 90
        stdev = 8 if lot == 'Lot D (New)' else 5
        values = np.random.normal(loc=mean_yield, scale=stdev, size=20)
        for val in values: data.append({'Lot ID': lot, 'Library Yield (ng)': val})
    return pd.DataFrame(data)

def calculate_cpk(data_series, usl, lsl):
    """Calculates the Cpk for a given series of data and spec limits."""
    mean = data_series.mean()
    std_dev = data_series.std()
    if std_dev == 0: return np.inf
    cpu = (usl - mean) / (3 * std_dev)
    cpl = (mean - lsl) / (3 * std_dev)
    return min(cpu, cpl)

def generate_doe_data():
    """Generates DOE data for an RT-PCR annealing step optimization."""
    np.random.seed(42)
    temp_levels = np.array([-1.414, -1, 1, -1, 1, 0, 0, 0, 0, -1.414, 1.414])
    primer_levels = np.array([0, -1, -1, 1, 1, -1.414, 1.414, 0, 0, 0, 0])
    temp_real = temp_levels * 2 + 60
    primer_real = primer_levels * 50 + 200
    true_efficiency = 95 - (2*temp_levels**2) - (3*primer_levels**2) + (1.5*temp_levels*primer_levels)
    measured_efficiency = true_efficiency + np.random.normal(0, 0.8, len(temp_real))
    return pd.DataFrame({'Annealing Temp (°C)': temp_real, 'Primer Conc. (nM)': primer_real, 'PCR Efficiency (%)': measured_efficiency})

def fit_rsm_model_and_optimize(df):
    """Fits a quadratic response surface model and finds the settings for maximum efficiency."""
    X = df.iloc[:, :-1]; y = df.iloc[:, -1]
    poly = PolynomialFeatures(degree=2, include_bias=False); X_poly = poly.fit_transform(X)
    model = LinearRegression(); model.fit(X_poly, y)
    def neg_response(params):
        x_in = pd.DataFrame([params], columns=X.columns)
        x_in_poly = poly.transform(x_in)
        return -model.predict(x_in_poly)[0]
    initial_guess = [X.iloc[:,0].mean(), X.iloc[:,1].mean()]
    bounds = [(X.iloc[:,0].min(), X.iloc[:,0].max()), (X.iloc[:,1].min(), X.iloc[:,1].max())]
    result = minimize(neg_response, initial_guess, method='L-BFGS-B', bounds=bounds)
    return model, poly, result.x, -result.fun

def generate_instrument_health_data():
    """Generates health data for an NGS Sequencer (e.g., NovaSeq)."""
    np.random.seed(101); runs = 100
    laser_power_A = 85 - np.linspace(0, 5, runs) + np.random.normal(0, 0.2, runs)
    flow_cell_temp_C = 50 + np.random.normal(0, 0.1, runs)
    pump_pressure_psi = 10 + np.sin(np.linspace(0, 10 * np.pi, runs)) * 0.5 + np.random.normal(0, 0.1, runs)
    laser_power_A[85:] -= np.logspace(0, 1, 15)
    flow_cell_temp_C[90:] += np.random.normal(0.5, 0.2, 10)
    df = pd.DataFrame({
        'Run_ID': range(runs), 'Laser_A_Power': laser_power_A,
        'Flow_Cell_Temp_C': flow_cell_temp_C, 'Pump_Pressure_psi': pump_pressure_psi
    })
    df['Failure'] = 0; df.loc[92:, 'Failure'] = 1
    return df

def train_instrument_model(df):
    """Trains a LightGBM model to predict sequencer failure."""
    X = df.drop(['Failure', 'Run_ID'], axis=1); y = df['Failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
    model.fit(X_train, y_train)
    return model, X

def generate_rca_data():
    """Generates RAW Root Cause Analysis data for Cologuard® assay failures."""
    np.random.seed(0); n_samples = 200
    reagent_lot_age = np.random.randint(1, 120, n_samples)
    operator_id = np.random.choice([101, 102, 103, 104], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    instrument_id = np.random.choice(['HML-01', 'HML-02', 'PCR-05'], n_samples, p=[0.5, 0.3, 0.2])
    causes = []
    for i in range(n_samples):
        if 'HML-02' == instrument_id[i] and np.random.rand() > 0.4: causes.append('Liquid Handler Pipetting Error')
        elif reagent_lot_age[i] > 90 or (reagent_lot_age[i] > 75 and np.random.rand() > 0.5): causes.append('Reagent Degradation')
        elif 104 == operator_id[i] and np.random.rand() > 0.6: causes.append('Operator/Sample Handling')
        else: causes.append('No Fault Found')
    df = pd.DataFrame({'Reagent Lot Age (days)': reagent_lot_age, 'Operator ID': operator_id, 'Instrument ID': instrument_id, 'Root Cause': causes})
    return df

def train_rca_model(df):
    """Trains a RandomForest model for Cologuard® Root Cause Analysis."""
    X = df.drop('Root Cause', axis=1); y = df['Root Cause']
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, class_weight='balanced')
    model.fit(X, y)
    return model, X, y

def generate_traceability_data():
    """Generates traceability data for an Oncotype DX® QC software validation."""
    return {
        'Requirement ID': ['URS-01', 'URS-02', 'URS-03', 'URS-04'],
        'User Requirement (URS)': [
            'System shall correctly calculate Recurrence Score® from input Ct values.',
            'System shall flag runs if Positive Control Ct is outside of spec.',
            'System must be 21 CFR Part 11 compliant with audit trails.',
            'System shall generate a locked PDF report of results.'
        ],
        'Functional Spec (FRS)': ['FRS-1.1, FRS-1.2', 'FRS-2.1', 'FRS-3.1, FRS-3.2', 'FRS-4.1'],
        'Test Case ID': ['TC-CALC-001-005', 'TC-FLAG-001-003', 'TC-P11-001-008', 'TC-RPT-001'],
        'Test Status': ['Pass', 'Pass', 'Fail', 'In Progress']
    }

def generate_defect_category_data():
    """Generates defect Pareto data for a bioinformatics pipeline."""
    return pd.DataFrame({
        'Category': ['Bioinformatics Algorithm', 'Data I/O & Parsing', 'UI/UX', 'Database Integration', 'Performance/Speed', 'Reporting'],
        'Count': [15, 8, 5, 3, 2, 1]
    }).sort_values('Count', ascending=False)

def generate_instrument_schedule_data():
    """Generates schedule data for high-complexity molecular diagnostics instruments."""
    today = pd.Timestamp.now().normalize()
    data = [
        {'Instrument': 'NovaSeq-01', 'Start': today - timedelta(hours=8), 'Finish': today + timedelta(days=2), 'Status': 'In Use', 'Details': 'OncoExTra Batch 24-101'},
        {'Instrument': 'NovaSeq-02', 'Start': today - timedelta(days=1), 'Finish': today + timedelta(days=1), 'Status': 'OOS', 'Details': 'OOS-451: Flow Cell Temp Failure'},
        {'Instrument': 'Hamilton-01', 'Start': today + timedelta(hours=1), 'Finish': today + timedelta(hours=5), 'Status': 'Scheduled', 'Details': 'Cologuard Lib Prep Validation'},
        {'Instrument': 'Hamilton-02', 'Start': today, 'Finish': today + timedelta(days=2), 'Status': 'Available', 'Details': 'Open for scheduling'},
        {'Instrument': 'QuantStudio-01', 'Start': today - timedelta(days=2), 'Finish': today, 'Status': 'PM Due', 'Details': 'Annual Preventative Maintenance'},
        {'Instrument': 'QuantStudio-02', 'Start': today, 'Finish': today + timedelta(hours=3), 'Status': 'In Use', 'Details': 'Oncotype DX Batch 24-305'},
    ]
    return pd.DataFrame(data)

def generate_training_data_for_heatmap():
    """Generates training competency data for QC analysts at Exact Sciences."""
    data = {
        'SOP-001 (Safety)': [2, 2, 2, 2],
        'TM-101 (HPLC)': [2, 2, 1, 0],
        'TM-201 (Oncotype RT-PCR)': [2, 2, 2, 1],
        'TM-202 (OncoExTra NGS Lib Prep)': [2, 1, 0, 0],
        'SW-301 (OncoExTra Bioinformatics)': [2, 0, 0, 0],
    }
    df = pd.DataFrame(data, index=['S. Scientist (Lead)', 'Jane Smith (Sr. Analyst)', 'John Doe (Analyst)', 'Peter Jones (New Hire)'])
    return df

def generate_reagent_lot_status_data():
    """Generates status data for critical reagent and consumable lots."""
    today = date.today()
    data = {
        'Lot ID': ['CG-BC-2401', 'ODX-PK-2399', 'OEX-LPK-2405', 'OEX-LPK-2406', 'CG-EB-2350'],
        'Reagent/Kit': ['Cologuard Bisulfite Conv.', 'Oncotype DX Probe Kit', 'OncoExTra Lib Prep Kit', 'OncoExTra Lib Prep Kit', 'Cologuard Elution Buffer'],
        'Status': ['In Use', 'In Use', 'In Qualification', 'On Hold', 'Expired'],
        'Expiry Date': [today + timedelta(days=90), today + timedelta(days=45), today + timedelta(days=180), today + timedelta(days=182), today - timedelta(days=5)],
        'Quantity Remaining (%)': [60, 35, 100, 100, 0],
        'Notes': ['Nominal performance.', 'Monitor closely, nearing re-order point.', 'Awaiting OOS investigation results.', 'DO NOT USE - Failed incoming QC.', 'Remove from inventory.']
    }
    return pd.DataFrame(data)

def generate_v_model_data():
    """Returns coordinates and labels for drawing a V-Model diagram."""
    data = {
        'x': [1, 2, 3, 4, 5, 6, 7, 8],
        'y': [4, 3, 2, 1, 1, 2, 3, 4],
        'text': [
            "User Requirements<br>(URS)", "System & Software<br>Requirements (SRS)", "Architectural Design", "Module Design",
            "Coding &<br>Implementation", "Unit &<br>Component Testing", "Integration Testing", "System & Acceptance<br>Testing (V&V)"
        ]
    }
    return pd.DataFrame(data)

def generate_defect_trend_data():
    """Generates time-series data for a defect burnup/burndown chart."""
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start="2024-06-01", periods=30, freq='D'))
    opened = np.random.randint(0, 4, size=30).cumsum() + 5
    closed = (opened * np.random.uniform(0.5, 0.9, size=30)).astype(int)
    closed[20:] = np.clip(closed[20:] + 5, 0, opened[20:])
    return pd.DataFrame({'Date': dates, 'Opened': opened, 'Closed': closed})

def generate_capa_source_data():
    """Generates data for a CAPA source Pareto chart specific to a diagnostics company."""
    return pd.DataFrame({
        'Source': ['Test Method OOS', 'Internal Audit', 'Bioinformatics Anomaly', 'Supplier Non-conformance', 'Customer Complaint (Clinical)', 'Process Trend'],
        'Count': [12, 7, 5, 3, 2, 1]
    }).sort_values('Count', ascending=False)

# ==============================================================================
# --- NEW FUNCTIONS FOR AI-DRIVEN FEATURES (EXPERT INTEGRATION) ---
# ==============================================================================
# In utils.py

# ... (other functions)

def analyze_fto(invention_desc: str):
    """Simulates an LLM call to analyze Freedom to Operate for a PROTAC, now with a novelty score."""
    # CORRECTED: Added a 'Recommendation' key to each dictionary to match the UI's expectation.
    return [
        {
            'Aspect of Invention': 'General PROTAC Scaffold',
            'Analysis': "Broadly claimed. Not novel.",
            'Risk Level': 'High',
            'Novelty Score': 1,
            'Recommendation': 'Requires detailed legal review. Avoid broad claims in patent application.'
        },
        {
            'Aspect of Invention': 'Target Protein (BTK)',
            'Analysis': "Targeting BTK for degradation is known.",
            'Risk Level': 'High',
            'Novelty Score': 2,
            'Recommendation': 'Ensure specific BTK binder moiety is not covered by existing chemical matter patents.'
        },
        {
            'Aspect of Invention': 'E3 Ligase Binder (RNF114)',
            'Analysis': "Novel ligase not claimed in key competitor IP.",
            'Risk Level': 'Low',
            'Novelty Score': 9,
            'Recommendation': 'Focus patent strategy here. This is the core inventive step. Generate extensive data.'
        },
        {
            'Aspect of Invention': 'Linker (PEG, 5-8 units)',
            'Analysis': "Specific length/composition may be novel but could be seen as obvious.",
            'Risk Level': 'Medium',
            'Novelty Score': 5,
            'Recommendation': 'Characterize if this specific linker composition provides unexpected advantages (e.g., stability, PK).'
        }
    ]

# ... (other functions)
def mock_uniprot_api(target_id: str):
    """Simulates a call to the UniProt API to get protein information."""
    if target_id == "P01116":  # UniProt ID for KRAS
        return {
            "Function": "Ras proteins bind GDP/GTP and possess intrinsic GTPase activity. Plays an important role in the regulation of cell proliferation and differentiation. Mutations in the KRAS gene are associated with various malignancies, including colorectal cancer.",
            "Pathway Association": "EGFR signaling pathway, MAP Kinase signaling pathway"
        }
    return {}

def mock_pubmed_api(query: str):
    """Simulates a call to the PubMed API to get recent publication titles."""
    if "KRAS G12C" in query:
        return [
            "Combined KRAS G12C and EGFR Inhibition in Colorectal Cancer.",
            "Mechanisms of acquired resistance to KRAS G12C-selective inhibitors.",
            "SHP2 inhibition as a strategy to overcome resistance to KRAS G12C inhibitors."
        ]
    return []

def generate_hypothesis_data(internal_data: dict, external_data: dict):
    """Simulates an LLM call to generate therapeutic hypotheses and the data for plotting."""
    hypotheses = [
        {
            "#": 1,
            "Therapeutic Hypothesis": "Overcome Acquired Resistance to KRAS G12C Inhibitors",
            "Scientific Rationale": "The literature clearly shows adaptive resistance to KRAS G12C inhibitors (like Sotorasib) often involves reactivation of the EGFR/MAPK pathway. Our compound, targeting a downstream node, could re-sensitize resistant tumors.",
            "Suggested \"Next Experiment\"": "Test Cmpd-X in combination with Sotorasib on a colorectal cancer cell line made resistant to Sotorasib (e.g., SW837-Resistant). Look for synergistic cell killing and inhibition of p-ERK."
        },
        {
            "#": 2,
            "Therapeutic Hypothesis": "Companion Diagnostic for Cologuard® Positive Patients",
            "Scientific Rationale": "A subset of Cologuard-positive patients have KRAS G12C mutations. A potent inhibitor of a key downstream target could be a viable therapeutic strategy for this specific, pre-identified patient population.",
            "Suggested \"Next Experiment\"": "Screen Cmpd-X against a panel of KRAS G12C-mutant colorectal cancer cell lines (e.g., SW837, HCT-116). Assess for antiproliferative effects as a monotherapy."
        },
        {
            "#": 3,
            "Therapeutic Hypothesis": "Synthetic Lethality with SHP2 Inhibition",
            "Scientific Rationale": "PubMed data suggests SHP2 inhibition is an emerging strategy to potentiate KRAS G12C blockade. Combining our compound with a SHP2 inhibitor could provide a powerful, dual-node blockade of the MAPK pathway.",
            "Suggested \"Next Experiment\"": "Perform a combination matrix study with Cmpd-X and a SHP2 inhibitor (e.g., TNO155) on multiple KRAS G12C-mutant cell lines. Calculate a synergy score (e.g., using Bliss or HSA models)."
        }
    ]
    graph_data = {
        'nodes': {'KRAS G12C': (2, 3), 'MAPK/ERK Pathway': (4, 3), 'Lung Cancer': (6, 4), 'Melanoma': (6, 3), 'CRC': (6, 2)},
        'edges': [('KRAS G12C', 'MAPK/ERK Pathway')],
        'annotations': [
            {'source': 'MAPK/ERK Pathway', 'target': 'Lung Cancer', 'text': 'Resistance to Osimertinib'},
            {'source': 'MAPK/ERK Pathway', 'target': 'Melanoma', 'text': 'Known Escape Pathway'},
            {'source': 'KRAS G12C', 'target': 'CRC', 'text': 'Cologuard® Dx Link'}
        ]
    }
    return hypotheses, graph_data

def mock_patent_api(competitors: list):
    """Simulates a call to a patent database API for PROTACs."""
    return [
        {"id": "US-10,123,456 B2", "owner": "Arvinas", "claim": "Claim 1: A bifunctional compound comprising a ubiquitin ligase binding moiety covalently linked to a protein target binding moiety, wherein the ubiquitin ligase is Cereblon."},
        {"id": "US-11,234,567 B2", "owner": "Kymera Therapeutics", "claim": "Claim 1: A compound represented by the structure T-L-E, wherein T is a moiety that binds a target protein, L is a linker, and E is a moiety that binds to the VHL E3 ubiquitin ligase."},
        {"id": "WO-2023-12345 A1", "owner": "Genentech", "claim": "Claim 1: A compound that effectuates the degradation of a target protein, wherein said compound comprises a moiety that binds a KRAS protein."}
    ]

def analyze_fto(invention_desc: str):
    """Simulates an LLM call to analyze Freedom to Operate for a PROTAC, now with a novelty score."""
    return [
        {'Aspect of Invention': 'General PROTAC Scaffold', 'Analysis': "Broadly claimed. Not novel.", 'Risk Level': 'High', 'Novelty Score': 1},
        {'Aspect of Invention': 'Target Protein (BTK)', 'Analysis': "Targeting BTK for degradation is known.", 'Risk Level': 'High', 'Novelty Score': 2},
        {'Aspect of Invention': 'E3 Ligase Binder (RNF114)', 'Analysis': "Novel ligase not claimed in key competitor IP.", 'Risk Level': 'Low', 'Novelty Score': 9},
        {'Aspect of Invention': 'Linker (PEG, 5-8 units)', 'Analysis': "Specific length/composition may be novel but could be seen as obvious.", 'Risk Level': 'Medium', 'Novelty Score': 5}
    ]

def mock_get_sop(sop_id: str):
    """Simulates retrieving an SOP from a document control system for an NGS workflow."""
    if sop_id == "LIBPREP-OEX-003B":
        return "Input DNA must be quantified via Qubit. Normalize all samples to 100ng total input... After ligation, perform a double-sided SPRI bead cleanup using a 0.55X ratio followed by a 0.85X ratio... Elute in 22 µL of elution buffer."
    return "SOP not found."

def mock_get_reagent_info(lot_id: str):
    """Simulates retrieving reagent lot information for NGS kits."""
    if lot_id == "LPK-23-9981":
        return "Kit Age: 9 months. QC Note (J. Doe): 'This lot showed a trend of increased adapter-dimer formation compared to previous lots. Recommend using with KAPA Pure Beads instead of standard SPRI.'"
    return "Lot information not found."

def mock_get_instrument_log(instrument_id: str):
    """Simulates retrieving an instrument log for an NGS sequencer."""
    if instrument_id == "NovaSeq-01":
        return "Last PM: 2 months ago. Status: Active. Last 5 runs show average %Q30 > 92%. No errors reported."
    return "Instrument log not found."

def troubleshoot_experiment(protocol: dict):
    """Simulates an LLM call to troubleshoot a failed experiment, returning structured data for visualization."""
    report = [
        {'Rank': 1, 'Most Likely Root Cause': 'Reagent Quality (Library Prep Kit)', 'Evidence': "Reagent Hub data for Lot #LPK-23-9981 explicitly notes a history of increased adapter-dimer formation. High adapter content is a direct cause of low usable reads and poor Q30 scores.", 'Corrective Action': "Immediately quarantine Lot #LPK-23-9981. Re-prep the failed libraries using a different, qualified lot (e.g., LPK-24-0112). Escalate to Supply Chain to investigate the suspect lot with the vendor."},
        {'Rank': 2, 'Most Likely Root Cause': 'Sub-optimal DNA Input', 'Evidence': f"Your protocol used {protocol['DNA Input']}ng of DNA, which is a significant deviation from the SOP (LIBPREP-OEX-003B) that specifies 100ng. Low input can lead to low library complexity, high duplication rates, and poor quality scores.", 'Corrective Action': "Strictly adhere to the SOP for DNA input. If sample is limited, consult with an SME on low-input-specific protocols, which may require different reagent concentrations or cycle numbers."},
        {'Rank': 3, 'Most Likely Root Cause': 'Bead Cleanup Ratio', 'Evidence': "Your protocol used a 'Standard SPRI' cleanup. The SOP specifies a 'double-sided' cleanup with specific ratios (0.55X / 0.85X) designed to remove small fragments like adapter-dimers. This deviation likely contributed to the high adapter content.", 'Corrective Action': "Follow the double-sided SPRI cleanup protocol exactly as written in the SOP for the next attempt."},
        {'Rank': 4, 'Most Likely Root Cause': 'Sequencer Issue (Unlikely)', 'Evidence': "The instrument log for NovaSeq-01 shows excellent recent performance (Q30 > 92%) and no reported errors. While a sudden instrument failure is possible, it is far less likely than the multiple protocol and reagent deviations noted above.", 'Corrective Action': 'No action required for the instrument at this time. Focus on correcting the library preparation process first.'}
    ]
    comparison_data = [
        {'Parameter': 'DNA Input', 'User Protocol': f"{protocol['DNA Input']} ng", 'SOP Requirement': '100 ng', 'Deviation': protocol['DNA Input'] != 100},
        {'Parameter': 'Bead Cleanup', 'User Protocol': protocol['Cleanup Method'], 'SOP Requirement': 'Double-sided SPRI', 'Deviation': protocol['Cleanup Method'] != 'Double-sided SPRI'},
        {'Parameter': 'Reagent Lot', 'User Protocol': protocol['Library Kit Lot'], 'SOP Requirement': 'Use in-date, qualified lot', 'Deviation': True}
    ]
    return report, comparison_data
