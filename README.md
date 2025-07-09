# Staff Scientist Command Center

## Project Overview

The **Staff Scientist Command Center** is a sophisticated, multi-page web dashboard designed to empower a Staff Scientist in a regulated (FDA/ISO) environment. It provides a centralized, interactive, and data-driven view for leading and coordinating QC Software and Assay Transfer activities.

Built with **Streamlit** and **Plotly**, this dashboard moves beyond simple KPI tracking to offer deep statistical analysis, test method validation insights, post-transfer process monitoring, and cutting-edge machine learning analytics for proactive process control.

This tool is designed to be both an operational command center for daily management and a strategic asset for demonstrating compliance and driving continual improvement.

---

## Features

This dashboard is organized into logical, role-specific pages. The main page is the **Master Dashboard**, and additional pages are available in the sidebar, sorted alphabetically:

#### 🏠 **Master Dashboard (Home Page)**
A high-level command center providing an at-a-glance summary of all active initiatives.
- **Executive KPIs**: Tracks active projects, high-priority risks, and completion rates.
- **Interactive Gantt Chart**: Visualizes project timelines, phases, and dependencies.
- **Risk Matrix Heatmap**: Prioritizes project risks based on impact and probability for effective resource allocation.

#### 📈 **Assay Validation Dashboard**
A scientific deep-dive into the data required for Test Method Validation (TMV).
- **Linearity & Accuracy Analysis**: Includes regression statistics and residual plots to detect bias and non-linearity.
- **Measurement System Analysis (MSA)**: Utilizes sunburst charts and calculates key metrics like Gage R&R and `ndc` to prove measurement system suitability.
- **Precision Analysis (Repeatability & Reproducibility)**: Employs CLSI-style box plots and calculates %CV for different conditions.

#### 🤖 **ML-Driven Analytics**
An investigational page that uses machine learning for proactive and predictive process understanding.
- **Predictive Instrument Health**: A Random Forest model predicts instrument failure from sensor data, explained with an interactive **SHAP plot**.
- **Multivariate Anomaly Detection**: An Isolation Forest model identifies subtle process drifts in a **3D interactive plot** that univariate rules would miss.
- **Automated Root Cause Insights**: A Decision Tree classifier suggests the most likely root cause of a failure, visualized with a clear tree diagram.

#### 🚚 **Project Transfer Hub**
A tactical dashboard for managing the end-to-end transfer process for individual projects.
- **Kanban Board**: Tracks the lifecycle of tasks from design to monitoring.
- **Document Control Tracker**: Monitors the status of critical compliance documents (e.g., Validation Protocols, Reports, SOPs).
- **Risk Mitigation Plan**: Details individual risk items and their corresponding risk scores.

#### 📊 **QC Performance Analytics**
Monitors the health and performance of methods *after* they have been transferred into the operational QC environment.
- **Advanced SPC (Levey-Jennings Chart)**: Implements automated **Westgard Rule** detection to identify out-of-control conditions.
- **Lot-to-Lot Trending with ANOVA**: Uses box plots and statistical tests (ANOVA) to confirm the consistency of critical reagents.


---

## 🏛️ Regulatory Context & Compliance

This dashboard is built with a deep understanding of the regulatory landscape for medical devices and in vitro diagnostics (IVDs). Each page contains expandable "Regulatory Context" legends that explicitly link the visualizations and metrics to relevant standards, including:

-   **FDA Title 21 CFR Part 820**: Quality System Regulation (Design Controls, Document Controls, CAPA, Statistical Techniques).
-   **ISO 13485:2016**: Medical devices — Quality management systems.
-   **ISO 14971:2019**: Medical devices — Application of risk management.
-   **Clinical and Laboratory Standards Institute (CLSI)**: Best-practice guidelines (e.g., EP05, EP06) for assay validation.
-   **ICH Q10**: Pharmaceutical Quality System principles for continual improvement (especially relevant for the ML page).

This makes the dashboard an invaluable tool for internal management, leadership reviews, and external audits.

---

## 🛠️ Installation & Setup

Follow these steps to get the dashboard running locally.

### Prerequisites

-   Python 3.8+
-   `pip` (Python package installer)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd staff_scientist_dashboard
