# Healthcare Patient Analytics
### End-to-End Data Analytics Project — Akansha Chowdhury

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.3.3-150458?logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-4C72B0)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.8-11557C)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Information](#2-dataset-information)
3. [Tech Stack](#3-tech-stack)
4. [Key Findings](#4-key-findings)
5. [Setup & Execution Guide](#5-setup--execution-guide)
6. [Repository File Structure](#6-repository-file-structure)
7. [Visualisations](#7-visualisations)
8. [Recommendations Summary](#8-recommendations-summary)
9. [Author](#9-author)

---

## 1. Project Overview

This project applies a **six-step end-to-end data analytics workflow** to 5,000 structured patient visit records from a real-world healthcare network. The goal is to convert raw clinical and administrative data into quantified, evidence-based operational recommendations across three domains:

| Domain | Operational Question |
|---|---|
| **Cost Intelligence** | Which departments and patient segments drive cost variance, and does higher spend yield better outcomes? |
| **Operational Efficiency** | Does length of stay correlate with recovery scores, and where can bed utilisation be safely reduced? |
| **Readmission Prevention** | Which patients, visit types, and demographics carry the highest 30-day readmission risk — and where should intervention resources be deployed? |

### Analytical Workflow

```
Step 1: Data Ingestion        →  Load CSV, audit shape, dtypes, descriptive stats
Step 2: Data Cleaning         →  Null handling, deduplication, date parsing, dtype enforcement
Step 2b: Categorical Encoding →  Ordinal encoding (age_group), one-hot encoding (nominal cols)
Step 3: Feature Engineering   →  6 derived features: daily_cost, is_high_risk, is_high_cost, temporal cols
Step 4: Aggregations & KPIs   →  5 named groupby views: dept, age group, visit type, time, region
Step 5: Visual Analytics      →  7 publication-quality figures (Matplotlib + Seaborn)
Step 6: Business Application  →  4 operational programmes, 16 named initiatives with owners & timelines
```

---

## 2. Dataset Information

| Attribute | Detail |
|---|---|
| **Source** | Kaggle — Healthcare Patient Analytics Dataset |
| **Records** | 5,000 patient visits |
| **Columns** | 12 attributes |
| **Time span** | January 2022 – July 2022 |
| **Nulls** | 0 (zero null values confirmed) |
| **Duplicates** | 0 (zero duplicate rows confirmed) |

**Direct link:** [https://www.kaggle.com/datasets/abbas829/healthcare-patient-analytics-dataset](https://www.kaggle.com/datasets/abbas829/healthcare-patient-analytics-dataset)

### Schema

| Column | Type | Description |
|---|---|---|
| `patient_id` | string | Synthetic unique identifier (not PII) |
| `age_group` | categorical (ordinal) | 18-30 / 31-45 / 46-60 / 60+ |
| `gender` | categorical | Male / Female |
| `region` | categorical | North / South / East / West |
| `department` | categorical | Cardiology / General Medicine / Neurology / Orthopedics / Pediatrics |
| `treatment_type` | categorical | Medication / Observation / Surgery / Therapy |
| `visit_type` | categorical | Routine / Emergency |
| `visit_date` | datetime | Date of patient encounter |
| `length_of_stay_days` | float | Inpatient bed-days |
| `treatment_cost` | float | Total cost of visit in USD |
| `recovery_score` | float | Post-treatment recovery score (0–100) |
| `readmission_risk` | float | 30-day readmission risk score (0.00–1.00) |

---

## 3. Tech Stack

| Tool / Library | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Core language |
| **Pandas** | 2.3.3 | Data ingestion, cleaning, feature engineering, aggregations |
| **NumPy** | 2.4.1 | Numerical operations, IQR calculations |
| **Matplotlib** | 3.10.8 | Base plotting engine, multi-panel dashboards |
| **Seaborn** | 0.13.2 | Statistical charts — scatter, bar, heatmap |
| **SciPy** | 1.17.1 | Pearson correlation, p-value calculation |
| **Jupyter / JupyterLab** | 1.1.1 / 4.5.1 | Interactive notebook environment |
| **openpyxl** | 3.1.5 | Excel I/O support for pandas |
| **IBM Bob** | — | AI-assisted development, code generation, report authoring |

---

## 4. Key Findings

### Hospital-Wide KPIs

| Metric | Value |
|---|---|
| Total patients | 5,000 |
| Total network spend | $274,577,359 |
| Average treatment cost | $54,915 (CV = 35.5%) |
| Median treatment cost | $55,124 |
| Average length of stay | 4.06 days (P75 = 5.4 days) |
| Average recovery score | 74.72 / 100 |
| High-risk patients (risk ≥ 0.40) | 1,152 (23.0%) |
| Routine : Emergency ratio | 2.19 : 1 |

### Finding 1 — Department Cost Analysis

| Department | Visits | Avg Cost | Total Spend | Avg LOS | Risk Rate |
|---|---|---|---|---|---|
| Neurology | 967 | $55,762 | $53.9M | 4.06d | 23.3% |
| Cardiology | 995 | $55,633 | $55.4M | 4.03d | 22.2% |
| Orthopedics | 1,058 | $54,913 | **$58.1M** | 4.01d | 22.6% |
| Pediatrics | 989 | $54,782 | $54.2M | 4.14d | 22.9% |
| General Medicine | 991 | $53,506 | $53.0M | 4.07d | **24.3%** |

> **Insight:** All departments sit within a $2,256 per-visit spread — cost uniformity, not a single outlier. Orthopedics leads total spend by volume; General Medicine leads readmission risk.

### Finding 2 — Readmission Risk by Visit Type

| Visit Type | Visits | High-Risk Count | High-Risk Rate |
|---|---|---|---|
| Routine | 3,432 (68.6%) | **802** | 23.4% |
| Emergency | 1,568 (31.4%) | 350 | 22.3% |

> **Insight:** An Emergency-only discharge screen misses **69.6%** of all high-risk patients. Routine pathways are the primary intervention target.

### Finding 3 — LOS vs Recovery Correlation

| Metric | Value |
|---|---|
| Pearson r (LOS vs recovery) | **+0.012** |
| p-value | 0.3891 (not significant) |
| Recovery range across all LOS bins | 74.52 → 75.93 (1.41 pts) |

> **Insight:** Length of stay has **zero clinical impact** on recovery score. LOS-reduction initiatives carry no evidence-based outcome risk.

### Finding 4 — Age Group Cost-Recovery Inversion

| Age Group | Avg Cost | Avg Recovery | Risk Rate |
|---|---|---|---|
| 18–30 | **$55,871** | **74.48** (lowest) | 23.0% |
| 31–45 | $54,748 | 74.75 | 23.6% |
| 46–60 | $54,749 | 74.76 | **23.8%** (highest) |
| 60+ | $53,920 (lowest) | 74.89 | 22.7% |

> **Insight:** The 18–30 cohort is the **most expensive and lowest-recovering** age group — the opposite of the expected age-cost relationship.

### Finding 5 — Treatment Type Efficacy

| Treatment Type | Avg Recovery | Patients |
|---|---|---|
| Observation | 74.81 | 1,270 |
| Therapy | 74.81 | 1,232 |
| Surgery | 74.70 | 1,262 |
| Medication | 74.55 | 1,236 |

> **Insight:** Total spread of only **0.27 points** — treatment type is a negligible recovery predictor.

---

## 5. Setup & Execution Guide

### Prerequisites

- Python 3.10 or higher
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/healthcare-patient-analytics.git
cd healthcare-patient-analytics
```

### Step 2 — Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the main analytics script

```bash
python AkanshaChowdhury_HealthcareAnalyticsProject.py
```

This will:
- Ingest and clean the dataset
- Engineer all derived features
- Compute all KPI aggregations
- Generate 4 plots saved to `outputs/plots/`
- Print a full console summary of all findings

### Step 5 — Export all 7 report figures

```bash
python export_figures.py
```

Saves `fig1_cost_by_dept.png` through `fig7_executive_kpi.png` to `outputs/plots/` at 150 DPI.

### Step 6 — Launch the Jupyter Notebook (interactive exploration)

```bash
jupyter notebook AkanshaChowdhury_HealthcareAnalyticsProject.ipynb
```

Or with JupyterLab:

```bash
jupyter lab
```

---

## 6. Repository File Structure

```
healthcare-patient-analytics/
│
├── AkanshaChowdhury_HealthcareAnalyticsProject.ipynb   # Jupyter Notebook (interactive)
├── AkanshaChowdhury_HealthcareAnalyticsProject.py      # Main executable analytics script
├── AkanshaChowdhury_HealthcareAnalyticsProject_Report.docx  # Full Word report (6 sections, 7 figures)
├── export_figures.py                                   # 7-figure PNG export script
├── requirements.txt                                    # Pinned Python dependencies
├── README.md                                           # This file
│
├── healthcare_patient_analytics_seaborn.csv            # Source dataset (5,000 rows x 12 cols)
│
└── outputs/
    └── plots/
        ├── plot1_avg_cost_by_dept.png                  # Dept cost bar chart (main script)
        ├── plot2_readmission_risk_by_dept.png          # Risk stacked bar (main script)
        ├── plot3_los_vs_recovery.png                   # LOS vs recovery scatter (main script)
        ├── plot4_visit_type_donut.png                  # Visit type donut (main script)
        ├── fig1_cost_by_dept.png                       # Avg cost by dept (report figure)
        ├── fig2_readmission_risk.png                   # Readmission risk by dept (report figure)
        ├── fig3_stay_vs_recovery.png                   # LOS vs recovery (report figure)
        ├── fig4_visit_distribution.png                 # Visit type distribution (report figure)
        ├── fig5_treatment_efficacy.png                 # Treatment efficacy (report figure)
        ├── fig6_age_region_heatmap.png                 # Age group & region heatmap (report figure)
        └── fig7_executive_kpi.png                      # Executive KPI dashboard (report figure)
```

---

## 7. Visualisations

All figures are embedded in the Word report and available as 150 DPI PNGs in `outputs/plots/`.

| Figure | Chart Type | Finding |
|---|---|---|
| Fig 1 | Horizontal bar chart | Avg treatment cost by department |
| Fig 2 | Stacked bar chart | High-risk readmission rate by department |
| Fig 3 | Scatter + regression line | LOS vs recovery score (r = +0.012) |
| Fig 4 | Donut chart | Visit type distribution (Routine 68.6% / Emergency 31.4%) |
| Fig 5 | Horizontal bar chart | Avg recovery score by treatment type |
| Fig 6 | Grouped bar / heatmap | Cost and recovery by age group and region |
| Fig 7 | Four-panel KPI dashboard | Hospital-wide executive summary |

---

## 8. Recommendations Summary

| # | Programme | Data Anchor | Owner | Target |
|---|---|---|---|---|
| 1 | **Bed & Capacity Management** | P75 LOS = 5.4d; r = +0.012 (zero clinical risk) | Bed Management Lead | Reduce mean LOS from 4.06d → 3.80d within 2 quarters |
| 2 | **Post-Discharge Readmission Protocol** | 23.0% high-risk; 69.6% via Routine pathways | Chief Nursing Officer | Universal pre-discharge screen for all 5 departments |
| 3 | **Cost Control & Resource Allocation** | CV = 35.5%; Orthopedics CV 36.3% on $58.1M | CFO + Dept Heads | Reduce Orthopedics cost CV below 35.0% within 2 quarters |
| 4 | **Targeted Demographic & Regional Care** | 18-30 highest cost + lowest recovery; East Region highest risk at lowest cost | Community Health Director + Regional Directors | Reduce 18-30 avg cost by $500/visit within 12 months |

---

## 9. Author

**Akansha Chowdhury**

- Project: End-to-End Healthcare Patient Analytics
- Dataset: [Kaggle — Healthcare Patient Analytics](https://www.kaggle.com/datasets/abbas829/healthcare-patient-analytics-dataset)
- Tools: Python · Pandas · NumPy · Matplotlib · Seaborn · SciPy · Jupyter · IBM Bob

---

*Generated with IBM Bob — AI-assisted analytics development.*
