# CUSTOMER CHURN ANALYSIS

**Author:** Group 5
**Course:** MLOps Engineering - MsC in Business Analytics and Data Science (IE University)
**Date:** February 2026
**Status:** 1st Group Assignment - Phase 1

---

## 1. Business Case

### Client
- **Company:** Mid-size telecommunications provider
- **Industry:** Telecommunications / Subscription Services
- **Business Unit:** Customer Retention & Marketing

### Client Maturity
- **Data:** Structured transactional data available (call logs, billing, account metadata). No real-time streaming yet.
- **Tools:** Standard BI stack; moving toward ML-driven decision-making.
- **Processes:** Retention campaigns are currently rule-based (e.g., tenure thresholds). No predictive targeting.
- **People:** Analytics team in place; limited ML engineering experience.
- **Strategy:** Executive buy-in for AI-driven churn reduction as a strategic priority.

### Goal of Project
Build a classification model to predict which customers are likely to churn, enabling proactive retention campaigns.
- **Objective metric:** Weighted F1 Score (handles class imbalance between churners and non-churners).
- **Improvement over baseline:** The model must outperform the majority-class baseline (~85% non-churn).
- **Quantifiable KPI:** Reduce monthly churn rate by at least 5% among targeted high-risk customers.

### Problem Statement
The client loses approximately 14.5% of customers per cycle. Current retention efforts are reactive (triggered only after a cancellation request). There is no systematic way to identify at-risk customers before they leave.
- **Current baseline:** No predictive model in place; churn intervention is manual and post-hoc.
- **KPI:** Monthly churn rate (%) and revenue at risk (sum of monthly charges for predicted churners).

### Solution Description & Key Functionalities
A machine learning pipeline that ingests customer account and usage data, engineers predictive features, trains a classification model, and outputs a ranked list of customers with their churn probability. The marketing team receives this list to prioritize retention outreach.
- End-to-end automated pipeline (load, clean, validate, feature engineer, train, evaluate, infer)
- Configurable via `config.yaml` (no code changes needed to retrain or adjust)
- Produces a serialized model artifact (`.pkl`) ready for production deployment

### Solution Scalability
- **Other use cases:** The same pipeline architecture can be adapted for upsell/cross-sell propensity, credit risk, or subscription renewal prediction.
- **Other industries:** Insurance (policy lapse), SaaS (user churn), banking (account closure).
- **Growth:** The pipeline can scale by swapping the model (e.g., from Logistic Regression to Gradient Boosting), adding new features, or connecting to real-time data sources.

### Client Benefit (Over Non-AI Approach)
- **Short term:** Targeted retention campaigns reduce wasted marketing spend by focusing on truly at-risk customers instead of blanket offers.
- **Long term:** Increased customer lifetime value (CLV) and reduced acquisition costs (acquiring a new customer costs 5-7x more than retaining one).
- **Competitiveness:** Data-driven retention becomes a sustainable competitive advantage in a commoditized market.
- **KPI:** Projected 10-15% reduction in churn-related revenue loss within the first quarter of deployment.

### Cost Estimation ($000, Ballpark)
| Role | Estimated Cost |
|---|---|
| AI / ML Specialist | $15-25k |
| Product Manager | $10-15k |
| ML / Software Engineer | $20-30k |
| Data Engineer | $15-20k |
| Subject Matter Expert (Telecom) | $5-10k |
| **Total Talent** | **$65-100k** |

Client to cover:
- **Data:** Internal CRM and billing data access
- **Infrastructure:** Cloud compute (AWS/GCP) ~$1-3k/month
- **Licenses:** BI tools, monitoring dashboards
- **Timeline:** 12+ weeks (MVP to production)

### Risks and Challenges
| Risk | Mitigation |
|---|---|
| Data quality issues (missing values, inconsistent formats) | Automated validation gate (`validate.py`) catches bad data before training |
| Class imbalance (only ~14.5% churners) | Weighted F1 metric; consider SMOTE or class weighting in future iterations |
| Model drift after deployment | Plan for periodic retraining and monitoring (Phase 2) |
| Skills gap in ML engineering | Modular codebase with documentation enables knowledge transfer |
| Security / data privacy | No PII in dataset; `data/` excluded from version control via `.gitignore` |

---

## 2. Success Metrics

* **Business KPI (The "Why"):**
  - Reduce churn rate among high-value customers.
  - Minimize revenue at risk (sum of monthly charges for predicted churners).

* **Technical Metric (The "How"):**
  Weighted F1 Score for classification, since it handles class imbalance between churners and non-churners.

* **Acceptance Criteria:**
  The model must outperform the majority-class baseline (~85% accuracy).

---

## 3. The Data

* **Source:** Kaggle CSV - Telecom Customer Churn dataset (3,333 records, 11 features)
* **Target Variable:** `Churn` - binary (1 = Customer churned, 0 = Customer retained)
* **Features:** `AccountWeeks`, `ContractRenewal`, `DataPlan`, `DataUsage`, `CustServCalls`, `DayMins`, `DayCalls`, `MonthlyCharge`, `OverageFee`, `RoamMins`
* **Sensitive Info:** No emails, credit cards, or PII in the dataset.

> **WARNING:** If the dataset contains sensitive data, it must NEVER be committed to GitHub. Ensure `data/` is in your `.gitignore`.

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                       # Project definition and business case
├── pytest.ini                      # Pytest configuration
├── environment.yml                 # Dependencies (Conda)
├── config.yaml                     # Global configuration (paths, params, features)
│
├── notebooks/                      # Experimental sandbox
│   └── 01_opioid_analysis_vExp.ipynb
│
├── src/                            # Production code
│   ├── __init__.py                 # Python package marker
│   ├── load_data.py                # Data ingestion with validation and logging
│   ├── clean_data.py               # Column standardization, dedup, missing values
│   ├── validate.py                 # Schema and data quality gate (GIGO)
│   ├── feature_engineering.py      # Feature creation, encoding, scaling
│   ├── train.py                    # Model training, pipeline bundling, artifact saving
│   ├── evaluate.py                 # Metrics computation and diagnostic plots
│   ├── infer.py                    # Inference / prediction on new data
│   └── main.py                     # Pipeline orchestrator (entry point)
│
├── data/                           # Local storage (IGNORED by Git)
│   ├── raw/                        # Immutable input data
│   ├── processed/                  # Cleaned data
│   └── inference/                  # Prediction outputs
│
├── models/                         # Serialized model artifacts (IGNORED by Git)
│
├── reports/                        # Generated metrics, plots, and figures
│   └── figures/
│
└── tests/                          # Automated test suite
    ├── test_load_data.py
    ├── test_clean_data.py
    ├── test_validate.py
    ├── test_feature_engineering.py
    ├── test_train.py
    ├── test_evaluate.py
    ├── test_infer.py
    └── test_main.py
```

---

## 5. Setup & Execution

### Prerequisites
- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd 1-mlops-kickoff-repo
```

### 2. Create and activate the environment
```bash
conda env create -f environment.yml
conda activate test_env
```

### 3. Run the full ML pipeline
```bash
python -m src.main
```
This will: load data, validate, clean, engineer features, train a model, evaluate it, and run inference. The trained model is saved to `models/model.pkl` and evaluation plots to `reports/figures/`.

### 4. Run the test suite
```bash
python -m pytest
```
To see verbose output with coverage:
```bash
python -m pytest -v --cov=src --cov-report=term-missing
```

---

## 6. ML Pipeline Flow

```
load_data.py → validate.py → clean_data.py → feature_engineering.py → train.py → evaluate.py → infer.py
                                                                         │
                                                                    main.py (orchestrator)
```

| Step | Module | Responsibility |
|---|---|---|
| 1 | `load_data.py` | Load CSV, validate file exists, log shape |
| 2 | `validate.py` | Check schema, types, missing values, domain rules |
| 3 | `clean_data.py` | Standardize columns, drop duplicates, normalize missing values |
| 4 | `feature_engineering.py` | Encode categoricals, scale numerics, create derived features |
| 5 | `train.py` | Split data, build sklearn Pipeline, fit model, save artifact |
| 6 | `evaluate.py` | Compute F1/RMSE, generate confusion matrix / residual plots |
| 7 | `infer.py` | Run predictions on new data, return standardized DataFrame |

---

## 7. Configuration

All pipeline parameters are centralized in `config.yaml`:
- Data paths (`data/raw/`, `data/processed/`, `data/inference/`)
- Model artifact path and problem type
- Target column and feature lists (numeric / categorical)
- Validation rules (required columns)
- Training hyperparameters (test size, random state)
- Logging settings

---

## 8. Version Control Workflow

- **Branches:** `main` (protected, production-ready), `dev` (integration), `feature/*` (individual work)
- **Process:** Each collaborator works on a `feature/` branch, opens a PR to `dev`, and the team reviews before merging.
- **Commits:** "Commit early, commit often, and push."
