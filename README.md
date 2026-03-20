# CUSTOMER CHURN ANALYSIS

**Author:** Group 5
**Course:** MLOps Engineering - MsC in Business Analytics and Data Science (IE University)
**Date:** March 2026
**Status:** Final Group Assignment

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
├── README.md                       # Project documentation, model card, changelog
├── config.yaml                     # Centralized runtime configuration
├── environment.yml                 # Conda environment specification
├── pytest.ini                      # Pytest configuration
├── Dockerfile                      # Lean serving container
├── .dockerignore                   # Strict Docker ignore policy
├── .env.example                    # Template for secrets (WANDB_API_KEY)
│
├── .github/workflows/
│   ├── ci.yml                      # CI: runs tests on Pull Requests
│   └── deploy.yml                  # CD: deploys on GitHub Release
│
├── src/                            # Production code
│   ├── __init__.py                 # Python package marker
│   ├── logger.py                   # Centralized dual-output logger (console + file)
│   ├── utils.py                    # Shared helpers (load_config, get_project_root)
│   ├── load_data.py                # Data ingestion with validation and logging
│   ├── clean_data.py               # Column standardization, dedup, missing values
│   ├── validate.py                 # Schema and data quality gate (GIGO)
│   ├── feature_engineering.py      # Feature creation, encoding, scaling
│   ├── train.py                    # Model training, pipeline bundling, artifact saving
│   ├── evaluate.py                 # Metrics computation and diagnostic plots
│   ├── infer.py                    # Inference + W&B model loading
│   ├── main.py                     # Pipeline orchestrator + W&B tracking
│   └── api.py                      # FastAPI serving layer (/health, /predict)
│
├── data/                           # Local storage (IGNORED by Git)
│   ├── raw/                        # Immutable input data
│   ├── processed/                  # Cleaned data
│   └── inference/                  # Prediction outputs
│
├── models/                         # Serialized model artifacts (IGNORED by Git)
├── logs/                           # Pipeline log files (IGNORED by Git)
│
├── reports/                        # Generated metrics, plots, and figures
│   └── figures/
│
├── notebooks/                      # Experimental sandbox
│
└── tests/                          # Automated test suite (119 tests)
    ├── test_load_data.py
    ├── test_clean_data.py
    ├── test_validate.py
    ├── test_feature_engineering.py
    ├── test_train.py
    ├── test_evaluate.py
    ├── test_infer.py
    ├── test_main.py
    ├── test_utils.py
    └── test_api.py
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
conda activate mlops-churn
```

### 3. Set up secrets
```bash
cp .env.example .env
# Edit .env and add your WANDB_API_KEY
```

### 4. Run the full ML pipeline
```bash
python -m src.main
```
This will: load data, validate, clean, engineer features, train a model, evaluate it, run inference, and log everything to W&B. The model artifact is promoted with alias `prod` in the W&B registry.

### 5. Run the test suite
```bash
python -m pytest
```

### 6. Start the API server
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```
Test it:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customers":[{"accountweeks":120,"datausage":2.7,"custservcalls":3,"daymins":200.5,"daycalls":90,"monthlycharge":65.0,"overagefee":10.25,"roammins":8.3,"contractrenewal":1,"dataplan":0}]}'
```

### 7. Build and run with Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 --env-file .env churn-api
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
- **CI:** Pull Requests trigger `ci.yml` which runs the full test suite. PRs must pass before merge.
- **CD:** Publishing a GitHub Release triggers `deploy.yml` which deploys to Render.

---

## 9. Experiment Tracking & Model Registry

- **Platform:** Weights & Biases (W&B)
- **Project:** [MLOPs-Group5-telecomChurn](https://wandb.ai/diego08-ie-university/MLOPs-Group5-telecomChurn)
- **What is tracked:** Run config, training/evaluation metrics, confusion matrix plots, pipeline logs, and model artifacts.
- **Model registry:** The trained model is stored as a W&B artifact and promoted with alias `prod`. The API loads the `prod` artifact on startup — no local `.pkl` files in production.

---

## 10. API Documentation

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "healthy", "model_loaded": true}` |
| `/predict` | POST | Accepts JSON with customer data, returns churn predictions |
| `/docs` | GET | Interactive Swagger UI (auto-generated by FastAPI) |

**Request schema** (`/predict`):
```json
{
  "customers": [
    {
      "accountweeks": 120,
      "datausage": 2.7,
      "custservcalls": 3,
      "daymins": 200.5,
      "daycalls": 90,
      "monthlycharge": 65.0,
      "overagefee": 10.25,
      "roammins": 8.3,
      "contractrenewal": 1,
      "dataplan": 0
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {"prediction": 0, "label": "No Churn"}
  ]
}
```

---

## 11. Model Card

| Field | Details |
|---|---|
| **Model name** | Telecom Churn Classifier |
| **Model type** | Logistic Regression (sklearn Pipeline) |
| **Task** | Binary classification (Churn / No Churn) |
| **Training data** | Kaggle Telecom Churn dataset — 3,333 customers, 10 features |
| **Evaluation metric** | Weighted F1 Score |
| **Performance** | F1 (weighted) = 0.827 on held-out test set (667 samples) |
| **Known limitations** | Class imbalance (~14.5% churn rate) limits recall on the minority class (Churn recall ~18%). Not suitable for high-stakes decisions without human review. |
| **Intended use** | Prioritize retention outreach for at-risk customers. Marketing teams use the ranked list to allocate campaign budgets. |
| **Out-of-scope use** | Automated account termination, credit decisions, or any use without human oversight. |
| **Ethical considerations** | No PII in training data. Model should be monitored for demographic bias if deployed on real customer data. |
| **Retraining cadence** | Recommended quarterly or when churn rate shifts by more than 2 percentage points. |

---

## 12. Changelog

| Version | Date | Changes |
|---|---|---|
| **v1.0.0** | 2026-03-20 | Production-ready release: full ML pipeline, W&B tracking with `prod` alias, FastAPI serving (`/health`, `/predict`), Dockerfile, CI/CD workflows, 119 tests |
| **v0.2.0** | 2026-03-15 | Repository & engineering overhaul: centralized `config.yaml`, `src/logger.py` (dual-output), zero `print()` in production code, import order cleanup, `.env` for secrets |
| **v0.1.0** | 2026-02-28 | Initial pipeline: modular `src/` modules (load, clean, validate, features, train, evaluate, infer), `main.py` orchestrator, test suite, business case README |
