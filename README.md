# CUSTOMER CHURN ANALYSIS 

**Author:** Group 5 (Your Group Name or number)  
**Course:** MLOps: Master in Business Analytics and Data Sciense
**Status:** Session 1 (Initialization)

---

## 1. Business Objective
*Replace this section with your project definition.*

* **The Goal:** What business value does this model create?
Build a machine learning model to predict customer churn for a telecommunications company. They can then identify customers at risk of leaving and take retention actions.

The model creates business value by:
  - Reducing customer acquisition costs
  - Increasing customer lifetime value
  - Improving their retention strategy

* **The User:** Who consumes the output and how?
Marketing team receives an Excel file report listing customers and their predicted churn probability.

---

## 2. Success Metrics
*How do we know if the project is successful?*

* **Business KPI (The "Why"):**
 - Reduce churn rate among high-value customer.
 - Minimize revenue at risk (sum of monthly charge x churn).

* **Technical Metric (The "How"):**
 Weighted F1 Score for Classification, since it will handle class imbalance.

* **Acceptance Criteria:**
The model must outperform the majority-class baseline.

---

## 3. The Data

* **Source:** Kaggle CSV (Customer Churn Analysis and Classification)
* **Target Variable:** Churn as binary (1 for Customer churned, 0 for Customer retained)
* **Sensitive Info:** No emails, credit cards, or PII (Personally Identifiable Information) in the dataset.
  > *⚠️ **WARNING:** If the dataset contains sensitive data, it must NEVER be committed to GitHub. Ensure `data/` is in your `.gitignore`.*

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                # This file (Project definition)
├── environment.yml          # Dependencies (Conda/Pip)
├── config.yaml              # Global configuration (paths, params)
├── .env                     # Secrets placeholder
│
├── notebooks/               # Experimental sandbox
│   └── yourbaseline.ipynb   # From previous work
│
├── src/                     # Production code (The "Factory")
│   ├── __init__.py          # Python package
│   ├── load_data.py         # Ingest raw data
│   ├── clean_data.py        # Preprocessing & cleaning
│   ├── features.py          # Feature engineering
│   ├── validate.py          # Data quality checks
│   ├── train.py             # Model training & saving
│   ├── evaluate.py          # Metrics & plotting
│   ├── infer.py             # Inference logic
│   └── main.py              # Pipeline orchestrator
│
├── data/                    # Local storage (IGNORED by Git)
│   ├── raw/                 # Immutable input data
│   └── processed/           # Cleaned data ready for training
│
├── models/                  # Serialized artifacts (IGNORED by Git)
│
├── reports/                 # Generated metrics, plots, and figures
│
└── tests/                   # Automated tests
```

## 5. Setup & Execution
**1. Create and activate the environment:**
```bash
conda env create -f environment.yml
conda activate test_env
```

**2. Run the pipeline:**
```bash
python -m src.main
```
