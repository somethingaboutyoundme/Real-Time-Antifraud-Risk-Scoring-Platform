# 🛡️ Freedom Loan Application Fraud Scoring Platform

**An end-to-end Machine Learning platform for detecting suspicious and high-risk loan applications using advanced tabular modeling, explainable AI, and production-ready API deployment.**

---

## 📋 Executive Summary

This project was developed as a production-ready solution for a credit risk and fraud detection assignment provided by **Freedom Bank**. The primary objective was to build a machine learning model that identifies suspicious loans by generating a `scoreball` (fraud probability), where a `scoreball >= 0.60` classifies the loan as suspicious/fraudulent.

Rather than delivering a simple Jupyter notebook, this project spans the full ML lifecycle: from strict data leakage auditing and handling severe class imbalance (0.46% positive rate) to deploying a containerized FastAPI scoring service with MLflow tracking and PostgreSQL persistence.

It serves as a robust portfolio piece demonstrating applied Data Science, Machine Learning Engineering, and Banking Domain Expertise.

---

## 🎯 Business Problem & Context

In the banking sector, identifying early-default and fraudulent loan applications is critical to minimizing capital loss.

**The Task:**

> *Find suspicious loans using machine learning models, where `scoreball ≥ 0.6` refers to fraudulent loans.*

**The Solution:**
The pipeline ingests historical loan application data (including credit bureau histories) and outputs a calibrated probability (`scoreball`). Applications exceeding the strict `0.60` threshold are flagged for manual review or automatic rejection.

---

## 🧠 Technical Approach & Domain Insights

### 1. Target Definition & Class Imbalance

The project utilizes **`FPD_15`** (First Payment Delinquency within 15 days) as the primary fraud/risk proxy.

* **Filtering:** Only matured applications (`CENZ_15_1_Maturity_Flag == 1`) were used.
* **Imbalance:** The dataset has a severe imbalance of **~0.46% positive rate** (111 cases out of 24,142).
* *Insight:* Because of this imbalance, standard `Accuracy` is completely useless (a dummy model achieves 99.54% accuracy). The primary metric used was **PR-AUC (Precision-Recall AUC)**.

### 2. Strict Leakage Audit

In financial ML, data leakage is the most common cause of "too good to be true" models. Before modeling, a rigorous audit was conducted:

* **Excluded:** `Active_Overdue_Days` was identified as a post-outcome feature (data available *after* the loan was issued) and was strictly removed from the feature set to prevent target leakage.
* **Excluded:** `Application_Code` (ID only).
* **Filtered (Not Features):** `CENZ` maturity flags were used strictly for data filtering, not as model inputs.

### 3. Feature Engineering

Domain-driven features were engineered to capture credit risk signals:

* `loan_income_ratio`: Requested loan amount divided by declared official income.
* `outstanding_income_ratio`: Total outstanding debt relative to income.
* `had_previous_overdue`: Binary flag for previous 6-month delinquency.
* Time-based features extracted from `Application_Date`.

### 4. Validation Strategy

To simulate a real-world banking environment where models predict the future based on the past, a **Temporal Split** was used as the primary validation strategy, alongside a stratified baseline:

* **Train:** April 1, 2022 — May 15, 2022
* **Validation:** May 16, 2022 — June 7, 2022
* **Test:** June 8, 2022 — June 30, 2022

---

## ⚙️ Models & Evaluation

Two distinct algorithms were trained and compared to provide both interpretability and high performance:

1. **Logistic Regression:** A linear, highly interpretable baseline model. Served as the benchmark for risk compliance. Handled imbalanced classes via `class_weight="balanced"`.
2. **CatBoostClassifier:** A gradient boosting model chosen specifically for its native handling of high-cardinality categorical features (e.g., `City`, `Employer_BIN`) and missing values without requiring heavy preprocessing.

### Evaluation Metrics

Metrics were evaluated specifically at the Freedom-mandated **Threshold = 0.60**:

| Model                      |    PR-AUC    |   ROC-AUC   | Precision@0.6 |  Recall@0.6  |    F2@0.6    |
| :------------------------- | :----------: | :----------: | :-----------: | :----------: | :----------: |
| Dummy Baseline             |      -      |     0.50     |       -       |      -      |      -      |
| Logistic Regression        | *computed* | *computed* | *computed* | *computed* | *computed* |
| **CatBoost (Final)** | *computed* | *computed* | *computed* | *computed* | *computed* |

*Note: F2-Score was prioritized over F1 because fraud detection heavily penalizes False Negatives (missing a fraudulent loan is more costly than a false alarm).*

### Explainability (SHAP)

To satisfy banking audit requirements, SHAP (SHapley Additive exPlanations) was integrated to explain individual predictions. For any given application flagged as `scoreball >= 0.60`, the system can output the exact feature contributions (e.g., *"High loan-to-income ratio increased risk by 0.15"*).

---

## 🚀 Production Architecture (ML Engineering)

The ML model is not just a notebook; it is a deployable microservice. The production architecture includes:

* **FastAPI:** Exposes a RESTful endpoint (`POST /application/score`) allowing loan officers or downstream systems to submit application data and receive a `scoreball` in real-time.
* **PostgreSQL:** Persists application data, model scores, and final decisions for audit logging.
* **MLflow:** Tracks experiment parameters, data schemas, and model versions to prevent model versioning chaos (`model_final_v2.pkl`).
* **Docker:** Containerizes the FastAPI app, database, and MLflow server for one-command deployment (`docker compose up`).

---

## 📁 Repository Structure

```text
freedom-loan-fraud-scoring/
│
├── data/
│   ├── raw/                  # Original Freedom Bank Excel file (.gitignored)
│   ├── interim/              # Intermediate cleaned data
│   └── processed/            # Final train/validation/test splits
│
├── notebooks/                # Exploratory and modeling notebooks (Levels 1-7)
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_quality.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_logistic_baseline.ipynb
│   ├── 06_catboost.ipynb
│   └── 07_model_comparison.ipynb
│
├── src/                      # Production Python package
│   └── fraud_scoring/
│       ├── data.py           # Data loading
│       ├── preprocessing.py  # Imputation & encoding pipelines
│       ├── features.py       # Domain-specific feature generation
│       ├── split.py          # Temporal splitting logic
│       ├── models/           # Model definitions
│       └── evaluation/       # Metrics & Plots
│
├── tests/                    # Pytest suite for data & features
├── reports/                  # Final PDF report & SHAP figures
├── artifacts/                # Saved models & test_predictions.csv
│
├── api/                      # FastAPI application code
├── docker-compose.yml        # Spins up API, Postgres, and MLflow
├── requirements.txt
└── README.md
```

---

## 💻 Getting Started

### Prerequisites

- Python 3.9+
- Docker & Docker Compose

### Installation & Running the Pipeline

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/freedom-loan-fraud-scoring.git
   cd freedom-loan-fraud-scoring
   ```
2. **Set up the Python environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Run the ML Pipeline (Notebooks)**
   Execute the notebooks in the `notebooks/` folder sequentially to understand the EDA, modeling, and evaluation process.
4. **Deploy the Production System**
   To spin up the FastAPI application, PostgreSQL database, and MLflow tracking server:

   ```bash
   docker-compose up --build
   ```
5. **Score a New Application**
   Once the API is running, send a POST request to score an application:

   ```bash
   curl -X 'POST' \
     'http://localhost:8000/application/score' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
       "credit_product": "Кредит налич",
       "official_income": 350000,
       "requested_amount": 3000000,
       "city": "Алматы",
       "previous_overdue_days": 4,
       "active_loans": 2,
       "pcb_inquiries_30d": 7
     }'
   ```

   **Expected Response:**

   ```json
   {
     "scoreball": 0.73,
     "suspicious": true,
     "threshold": 0.60,
     "model_version": "catboost_v1"
   }
   ```

---

## 🛠️ Tech Stack

| Category                 | Technologies                                     |
| :----------------------- | :----------------------------------------------- |
| **Language**       | Python, SQL                                      |
| **Data & ML**      | Pandas, Scikit-Learn, CatBoost, Imbalanced-Learn |
| **Explainability** | SHAP                                             |
| **API**            | FastAPI, Pydantic, Uvicorn                       |
| **MLOps & DB**     | MLflow, PostgreSQL, Docker, Docker Compose       |
| **Testing**        | Pytest                                           |

---

## 📄 License & Data Disclaimer

This project code is licensed under the MIT License.
*Disclaimer: The raw dataset provided by Freedom Bank is strictly excluded from this repository via `.gitignore` to comply with banking data privacy regulations.*
