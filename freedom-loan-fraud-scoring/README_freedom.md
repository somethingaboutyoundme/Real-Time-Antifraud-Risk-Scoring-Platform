1. The Essence of the Freedom Bank Task

**Task formulation:** Find suspicious loans using machine learning models, where a `scoreball >= 0.6` indicates a fraudulent loan.

**How it is implemented:**

1. There is no pre-existing `scoreball` column in the Excel file; your ML model creates it.
2. `scoreball = model.predict_proba(X)[:, 1]`
3. If `scoreball >= 0.60` -> the loan is flagged as suspicious/fraud. If `< 0.60` -> normal.

**Target Selection:**

* **Primary target:** `FPD_15` (First Payment Delinquency). Used only for observations where `CENZ_15_1_Maturity_Flag == 1` (verified matured applications).
  * *Statistics:* 24,142 applications, of which 111 are fraudulent. Positive rate ≈ 0.46%. This is a severe class imbalance, which is good for antifraud ML. `FPD_15` is logically called a **fraud/risk proxy**.
* **Secondary target:** `SPD_15` with the filter `CENZ_15_2_Maturity_Flag == 1`.
  * *Statistics:* 23,886 applications, 199 positives. Positive rate ≈ 0.83%.
  * Allows you to compare Model A (predicting FPD) vs. Model B (predicting SPD).

**Leakage Audit:**
Before building the first model, classify the 20 fields:

* `Application_Date`: ✅ use / create time features.
* `Application_Code`: ❌ ID only.
* `Credit_Product`, `Declared_Official_Income`, `Requested_Loan_Amount`, `City`, `Employer_BIN`, `Days_Overdue_Last_6_Months`, `Number_of_Completed_Loans`, `Total_Outstanding_Balance`, `Number_of_Active_Loans`, `Number_of_PCB_Inquiries`, `Loan_Term`: ✅ use.
* `Manager`: ⚠️ conduct a separate experiment.
* `CENZ_15_1_Maturity_Flag` / `CENZ_15_2_Maturity_Flag`: ❌ these are filters, not features.
* `FPD_15`, `SPD_15`: 🎯 targets.
* `Active_Overdue_Days`: 🚨 **PROBABLE LEAKAGE**. This column must not be included in the model; otherwise, the model will "predict" delinquency simply by looking at how many days the client is already overdue.

---

### 2. Step-by-Step Instructions (From LEVEL 0 to LEVEL 15)

**LEVEL 0: Basic Structure**
Create a repository with folders `data/`, `notebooks/`, `src/`, `reports/`, `tests/`. No Docker, Kafka, or MLflow at this stage. Place `freedom_sample.xlsx` in `data/raw/` (and add to `.gitignore` if the data cannot be published publicly).

**LEVEL 1: Data Understanding**
Notebook `01_data_understanding.ipynb`. No ML. Check shape, dtypes, duplicates, missing values, target distribution. Answer the questions: how many applications? how much fraud? what products/cities? what features are categorical/numerical? where are the missing values? where is potential leakage?

**LEVEL 2: EDA**
Notebook `02_eda.ipynb`. Explore FPD by products, loan amount, income, previous delinquencies, active loans, PCB inquiries, and fraud distribution over time.

**LEVEL 3: Data Quality**
Notebook `03_data_quality.ipynb`. Study missing values (Employer BIN, Loan Term, etc.). Do not use `fillna(0)` automatically. Fill missing categorical values with `"UNKNOWN"`, as missing data itself can be a signal.

**LEVEL 4: Outliers**
Incomes have a huge spread. Do not delete them automatically. Look at percentiles. For Logistic Regression, use `log1p()` or winsorization (1st and 99th percentiles). For CatBoost, outliers are less destructive.

**LEVEL 5: Feature Engineering**
Notebook `04_feature_engineering.ipynb`. Create clear features:

* `loan_income_ratio` (loan amount / income)
* `outstanding_income_ratio` (outstanding debt / income)
* `total_previous_loans` (active + completed)
* `active_loan_ratio`
* `had_previous_overdue` (overdue days > 0)
* `high_inquiry_flag`
* Time features from the application date (month, week, day of the week).

**LEVEL 6: Proper Split Strategy**
Do not use a random split as the primary one. Do a **Temporal split** (by time):

* TRAIN: 2022-04-01 → 2022-05-15
* VALIDATION: 2022-05-16 → 2022-06-07
* TEST: 2022-06-08 → 2022-06-30
  For the baseline, you can do a Stratified split, but the final result must be checked with a temporal holdout.

**LEVEL 7: Dummy Baseline**
Train a `DummyClassifier` (predicts everything as 0). With a 0.46% fraud rate, accuracy will be 99.54%. This proves that the accuracy metric is useless here.

**LEVEL 8: Model №1 — Logistic Regression**
Simple, interpretable baseline. Pipeline: median imputation + scaling for numerical; filling with `UNKNOWN` + OneHotEncoder for categorical. Use `class_weight="balanced"`.

**LEVEL 9: Model №2 — CatBoostClassifier**
Ideal for tabular data, high-cardinality categorical features (City, Employer_BIN), missing values, and class imbalance.

**LEVEL 10: Metrics**
The main metric is **PR-AUC** (due to severe imbalance). Additionally: Recall, Precision, F1, F2, ROC-AUC. Be sure to show metrics **at threshold = 0.60** (TP, FP, TN, FN, Precision@0.6, Recall@0.6).

**LEVEL 11: Threshold Analysis**
Explore thresholds from 0.10 to 0.90. Plot threshold vs. recall/precision/F2. But in the final comparison, keep Freedom's strict requirement: `0.60`.

**LEVEL 12: Calibration**
Check the Calibration Curve and Brier Score. Apply Platt scaling or Isotonic calibration so that a `scoreball = 0.70` truly reflects the probability of fraud.

**LEVEL 13: Model Comparison**
Create a comparison table: Dummy, Logistic Regression, CatBoost by metrics PR-AUC, ROC-AUC, Precision@0.6, Recall@0.6, F2@0.6.

**LEVEL 14: Error Analysis**
Analyze False Positives and False Negatives. Find patterns (e.g., the model poorly catches a specific credit product).

**LEVEL 15: Explainability**
Output Logistic Regression coefficients and SHAP for CatBoost. Show Global feature importance and an example of an individual prediction (Application #123, scoreball = 0.81, main risk factors).

---

### 3. Final Repository Structure (Freedom Assignment + Production)

After completing the base assignment (V1.0), the project turns into a production module:

```text
freedom-loan-fraud-scoring/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_quality.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_logistic_baseline.ipynb
│   ├── 06_catboost.ipynb
│   └── 07_model_comparison.ipynb
│
├── src/
│   └── fraud_scoring/
│       ├── __init__.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── split.py
│       ├── models/
│       │   ├── logistic.py
│       │   └── catboost.py
│       └── evaluation/
│           ├── metrics.py
│           └── plots.py
│
├── tests/
│   ├── test_features.py
│   └── test_preprocessing.py
│
├── reports/
│   ├── figures/
│   └── freedom_antifraud_report.pdf
│
├── artifacts/
│   ├── models/
│   └── predictions/    # Save test_predictions.csv (application_code, actual_fpd, scoreball, predicted_fraud)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

### 4. Production Deployment (Hero Levels for the Freedom model)

After V1.0, the model can be "productionized":

1. **FastAPI:** Create an endpoint `POST /application/score` that accepts a JSON with application parameters and returns `scoreball`, `suspicious` (bool), `threshold`, and `model_version`.
2. **PostgreSQL:** Set up tables `applications`, `scores`, `model_versions`, `decisions` so the flow is: application -> DB -> ML model -> scoreball -> decision -> DB.
3. **MLflow:** Track experiments, parameters, feature versions, metrics (PR-AUC, Brier Score), and artifacts.
4. **Docker:** Containerize FastAPI, PostgreSQL, and MLflow to run via `docker compose up`.

---

### 5. Freedom Bank Report Requirements

Structure of the final report:

1. Business Problem
2. Dataset Overview
3. Target Definition
4. Leakage Audit
5. Data Quality
6. Exploratory Data Analysis
7. Feature Engineering
8. Validation Strategy
9. Logistic Regression
10. CatBoost
11. Metrics
12. Threshold = 0.60 Analysis
13. Model Comparison
14. Error Analysis
15. Explainability
16. Limitations
17. Conclusion

---

### 6. 7-Day Plan (Minimal Version V1.0)

* **Day 1:** Repository, Data dictionary, Feature classification, Leakage audit, Target definition. (Result: `docs/data_contract.md`)
* **Day 2:** Missing values, Outliers, Target imbalance, Categorical/numerical distributions. (Result: `01_data_understanding.ipynb`)
* **Day 3:** EDA (FPD by products, cities, incomes, amounts, delinquencies, inquiries).
* **Day 4:** Preprocessing (missing values, encoding, logarithms, feature engineering).
* **Day 5:** DummyClassifier, LogisticRegression. Metrics.
* **Day 6:** CatBoostClassifier (with class weights). Metrics.
* **Day 7:** Compare Logistic vs. CatBoost at `threshold = 0.60`. Confusion matrix, precision, recall, F2, PR-AUC, ROC-AUC.

*Result after a week:* a working ML task that does exactly what the bank asked.

---

### 7. Releases (Freedom part only)

```text
v0.1 — Data understanding
v0.2 — EDA + leakage audit
v0.3 — Logistic baseline
v0.4 — CatBoost
v0.5 — Threshold = 0.60 evaluation
v0.6 — SHAP + error analysis
v1.0 — Freedom Bank assignment complete (Sent to the bank)
v1.1 — Clean Python package
v1.2 — FastAPI
v1.3 — PostgreSQL
v1.4 — MLflow
v1.5 — Docker
```

---

### 8. Signs You've Reached HERO Level (within the Freedom project)

You should be able to explain without a notebook:

* Why was FPD used as the target?
* Why is CENZ needed for filtering?
* Why was `Active_Overdue_Days` excluded?
* Why is accuracy useless at a 0.46% positive class?
* Why was PR-AUC chosen?
* What does Recall@0.6 mean?
* Why is a temporal split necessary?
* How did you handle missing values?
* Why was Logistic Regression used as a baseline?
* Why is CatBoost suitable for high-cardinality categorical variables?
* Why do probabilities need to be calibrated?
* How to send a new application to the API and get a scoreball?



























































Вот полная выжимка всех материалов, касающихся **исключительно проекта Freedom Credit Score** (от понимания задачи до.production-развертывания модели), без каких-либо упоминаний второго проекта или стриминг-систем.

---

### 1. Суть задачи Freedom Bank

**Формулировка задания:** Найти подозрительные займы с помощью моделей машинного обучения, где `scoreball ≥ 0.6` относится к фродовым займам.

**Как это реализуется:**

1. В Excel нет готового столбца `scoreball`, его создает ваша ML-модель.
2. `scoreball = model.predict_proba(X)[:, 1]`
3. Если `scoreball >= 0.60` → займ помечается как подозрительный/фродовый (Suspicious / Fraud). Если `< 0.60` → нормальный (normal).

**Выбор таргета (Target):**

* **Основной таргет:** `FPD_15` (First Payment Delinquency). Используется только для наблюдений, где `CENZ_15_1_Maturity_Flag == 1` (проверенные вызревшие заявки).
  * *Статистика:* 24,142 заявок, из них 111 фродовых. Positive rate ≈ 0.46%. Это сильный дисбаланс, что хорошо для antifraud ML. `FPD_15` логично называть **fraud/risk proxy**.
* **Второй таргет:** `SPD_15` с фильтром `CENZ_15_2_Maturity_Flag == 1`.
  * *Статистика:* 23,886 заявок, 199 позитивных. Positive rate ≈ 0.83%.
  * Позволяет сравнить Model A (FPD) vs Model B (SPD).

**Leakage Audit (Аудит утечек данных):**
До первой модели необходимо классифицировать 20 полей:

* `Application_Date` — ✅ использовать / создавать time features.
* `Application_Code` — ❌ только ID.
* `Credit_Product`, `Declared_Official_Income`, `Requested_Loan_Amount`, `City`, `Employer_BIN`, `Days_Overdue_Last_6_Months`, `Number_of_Completed_Loans`, `Total_Outstanding_Balance`, `Number_of_Active_Loans`, `Number_of_PCB_Inquiries`, `Loan_Term` — ✅ использовать.
* `Manager` — ⚠️ отдельный эксперимент.
* `CENZ_15_1_Maturity_Flag` / `CENZ_15_2_Maturity_Flag` — ❌ это фильтры, не фичи.
* `FPD_15`, `SPD_15` — 🎯 таргеты.
* `Active_Overdue_Days` — 🚨 **ВЕРОЯТНАЯ УТЕЧКА**. Столбец нельзя включать в модель, иначе она будет «предсказывать» просрочку, смотря на то, сколько дней клиент уже просрочен.

---

### 2. Пошаговые инструкции (От LEVEL 0 до LEVEL 15)

**LEVEL 0: Базовая структура**
Создать репозиторий с папками `data/`, `notebooks/`, `src/`, `reports/`, `tests/`. Никаких Docker, Kafka или MLflow на этом этапе. Файл `freedom_sample.xlsx` положить в `data/raw/` (и добавить в `.gitignore`, если данные нельзя публиковать).

**LEVEL 1: Data Understanding**
Ноутбук `01_data_understanding.ipynb`. Без ML. Проверить shape, dtypes, duplicates, missing values, target distribution. Ответить на вопросы: сколько заявок? сколько фрода? какие продукты/города? какие признаки категориальные/числовые? где пропуски? где возможна утечка?

**LEVEL 2: EDA**
Ноутбук `02_eda.ipynb`. Исследовать FPD по продуктам, сумму займа, доход, предыдущие просрочки, активные займы, запросы в ПКБ и распределение фрода по времени.

**LEVEL 3: Data Quality**
Ноутбук `03_data_quality.ipynb`. Изучить пропуски (Employer BIN, Loan Term и т.д.). Не делать `fillna(0)` автоматически. Пропуски в категориальных лучше заполнять `"UNKNOWN"`, так как отсутствие данных может быть сигналом.

**LEVEL 4: Outliers**
В доходах огромный разброс. Не удалять автоматически. Смотреть перцентили. Для Logistic Regression использовать `log1p()` или winsorization (1-й и 99-й перцентили). Для CatBoost выбросы менее разрушительны.

**LEVEL 5: Feature Engineering**
Ноутбук `04_feature_engineering.ipynb`. Создать понятные фичи:

* `loan_income_ratio` (сумма займа / доход)
* `outstanding_income_ratio` (остаток долга / доход)
* `total_previous_loans` (активные + завершенные)
* `active_loan_ratio`
* `had_previous_overdue` (дни просрочки > 0)
* `high_inquiry_flag`
* Time features из даты заявки (месяц, неделя, день недели).

**LEVEL 6: Правильный Split**
Не использовать random split как основной. Сделать **Temporal split** (по времени):

* TRAIN: 2022-04-01 → 2022-05-15
* VALIDATION: 2022-05-16 → 2022-06-07
* TEST: 2022-06-08 → 2022-06-30
  Для baseline можно сделать Stratified split, но финальный результат проверять temporal holdout.

**LEVEL 7: Dummy Baseline**
Обучить `DummyClassifier` (предсказывает всё как 0). При 0.46% фрода accuracy будет 99.54%. Это доказывает, что метрика accuracy бесполезна.

**LEVEL 8: Model №1 — Logistic Regression**
Простой, интерпретируемый baseline. Пайплайн: медианная импутация + масштабирование для числовых; заполнение `UNKNOWN` + OneHotEncoder для категориальных. Использовать `class_weight="balanced"`.

**LEVEL 9: Model №2 — CatBoostClassifier**
Идеален для табличных данных, категориальных фичей высокой кардинальности (City, Employer_BIN), пропусков и дисбаланса классов.

**LEVEL 10: Metrics**
Основная метрика — **PR-AUC** (из-за сильного дисбаланса). Дополнительно: Recall, Precision, F1, F2, ROC-AUC. Обязательно показать метрики **при threshold = 0.60** (TP, FP, TN, FN, Precision@0.6, Recall@0.6).

**LEVEL 11: Threshold Analysis**
Исследовать пороги от 0.10 до 0.90. Построить графики threshold vs recall/precision/F2. Но в финальном сравнении оставить требование Freedom: `0.60`.

**LEVEL 12: Calibration**
Проверить Calibration Curve и Brier Score. Применить Platt scaling или Isotonic calibration, чтобы `scoreball = 0.70` действительно отражало вероятность фрода.

**LEVEL 13: Model Comparison**
Создать таблицу сравнения: Dummy, Logistic Regression, CatBoost по метрикам PR-AUC, ROC-AUC, Precision@0.6, Recall@0.6, F2@0.6.

**LEVEL 14: Error Analysis**
Анализ False Positives и False Negatives. Найти закономерности (например, модель плохо ловит определенный кредитный продукт).

**LEVEL 15: Explainability**
Вывести коэффициенты Logistic Regression и SHAP для CatBoost. Показать Global feature importance и пример индивидуального предсказания (Application #123, scoreball = 0.81, главные факторы риска).

---

### 3. Финальная структура репозитория (Freedom Assignment + Production)

После выполнения базового задания (V1.0), проект превращается в production-модуль:

```text
freedom-loan-fraud-scoring/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_quality.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_logistic_baseline.ipynb
│   ├── 06_catboost.ipynb
│   └── 07_model_comparison.ipynb
│
├── src/
│   └── fraud_scoring/
│       ├── __init__.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── split.py
│       ├── models/
│       │   ├── logistic.py
│       │   └── catboost.py
│       └── evaluation/
│           ├── metrics.py
│           └── plots.py
│
├── tests/
│   ├── test_features.py
│   └── test_preprocessing.py
│
├── reports/
│   ├── figures/
│   └── freedom_antifraud_report.pdf
│
├── artifacts/
│   ├── models/
│   └── predictions/    # Сохранять test_predictions.csv (application_code, actual_fpd, scoreball, predicted_fraud)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

### 4. Production-развертывание (Hero Levels для модели Freedom)

После V1.0 модель можно "продакшенить":

1. **FastAPI:** Создать эндпоинт `POST /application/score`, который принимает JSON с параметрами заявки и возвращает `scoreball`, `suspicious` (bool), `threshold` и `model_version`.
2. **PostgreSQL:** Завести таблицы `applications`, `scores`, `model_versions`, `decisions`, чтобы заявка → БД → ML модель → scoreball → решение → БД.
3. **MLflow:** Трекать эксперименты, параметры, версии фичей, метрики (PR-AUC, Brier Score) и артефакты.
4. **Docker:** Упаковать FastAPI, PostgreSQL и MLflow в контейнеры для запуска через `docker compose up`.

---

### 5. Требования к Отчету Freedom Bank

Структура финального отчета:

1. Business Problem
2. Dataset Overview
3. Target Definition
4. Leakage Audit
5. Data Quality
6. Exploratory Data Analysis
7. Feature Engineering
8. Validation Strategy
9. Logistic Regression
10. CatBoost
11. Metrics
12. Threshold = 0.60 Analysis
13. Model Comparison
14. Error Analysis
15. Explainability
16. Limitations
17. Conclusion

---

### 6. План на первые 7 дней (Минимальная версия V1.0)

* **Day 1:** Repository, Data dictionary, Feature classification, Leakage audit, Target definition. (Результат: `docs/data_contract.md`)
* **Day 2:** Missing values, Outliers, Target imbalance, Categorical/numerical distributions. (Результат: `01_data_understanding.ipynb`)
* **Day 3:** EDA (FPD по продуктам, городам, доходам, суммам, просрочкам, запросам).
* **Day 4:** Preprocessing (пропуски, кодирование, логарифмирование, feature engineering).
* **Day 5:** DummyClassifier, LogisticRegression. Метрики.
* **Day 6:** CatBoostClassifier (с class weights). Метрики.
* **Day 7:** Сравнение Logistic vs CatBoost при `threshold = 0.60`. Confusion matrix, precision, recall, F2, PR-AUC, ROC-AUC.

Итог через неделю: работающая ML-задача, решающая ровно то, что просил банк.

---

### 7. Релизы (только для части Freedom)

```text
v0.1 — Data understanding
v0.2 — EDA + leakage audit
v0.3 — Logistic baseline
v0.4 — CatBoost
v0.5 — Threshold = 0.60 evaluation
v0.6 — SHAP + error analysis
v1.0 — Freedom Bank assignment complete (Отправлено банку)
v1.1 — Clean Python package
v1.2 — FastAPI
v1.3 — PostgreSQL
v1.4 — MLflow
v1.5 — Docker
```

### 8. Признаки того, что вы достигли уровня HERO (в рамках проекта Freedom)

Вы должны уметь без ноутбука объяснить:

* Почему FPD использован как target?
* Почему CENZ нужен для filtering?
* Почему `Active_Overdue_Days` исключён?
* Почему accuracy бесполезна при 0.46% positive class?
* Почему выбран PR-AUC?
* Что означает Recall@0.6?
* Почему нужен temporal split?
* Как вы обрабатывали missing values?
* Почему Logistic Regression использована как baseline?
* Почему CatBoost подходит для high-cardinality categorical variables?
* Почему probabilities нужно калибровать?
* Как отправить новую заявку в API и получить scoreball?
