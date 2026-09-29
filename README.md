* [ ] Credit Risk & Early-Default ML Platform


Да. Теперь, когда видна **реальная формулировка задания Freedom Bank**, я бы немного перестроил наш roadmap.

Ваш Freedom dataset должен стать **первым модулем большого проекта**, а не просто одноразовым тестовым заданием:

> **Этап A: Loan/Application Fraud Scoring на данных Freedom**
> ↓
> **Этап B: Production Application-Risk Service**
> ↓
> **Этап C: Real-Time Transaction Antifraud на PaySim**
> ↓
> **Финал: Financial Risk & Real-Time Antifraud Platform**

Так вы сначала решаете ровно ту задачу, которую дал банк, а уже потом превращаете её в серьезный portfolio project. Никакого Kafka на второй день. Человечество переживет.

---

# 1. Сначала правильно поймём задачу Freedom

Формулировка:

> Найти подозрительные займы с помощью моделей машинного обучения, где scoreball ≥ 0.6 относится к фродовым займам.

Я бы интерпретировал это так:

```text
Application
    ↓
ML model
    ↓
predict_proba()
    ↓
scoreball
    ↓
scoreball >= 0.60
    ↓
Suspicious / Fraud
```

То есть:

```python
scoreball = model.predict_proba(X)[:, 1]
```

И затем:

```text
scoreball < 0.60
→ normal

scoreball >= 0.60
→ suspicious/fraud
```

Но важно: в Excel **нет готового столбца `scoreball`**.

Его создаёт ваша модель.

---

# 2. Что использовать как target?

Я бы начал с:

```text
TARGET = FPD_15
```

и только с наблюдений:

```text
CENZ_15_1_Maturity_Flag == 1
```

В вашем файле это даёт:

```text
Mature applications: 24,142

FPD_15 = 1:      111
FPD_15 = 0:   24,031
```

То есть positive rate:

**≈ 0.46%**

Очень сильный дисбаланс.

И это хорошо для обучения antifraud ML.

Но есть важная терминологическая оговорка:

> `FPD_15 = 1` означает early/first-payment delinquency outcome, а не автоматически доказанное мошенничество.

Freedom, судя по заданию, использует такие outcomes для поиска подозрительных займов. В своём README я бы называл `FPD_15` **fraud/risk proxy**, если банк отдельно не дал более точного определения fraud label.

---

# 3. Второй target

После FPD:

```text
TARGET = SPD_15
```

с фильтром:

```text
CENZ_15_2_Maturity_Flag == 1
```

Там у вас:

```text
23,886 mature observations
199 SPD positives
```

Positive rate:

**≈ 0.83%**

То есть чуть больше положительных примеров.

В итоге вы сможете сравнить:

```text
Model A
Predict FPD_15

vs

Model B
Predict SPD_15
```

Это уже интереснее обычного Kaggle-ноутбука.

---

# 4. Сначала сделайте leakage audit

До первой модели.

Для ваших 20 полей я бы классифицировал их примерно так:

| Feature                                | Что делать                                                 |
| -------------------------------------- | ------------------------------------------------------------------- |
| `Application_Date`                   | ✅ использовать / создавать time features      |
| `Application_Code`                   | ❌ только ID                                                  |
| `Client_ID_Expiry_Date`              | ⚠️ проверить смысл, затем создать delta |
| `Credit_Product`                     | ✅                                                                  |
| `Declared_Official_Income_from_Form` | ✅                                                                  |
| `Manager`                            | ⚠️ отдельный эксперимент                      |
| `Requested_Loan_Amount`              | ✅                                                                  |
| `City`                               | ✅                                                                  |
| `Employer_BIN_from_GCVP_Report`      | ✅ high-cardinality                                                 |
| `Days_Overdue_Last_6_Months...`      | ✅ если известно на момент заявки         |
| `Number_of_Completed_Loans...`       | ✅                                                                  |
| `Total_Outstanding_Balance...`       | ✅                                                                  |
| `Number_of_Active_Loans...`          | ✅                                                                  |
| `Number_of_PCB_Inquiries...`         | ✅                                                                  |
| `Loan_Term_in_Months`                | ✅                                                                  |
| `CENZ_15_1_Maturity_Flag`            | ❌ фильтр, не feature                                       |
| `FPD_15`                             | 🎯 target                                                           |
| `CENZ_15_2_Maturity_Flag`            | ❌ фильтр                                                     |
| `SPD_15`                             | 🎯 второй target                                              |
| `Active_Overdue_Days`                | 🚨 probable leakage                                                 |

Особенно:

```text
Active_Overdue_Days
```

не включайте в основной FPD/SPD model.

В вашем файле среднее значение этого поля очень сильно различается между positive/negative классами. Это выглядит как информация, которая могла появиться уже **после выдачи кредита**.

Иначе модель будет отвечать:

> «Будет ли клиент просрочен?»

посмотрев на:

> «Сколько дней он уже просрочен».

Машинное обучение достигло бы небывалого уровня ясновидения.

---

# 5. LEVEL 0 — начинаем совсем просто

Создайте repository:

```text
freedom-loan-fraud-scoring/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── reports/
│
├── tests/
│
├── README.md
├── requirements.txt
└── .gitignore
```

И всё.

**Не создавайте пока:**

```text
Kafka/
Docker/
Redis/
MLflow/
Prometheus/
Grafana/
Airflow/
FeatureStore/
```

Они вам сейчас совершенно не нужны.

---

# 6. `data/raw/`

Положите туда:

```text
data/
└── raw/
    └── freedom_sample.xlsx
```

Но если данные Freedom нельзя публично распространять:

```text
data/raw/
```

обязательно добавьте в `.gitignore`.

На GitHub можете оставить:

```text
data/
└── README.md
```

с описанием schema без самого файла.

Это особенно разумно для данных, полученных напрямую от банка.

---

# 7. LEVEL 1 — Data Understanding

Первый notebook:

```text
notebooks/
└── 01_data_understanding.ipynb
```

Никакого ML.

Проверьте:

```text
shape
columns
dtypes
duplicates
missing values
unique values
target distribution
dates
outliers
```

Ответьте на вопросы:

```text
Сколько заявок?

Сколько fraud/risk cases?

Какие продукты?

Какие города?

Какие признаки категориальные?

Какие числовые?

Какие признаки имеют пропуски?

Какие признаки похожи на ID?

Какие признаки могут быть leakage?
```

---

# 8. LEVEL 2 — EDA

Создайте:

```text
02_eda.ipynb
```

Исследуйте:

### Target

```text
FPD_15 = 0/1
```

---

### Credit product

```text
FPD rate by Credit_Product
```

---

### Requested amount

```text
Requested_Loan_Amount
vs
FPD
```

---

### Income

```text
Declared_Official_Income
vs
FPD
```

---

### Previous delinquency

```text
Days_Overdue_Last_6_Months
vs
FPD
```

---

### Existing loans

```text
Number_of_Active_Loans
vs
FPD
```

---

### PCB inquiries

```text
Number_of_PCB_Inquiries_Last_30_Days
vs
FPD
```

---

### Time

```text
fraud/default rate
by week/month
```

---

# 9. LEVEL 3 — Data quality

Создайте:

```text
03_data_quality.ipynb
```

У вас, например, есть пропуски в:

```text
Employer BIN
Loan Term
bureau features
ID-related date
```

Не делайте автоматически:

```python
df.fillna(0)
```

Для каждого признака задавайте вопрос:

> Что означает missing?

Например:

```text
Employer_BIN missing
```

может означать:

```text
нет данных
самозанятый
ошибка системы
не найден работодатель
```

Это может быть сигналом.

Для categorical:

```text
missing → "UNKNOWN"
```

часто разумнее, чем случайная мода.

---

# 10. LEVEL 4 — Outliers

В income у вас огромный разброс.

Есть значения порядка:

```text
1,000,000,000
```

А median около:

```text
350,000
```

Не удаляйте выбросы автоматически.

Сначала:

```text
p50
p90
p95
p99
max
```

Потом для Logistic Regression можно попробовать:

```text
log1p(income)

log1p(loan_amount)
```

или winsorization:

```text
1st percentile
99th percentile
```

Для CatBoost сильные выбросы обычно менее разрушительны.

И это можно написать в отчёте:

> Для линейной модели skewed monetary variables были преобразованы логарифмически, тогда как tree-based model использовал исходные значения.

Хорошая инженерная аргументация.

---

# 11. LEVEL 5 — Feature engineering

Теперь создайте:

```text
04_feature_engineering.ipynb
```

Сначала всего несколько понятных features.

### Loan-to-income

```text
loan_income_ratio =
requested_loan /
declared_income
```

---

### Outstanding debt / income

```text
outstanding_income_ratio =
total_outstanding_balance /
declared_income
```

---

### Existing credit activity

```text
total_previous_loans =
active_loans
+
completed_loans
```

---

### Active-loan ratio

```text
active_loan_ratio =
active_loans /
(total_loans + 1)
```

---

### Previous delinquency indicator

```text
had_previous_overdue =
days_overdue_last_6m > 0
```

---

### Credit-seeking intensity

```text
pcb_inquiries_30d
```

можно оставить raw и позже попробовать:

```text
high_inquiry_flag
```

---

# 12. Date features

Из:

```text
Application_Date
```

можно получить:

```text
application_month
application_week
application_day_of_week
```

Но не переусложняйте.

Из ID date только после понимания её точного значения.

Если это expiry date:

```text
days_until_document_expiry
```

Если дата выдачи документа:

```text
document_age_days
```

Не угадывайте семантику в финальной модели.

---

# 13. LEVEL 6 — правильный split

Для финального результата я бы **не использовал обычный random split как основной**.

У вас апрель-июнь 2022.

Можно сделать:

```text
TRAIN
2022-04-01 → 2022-05-15

10,354 mature applications
42 FPD
```

```text
VALIDATION
2022-05-16 → 2022-06-07

7,876 applications
37 FPD
```

```text
TEST
2022-06-08 → 2022-06-30

5,912 applications
32 FPD
```

Это не огромные positive counts, но уже позволяет сделать temporal experiment.

Такой split отвечает на вопрос:

> Если модель обучилась на прошлом, сможет ли она работать на будущих заявках?

---

# 14. Но для самого entry task можно сделать два validation experiments

### Experiment A

```text
Stratified train/test split
```

Нужен для стабильного первого baseline.

### Experiment B

```text
Temporal split
```

Нужен как более реалистичная банковская проверка.

И затем объяснить:

> Stratified split использовался для baseline и проверки pipeline, однако основной результат дополнительно проверялся temporal holdout, поскольку production model будет скорить будущие заявки.

Это сильнее, чем слепо использовать одну функцию sklearn.

---

# 15. LEVEL 7 — Dummy baseline

Первый ML model:

```text
DummyClassifier
```

Серьёзно.

Он показывает:

> Что произойдёт без ML?

При 0.46% positive rate:

```text
predict everything as 0
```

получит примерно:

**99.54% accuracy.**

Поэтому accuracy должна перестать вас впечатлять примерно навсегда.

---

# 16. LEVEL 8 — Model №1

Используйте:

# Logistic Regression

Почему?

```text
Simple
Interpretable
Strong baseline
Probability output
Common for risk/scoring
```

Pipeline:

```text
Numerical
    ↓
median imputation
    ↓
scaling

Categorical
    ↓
missing="UNKNOWN"
    ↓
OneHotEncoder

        ↓

Logistic Regression
class_weight="balanced"
```

Результат:

```text
P(FPD=1)
```

и далее:

```text
scoreball >= 0.60
→ suspicious
```

---

# 17. LEVEL 9 — Model №2

После Logistic Regression используйте:

# CatBoostClassifier

Не neural network.

Не Transformer.

Не модель, обнаруженная вчера ночью на arXiv.

CatBoost прекрасно подходит для:

```text
tabular data
categorical features
missing values
class imbalance
high-cardinality categories
```

У вас есть:

```text
City
Credit_Product
Manager
Employer_BIN
```

Поэтому CatBoost здесь логичен.

---

# 18. Почему эти две модели хороши для задания Freedom

Вы сможете написать:

### Logistic Regression

> Линейная интерпретируемая baseline-модель.

### CatBoost

> Нелинейная gradient boosting модель, способная учитывать взаимодействия признаков и эффективно работать с категориальными признаками.

То есть вы выполнили:

> минимум два разных алгоритма

не ради галочки.

---

# 19. LEVEL 10 — Metrics

Для этой задачи основной metric я бы выбрал:

# PR-AUC

Потому что positive class около:

```text
0.46%
```

Также:

```text
Recall
Precision
F1
F2
ROC-AUC
```

Но обязательно отдельно показать:

# Metrics at scoreball = 0.60

Например:

```text
threshold = 0.60

TP =
FP =
TN =
FN =

Precision =
Recall =
F2 =
```

Потому что Freedom прямо зафиксировал:

```text
>= 0.6 → fraud
```

---

# 20. Почему F2 полезен

Fraud detection часто сильнее наказывает:

```text
False Negative
```

то есть:

> реальный suspicious loan пропустили.

F2 придаёт recall больший вес, чем precision.

Поэтому можете показать:

```text
Precision
Recall
F1
F2
```

не делая вид, что существует одна священная метрика.

---

# 21. LEVEL 11 — Threshold analysis

Хотя Freedom дал:

```text
0.60
```

всё равно исследуйте:

```text
0.10
0.20
0.30
...
0.90
```

Постройте:

```text
threshold vs recall
threshold vs precision
threshold vs F2
```

Но в финальном сравнении обязательно оставьте:

```text
Freedom required threshold = 0.60
```

Вы не заменяете требования заказчика. Вы показываете дополнительный анализ.

---

# 22. LEVEL 12 — Calibration

Вот это уже сделает работу сильнее.

Если:

```text
scoreball = 0.70
```

желательно понимать, насколько probability meaningful.

Проверьте:

```text
Calibration Curve
Brier Score
```

Позже:

```text
Platt scaling
Isotonic calibration
```

Почему это особенно важно?

Потому что бизнес использует конкретный threshold:

```text
0.60
```

Если probabilities плохо calibrated, красивое число `0.60` может означать не то, что кажется.

---

# 23. LEVEL 13 — Model comparison

Сделайте таблицу:

| Model               | PR-AUC | ROC-AUC | Precision@0.6 | Recall@0.6 | F2@0.6 |
| ------------------- | -----: | ------: | ------------: | ---------: | -----: |
| Dummy               |    ... |     ... |           ... |        ... |    ... |
| Logistic Regression |    ... |     ... |           ... |        ... |    ... |
| CatBoost            |    ... |     ... |           ... |        ... |    ... |

Вот такая таблица должна быть центральной в отчёте.

---

# 24. LEVEL 14 — Error analysis

Это уже отличает вас от человека, который просто нажал `.fit()`.

Посмотрите:

### False positives

```text
Model:
fraud

Reality:
normal
```

Что у них общего?

---

### False negatives

```text
Model:
normal

Reality:
fraud/risk
```

Особенно важно.

Исследуйте:

```text
loan product
amount
income
city
previous overdue
active loans
inquiries
```

Может оказаться, например, что модель плохо ловит определённый credit product.

Это уже business insight.

---

# 25. LEVEL 15 — Explainability

Добавьте:

```text
Logistic Regression coefficients
```

и:

```text
SHAP for CatBoost
```

Покажите:

```text
Global feature importance
```

и:

```text
Individual fraud prediction
```

Например:

```text
Application #123

scoreball = 0.81

main risk factors:
↑ high loan/income ratio
↑ previous overdue
↑ many PCB inquiries
↓ high official income
```

Это очень хорошо смотрится на защите.

---

# 26. Первая нормальная architecture после этого

Теперь можно немного расширить repo:

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
│       │
│       ├── data.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── split.py
│       │
│       ├── models/
│       │   ├── logistic.py
│       │   └── catboost.py
│       │
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
│   └── predictions/
│
├── README.md
├── requirements.txt
└── .gitignore
```

Вот это уже хороший **Freedom assignment repository**.

---

# 27. Что сохранять после prediction

Создайте файл:

```text
artifacts/predictions/test_predictions.csv
```

Пример:

```text
application_code
application_date
actual_fpd
scoreball
predicted_fraud
model_name
```

Например:

```text
KAR.12345
2022-06-22
1
0.847
1
catboost_v1
```

Где:

```text
predicted_fraud =
1 if scoreball >= 0.60 else 0
```

Это напрямую отвечает заданию банка.

---

# 28. Что должно быть в Freedom report

Структура отчёта:

```text
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
```

Вот это уже полноценный ответ Freedom.

---

# 29. Минимальная версия задания

Если задача только:

> выполнить entry assignment

вам достаточно:

```text
EDA
↓
cleaning
↓
feature engineering
↓
Logistic Regression
↓
CatBoost
↓
PR-AUC / Recall / Precision
↓
threshold 0.60
↓
comparison
↓
report
```

Это **Version 1.0**.

Примерно:

**7–10 хороших рабочих дней**.

---

# 30. После выполнения задания не выбрасывайте repo

Вот здесь начинается ваш **Zero → Hero**.

Переименуйте conceptual project из:

```text
freedom-loan-fraud-scoring
```

в модуль:

```text
financial-risk-antifraud-platform/
└── application_risk/
```

Freedom становится:

```text
application_risk/
```

---

# 31. HERO LEVEL 1 — productionize Freedom model

Добавьте:

```text
FastAPI
```

Endpoint:

```text
POST /application/score
```

Request:

```json
{
  "credit_product": "...",
  "official_income": 350000,
  "requested_amount": 3000000,
  "city": "...",
  "previous_overdue_days": 4,
  "active_loans": 2,
  "pcb_inquiries_30d": 7
}
```

Response:

```json
{
  "scoreball": 0.73,
  "suspicious": true,
  "threshold": 0.60,
  "model_version": "catboost_v1"
}
```

Теперь модель реально используется.

---

# 32. HERO LEVEL 2 — PostgreSQL

Добавьте таблицы:

```text
applications
scores
model_versions
decisions
```

Теперь flow:

```text
Application
    ↓
PostgreSQL
    ↓
ML model
    ↓
scoreball
    ↓
decision
    ↓
PostgreSQL
```

---

# 33. HERO LEVEL 3 — MLflow

Теперь tracking:

```text
experiment
model
parameters
feature version
PR-AUC
ROC-AUC
Recall@0.6
Precision@0.6
Brier Score
artifact
```

И вы уже не храните модели как:

```text
model_final.pkl
model_final2.pkl
model_really_final.pkl
```

Видите, цивилизация всё-таки может развиваться.

---

# 34. HERO LEVEL 4 — Docker

Только теперь:

```text
Docker
```

Контейнеры:

```text
FastAPI
PostgreSQL
MLflow
```

и:

```bash
docker compose up
```

поднимает систему.

---

# 35. Но это всё ещё не Real-Time Transaction Antifraud

Freedom dataset отвечает на:

> Насколько подозрительна **заявка/заём**?

Он не отвечает на:

> Насколько подозрительна **транзакция прямо сейчас**?

Для полного проекта теперь добавляем:

```text
PaySim
```

---

# 36. HERO LEVEL 5 — PaySim module

Теперь структура превращается в:

```text
financial-risk-antifraud-platform/
│
├── application_risk/
│   └── freedom/
│
└── transaction_fraud/
    └── paysim/
```

Freedom:

```text
Loan/application scoring
```

PaySim:

```text
Transaction scoring
```

---

# 37. Начните PaySim тоже просто

Не Kafka.

Сначала:

```text
PaySim
 ↓
EDA
 ↓
temporal split
 ↓
Logistic Regression
 ↓
CatBoost / LightGBM
 ↓
PR-AUC
 ↓
threshold optimization
```

Только после этого:

```text
real-time simulation
```

---

# 38. HERO LEVEL 6 — simulate transactions

Берёте PaySim:

```text
transaction #1
transaction #2
transaction #3
...
```

и replay по времени:

```text
historical transactions
        ↓
Python producer
        ↓
one event at a time
```

Теперь уже можно говорить:

> streaming simulation

---

# 39. HERO LEVEL 7 — Kafka / Redpanda

Вот **только здесь**:

```text
PaySim
  ↓
Producer
  ↓
Kafka topic
  ↓
Fraud scoring consumer
  ↓
ML model
  ↓
Decision engine
```

---

# 40. HERO LEVEL 8 — объединяем оба мира

Финальная структура:

```text
financial-risk-antifraud-platform/
│
├── application_risk/
│   │
│   ├── freedom/
│   ├── features/
│   ├── models/
│   └── scoring/
│
├── transaction_fraud/
│   │
│   ├── paysim/
│   ├── features/
│   ├── models/
│   └── streaming/
│
├── common/
│   ├── config/
│   ├── logging/
│   └── database/
│
├── decision_engine/
│   ├── application_policy.py
│   └── transaction_policy.py
│
├── api/
│   ├── application_routes.py
│   └── fraud_routes.py
│
├── streaming/
│   ├── producer.py
│   └── consumer.py
│
├── database/
│   ├── models.py
│   └── migrations/
│
├── monitoring/
│   ├── drift.py
│   ├── psi.py
│   └── latency.py
│
├── tests/
│
├── configs/
│
├── docker/
│
├── docs/
│
├── reports/
│
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

Но обратите внимание:

**это final architecture.**

Не структура вашего первого дня.

---

# 41. Финальный HERO flow

В конце проект должен выглядеть концептуально так:

```text
                     FINANCIAL RISK PLATFORM
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
       LOAN APPLICATION                 TRANSACTION
              │                               │
              ▼                               ▼
       Freedom Features               Real-Time Features
              │                               │
              ▼                               ▼
        Application Model                Fraud Model
              │                               │
              ▼                               ▼
         Scoreball                    Fraud Probability
              │                               │
              └───────────────┬───────────────┘
                              ▼
                       Decision Engine
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
              APPROVE       REVIEW        BLOCK
                              │
                              ▼
                         PostgreSQL
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
           MLflow          Monitoring       Audit Log
```

Для real-time side:

```text
PaySim
  ↓
Kafka / Redpanda
  ↓
Feature Pipeline
  ↓
Fraud Model
  ↓
Decision
```

---

# 42. В каком порядке всё это делать

Вот ваш настоящий roadmap:

| Stage        | Что строите        | Примерный результат           |
| ------------ | ---------------------------- | ----------------------------------------------- |
| **1**  | Freedom data understanding   | понимаете dataset                      |
| **2**  | Leakage audit                | знаете допустимые features      |
| **3**  | EDA                          | понимаете fraud/default patterns       |
| **4**  | Preprocessing                | чистый dataset                            |
| **5**  | Logistic Regression          | baseline                                        |
| **6**  | CatBoost                     | сильная model                            |
| **7**  | Metrics + threshold 0.6      | выполнено требование Freedom |
| **8**  | Error analysis               | понимаете ошибки                 |
| **9**  | SHAP                         | объясняете predictions                |
| **10** | Final Freedom report         | entry task завершён                     |
| **11** | Refactor notebooks → Python | нормальный repo                       |
| **12** | FastAPI                      | application scoring service                     |
| **13** | PostgreSQL                   | persistence                                     |
| **14** | MLflow                       | experiment tracking                             |
| **15** | Docker                       | reproducible system                             |
| **16** | PaySim                       | transaction fraud                               |
| **17** | temporal fraud features      | stronger fraud model                            |
| **18** | event replay                 | pseudo real-time                                |
| **19** | Kafka/Redpanda               | real streaming                                  |
| **20** | Monitoring                   | drift + latency                                 |
| **21** | Combined platform            | hero version                                    |

---

# 43. Я бы разбил это на releases

```text
v0.1
Data understanding

v0.2
EDA + leakage audit

v0.3
Logistic baseline

v0.4
CatBoost

v0.5
Threshold = 0.60 evaluation

v0.6
SHAP + error analysis

v1.0
Freedom Bank assignment complete
```

Вот здесь уже можно остановиться и отправить работу Freedom.

После этого:

```text
v1.1
Clean Python package

v1.2
FastAPI

v1.3
PostgreSQL

v1.4
MLflow

v1.5
Docker
```

А затем:

```text
v2.0
PaySim transaction fraud

v2.1
Streaming simulation

v2.2
Kafka / Redpanda

v2.3
Monitoring

v3.0
Financial Risk & Real-Time Antifraud Platform
```

Это намного здоровее, чем пытаться закончить `v3.0` прежде, чем вы вообще обучили первую Logistic Regression.

---

# 44. Ваши первые 7 дней должны выглядеть вот так

### Day 1

```text
Repository
Data dictionary
Feature classification
Leakage audit
Target definition
```

Результат:

```text
docs/data_contract.md
```

---

### Day 2

```text
Missing values
Outliers
Target imbalance
Categorical values
Numerical distributions
```

Результат:

```text
01_data_understanding.ipynb
```

---

### Day 3

EDA:

```text
FPD by product
FPD by city
income
loan amount
overdue history
active loans
PCB inquiries
```

---

### Day 4

Preprocessing:

```text
missing values
categorical encoding
numerical transformation
feature engineering
```

---

### Day 5

```text
DummyClassifier
LogisticRegression
```

Метрики.

---

### Day 6

```text
CatBoostClassifier
class weights
```

Метрики.

---

### Day 7

Сравнить:

```text
Logistic
vs
CatBoost
```

при:

```text
threshold = 0.60
```

И сделать:

```text
confusion matrix
precision
recall
F2
PR-AUC
ROC-AUC
```

Через неделю у вас уже будет **работающая версия задания Freedom**.

Не Kafka.

Не Docker.

Не красивые облака на архитектурной диаграмме.

**Работающая ML-задача.**

---

# 45. Когда считать себя дошедшим до HERO

Не тогда, когда установлен Kafka.

А когда вы сможете без ноутбука объяснить:

> Почему FPD использован как target?

> Почему CENZ нужен для filtering?

> Почему `Active_Overdue_Days` исключён?

> Почему accuracy бесполезна при 0.46% positive class?

> Почему выбран PR-AUC?

> Что означает Recall@0.6?

> Почему нужен temporal split?

> Как вы обрабатывали missing values?

> Почему Logistic Regression использована как baseline?

> Почему CatBoost подходит high-cardinality categorical variables?

> Почему probabilities нужно калибровать?

> Как изменить threshold при изменении стоимости false positives/false negatives?

> Как предотвратить leakage при создании historical features?

> Как отправить новую заявку в API и получить scoreball?

> Как новая транзакция проходит через streaming antifraud system?

Вот тогда ваш проект действительно перестаёт быть учебным notebook и превращается в **банковский ML engineering portfolio project**.

И самое удачное здесь то, что Freedom уже дал вам отличный первый реальный use case. Вы можете выполнить исходное задание **в v1.0**, а затем использовать его как фундамент для всего `Financial Risk & Real-Time Antifraud Platform`, вместо того чтобы выбрасывать работу после тестового.
