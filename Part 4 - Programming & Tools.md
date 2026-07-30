# Part 4: Programming & Tools

> Comprehensive Lecture Notes for BS Data Science (4th Semester)

---

## 17. Python for Data Science

### Simple Explanation
Python is the most popular programming language for data science because it's easy to learn, has a huge ecosystem of libraries, and is used across the entire data pipeline — from data collection to model deployment.

### Core Libraries

| Library | Simple Explanation | Analogy |
|---------|-------------------|---------|
| **NumPy** | Numerical computing — arrays, math operations | The engine of a car |
| **Pandas** | Data manipulation — tables, CSV, Excel, SQL | The dashboard and controls |
| **Matplotlib** | Basic plotting (static) | A sketchpad |
| **Seaborn** | Beautiful statistical plots (built on Matplotlib) | A professional art kit |
| **Scikit-learn** | Machine learning — models, preprocessing, evaluation | A toolbox of ML algorithms |

### Simple Example
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# NumPy: fast array operations
arr = np.array([1, 2, 3, 4, 5])
print(arr * 2)  # [2, 4, 6, 8, 10]

# Pandas: load and explore data
df = pd.read_csv('sales.csv')
print(df.head())
print(df.describe())

# Matplotlib + Seaborn: visualize
sns.scatterplot(data=df, x='ad_spend', y='revenue')
plt.show()

# Scikit-learn: train model
model = LinearRegression()
model.fit(df[['ad_spend']], df['revenue'])
```

### Expert Example
Full data science pipeline using library best practices:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_parquet('transactions.parquet')

# Preprocessing pipeline
numeric_features = ['age', 'income', 'tenure']
categoric_features = ['education', 'region', 'gender']

numeric_transformer = Pipeline([
    ('scaler', StandardScaler()),
    ('log_transform', FunctionTransformer(np.log1p, validate=True))
])

categoric_transformer = Pipeline([
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categoric_transformer, categoric_features)
])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10,
        early_stopping=True
    ))
])

# Train-test split (time-based for time series)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False  # No shuffle for time series
)

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Train final model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"Test R²: {r2_score(y_test, y_pred):.3f}")

# Feature importance
feature_names = (numeric_features + 
    list(pipeline.named_steps['preprocessor']
         .named_transformers_['cat']
         .named_steps['onehot']
         .get_feature_names_out(categoric_features)))
importances = pipeline.named_steps['model'].feature_importances_
feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False)
print(feat_imp.head(10))

# Save model
joblib.dump(pipeline, 'model.pkl')
```

---

### Jupyter Notebook / JupyterLab

**Simple Explanation**: An interactive computing environment where you can combine code, visualizations, and explanatory text in a single document. Great for exploration and sharing.

### Simple Example
A Jupyter Notebook for a data science project:
```
Cell 1: [Markdown] "# Customer Churn Analysis"
Cell 2: [Code] import pandas as pd; df = pd.read_csv('customers.csv')
Cell 3: [Code] df.head()
Cell 4: [Code] sns.countplot(x='churn', data=df)
Cell 5: [Markdown] "## Key Findings"
Cell 6: [Code] model.fit(X, y)
```

### Expert Example
Production-grade Jupyter workflow using nbdev and papermill:

```python
# Parametrized notebook (useful for running with different parameters)
# Cell tagged as "parameters":
date_range = "2024-01-01 to 2024-03-31"
model_type = "xgboost"
threshold = 0.5

# Execute notebook with parameters from command line
# papermill analysis.ipynb output.ipynb -p date_range "2024-Q1" -p model_type "lightgbm"

# Use JupyterLab extensions:
# - jupyterlab-git: version control
# - jupyterlab-toc: table of contents
# - jupyter-resource-usage: monitor memory
# - nb_black: auto-formatter
```

---

### Basic Python: Data Types, Loops, Functions, List Comprehensions

| Concept | Simple Explanation | Code Example |
|---------|-------------------|--------------|
| **Data Types** | Types of values Python can hold | int, float, str, bool, list, dict |
| **Loops** | Repeat operations | for i in range(10) |
| **Functions** | Reusable blocks of code | def clean_data(df): |
| **List Comprehension** | Compact way to create lists | [x*2 for x in range(10)] |

### Expert Example
Performance-optimized Python patterns:

```python
# Avoid: slow explicit loops
result = []
for i in range(len(df)):
    if df['value'][i] > 0:
        result.append(df['value'][i] * 2)

# Prefer: vectorized operations (10-100x faster)
condition = df['value'] > 0
result = df.loc[condition, 'value'] * 2

# Avoid: row-by-row apply
df['category'] = df['score'].apply(lambda x: 'High' if x > 80 else 'Medium' if x > 50 else 'Low')

# Prefer: np.select (vectorized)
conditions = [df['score'] > 80, df['score'] > 50]
choices = ['High', 'Medium']
df['category'] = np.select(conditions, choices, default='Low')

# Memory optimization: downcast dtypes
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
df['float_col'] = pd.to_numeric(df['float_col'], downcast='float')
# Can reduce memory by 50-75%

# Generator for memory-efficient iteration
def read_large_csv_in_chunks(filepath, chunksize=10000):
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        yield chunk

for chunk in read_large_csv_in_chunks('huge_file.csv'):
    process_chunk(chunk)
```

---

## 18. R for Data Science (Overview)

### Key Packages

| Package | Purpose | Simple Example |
|---------|---------|---------------|
| **dplyr** | Data manipulation (filter, mutate, group_by) | `filter(df, age > 30)` |
| **ggplot2** | Beautiful, layered plotting | `ggplot(df, aes(x=age, y=income)) + geom_point()` |
| **tidyr** | Data reshaping (pivot, separate, unite) | `pivot_wider(data, names_from=year, values_from=value)` |
| **readr** | Fast data reading | `read_csv('file.csv')` |

### Simple Example
```r
library(tidyverse)

# Read, clean, plot in R
df <- read_csv('sales.csv')

df %>%
  filter(revenue > 1000) %>%
  mutate(profit_margin = (revenue - cost) / revenue * 100) %>%
  group_by(region) %>%
  summarise(
    avg_margin = mean(profit_margin),
    total_revenue = sum(revenue),
    n = n()
  ) %>%
  ggplot(aes(x = region, y = avg_margin)) +
  geom_col(fill = "steelblue") +
  labs(title = "Average Profit Margin by Region") +
  theme_minimal()
```

### When to Use R vs. Python

| Scenario | Better Choice | Why |
|----------|--------------|-----|
| Statistical modeling | R | Built for statistics, vast package library |
| Machine learning / deployment | Python | Scikit-learn, TensorFlow, production tools |
| Data visualization | R (ggplot2) | Grammar of graphics is powerful and elegant |
| Deep learning | Python | TensorFlow, PyTorch built for Python |
| Web apps / APIs | Python | Flask, FastAPI, Django |
| Academic research / reports | R | RMarkdown, knitr, excellent for reproducible research |
| General-purpose programming | Python | More versatile outside data science |

### Expert Example
R for statistical modeling with tidyverse + broom:

```r
library(tidyverse)
library(broom)
library(caret)

# Multi-model comparison
df <- read_csv('churn_data.csv')

# Logistic regression
model1 <- glm(churn ~ tenure + contract_type + monthly_charges, 
              data = df, family = binomial)

# Mixed effects model
library(lme4)
model2 <- glmer(churn ~ tenure + monthly_charges + (1 | region), 
                data = df, family = binomial)

# Extract tidy results
model1_results <- tidy(model1, conf.int = TRUE)
model1_metrics <- glance(model1)

# Compare models with cross-validation
control <- trainControl(method = "cv", number = 5)
models <- list(
  logistic = train(churn ~ ., data = df, method = "glm", trControl = control),
  rf = train(churn ~ ., data = df, method = "rf", trControl = control),
  xgb = train(churn ~ ., data = df, method = "xgbTree", trControl = control)
)

results <- resamples(models)
summary(results)
dotplot(results)
```

---

## 19. SQL for Data Science

### Simple Explanation
SQL (Structured Query Language) is the language for talking to databases. It lets you extract, filter, aggregate, and join data. Most real-world data lives in databases, so SQL is a non-negotiable skill for data scientists.

### Simple Example
```sql
-- Find top 10 customers by total orders
SELECT 
    c.customer_name,
    COUNT(o.order_id) AS order_count,
    SUM(o.order_amount) AS total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= '2024-01-01'
GROUP BY c.customer_id, c.customer_name
HAVING COUNT(o.order_id) >= 5
ORDER BY total_spent DESC
LIMIT 10;
```

### SELECT, WHERE, GROUP BY, HAVING, ORDER BY

| Clause | Simple Explanation | Example |
|--------|-------------------|---------|
| **SELECT** | Choose columns | `SELECT name, salary` |
| **WHERE** | Filter rows | `WHERE age > 30` |
| **GROUP BY** | Group rows for aggregation | `GROUP BY department` |
| **HAVING** | Filter groups (like WHERE for groups) | `HAVING AVG(salary) > 50000` |
| **ORDER BY** | Sort results | `ORDER BY salary DESC` |

### Expert Example
Advanced analytical SQL with window functions:

```sql
-- Customer segmentation with RFM analysis (Recency, Frequency, Monetary)
WITH customer_rfm AS (
    SELECT 
        customer_id,
        DATE_DIFF(CURRENT_DATE(), MAX(order_date), DAY) AS recency,
        COUNT(DISTINCT order_id) AS frequency,
        SUM(order_amount) AS monetary,
        AVG(order_amount) AS avg_order_value,
        -- Percentile ranks
        NTILE(5) OVER (ORDER BY DATE_DIFF(CURRENT_DATE(), MAX(order_date), DAY) DESC) AS r_score,
        NTILE(5) OVER (ORDER BY COUNT(DISTINCT order_id)) AS f_score,
        NTILE(5) OVER (ORDER BY SUM(order_amount)) AS m_score,
        -- Running total of revenue by month
        SUM(SUM(order_amount)) OVER (
            PARTITION BY customer_id 
            ORDER BY DATE_TRUNC(order_date, MONTH)
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_revenue
    FROM orders
    WHERE order_date >= '2023-01-01'
    GROUP BY customer_id
)
SELECT 
    customer_id,
    recency, frequency, monetary,
    CONCAT(r_score, f_score, m_score) AS rfm_segment,
    CASE 
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Best Customers'
        WHEN r_score >= 4 AND f_score >= 3 THEN 'Loyal Customers'
        WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
        WHEN r_score <= 2 AND f_score <= 2 THEN 'Lost'
        ELSE 'Needs Attention'
    END AS customer_segment,
    cumulative_revenue
FROM customer_rfm
ORDER BY monetary DESC;
```

---

### Joins (INNER, LEFT, RIGHT, FULL)

| Join | Simple Explanation | Venn |
|------|-------------------|------|
| **INNER** | Rows that exist in BOTH tables | Intersection |
| **LEFT** | All rows from left + matching from right | Left circle |
| **RIGHT** | All rows from right + matching from left | Right circle |
| **FULL** | All rows from both | Union |

### Expert Example
Multi-table join for a data warehouse star schema:

```sql
-- Star schema: fact_sales + dimension tables
SELECT 
    d_date.year,
    d_product.category,
    d_store.region,
    d_customer.segment,
    SUM(f.sales_amount) AS total_sales,
    COUNT(DISTINCT f.customer_id) AS unique_customers,
    SUM(f.quantity) AS total_units,
    ROUND(AVG(f.discount_pct), 2) AS avg_discount,
    -- Year-over-year growth
    (SUM(f.sales_amount) - LAG(SUM(f.sales_amount)) 
        OVER (PARTITION BY d_product.category, d_store.region 
              ORDER BY d_date.year)) 
    / NULLIF(LAG(SUM(f.sales_amount)) 
        OVER (PARTITION BY d_product.category, d_store.region 
              ORDER BY d_date.year), 0) * 100 AS yoy_growth_pct
FROM fact_sales f
INNER JOIN dim_date d_date ON f.date_key = d_date.date_key
LEFT JOIN dim_product d_product ON f.product_key = d_product.product_key
LEFT JOIN dim_store d_store ON f.store_key = d_store.store_key
LEFT JOIN dim_customer d_customer ON f.customer_key = d_customer.customer_key
WHERE d_date.year >= 2022
GROUP BY 
    d_date.year,
    d_product.category,
    d_store.region,
    d_customer.segment
HAVING SUM(f.sales_amount) > 10000
ORDER BY d_date.year DESC, total_sales DESC;
```

---

### Subqueries and CTEs

| Feature | Simple Explanation | When to Use |
|---------|-------------------|-------------|
| **Subquery** | A query inside another query | Simple one-time calculations |
| **CTE (WITH)** | Named temporary result set | Complex multi-step queries, readability |

### Expert Example
Recursive CTE — hierarchical data traversal:

```sql
-- Employee hierarchy (who reports to whom)
WITH RECURSIVE org_chart AS (
    -- Anchor: top-level managers
    SELECT 
        employee_id,
        employee_name,
        manager_id,
        employee_name AS path,
        0 AS level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive: subordinates
    SELECT 
        e.employee_id,
        e.employee_name,
        e.manager_id,
        CONCAT(oc.path, ' → ', e.employee_name) AS path,
        oc.level + 1
    FROM employees e
    INNER JOIN org_chart oc ON e.manager_id = oc.employee_id
)
SELECT 
    employee_id,
    employee_name,
    level,
    path
FROM org_chart
ORDER BY path;
```

---

### Aggregation Functions

| Function | Simple Explanation | Example |
|----------|-------------------|---------|
| **COUNT** | Count rows | `COUNT(*)` or `COUNT(DISTINCT id)` |
| **SUM** | Sum values | `SUM(revenue)` |
| **AVG** | Average | `AVG(salary)` |
| **MIN/MAX** | Minimum/Maximum | `MIN(age)`, `MAX(age)` |

### Expert Example

```sql
-- Advanced aggregations for data quality and reporting
SELECT 
    department,
    COUNT(*) AS total_employees,
    COUNT(DISTINCT job_title) AS unique_titles,
    COUNT(email) AS has_email,  -- NULLs excluded from count
    COUNT(CASE WHEN salary IS NULL THEN 1 END) AS missing_salary,
    AVG(salary) AS mean_salary,
    -- Robust statistics (no built-in median in most SQL)
    AVG(CASE 
        WHEN percentile <= 0.5 AND percentile > 0.0 THEN salary 
    END) AS median_salary,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_sql,
    STDDEV(salary) AS std_salary,
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary,
    -- Skewness (third moment)
    AVG(POWER((salary - AVG(salary)) / NULLIF(STDDEV(salary), 0), 3)) AS skewness
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary) / 
        COUNT(*) OVER (PARTITION BY department) AS percentile
    FROM employees
) sub
GROUP BY department;
```

---

### Window Functions

**Simple Explanation**: Functions that perform calculations across a set of rows related to the current row, without collapsing them into a single output row.

| Function | Simple Explanation | Example |
|----------|-------------------|---------|
| **ROW_NUMBER** | Sequential row number | 1, 2, 3, 4... |
| **RANK** | Rank with gaps for ties | 1, 1, 3, 4... |
| **DENSE_RANK** | Rank without gaps | 1, 1, 2, 3... |
| **LAG** | Previous row's value | Previous month's sales |
| **LEAD** | Next row's value | Next month's forecast |

### Expert Example

```sql
-- Time-series anomaly detection using window functions
WITH daily_metrics AS (
    SELECT 
        date,
        SUM(revenue) AS daily_revenue,
        COUNT(DISTINCT user_id) AS daily_users
    FROM transactions
    GROUP BY date
),
rolling_stats AS (
    SELECT 
        date,
        daily_revenue,
        daily_users,
        -- 7-day rolling average
        AVG(daily_revenue) OVER (
            ORDER BY date 
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) AS rolling_avg_revenue,
        -- 7-day rolling standard deviation
        STDDEV(daily_revenue) OVER (
            ORDER BY date 
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) AS rolling_std_revenue,
        -- Year-over-year comparison
        LAG(daily_revenue, 365) OVER (ORDER BY date) AS revenue_same_day_last_year,
        -- Running total for the month
        SUM(daily_revenue) OVER (
            PARTITION BY DATE_TRUNC(date, MONTH)
            ORDER BY date
        ) AS month_to_date_revenue,
        -- Same day last week
        LAG(daily_revenue, 7) OVER (ORDER BY date) AS revenue_last_week
    FROM daily_metrics
)
SELECT 
    date,
    daily_revenue,
    rolling_avg_revenue,
    CASE 
        WHEN daily_revenue > rolling_avg_revenue + 3 * rolling_std_revenue 
            THEN 'SPIKE'
        WHEN daily_revenue < rolling_avg_revenue - 3 * rolling_std_revenue
            THEN 'DROP'
        ELSE 'NORMAL'
    END AS anomaly_flag,
    -- Revenue growth vs last year
    ROUND((daily_revenue - revenue_same_day_last_year) / 
          NULLIF(revenue_same_day_last_year, 0) * 100, 2) AS yoy_growth_pct,
    -- Week-over-week change
    daily_revenue - revenue_last_week AS wow_change
FROM rolling_stats
WHERE date >= '2024-01-01'
ORDER BY date;
```

---

## 20. Version Control (Git)

### Simple Explanation
Git is like a "save" button for your entire project — it tracks every change, lets you go back to any previous version, and makes collaboration possible.

### Simple Example
```bash
# Basic workflow
git init                      # Start tracking a project
git add file.py               # Stage a file
git commit -m "Add analysis script"  # Save a snapshot
git log                       # See all snapshots

# Collaboration
git clone https://github.com/...  # Download a project
git pull                      # Get latest changes
git push                      # Upload your changes

# Branching (experiment safely)
git branch feature-analysis   # Create a new branch
git checkout feature-analysis # Switch to it
# ... make changes ... 
git checkout main
git merge feature-analysis    # Bring changes back
```

### Expert Example
Git workflow for data science projects:

```bash
# Branch strategy for experiments
git checkout -b experiment/xgboost-hyperopt

# Run experiment, then commit results
git add notebooks/experiment_results.ipynb
git commit -m "Experiment: XGBoost with Hyperopt — AUC improved from 0.82 to 0.85"

# Use .gitignore for data science
# .gitignore
*.pyc
__pycache__/
/data/raw/        # Raw data too large
/data/processed/
*.parquet
*.csv
*.pkl
models/*.pkl
.DS_Store
.ipynb_checkpoints/
.env

# Track only code and small processed data samples
# Use DVC (Data Version Control) for large files
```

---

## 21. Command Line / Shell Basics

### Simple Explanation
The command line is a text-based way to interact with your computer. It's faster than GUIs for many tasks and essential for working with servers and data pipelines.

### File Navigation
```bash
pwd                    # Print working directory — where am I?
ls -la                 # List all files (including hidden) with details
cd /path/to/folder     # Change directory
mkdir project          # Create a directory
```

### File Operations
```bash
cp file.txt backup.txt    # Copy file
mv file.txt newname.txt   # Move or rename
rm oldfile.txt            # Remove (careful!)
cat file.txt              # Display file content
less file.txt             # View file page by page (q to quit)
head -n 20 file.csv       # First 20 lines
tail -n 20 file.csv       # Last 20 lines
wc -l file.csv            # Count lines
```

### Pipes and Redirection
```bash
# Pipe: send output of one command to another
cat sales.csv | head -n 5

# Redirection: save output to file
python analysis.py > results.txt

# Chain commands: count unique IPs in log
cat server.log | grep "ERROR" | awk '{print $1}' | sort | uniq -c | sort -nr | head -10

# Data pipeline example (extract-transform-display)
cat logfile.csv \
  | grep "2024-10" \
  | awk -F',' '{print $2, $5}' \
  | sort \
  | uniq -c \
  | sort -rn \
  | head -5
```

### Expert Example
Shell data pipeline for big data:

```bash
# Process 10GB CSV without loading into memory
# Find top 10 products by sales
zcat sales_2024.csv.gz \
  | awk -F',' 'NR>1 {sales[$3]+=$5; count[$3]++} END {for (p in sales) print p, sales[p], count[p]}' \
  | sort -k2 -rn \
  | head -10 \
  > top_products.txt

# Parallel processing with xargs
cat urls.txt | xargs -P 8 -I {} curl -s {} > results.json

# Monitor and alert
while true; do
  errors=$(tail -100 app.log | grep -c "ERROR")
  if [ $errors -gt 10 ]; then
    echo "ALERT: $errors errors in last 100 lines" | mail -s "Error Alert" team@company.com
  fi
  sleep 60
done
```

---

> **Summary**: Part 4 covers the essential programming and tooling skills every data scientist needs. Python is the primary language, but SQL is equally important for data extraction. Git ensures you don't lose work, and the command line makes you efficient. Master these tools early — they form the foundation for everything else in data science.
