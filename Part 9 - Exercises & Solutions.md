# Part 9: Exercises & Practice Problems

> Practice problems with solutions for BS Data Science.
> Work through each problem before reading the solution. Problems are marked **[Conceptual]**, **[Coding]**, or **[Challenge]** (expert-level).

---

## Part 1: Core Concepts & Foundations

**1.1 [Conceptual]** What is the difference between structured and unstructured data? Give two real-world examples of each.

**1.2 [Conceptual]** A temperature reading of 30°C is measured on which scale of measurement? Explain why.

**1.3 [Coding]** Given a Python dictionary of survey responses, classify each value as discrete, continuous, nominal, or ordinal:

```python
data = {
    'age': 28,
    'income': 85000.50,
    'education': 'Masters',
    'rating': 4.5,
    'num_children': 2
}
```

**1.4 [Challenge]** A hospital stores patient records in a relational database, MRI images in an object store, and doctor notes in XML files. Identify each as structured/semi-structured/unstructured, and explain the challenges of combining them for a research study.

### Solutions

**1.1** Structured data fits neatly into rows and columns (e.g., a sales spreadsheet, a customer table in a database). Unstructured data has no fixed format (e.g., emails, photos, videos).

**1.2** Interval scale — zero is arbitrary (30°C is not "twice as hot" as 15°C in a physical sense, and 0°C does not mean "no heat"). Ratio scale requires a true zero (e.g., weight, income).

**1.3** `age`: discrete numerical; `income`: continuous numerical; `education`: ordinal categorical; `rating`: continuous numerical (or ordinal if it is a 1–5 star rating); `num_children`: discrete numerical.

**1.4** Patient records = structured; MRI images = unstructured; doctor notes (XML) = semi-structured. Challenges: aligning identifiers across systems, matching the same patient in all three sources (entity resolution), handling different update frequencies, and conforming heterogeneous formats into a common schema.

---

## Part 2: Data Processing & Wrangling

**2.1 [Conceptual]** Explain the difference between dropping, imputing, and flagging missing values. When is each appropriate?

**2.2 [Coding]** A DataFrame has a `salary` column with a few extreme outliers. Detect them with the IQR method and clip (winsorize) them instead of removing them.

**2.3 [Coding]** Combine two customer tables — one from PostgreSQL, one from MongoDB — using a full outer join, then drop duplicate customer IDs keeping the latest record.

**2.4 [Challenge]** A sensor emits readings every second but some readings are corrupted (negative where impossible). Design a cleaning strategy: 3-sigma clipping, then linear interpolation, then flag any remaining suspicious values. Write the pandas code.

### Solutions

**2.1** Drop: remove rows with missing values — fine when missingness is rare and random. Impute: fill missing values (mean/median/mode, or model-based like KNN/MICE) — keeps data volume but can introduce bias. Flag: add an indicator column noting the value was missing — preserves information about *why* it is missing (often itself predictive).

**2.2**
```python
q1, q3 = df['salary'].quantile([0.25, 0.75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
df['salary'] = df['salary'].clip(lower=lower, upper=upper)
```

**2.3**
```python
merged = pd.merge(sql_customers, mongo_customers, on='customer_id',
                  how='outer', suffixes=('_sql', '_mongo'))
# Fill missing values from one source with the other, then keep one row per customer
merged = merged.sort_values('updated_at').drop_duplicates('customer_id', keep='last')
```

**2.4**
```python
# Step 1: 3-sigma clipping
mean, std = df['reading'].mean(), df['reading'].std()
df['reading'] = df['reading'].clip(mean - 3*std, mean + 3*std)

# Step 2: linear interpolation of remaining missing/flagged values
df['reading'] = df['reading'].interpolate(method='time')

# Step 3: flag suspicious values (impossible negatives, jumps > 5x previous)
df['flagged'] = (df['reading'] < 0) | (df['reading'].diff().abs() > 5 * df['reading'].shift())
```

---

## Part 3: EDA & Visualization

**3.1 [Conceptual]** Your boss asks "what does skewness tell you?" Answer in one plain-English sentence.

**3.2 [Coding]** Given a DataFrame with `age` and `income`, produce: a histogram of income, a box plot of income grouped by education level, and a scatter of age vs income with a regression line.

**3.3 [Coding]** A product manager reports "conversion went up 2% in the new design." Check whether this is likely real: run a two-sample proportion test (or chi-square) and report the p-value with a plain-English conclusion.

**3.4 [Challenge]** You have 4 years of daily sales. Check for stationarity with the Dickey-Fuller test, plot ACF/PACF, fit a SARIMA model, and evaluate with a walk-forward backtest. Which horizon (1-day vs 30-day) is more accurate, and why?

### Solutions

**3.1** Skewness tells you whether the data is lopsided: a long tail to the right (positive skew, e.g., income) or to the left (negative skew). It matters because many statistical methods assume symmetry.

**3.2**
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df['income'], bins=30, ax=axes[0])
sns.boxplot(data=df, x='education', y='income', ax=axes[1])
sns.regplot(data=df, x='age', y='income', ax=axes[2], scatter_kws={'alpha': 0.3})
plt.tight_layout()
plt.show()
```

**3.3**
```python
from statsmodels.stats.proportion import proportions_ztest

old_success, old_total = 800, 10000
new_success, new_total = 816, 10000
stat, p_value = proportions_ztest([new_success, old_success],
                                  [new_total, old_total])
print(f"p-value = {p_value:.4f}")
# p = 0.77 > 0.05 → not statistically significant; the 2% difference is likely noise
```

**3.4** Fit `SARIMAX(sales, order=(1,1,1), seasonal_order=(1,1,1,12))` (or ARIMA after checking ADF and ACF/PACF). Backtest by walking forward, always training only on past data. 1-day forecasts are almost always more accurate than 30-day forecasts because errors accumulate: uncertainty compounds with horizon, and trend/seasonal assumptions degrade over longer windows.

---

## Part 4: Programming & Tools

**4.1 [Coding]** Rewrite the slow loop below with vectorized pandas (10–100x faster):

```python
slow = []
for i in range(len(df)):
    if df['score'].iloc[i] >= 0.5:
        slow.append('high')
    else:
        slow.append('low')
```

**4.2 [Coding]** Write a SQL query using a window function to rank customers by total spend, then keep only the top 3 per region.

**4.3 [Coding]** From the command line, find the top 10 most frequent IPs in a large web server log using pipes.

**4.4 [Challenge]** You accidentally committed a 2GB data file to git. What's the problem, and what steps would you take to fix it (including preventing it from happening again)?

### Solutions

**4.1**
```python
df['bucket'] = np.where(df['score'] >= 0.5, 'high', 'low')
```

**4.2**
```sql
WITH ranked AS (
    SELECT customer_id, region, SUM(spend) AS total_spend,
           ROW_NUMBER() OVER (PARTITION BY region ORDER BY SUM(spend) DESC) AS rn
    FROM orders
    GROUP BY customer_id, region
)
SELECT customer_id, region, total_spend
FROM ranked
WHERE rn <= 3;
```

**4.3**
```bash
# Extract the IP field, sort, count unique, take the top 10
awk '{print $1}' access.log | sort | uniq -c | sort -rn | head -10
```

**4.4** Git history is permanent — anyone cloning the repo downloads the full history, so the 2GB file bloat is paid by everyone forever. Fix: use `git filter-repo` (or `git filter-branch`) to remove the file from history, force-push, and tell collaborators to re-clone. Prevent: add the file to `.gitignore`, never commit large data, and use a data versioning tool (DVC) or object storage for big files.

---

## Part 5: Machine Learning Foundations

**5.1 [Conceptual]** A model gets 99% accuracy on training data but 72% on test data. What is happening, and name two fixes.

**5.2 [Coding]** Train a Random Forest and an XGBoost classifier on the breast cancer dataset, tune each with `GridSearchCV` (or `RandomizedSearchCV`), and report cross-validated ROC-AUC.

**5.3 [Coding]** A churn dataset has only 10% churners. Why is accuracy misleading here? Compute precision, recall, and F1 instead.

**5.4 [Challenge]** Explain the bias-variance tradeoff using a decision tree whose depth you vary from 1 to 50. Plot training vs test error and identify the regions of high bias, optimal, and high variance.

### Solutions

**5.1** Overfitting — the model memorized the training data instead of learning general patterns. Fixes: more data, regularization (L1/L2), simpler model (fewer features, shallower trees), early stopping, or cross-validation for model selection.

**5.2**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

rf = RandomForestClassifier(random_state=42)
grid = {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}
search = GridSearchCV(rf, grid, cv=5, scoring='roc_auc')
search.fit(X, y)
print(f"Best RF ROC-AUC: {search.best_score_:.4f}")

# Same pattern for XGBoost — add early_stopping_rounds and learning_rate to the grid
```

**5.3** With 10% churners, a model that predicts "no churn" for everyone gets 90% accuracy while being useless. Precision, recall, and F1 focus on the minority (positive) class: precision = "of those we flagged as churners, how many were right?", recall = "of actual churners, how many did we catch?". F1 balances both.

**5.4** Vary `max_depth` from 1 to 50. Train error drops quickly and keeps falling (the tree memorizes). Test error initially falls (underfitting / high bias, depth 1–3), then rises as depth grows (overfitting / high variance, depth > ~10). The optimal depth is where test error is minimized — the sweet spot of the bias-variance tradeoff.

---

## Part 6: Big Data & Databases

**6.1 [Conceptual]** List the 4 V's of Big Data and give a real-world example of each from social media.

**6.2 [Conceptual]** What is the difference between a data warehouse, a data lake, and a lakehouse?

**6.3 [Coding]** Using PySpark, load a 10GB Parquet dataset and compute total sales per region, using a broadcast hint for a small lookup table.

**6.4 [Challenge]** Compare batch vs stream processing for a ride-hailing platform. Which use cases need streaming (surge pricing, driver ETA) and which can wait for batch (weekly earnings report)? Name one tool for each.

### Solutions

**6.1** Volume: billions of posts. Velocity: millions of posts/minute in real-time. Variety: text, images, video, likes, shares, metadata. Veracity: fake accounts, spam, and bots introduce uncertainty and noise.

**6.2** Warehouse: cleaned, structured, schema-on-write — great for BI and SQL reporting. Data lake: raw, any format, schema-on-read — cheap storage, but can become a "data swamp" without governance. Lakehouse: combines both — ACID transactions and SQL on raw data in object storage (e.g., Delta Lake, Iceberg).

**6.3**
```python
from pyspark.sql import functions as F

sales = spark.read.parquet('s3://bucket/sales/')
region_lookup = spark.read.parquet('s3://bucket/regions/').hint('broadcast')

result = (sales.join(region_lookup, 'region_id')
               .groupBy('region')
               .agg(F.sum('amount').alias('total_sales')))
result.show()
```

**6.4** Streaming (Apache Kafka + Flink/Spark Streaming): surge pricing, real-time driver dispatch, fraud detection — these need sub-second decisions. Batch (Spark, Hive nightly jobs): weekly earnings reports, monthly aggregates, model retraining — latency-tolerant and cheaper to operate.

---

## Part 7: Data Ethics, Governance & Communication

**7.1 [Conceptual]** What is the difference between anonymization and differential privacy?

**7.2 [Conceptual]** Define k-anonymity and give an example where a k=5 anonymized dataset could still leak information.

**7.3 [Coding]** A fairness audit finds the model accepts 70% of men but 45% of women. Compute the disparate impact ratio and interpret it.

**7.4 [Challenge]** Your dashboard shows overall growth while every department shows decline. Diagnose the paradox, name it, and explain how you would present the data honestly.

### Solutions

**7.1** Anonymization removes/replaces identifiers (names, emails) so individuals can't be directly identified — but it can fail against re-identification attacks. Differential privacy is a formal guarantee: it adds calibrated random noise so the output of a query is almost the same whether or not any single person is in the dataset, bounding how much any individual can be learned about.

**7.2** k-anonymity means every combination of quasi-identifiers (age, zip, gender) appears in at least k rows, so an individual is indistinguishable from k-1 others. Leak example: k=5 table still leaks if one small group of 5 shares a rare combination (e.g., all 5 are the only pancreatic cancer patients in a small town) — the group is still identifiable as a whole, or a privacy leak occurs if the group is homogeneous on a sensitive attribute.

**7.3**
```python
# Disparate impact = minority acceptance rate / majority acceptance rate
ratio = 0.45 / 0.70   # 0.64
# A ratio below 0.80 is generally considered evidence of adverse impact
print(f"Disparate impact ratio: {ratio:.2f}  → violates the 80% rule")
```

**7.4** Simpson's Paradox — the overall result is driven by the size of subgroups, not the trend within them (e.g., women applied more to competitive departments). Present the data stratified by department, explain the paradox plainly, and never let an aggregate number hide the segmented truth.

---

## Part 8: Emerging Topics

**8.1 [Conceptual]** In one sentence each: what is the difference between AI, ML, and deep learning?

**8.2 [Conceptual]** Why does ReLU largely replace Sigmoid in modern deep networks?

**8.3 [Coding]** Tokenize a sentence, remove stopwords, stem, and lemmatize it — and explain why lemmatization is "smarter" than stemming.

**8.4 [Challenge]** A bank wants to deploy a churn model as a REST API. Name the key components of a production MLOps setup (serving, monitoring, retraining) and one tool for each stage.

### Solutions

**8.1** AI is machines doing tasks that need human intelligence; ML is a subset that learns patterns from data without explicit programming; deep learning is ML using multi-layer neural networks that learn features automatically.

**8.2** ReLU's gradient is 1 for all positive inputs, so gradients flow through many layers without shrinking — avoiding the vanishing gradient problem that makes Sigmoid (gradient ≈ 0 for large |x|) stall training in deep networks.

**8.3**
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "The children are running faster than their brothers."
tokens = word_tokenize(text.lower())
tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words('english')]

stemmed = [PorterStemmer().stem(t) for t in tokens]   # 'running' → 'run', crude
lemmatized = [WordNetLemmatizer().lemmatize(t, pos='v') for t in tokens]  # 'children' → 'child'
```
Stemming chops endings heuristically (`children` → `children`; can produce non-words like `run` → `run`, `brothers` → `brother` but also wrong forms). Lemmatization uses a dictionary + part-of-speech to return real dictionary words (`children` → `child`, `running` → `run`). Lemmatization is more accurate but slower and needs POS tagging.

**8.4** Serving: FastAPI + uvicorn (or Triton/TensorFlow Serving for scale). Monitoring: drift detection on inputs and predictions (e.g., Evidently AI, Grafana) + logging to track performance decay. Retraining: scheduled pipelines (Airflow/Prefect) with experiment tracking (MLflow) and A/B or shadow deployment before rollout.

---

## Answer Bank Tips

- **Show your reasoning**: A correct answer with a clear method is worth more than a lucky guess.
- **Sanity-check outputs**: If your RMSE is huge or your plot looks empty, debug the data first — not the model.
- **Reproducibility**: set `random_state`/seed in every exercise so results can be compared.

---

> **Summary**: Practice is the only way to make these concepts stick. Finish each part's notes, then work the corresponding problems here before moving on. If a solution surprises you, re-read that part's notes and re-run the code yourself.
