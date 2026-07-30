# Part 2: Data Processing & Wrangling

> Comprehensive Lecture Notes for BS Data Science (3rd Semester)

---

## 6. Data Wrangling (Data Munging)

### Simple Explanation
Data wrangling is the process of cleaning, transforming, and preparing raw data into a format suitable for analysis. Raw data is almost never ready to use — it's messy, incomplete, and inconsistent. Wrangling is like preparing ingredients before cooking: you wash, chop, and measure before you start cooking.

### Simple Example
You download a CSV of customer data. It has:
- Missing phone numbers
- Names in different formats ("JOHN DOE", "Doe, John", "John D.")
- Ages like "twenty-five" instead of "25"
- Some rows with extra blank columns

Wrangling fixes all of this before you analyze it.

### Expert Example
A data pipeline in PySpark processes 10 TB of raw clickstream data daily:

```python
from pyspark.sql import functions as F

df = spark.read.parquet("s3://raw-data/clickstream/2024/10/*")

df_clean = (df
    .dropDuplicates(["session_id", "event_timestamp"])
    .filter(F.col("user_id").isNotNull())
    .withColumn("event_time", F.to_timestamp("event_timestamp", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("session_duration", 
        F.col("session_end") - F.col("session_start"))
    .withColumn("page_category", 
        F.when(F.col("url").contains("/products"), "product")
         .when(F.col("url").contains("/checkout"), "checkout")
         .otherwise("other"))
    .fillna({"device_type": "unknown", "referrer": "direct"})
)
```

---

### Filtering, Sorting, Merging, Reshaping

**Filtering**: Selecting specific rows based on conditions.

*Simple Example*: From a student dataset, keep only students with GPA > 3.5.

*Expert Example*: In PySpark, filter using predicate pushdown to reduce I/O:
```python
df_filtered = spark.read.parquet("sales/").filter(
    (F.col("amount") > 1000) & 
    (F.col("region").isin(["APAC", "EMEA"])) &
    (F.year("order_date") >= 2024)
)
# Predicate pushdown means Parquet only reads matching row groups
```

---

**Sorting**: Arranging data by column values.

*Simple Example*: Sort employees by salary (highest first).

*Expert Example*: Secondary sort in Spark to avoid shuffle overhead:
```python
from pyspark.sql import Window
window_spec = Window.partitionBy("department").orderBy(F.desc("salary"))
df.withColumn("rank", F.rank().over(window_spec))
```

---

**Merging**: Combining datasets by common columns (joins).

*Simple Example*: Join a customer table (ID, name) with an orders table (ID, product) on Customer ID.

*Expert Example*: A complex merge in Spark with broadcast join optimization:
```python
# Small lookup table broadcasted to all executors
lookup_df = spark.read.parquet("product_categories").cache()
lookup_df.count()  # Force caching, 200KB only

# Broadcast join — no shuffle for the small table
result = large_transactions_df.join(
    F.broadcast(lookup_df),
    on="product_id",
    how="left"
)
```

---

**Reshaping**: Changing the structure (pivot — rows to columns; melt — columns to rows).

*Simple Example*: Pivot — Convert month-wise sales data (Jan, Feb, Mar as columns) from long to wide format.

*Expert Example*: Unpivoting (melting) in Pandas for ML features:
```python
# Before: wide format — one column per sensor
# sensor_1 | sensor_2 | sensor_3 | timestamp
#    72    |    68    |    75    | 10:00:01

# After: long format — sensor readings as rows
df_melted = df.melt(
    id_vars=["timestamp"],
    value_vars=["sensor_1", "sensor_2", "sensor_3"],
    var_name="sensor",
    value_name="reading"
)
# timestamp |  sensor   | reading
# 10:00:01  | sensor_1  |   72
# 10:00:01  | sensor_2  |   68
```

---

### Handling Missing Values

| Method | Simple Explanation | When to Use |
|--------|-------------------|-------------|
| **Drop** | Delete rows/columns with missing data | When < 5% of data is missing randomly |
| **Impute** | Fill missing values with estimates | When data is missing but pattern exists |
| **Flag** | Add a column indicating it was missing | When the "missingness" itself is informative |

### Simple Example
A survey has 1,000 responses but 50 people didn't answer "income":
- **Drop**: Remove those 50 rows (only 5%)
- **Impute (mean)**: Fill with the average income of responders
- **Impute (median)**: Fill with the median (better if income is skewed)
- **Flag**: Add a column `income_missing = 1` for those 50 people
- **Impute (regression)**: Predict income from education, age, and job title

### Expert Example
Multiple imputation using MICE (Multiple Imputation by Chained Equations) in R:

```r
library(mice)
library(miceRanger)

# Pattern: 15% missing randomly across 50 variables
# Using MICE with predictive mean matching (PMM)
imputed_data <- mice(
  raw_data,
  m = 5,              # 5 imputed datasets
  method = "pmm",     # Predictive Mean Matching
  maxit = 50,         # 50 iterations
  seed = 42,
  predictorMatrix = quickpred(raw_data, mincor = 0.3)
)

# Fit model on each imputed dataset
fit <- with(imputed_data, lm(salary ~ education + experience + age))

# Pool results using Rubin's rules
pooled <- pool(fit)
summary(pooled)
```

For time-series data, use interpolation methods:
```python
# Forward fill + backward fill for stock price data
df["price"].fillna(method="ffill", inplace=True)   
df["price"].fillna(method="bfill", inplace=True)   

# Or use time-weighted interpolation
df["price"].interpolate(method="time", inplace=True)
```

---

### Handling Outliers

**Simple Explanation**: Outliers are data points that are significantly different from others. They could be errors (a person's age as 500) or genuine rare events (a billionaire's income).

**Simple Example**: In a class of 30 students with ages 18-22, one student is 45 (a returning adult). Stats: mean age = 19.5 with the outlier, 18.9 without.

### Detection Methods

| Method | How It Works | Example |
|--------|-------------|---------|
| **Z-score** | Points beyond 3 standard deviations from mean | Z > 3 or Z < -3 |
| **IQR** | Points below Q1 - 1.5*IQR or above Q3 + 1.5*IQR | More robust for skewed data |
| **Isolation Forest** | ML algorithm that isolates anomalies | Works well for high-dimensional data |
| **DBSCAN** | Points not assigned to any cluster | Good for spatial data |

### Expert Example
Robust outlier detection using the Hampel identifier and winsorization for financial data:
```python
import numpy as np
from scipy import stats

def hampel_filter(data, window=5, n_sigma=3.0):
    """Hampel identifier using rolling median and MAD"""
    n = len(data)
    filtered = data.copy()
    
    for i in range(window, n - window):
        segment = data[i - window : i + window]
        med = np.median(segment)
        mad = np.median(np.abs(segment - med))  # Median Absolute Deviation
        if mad == 0:
            continue
        z_score = 0.6745 * (data[i] - med) / mad  # Normalize MAD to sigma
        if np.abs(z_score) > n_sigma:
            filtered[i] = med  # Replace with median
    
    return filtered

# For extreme outliers, use winsorization (cap, not remove)
from scipy.stats.mstats import winsorize
capped_data = winsorize(
    df["daily_returns"], 
    limits=[0.01, 0.01],  # Cap bottom and top 1%
    inplace=False
)
```

---

## 7. Data Cleaning

### Deduplication

**Simple Explanation**: Finding and removing duplicate records.

**Simple Example**: A mailing list has "john@email.com" entered three times. Keep one, remove the rest.

**Expert Example**: Fuzzy deduplication using record linkage at scale using Spark:

```python
# Exact dedup
df_clean = df.dropDuplicates(["email", "phone"])

# Fuzzy dedup using blocking and Jaccard similarity
from spark_string_similarity import jaccard_similarity

# Step 1: Blocking — group by zip code to reduce comparisons
# Step 2: Compare within blocks
df_blocked = df.alias("a").join(df.alias("b"), 
    on=F.col("a.zip") == F.col("b.zip"),
    how="inner")

df_blocked = df_blocked.filter(
    jaccard_similarity(F.col("a.name"), F.col("b.name")) > 0.85
)
```

---

### Standardization and Normalization

**Simple Explanation**: Making data consistent in format and scale.

**Simple Example**:
- Standardize phone numbers: "(555) 123-4567" → "5551234567"
- Standardize dates: "10/15/24", "Oct 15 2024", "2024-10-15" → all to "2024-10-15"
- Standardize text: "New York", "new york", "NEW YORK" → all to "New York"

**Expert Example**: Automated data standardization pipeline using Great Expectations for data quality validation:

```python
import great_expectations as ge

df_ge = ge.dataset.PandasDataset(df)

# Define expectations
df_ge.expect_column_values_to_match_regex(
    "phone", r"^\d{10}$"  # Must be 10 digits
)
df_ge.expect_column_values_to_be_in_set(
    "gender", ["M", "F", "Non-binary", "Prefer not to say"]
)
df_ge.expect_column_values_to_match_strftime_format(
    "date", "%Y-%m-%d"
)

# Run validation
results = df_ge.validate()
print(f"Expectation suite passed: {results['success']}")

# Apply standardization
df["phone"] = df["phone"].str.replace(r"\D", "", regex=True)
df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True).dt.strftime("%Y-%m-%d")
df["country"] = df["country"].str.title().replace({
    "U.S.A": "United States",
    "USA": "United States",
    "Uk": "United Kingdom"
})
```

---

### Handling Inconsistent Formats

**Simple Example**: A dataset has salary in different currencies and formats:
- "$50,000/year"  
- "EUR 4,200/month"
- "35 lakh per annum"

**Expert Example**: Using declarative data transformation with a custom parser:

```python
import re
from currency_converter import CurrencyConverter
cc = CurrencyConverter()

def parse_salary(salary_str):
    """Parse salary strings into standardized USD/year"""
    if pd.isna(salary_str):
        return None
    
    s = salary_str.strip().lower()
    
    # Extract amount
    amount = re.search(r'[\d,]+(?:\.\d+)?', s.replace(',', ''))
    if not amount:
        return None
    amount = float(amount.group())
    
    # Determine currency and period
    if '$' in s or 'usd' in s:
        currency = 'USD'
    elif 'eur' in s or '€' in s:
        currency = 'EUR'; amount *= 12  # monthly to annual
    elif 'lakh' in s or 'inr' in s or '₹' in s:
        currency = 'INR'; amount *= 100000  # lakh to rupees
    else:
        return None
    
    # Convert to annual
    if 'month' in s or '/mo' in s:
        amount *= 12
    elif 'hour' in s:
        amount *= 40 * 52
    elif 'day' in s:
        amount *= 260
    
    # Convert to USD
    return cc.convert(amount, currency, 'USD')

df["salary_usd"] = df["salary_raw"].apply(parse_salary)
```

---

### Dealing with Noisy Data

**Simple Explanation**: Noisy data contains random errors, irrelevant information, or meaningless variations that obscure the true signal.

**Simple Example**: Temperature readings from a sensor that occasionally spikes to 200°C due to electrical interference, when the real temperature is 25°C.

**Expert Example**: Signal processing approach using rolling median + wavelet denoising:

```python
import pywt
import numpy as np
from scipy.signal import savgol_filter

# Method 1: Savitzky-Golay filter (preserves trends while smoothing)
df["price_smoothed"] = savgol_filter(
    df["price"], 
    window_length=11,   # Must be odd
    polyorder=3         # Polynomial degree
)

# Method 2: Wavelet denoising (for non-stationary signals)
def wavelet_denoise(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level)
    # Estimate noise standard deviation using robust MAD estimator
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    # Apply soft thresholding
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [
        pywt.threshold(c, threshold, mode='soft') if i > 0 else c 
        for i, c in enumerate(coeffs)
    ]
    return pywt.waverec(coeffs_thresh, wavelet, mode='per')

# Method 3: 3-sigma clipping for sensor data
df["temp_clean"] = df["temp"].mask(
    np.abs(stats.zscore(df["temp"])) > 3
).interpolate(method='linear')
```

---

## 8. Data Transformation

### Scaling (Min-Max, Standardization/Z-score)

**Simple Explanation**: Putting data on a common scale so features with large values don't dominate features with small values.

| Method | Formula | Range | When to Use |
|--------|---------|-------|-------------|
| **Min-Max** | (x - min) / (max - min) | [0, 1] | Neural networks (bounded activations) |
| **Z-score** | (x - mean) / std | ~[-3, 3] | Linear models, PCA, SVM |

### Simple Example
Feature A: Salary ($30K - $200K). Feature B: Age (18 - 65).
Without scaling, Salary dominates any distance calculation (A varies by 170K vs B varies by 47).
After scaling, both features contribute equally to distance.

### Expert Example
Robust scaling for financial data with outliers:
```python
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# RobustScaler uses median and IQR — resistant to outliers
robust_scaler = RobustScaler(quantile_range=(25, 75))
X_scaled = robust_scaler.fit_transform(X)

# For time-series, use rolling standardization
def rolling_zscore(series, window=30):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return (series - rolling_mean) / rolling_std

# For pipelines: StandardScaler after Winsorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

pipeline = Pipeline([
    ('winsorize', FunctionTransformer(
        lambda x: np.clip(x, 
            np.percentile(x, 1, axis=0), 
            np.percentile(x, 99, axis=0)
        )
    )),
    ('scale', RobustScaler()),
])
```

---

### Encoding Categorical Variables

**Simple Explanation**: Converting text categories into numbers that ML algorithms can process.

| Method | Example | Result | When to Use |
|--------|---------|--------|-------------|
| **Label Encoding** | Color: Red→0, Blue→1, Green→2 | Single column, ordinal | Tree-based models (ordinal) |
| **One-Hot Encoding** | Color: Red→[1,0,0], Blue→[0,1,0], Green→[0,0,1] | K new columns (binary) | Linear models, no category order |
| **Target Encoding** | Replace category with mean of target | Single column, continuous | High-cardinality features |

### Simple Example
City column: Islamabad, Lahore, Karachi

- **Label Encoding**: Islamabad=0, Lahore=1, Karachi=2
- **One-Hot**: Islamabad → [1,0,0], Lahore → [0,1,0], Karachi → [0,0,1]

### Expert Example
Advanced encoding for high-cardinality categorical variables (500+ categories):

```python
from category_encoders import TargetEncoder, CatBoostEncoder, WOEEncoder
import category_encoders as ce

# Target encoding with smoothing to prevent overfitting
encoder = TargetEncoder(
    cols=['zip_code', 'product_category'],
    smoothing=10,         # Higher = more regularization
    min_samples_leaf=20   # Minimum samples per category
)
X_encoded = encoder.fit_transform(X, y)

# For tree models: ordinal encoding based on category frequency
encoder = ce.OrdinalEncoder(
    mapping=[{
        'col': 'city',
        'mapping': {k: i for i, (k, v) in enumerate(
            df['city'].value_counts().items()
        )}
    }]
)

# Cyclical encoding for time features (preserves circular nature)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

---

### Log Transformation and Box-Cox

**Simple Explanation**: Making skewed data more normally distributed. Many real-world datasets (income, prices, population) have long right tails. Log transformation compresses the tail.

### Simple Example
Income data is right-skewed: most people earn $30-60K but a few earn millions.
- Before log: histogram shows a long right tail
- After log: histogram looks closer to bell-shaped
- Interpretation: "A 1% increase in X is associated with a β% change in Y" (elasticity)

### Expert Example
Power transformations with Yeo-Johnson (handles negative values too):

```python
from sklearn.preprocessing import PowerTransformer
from scipy import stats

# Box-Cox requires positive values
# lambda ≈ 0 → log transform ; lambda = 0.5 → square root
fitted_lambda, _ = stats.boxcox(df["positive_revenue"])
print(f"Optimal lambda: {fitted_lambda}")

# Yeo-Johnson handles negative values
pt = PowerTransformer(method='yeo-johnson', standardize=True)
df_transformed = pt.fit_transform(df[features])

# Apply to pipeline
from sklearn.compose import TransformedTargetRegressor
model = TransformedTargetRegressor(
    regressor=RandomForestRegressor(),
    transformer=PowerTransformer(method='box-cox')
)
```

---

### Binning / Discretization

**Simple Explanation**: Converting continuous values into discrete buckets or bins.

**Simple Example**: Converting ages into groups: Child (0-12), Teen (13-19), Adult (20-59), Senior (60+).

**Expert Example**: Optimal binning using decision tree splits:
```python
from sklearn.tree import DecisionTreeRegressor

def optimal_binning(X, y, max_bins=5):
    """Use decision tree to find optimal bin boundaries"""
    tree = DecisionTreeRegressor(
        max_leaf_nodes=max_bins,
        min_samples_leaf=50,
        random_state=42
    )
    tree.fit(X.reshape(-1, 1), y)
    thresholds = np.sort(tree.tree_.threshold[
        tree.tree_.feature != -2  # Internal nodes only
    ])
    return np.unique(thresholds)

# Adapt bins to monotonic trends for credit risk scoring
import optbinning
from optbinning import OptimalBinning

optb = OptimalBinning(
    name='age',
    dtype='numerical',
    solver='cp',        # Custom algorithm
    min_bin_size=0.05,  # Minimum 5% per bin
    max_n_bins=5,
    monotonic_trend='auto'  # Ensure monotonic WoE
)
optb.fit(X['age'], y)
binning_table = optb.binning_table
print(binning_table.build())
```

---

### Feature Extraction

**Simple Explanation**: Creating new informative features from existing raw data.

**Simple Example**: From a timestamp column "2024-10-15 14:30:00", extract:
- Hour: 14 (afternoon)
- Day of week: Tuesday
- Is weekend: No
- Month: October
- Season: Fall

**Expert Example**: Time-series feature extraction with tsfresh (automated):

```python
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

# Extract 700+ features automatically from time series
extraction_settings = ComprehensiveFCParameters()

# Per sensor: mean, variance, skew, kurtosis, FFT coefficients,
# autocorrelation, entropy, approximate entropy, etc.
features = extract_features(
    df_timeseries,
    column_id="sensor_id",
    column_sort="timestamp",
    default_fc_parameters=extraction_settings,
    impute_function=lambda x: x.fillna(0),
    n_jobs=8
)

# Text feature extraction with NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 3),
    stop_words='english',
    min_df=5
)
tfidf_features = vectorizer.fit_transform(corpus)

# Sentiment features
df['sentiment'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
```

---

## 9. Data Integration

### Combining Data from Multiple Sources

**Simple Explanation**: Merging data from different tables, files, or databases into one unified dataset.

**Simple Example**: You have three Excel files — Customers, Orders, Products. You merge them on CustomerID and ProductID to get a complete sales view.

### Expert Example**: A polyglot data integration pipeline across multiple systems:

```python
import pandas as pd
from sqlalchemy import create_engine
import requests
import boto3

# Source 1: PostgreSQL (customers)
pg_engine = create_engine('postgresql://user:pass@host:5432/db')
customers = pd.read_sql("SELECT * FROM customers WHERE created_at > '2024-01-01'", pg_engine)

# Source 2: MongoDB (transactions)
from pymongo import MongoClient
mongo_client = MongoClient('mongodb://host:27017')
transactions = pd.DataFrame(
    list(mongo_client.db.transactions.find(
        {"status": "completed"},
        {"_id": 0, "user_id": 1, "amount": 1, "date": 1}
    ))
)

# Source 3: S3 (Parquet files — product catalog)
s3 = boto3.client('s3')
products = pd.read_parquet('s3://data-lake/products/catalog.parquet')

# Source 4: REST API (fraud scores)
response = requests.get(
    "https://api.fraudservice.com/v2/scores",
    params={"batch_size": 1000}
)
fraud_scores = pd.DataFrame(response.json())

# Integration with quality checks
merged = (customers
    .merge(transactions, left_on="id", right_on="user_id", how="left", validate="one_to_many")
    .merge(products, on="product_id", how="left", validate="many_to_one")
    .merge(fraud_scores, on="user_id", how="left", validate="one_to_one")
)
assert len(merged) == len(transactions), "Lost records during merge!"
```

---

### Joins (Inner, Outer, Left, Right)

| Join Type | Simple Explanation | Venn Diagram |
|-----------|-------------------|--------------|
| **Inner** | Keep only rows that match in both tables | Intersection |
| **Left** | Keep all rows from left, fill right with NaN if no match | Left circle + intersection |
| **Right** | Keep all rows from right, fill left with NaN if no match | Right circle + intersection |
| **Outer** | Keep all rows from both tables | Union of both circles |

### Simple Example
Table A (Students): [1: Alice, 2: Bob, 3: Charlie]
Table B (Grades): [1: A, 2: B, 4: A]

- Inner: [1: Alice-A, 2: Bob-B]
- Left: [1: Alice-A, 2: Bob-B, 3: Charlie-NaN]
- Right: [1: Alice-A, 2: Bob-B, 4: NaN-A]
- Outer: [1: Alice-A, 2: Bob-B, 3: Charlie-NaN, 4: NaN-A]

### Expert Example
Complex multi-table join optimization in Spark:

```python
from pyspark.sql import functions as F

# Left anti-join: find customers who have never ordered
inactive_customers = customers.join(
    orders,
    on="customer_id",
    how="left_anti"
)

# Semi-join: find products that have been ordered at least once
ordered_products = products.join(
    orders.select("product_id").distinct(),
    on="product_id",
    how="left_semi"
)

# Multi-way join with broadcast hint for dimension tables
result = (fact_sales
    .join(F.broadcast(dim_customers), "customer_id", "inner")
    .join(F.broadcast(dim_products), "product_id", "inner")
    .join(F.broadcast(dim_store), "store_id", "inner")
    .join(dim_date, "date_id", "inner")
)
```

---

### Entity Resolution / Record Linkage

**Simple Explanation**: Identifying and merging records that refer to the same real-world entity but have slightly different representations.

**Simple Example**:
- Record 1: "John A. Smith", "555-1234", "123 Main St"
- Record 2: "John Smith", "5551234", "123 Main Street"
These are the same person. Entity resolution identifies this match.

### Expert Example
Probabilistic record linkage using Fellegi-Sunter model with blocking:

```python
import recordlinkage
from recordlinkage.index import Block

# Step 1: Blocking to reduce comparison pairs
indexer = recordlinkage.Index()
indexer.add(Block('zip_code'))   # Only compare within same zip
indexer.add(Block('last_name_first_letter'))  # Only same first letter of last name
pairs = indexer.index(df_a, df_b)

# Step 2: Compare fields
compare = recordlinkage.Compare()
compare.string('name', 'name', method='jarowinkler', threshold=0.85, label='name')
compare.string('address', 'address', method='levenshtein', threshold=0.80, label='address')
compare.exact('phone', 'phone', label='phone')
compare.exact('email', 'email', label='email')
compare.string('city', 'city', method='exact', label='city')

features = compare.compute(pairs, df_a, df_b)

# Step 3: Score and classify
ecm = recordlinkage.ECMClassifier()
matches = ecm.fit_predict(features)

print(f"Found {len(matches)} matched pairs out of {len(pairs)} compared pairs")
```

---

## 10. Data Reduction

### Sampling

| Method | Simple Explanation | Example |
|--------|-------------------|---------|
| **Random** | Every item has equal chance | Pick 100 random students |
| **Stratified** | Maintain group proportions | 50% male / 50% female in sample, matching population |
| **Systematic** | Every k-th item | Every 10th customer who walks in |

### Simple Example
A dataset has 1M records and you want 10,000 (1%):
- Random: `df.sample(n=10000)` — simple, but might miss rare groups
- Stratified: Ensure minority class (3% fraud) is represented at 3% in sample

### Expert Example
Reservoir sampling for streaming data (unknown total size):

```python
import random

def reservoir_sample(stream, k):
    """Reservoir sampling — maintain k random samples from infinite stream"""
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir

# Stratified sampling in Spark
from pyspark.sql import functions as F

fractions = df.groupBy("region").agg(F.count("*").alias("count")).collect()
fractions_dict = {row.region: min(1.0, 10000.0 / row.count) for row in fractions}

stratified_sample = df.sampleBy("region", fractions=fractions_dict, seed=42)
```

---

### Dimensionality Reduction

**Simple Explanation**: Reducing the number of features (columns) while preserving as much information as possible.

| Method | Simple Explanation | Best For |
|--------|-------------------|----------|
| **PCA** | Creates new features (principal components) that capture maximum variance | Linear relationships, dense data |
| **t-SNE** | Preserves local neighborhood structure | Visualization (2D/3D) |
| **UMAP** | Preserves both local and global structure | Visualization + faster than t-SNE |

### Simple Example
You have 100 features about customers (income, age, spending, etc.). PCA reduces it to 10 "super-features" that capture 90% of the information. Each super-feature is a weighted combination of the original features.

### Expert Example
PCA with automated component selection and interpretation:

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Preprocessing
X_scaled = StandardScaler().fit_transform(X)

# PCA with variance threshold
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

# Analysis
print(f"Original features: {X.shape[1]}")
print(f"Reduced features: {X_pca.shape[1]}")
print(f"Explained variance per component: {pca.explained_variance_ratio_}")
print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")

# Interpret components (top loadings)
components_df = pd.DataFrame(
    pca.components_.T,
    index=feature_names,
    columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
)
print(components_df['PC1'].abs().sort_values(ascending=False).head(10))

# For visualization: t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# For large-scale: UMAP (much faster than t-SNE)
import umap
reducer = umap.UMAP(n_components=2, min_dist=0.1, n_neighbors=15)
X_umap = reducer.fit_transform(X_scaled)
```

---

### Feature Selection

| Method | Simple Explanation | Example |
|--------|-------------------|---------|
| **Filter** | Rank features by statistical tests | Keep features with correlation > 0.3 with target |
| **Wrapper** | Try different subsets, pick best-performing | Forward selection: add features one by one |
| **Embedded** | Feature selection built into model training | Lasso regression automatically removes useless features |

### Simple Example
You have 50 features and want the 10 most important ones:
- Filter: Calculate correlation of each feature with target, keep top 10
- Wrapper: Try all combinations (too slow) or forward/backward stepwise
- Embedded: Train Lasso, keep features with non-zero coefficients

### Expert Example
Recursive Feature Elimination with Cross-Validation (RFECV):

```python
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# RFECV automatically selects optimal number of features
estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
selector = RFECV(
    estimator=estimator,
    step=1,                 # Remove one feature at a time
    cv=5,                   # 5-fold cross-validation
    scoring='f1_weighted',
    min_features_to_select=5,
    n_jobs=-1
)
selector.fit(X_train, y_train)

# Optimal number of features
print(f"Optimal features: {selector.n_features_}")
print(f"Selected features: {X.columns[selector.support_].tolist()}")

# Plot feature selection results
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1),
         selector.cv_results_['mean_test_score'])
plt.xlabel('Number of features')
plt.ylabel('Cross-validation score')
plt.axvline(selector.n_features_, color='red', linestyle='--')
plt.show()

# Embedded method: L1 regularization
from sklearn.linear_model import LogisticRegression
lasso = LogisticRegression(penalty='l1', solver='saga', C=0.1, random_state=42)
lasso.fit(X_train, y_train)
selected_mask = np.abs(lasso.coef_[0]) > 0
print(f"Lasso selected {selected_mask.sum()} features out of {X.shape[1]}")
```

---

> **Summary**: Part 2 covers the tools and techniques to transform raw, messy data into analysis-ready form. Data scientists spend 60-80% of their time on wrangling — mastering these skills is essential. Every dataset will have missing values, inconsistencies, and noise. Knowing how to handle each situation systematically separates professionals from beginners.
