# Part 3: Exploratory Data Analysis (EDA) & Visualization

> Comprehensive Lecture Notes for BS/MS Data Science

---

## 11. Exploratory Data Analysis (EDA)

### Simple Explanation
EDA is the process of investigating a dataset to discover patterns, spot anomalies, test hypotheses, and check assumptions using summary statistics and visualizations. It's like a doctor doing a check-up before making a diagnosis — you look at all vital signs before deciding what's wrong.

**The EDA Mantra**: "Plot first, ask questions later."

### Simple Example
You get a dataset of 10,000 customer transactions. Before building any fancy model, you:
1. Check summary statistics (mean spend = $47, median = $32 — right-skewed)
2. Plot a histogram of purchase amounts (long tail to the right)
3. Check for missing values (5% missing in "age" column)
4. Plot spend by gender (females spend 22% more on average)
5. Check correlation between age and spend (weak positive, r = 0.18)

**Insight gained**: Don't use mean for customer segmentation (skewed data). Focus on female shoppers aged 25-40.

### Expert Example
Multi-dimensional EDA on a customer churn dataset (100K customers, 200 features):

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from pandas_profiling import ProfileReport

# Auto-EDA with pandas-profiling
profile = ProfileReport(df, title="Churn EDA Report", explorative=True)
profile.to_file("churn_eda_report.html")

# Custom EDA: churn rate by segment
segment_analysis = (df.groupby(['tenure_group', 'contract_type'])
    .agg(
        churn_rate=('churn', 'mean'),
        avg_monthly_spend=('monthly_charges', 'mean'),
        customer_count=('customer_id', 'count')
    )
    .reset_index()
)

# For each segment, compare churners vs non-churners
features_to_examine = ['tenure', 'monthly_charges', 'num_services', 'support_calls']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, feature in zip(axes.flatten(), features_to_examine):
    sns.kdeplot(data=df, x=feature, hue='churn', ax=ax, fill=True)
    ax.set_title(f'{feature} Distribution by Churn Status')

# Statistical test: do churners and non-churners have different mean tenure?
from scipy import stats
churn_tenure = df[df['churn']==1]['tenure']
active_tenure = df[df['churn']==0]['tenure']
t_stat, p_value = stats.ttest_ind(churn_tenure, active_tenure)
print(f"t-test: t={t_stat:.3f}, p={p_value:.2e} (p < 0.05 = significant)")
```

---

### Summary Statistics

| Statistic | Simple Explanation | When to Use |
|-----------|-------------------|-------------|
| **Mean** | Average of all values | Symmetric distributions, no outliers |
| **Median** | Middle value when sorted | Skewed distributions, has outliers |
| **Mode** | Most frequent value | Categorical data |
| **Variance** | How spread out values are | Comparing variability |
| **Std Dev** | √variance, in original units | Interpreting spread |
| **IQR** | Range of middle 50% | Robust measure of spread |

### Simple Example
Income data: [$30K, $32K, $35K, $38K, $42K, $45K, $1M]
- Mean = $175K (misleading — one billionaire distorts it)
- Median = $38K (more representative of typical person)
- Mode = N/A (no repeated value)
- Variance = very large (outlier inflates it)
- IQR = $42K - $32K = $10K (middle 50% earn within $10K range)

**Lesson**: Always report median + IQR for income data, not mean.

### Expert Example
Descriptive statistics grouped by segments with custom aggregations:

```python
# Comprehensive summary by segment
summary = (df.groupby('customer_segment')
    .agg(
        count=('revenue', 'count'),
        mean_revenue=('revenue', 'mean'),
        median_revenue=('revenue', 'median'),
        std_revenue=('revenue', 'std'),
        q25_revenue=('revenue', lambda x: x.quantile(0.25)),
        q75_revenue=('revenue', lambda x: x.quantile(0.75)),
        skewness=('revenue', lambda x: stats.skew(x.dropna())),
        kurtosis=('revenue', lambda x: stats.kurtosis(x.dropna())),
        cv=('revenue', lambda x: x.std() / x.mean()),  # Coefficient of Variation
        missing_pct=('revenue', lambda x: x.isna().mean() * 100)
    )
    .round(2)
)
print(summary)
```

---

### Distribution Analysis

**Simple Explanation**: Understanding how values are spread across a variable — are they concentrated, spread out, symmetric, skewed?

### Simple Example
Exam scores in two classes:
- Class A: scores cluster around 75 (normal distribution)
- Class B: scores split — some very low (20-30), some very high (85-100) (bimodal)

**Insight**: Class B likely has two distinct groups of students. The teacher should investigate what's different.

### Expert Example
Distribution fitting and goodness-of-fit testing:

```python
import scipy.stats as stats

# Fit multiple distributions to the data
distributions = ['norm', 'lognorm', 'expon', 'gamma', 'weibull_min']
results = []

for dist_name in distributions:
    dist = getattr(stats, dist_name)
    params = dist.fit(data)
    ks_stat, p_value = stats.kstest(data, dist_name, args=params)
    results.append({
        'distribution': dist_name,
        'ks_stat': ks_stat,
        'p_value': p_value,
        'params': params
    })

# Best fit has lowest KS statistic
best_dist = min(results, key=lambda x: x['ks_stat'])
print(f"Best fit: {best_dist['distribution']} (KS={best_dist['ks_stat']:.4f}, p={best_dist['p_value']:.4f})")

# Q-Q plot for visual check
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
stats.probplot(data, dist=best_dist_name, plot=ax[0])
ax[0].set_title(f'Q-Q Plot: {best_dist_name}')
ax[1].hist(data, bins=50, density=True, alpha=0.6, label='Data')

# Overlay fitted distribution
x = np.linspace(data.min(), data.max(), 100)
pdf = getattr(stats, best_dist_name).pdf(x, *best_dist['params'])
ax[1].plot(x, pdf, 'r-', linewidth=2, label='Fitted')
ax[1].legend()
plt.show()
```

---

### Correlation Analysis

**Simple Explanation**: Measuring how two variables move together. Correlation ranges from -1 to +1.

| Value | Meaning | Example |
|-------|---------|---------|
| +1 | Perfect positive | Hours studied → Exam score |
| 0 | No relationship | Shoe size → IQ |
| -1 | Perfect negative | Speed → Travel time |

**Important**: Correlation ≠ Causation! Ice cream sales and drowning deaths are correlated (both increase in summer), but eating ice cream doesn't cause drowning.

### Simple Example
Hours studied vs. GPA for 100 students: r = 0.72

Interpretation: Strong positive correlation. More study hours are associated with higher GPA, but we can't say studying *causes* higher GPA (maybe motivated students study more AND do well).

### Expert Example
Partial correlation and network correlation analysis:

```python
import pingouin as pg

# Spearman (non-parametric) correlation matrix
corr_matrix = df.corr(method='spearman')

# Partial correlation — correlation between two variables controlling for others
partial_corr = pg.partial_corr(
    data=df,
    x='ad_spend',
    y='revenue',
    covar=['seasonality', 'competitor_activity', 'brand_awareness']
)
print(f"Partial correlation (controlling for confounders): r={partial_corr['r'].values[0]:.3f}")

# Correlation network graph
import networkx as nx

G = nx.Graph()
threshold = 0.3  # Only show strong correlations
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j],
                      weight=corr_matrix.iloc[i, j])

# Detect communities of correlated features
from networkx.algorithms.community import girvan_newman
communities = next(girvan_newman(G))
print(f"Found {len(list(communities))} feature communities")
```

---

### Identifying Patterns, Trends, and Anomalies

**Simple Example**: A retail store's daily sales:
- Pattern: Sales spike every weekend (weekly seasonality)
- Trend: Overall sales increasing 5% year-over-year
- Anomaly: Sudden 50% drop on March 15 (system outage day)

### Expert Example
Time-series decomposition and anomaly detection using STL (Seasonal-Trend decomposition using LOESS):

```python
from statsmodels.tsa.seasonal import STL
from scipy import stats

# Decompose time series
stl = STL(df['daily_sales'], period=7, robust=True)
result = stl.fit()

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
result.observed.plot(ax=axes[0], title='Observed')
result.trend.plot(ax=axes[1], title='Trend')
result.seasonal.plot(ax=axes[2], title='Seasonal (weekly)')
result.resid.plot(ax=axes[3], title='Residual (anomalies)')
plt.tight_layout()

# Detect anomalies in residuals (outside 3-sigma)
residual = result.resid.dropna()
z_scores = np.abs(stats.zscore(residual))
anomalies = residual[z_scores > 3]

# Mark anomalies on original data
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['daily_sales'], label='Sales', color='blue')
plt.scatter(anomalies.index, df.loc[anomalies.index, 'daily_sales'],
           color='red', s=100, label='Anomaly', zorder=5)
plt.legend()
plt.title(f'Detected {len(anomalies)} Anomalies')
plt.show()

# Change point detection (structural breaks)
import ruptures as rpt
signal = df['daily_sales'].values
algo = rpt.Pelt(model="rbf", min_size=7)
result = algo.fit_predict(signal, pen=10)
rpt.display(signal, result, figsize=(14, 4))
plt.title('Change Point Detection')
plt.show()
```

---

## 12. Descriptive Statistics

### Measures of Central Tendency

| Measure | Simple Explanation | Formula | Pros | Cons |
|---------|-------------------|---------|------|------|
| **Mean** | Arithmetic average | Σx / n | Uses all data | Sensitive to outliers |
| **Median** | Middle value | Sort, pick middle | Robust to outliers | Ignores distribution shape |
| **Mode** | Most common value | Frequency count | Works for categorical | May not exist or be unique |

### Simple Example
Dataset: [2, 3, 3, 4, 5, 5, 5, 100]
- Mean = 127/8 = 15.9 (misleading — 100 pulls it up)
- Median = (4+5)/2 = 4.5 (representative)
- Mode = 5 (most common value)

### Expert Example
Robust estimation with trimmed mean and Winsorized mean:

```python
from scipy import stats

# Compare different central tendency estimators
print(f"Standard mean: {np.mean(data):.2f}")
print(f"Trimmed mean (10%): {stats.trim_mean(data, 0.1):.2f}")  # Remove 10% from each end
print(f"Trimmed mean (20%): {stats.trim_mean(data, 0.2):.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Winsorized mean: {stats.mstats.winsorize(data, limits=[0.1, 0.1]).mean():.2f}")

# For grouped data, weighted mean is more appropriate
weighted_mean = np.average(df['score'], weights=df['weight'])
print(f"Weighted mean: {weighted_mean:.2f}")
```

---

### Measures of Dispersion

| Measure | Simple Explanation | Example |
|---------|-------------------|---------|
| **Range** | Max - Min | Sales range: $100K - $5M = $4.9M |
| **Variance** | Average squared distance from mean | High variance = spread out |
| **Std Dev** | √Variance — in original units | Height: σ = 3 inches (interpretable) |
| **IQR** | Q3 - Q1 (middle 50%) | Income IQR: $35K (most people's range) |

### Simple Example
Two classes, same mean (75/100) but different spread:
- Class A: Std = 5 — all students around 70-80
- Class B: Std = 20 — scores range from 20 to 100

**Insight**: Same average performance, but Class B is much more diverse in ability.

### Expert Example
Robust dispersion metrics for non-normal data:

```python
# Standard deviation (parametric)
std = df['feature'].std()

# MAD — Median Absolute Deviation (robust)
mad = np.median(np.abs(df['feature'] - np.median(df['feature'])))
# Convert to sigma-scale (consistent with std for normal data)
mad_sigma = mad * 1.4826

# Coefficient of Variation (relative spread)
cv = df['feature'].std() / df['feature'].mean() * 100
print(f"CV: {cv:.2f}% — {cv < 15 and 'low variability' or 'high variability'}")

# Range statistics for quality control
p1, p99 = np.percentile(df['feature'], [1, 99])
range_98 = p99 - p1

# Gini coefficient (inequality measure — common in economics)
def gini(x):
    x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

print(f"Gini coefficient: {gini(df['income']):.3f} (0=perfect equality, 1=perfect inequality)")
```

---

### Skewness and Kurtosis

| Metric | Simple Explanation | Value Meaning | Example |
|--------|-------------------|---------------|---------|
| **Skewness** | Symmetry of distribution | > 0 = right-tail (positive skew) | Income distribution |
| | | < 0 = left-tail (negative skew) | Exam scores (ceiling effect) |
| | | = 0 = symmetric | Height distribution |
| **Kurtosis** | "Tailedness" — how heavy the tails are | > 3 = heavy tails, more outliers | Stock market returns |
| | | < 3 = light tails, fewer outliers | Uniform distribution |
| | | = 3 = normal distribution (mesokurtic) | Bell curve |

### Simple Example
- **Positive skew**: Most people earn $30-60K, a few earn millions (right tail)
- **Negative skew**: Most students score 80-100%, a few fail (left tail)
- **High kurtosis**: Financial returns — most days are flat, but sometimes crash (fat tails)

### Expert Example
Quantitative skewness and kurtosis with bootstrap confidence intervals:

```python
def bootstrap_skewness(data, n_iterations=10000):
    """Bootstrap CI for skewness"""
    skew_vals = np.zeros(n_iterations)
    n = len(data)
    for i in range(n_iterations):
        sample = np.random.choice(data, size=n, replace=True)
        skew_vals[i] = stats.skew(sample)
    return {
        'skewness': stats.skew(data),
        'ci_lower': np.percentile(skew_vals, 2.5),
        'ci_upper': np.percentile(skew_vals, 97.5),
        'p_value':  2 * min(
            np.mean(skew_vals <= 0),
            np.mean(skew_vals >= 0)
        )
    }

result = bootstrap_skewness(df['stock_returns'])
print(f"Skewness: {result['skewness']:.3f}")
print(f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
print(f"p-value: {result['p_value']:.4f} — {'significant' if result['p_value'] < 0.05 else 'not significant'}")

# Jarque-Bera test for normality (uses both skewness and kurtosis)
jb_stat, jb_p = stats.jarque_bera(data)
print(f"Jarque-Bera test: JB={jb_stat:.2f}, p={jb_p:.4f}")
```

---

### Five-Number Summary

**Simple Explanation**: A quick summary of a distribution using five key values:

| Value | Meaning |
|-------|---------|
| **Minimum** | Smallest value |
| **Q1** | 25th percentile |
| **Median (Q2)** | 50th percentile |
| **Q3** | 75th percentile |
| **Maximum** | Largest value |

**Simple Example**: House prices in a neighborhood
- Min: $150K
- Q1: $220K
- Median: $310K
- Q3: $450K
- Max: $1.2M

**Interpretation**: 50% of houses are between $220K and $450K. The median ($310K) is more informative than the mean (which a few mansions would inflate).

### Expert Example
Extended summary with Tukey's letter values:

```python
# Standard five-number
q1, q2, q3 = np.percentile(data, [25, 50, 75])
five_num = {
    'min': data.min(),
    'q1': q1,
    'median': q2,
    'q3': q3,
    'max': data.max(),
    'iqr': q3 - q1,
    'range': data.max() - data.min()
}

# Tukey's letter values (midpoints at different depths)
def letter_values(data):
    n = len(data)
    values = {}
    # Depth of median
    depth = (n + 1) / 2
    values['M'] = np.median(data)
    
    while depth > 1:
        # Lower and upper hinges at this depth
        lower = np.percentile(data, (depth - 1) / (n - 1) * 100)
        upper = np.percentile(data, (1 - (depth - 1) / (n - 1)) * 100)
        mid = (lower + upper) / 2
        values[f'H{len(values)}'] = mid
        depth = (depth + 1) / 2
    
    return values

print(letter_values(df['revenue']))
```

---

## 13. Data Visualization

### Univariate Plots

**Simple Explanation**: Visualizing a single variable to understand its distribution.

| Plot Type | What It Shows | Best For |
|-----------|--------------|----------|
| **Histogram** | Distribution shape, bins | Continuous data |
| **Box plot** | Five-number summary, outliers | Comparing distributions |
| **Bar chart** | Category counts | Categorical data |
| **Pie chart** | Proportions (use sparingly) | A few categories (≤5) |
| **Density plot** | Smooth distribution curve | Continuous, larger datasets |

### Simple Example
```python
# Histogram
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(df['age'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Age Distribution (Histogram)')
plt.xlabel('Age')
plt.ylabel('Count')

# Box plot
plt.subplot(1, 3, 2)
plt.boxplot(df['age'], vert=False)
plt.title('Age Distribution (Box Plot)')
plt.xlabel('Age')

# Density plot
plt.subplot(1, 3, 3)
sns.kdeplot(df['age'], fill=True)
plt.title('Age Distribution (Density)')
plt.xlabel('Age')
plt.tight_layout()
plt.show()
```

### Expert Example
Violin plot (combines box plot + density) versus histogram with custom binning:

```python
# Violin plot (better than box plot for multimodal distributions)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.violinplot(data=df, x='segment', y='revenue', ax=axes[0],
               inner='box', palette='Set2')
axes[0].set_title('Revenue Distribution by Segment (Violin)')

# Strip plot (individual data points)
sns.stripplot(data=df, x='segment', y='revenue', ax=axes[1],
              jitter=True, alpha=0.3, palette='Set2')
axes[1].set_title('Revenue by Segment (Strip)')

# ECDF plot (Empirical Cumulative Distribution Function)
from statsmodels.distributions.empirical_distribution import ECDF
for group, grp_df in df.groupby('segment'):
    ecdf = ECDF(grp_df['revenue'])
    axes[2].plot(ecdf.x, ecdf.y, label=group)
axes[2].set_title('Revenue Cumulative Distribution')
axes[2].legend()
plt.tight_layout()
plt.show()
```

---

### Bivariate Plots

**Simple Explanation**: Visualizing the relationship between two variables.

| Plot Type | Variable Types | What It Reveals |
|-----------|---------------|-----------------|
| **Scatter plot** | Continuous × Continuous | Correlation, clusters, outliers |
| **Line chart** | Time × Continuous | Trends over time |
| **Stacked bar** | Categorical × Continuous | Composition across groups |
| **Heatmap** | Continuous × Continuous | Correlation matrix |

### Simple Example
```python
# Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='ad_spend', y='revenue', scatter_kws={'alpha': 0.5})
plt.title('Ad Spend vs. Revenue (r = 0.72)')
plt.show()

# Grouped bar chart
df.groupby('region')['sales'].sum().plot(kind='bar')
plt.title('Total Sales by Region')
plt.ylabel('Sales ($)')
plt.show()
```

### Expert Example
Hexbin plot (for dense scatter plots with overplotting) + joint plot:

```python
# Hexbin plot — handles millions of points
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Hexbin for dense data
hb = axes[0].hexbin(df['feature_a'], df['feature_b'], gridsize=30,
                    cmap='Blues', bins='log')
plt.colorbar(hb, ax=axes[0], label='log10(count)')
axes[0].set_title('Hexbin Plot (Overcome Overplotting)')

# Joint plot with marginals
jp = sns.jointplot(data=df, x='feature_a', y='feature_b',
                   kind='hex', gridsize=30, marginal_kws=dict(bins=50))

# 2D KDE contour plot (identifies clusters)
axes[2] = sns.kdeplot(data=df, x='feature_a', y='feature_b',
                      fill=True, thresh=0.05, cmap='viridis',
                      levels=20)
axes[2].set_title('2D KDE (Density Contours)')

plt.tight_layout()
plt.show()

# Pearson vs Spearman correlation
from scipy.stats import pearsonr, spearmanr
r_pearson, _ = pearsonr(df['feature_a'], df['feature_b'])
r_spearman, _ = spearmanr(df['feature_a'], df['feature_b'])
print(f"Pearson (linear) r = {r_pearson:.3f}")
print(f"Spearman (monotonic) r = {r_spearman:.3f}")
print(f"Difference suggests {'non-linear' if abs(r_pearson - r_spearman) > 0.1 else 'linear'} relationship")
```

---

### Multivariate Plots

**Simple Explanation**: Visualizing relationships between 3+ variables simultaneously.

| Plot Type | Dimensions | What It Shows |
|-----------|-----------|---------------|
| **Pair plot** | All numeric columns pairwise | Quick overview of all relationships |
| **Bubble chart** | X, Y, and size (3D in 2D) | Three numeric variables |
| **Facet grid** | Multiple subplots by category | Conditional relationships |
| **Parallel coordinates** | All features as parallel axes | Multi-dimensional patterns |

### Expert Example
Comprehensive multi-variable visualization:

```python
# Pair plot (limited to 5-6 features for readability)
sns.pairplot(
    df[sample_features], 
    hue='target',
    diag_kind='kde',
    palette='viridis',
    plot_kws={'alpha': 0.5, 's': 10}
)
plt.suptitle('Pairwise Relationships Colored by Target', y=1.02)
plt.show()

# Bubble chart (4 dimensions)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    x=df['gdp_per_capita'],
    y=df['life_expectancy'],
    s=df['population'] / 1e6,   # Size = population
    c=df['continent'].astype('category').cat.codes,  # Color = continent
    alpha=0.6, cmap='Set1'
)
plt.xlabel('GDP per Capita ($)')
plt.ylabel('Life Expectancy (years)')
plt.title('Health & Wealth by Country (Size = Population)')
plt.colorbar(scatter, label='Continent')
plt.show()

# Facet grid — conditional relationships
g = sns.FacetGrid(df, col='region', row='customer_type', margin_titles=True)
g.map(sns.scatterplot, 'ad_spend', 'revenue', alpha=0.5)
g.fig.suptitle('Ad Spend vs Revenue by Region and Customer Type', y=1.02)
plt.tight_layout()
plt.show()
```

---

### Time-series Visualization

**Simple Explanation**: Showing how data changes over time.

### Expert Example
Multi-panel time series with annotations:

```python
import matplotlib.dates as mdates

fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

# Raw series with rolling average
axes[0].plot(df.index, df['daily_sales'], alpha=0.5, label='Daily')
axes[0].plot(df.index, df['daily_sales'].rolling(7).mean(), 
            linewidth=2, color='red', label='7-day MA')
axes[0].plot(df.index, df['daily_sales'].rolling(30).mean(), 
            linewidth=2, color='green', label='30-day MA')
axes[0].set_ylabel('Daily Sales ($)')
axes[0].legend()
axes[0].set_title('Sales with Rolling Averages')

# Year-over-year comparison
df['prev_year_sales'] = df['daily_sales'].shift(365)
axes[1].fill_between(df.index, df['daily_sales'], df['prev_year_sales'],
                     where=df['daily_sales'] > df['prev_year_sales'],
                     color='green', alpha=0.3, label='YoY Growth')
axes[1].fill_between(df.index, df['daily_sales'], df['prev_year_sales'],
                     where=df['daily_sales'] <= df['prev_year_sales'],
                     color='red', alpha=0.3, label='YoY Decline')
axes[1].set_ylabel('Sales ($)')
axes[1].legend()
axes[1].set_title('Year-over-Year Comparison')

# Seasonal sub-series plot
df['month'] = df.index.month
df['year'] = df.index.year
sns.boxplot(data=df, x='month', y='daily_sales', ax=axes[2])
axes[2].set_title('Seasonal Distribution by Month')
axes[2].set_xlabel('Month')
axes[2].set_ylabel('Daily Sales ($)')

plt.tight_layout()
plt.show()
```

---

### Time-series Analysis & Forecasting Basics

**Simple Explanation**: Beyond *visualizing* a series over time, we often want to *predict* its future. Time series are special because observations are not independent — today's value depends on yesterday's. Forecasting means using that history to estimate what comes next (next month's sales, tomorrow's temperature, next week's server load).

**Key concepts**:
- **Trend**: Long-term upward or downward movement.
- **Seasonality**: Regular, repeating pattern (daily, weekly, yearly).
- **Stationarity**: A series is stationary if its mean and variance are constant over time and it has no trend/seasonality. Most forecasting models (like ARIMA) require stationarity. The **Dickey-Fuller test** checks it formally.
- **ACF (Autocorrelation Function)**: Correlates the series with a lagged version of itself — "how related is today to yesterday? to 2 days ago?"
- **PACF (Partial Autocorrelation Function)**: Same idea, but removes the effect of intermediate lags.
- **ARIMA(p, d, q)**: AutoRegressive Integrated Moving Average. `p` = autoregressive lags, `d` = times differenced to achieve stationarity, `q` = moving-average lags.

### Simple Example
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Monthly sales with a clear upward trend + yearly seasonality
dates = pd.date_range('2020-01-01', periods=60, freq='MS')
trend = np.linspace(50, 150, 60)
seasonal = 20 * np.sin(2 * np.pi * np.arange(60) / 12)
noise = np.random.normal(0, 5, 60)
sales = trend + seasonal + noise

series = pd.Series(sales, index=dates)

# 1. Rolling mean (smooths noise, shows the trend)
rolling = series.rolling(12).mean()

# 2. Simple exponential smoothing (weights recent points more)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
fitted = model.fit()
forecast = fitted.forecast(6)   # next 6 months

plt.figure(figsize=(12, 5))
plt.plot(series, label='Actual')
plt.plot(rolling, label='12-month rolling mean', color='red')
plt.plot(forecast, label='Forecast (next 6 months)', color='green', linewidth=2)
plt.legend()
plt.title('Sales with Rolling Mean and Forecast')
plt.show()
```

### Expert Example
ARIMA with stationarity checks, ACF/PACF analysis, and proper backtesting:

```python
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Stationarity check (Augmented Dickey-Fuller)
result = adfuller(series.dropna())
print(f"ADF p-value: {result[1]:.4f}  (p < 0.05 = stationary)")
# If not stationary, difference the series:
series_diff = series.diff().dropna()

# 2. Inspect ACF/PACF to choose p and q
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# plot_acf(series_diff, ax=axes[0])
# plot_pacf(series_diff, ax=axes[1])

# 3. Walk-forward backtest (train on all data up to t, predict t+1)
def backtest_arima(series, order, steps=12):
    preds = []
    history = list(series[:-steps])
    for i in range(steps):
        model = ARIMA(history, order=order).fit()
        preds.append(model.forecast()[0])
        history.append(series[len(history)])
    return np.array(preds)

# 4. Compare simple ARIMA vs SARIMA with weekly seasonality
test_size = 12
y_true = series[-test_size:]

arima_preds = backtest_arima(series, order=(2, 1, 2))
sarima = SARIMAX(series[:-test_size],
                 order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
sarima_preds = sarima.forecast(test_size)

print(f"ARIMA  MAE: {mean_absolute_error(y_true, arima_preds):.2f}")
print(f"SARIMA MAE: {mean_absolute_error(y_true, sarima_preds):.2f}")

# 5. Production tip: never retrain on every step for large data —
#    use a fixed model with periodic retraining (e.g., weekly) instead,
#    and always log your MAE/RMSE per horizon (1-step, 7-step, 30-step)
```

**Key takeaway**: Check stationarity before modeling, use ACF/PACF to pick orders, and always evaluate forecasts with a walk-forward backtest — never test on the future you "peeked" at.

---

## 14. Visualization Tools & Libraries

| Tool | Simple Explanation | Best For |
|------|-------------------|----------|
| **Matplotlib** | Low-level control, everything customizable | Publication-quality figures |
| **Seaborn** | Built on Matplotlib, statistical plots | EDA, beautiful defaults |
| **Plotly** | Interactive, zoomable, hoverable | Web dashboards, exploration |
| **Tableau** | Drag-and-drop, no coding | Business dashboards |
| **ggplot2 (R)** | Grammar of graphics philosophy | R users, layered plots |

### Simple Example
Same plot in 3 libraries:

**Matplotlib**:
```python
plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['income'], alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Age vs Income')
plt.show()
```

**Seaborn** (simpler code + better defaults):
```python
sns.scatterplot(data=df, x='age', y='income', hue='gender', style='education')
```

**Plotly** (interactive):
```python
import plotly.express as px
fig = px.scatter(df, x='age', y='income', color='gender', 
                 hover_data=['name', 'education'],
                 title='Age vs Income (Interactive)')
fig.show()
```

### Expert Example
Customizing Matplotlib with publication-quality settings:

```python
# Set global style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2,
    'figure.figsize': (10, 6)
})

# Create a complex, publication-ready plot
fig, ax = plt.subplots()
sns.kdeplot(data=df[df['churn']==0], x='tenure', fill=True, 
            alpha=0.3, label='Active', ax=ax)
sns.kdeplot(data=df[df['churn']==1], x='tenure', fill=True, 
            alpha=0.3, label='Churned', ax=ax)
ax.axvline(df[df['churn']==0]['tenure'].median(), color='blue', 
           linestyle='--', alpha=0.7, label=f"Active median: {df[df['churn']==0]['tenure'].median():.1f}")
ax.axvline(df[df['churn']==1]['tenure'].median(), color='orange', 
           linestyle='--', alpha=0.7, label=f"Churned median: {df[df['churn']==1]['tenure'].median():.1f}")
ax.set_xlabel('Tenure (months)')
ax.set_ylabel('Density')
ax.set_title('Customer Tenure Distribution by Churn Status')
ax.legend(frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig('churn_tenure_dist.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 15. Summary Tables

### Simple Explanation
Tables that summarize large datasets into digestible formats. While plots give visual intuition, summary tables provide precise numbers.

| Table Type | Simple Explanation | Example |
|-----------|-------------------|---------|
| **Frequency Table** | Count how many times each value appears | Number of students per grade |
| **Contingency Table** | Cross-tabulation of two categorical variables | Gender vs. Product Preference |
| **Pivot Table** | Multi-dimensional summary with aggregation | Average sales by region AND quarter |

### Expert Example

```python
# Frequency table with proportions
freq = df['category'].value_counts()
freq_pct = df['category'].value_counts(normalize=True) * 100
freq_table = pd.DataFrame({
    'count': freq,
    'percentage': freq_pct,
    'cumulative': freq_pct.cumsum()
})
freq_table.index.name = 'Category'
print(freq_table.round(2))

# Contingency table with expected values and chi-square
contingency = pd.crosstab(df['gender'], df['product_category'], margins=True, margins_name='Total')
print(contingency)

# Chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(
    pd.crosstab(df['gender'], df['product_category']).values
)
print(f"Chi-square: {chi2:.2f}, p-value: {p:.4f}")

# Pivot table with multiple aggregation
pivot = pd.pivot_table(
    df,
    values='revenue',
    index='region',
    columns='customer_segment',
    aggfunc=['mean', 'count', 'std'],
    fill_value=0,
    margins=True,
    margins_name='Total'
)
print(pivot.round(2))

# Advanced: cross-tab with proportions by row
prop_table = pd.crosstab(
    df['education'], df['churn'], 
    normalize='index'  # Row percentages
) * 100
prop_table.columns = ['Retained %', 'Churned %']
print(prop_table.round(2))
```

---

## 16. Statistical Foundations

### Population vs. Sample

| Term | Simple Explanation | Example |
|------|-------------------|---------|
| **Population** | The entire group you want to study | All 8 billion people on Earth |
| **Sample** | A subset you actually measure | 1,000 people surveyed |

### Simple Example
You want to know the average height of students at a university (population = 20,000 students). You measure 200 randomly selected students (sample). The sample mean is an estimate of the population mean.

### Expert Example

```python
# Central Limit Theorem demonstration
population = np.random.exponential(scale=50, size=100000)  # Non-normal population
sample_means = []

for _ in range(1000):
    sample = np.random.choice(population, size=100)
    sample_means.append(np.mean(sample))

# Sample means should be approximately normal (CLT)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(population, bins=50, density=True, alpha=0.7)
plt.title('Population Distribution (Exponential)')

plt.subplot(1, 2, 2)
plt.hist(sample_means, bins=50, density=True, alpha=0.7)
sns.kdeplot(sample_means, color='red', linewidth=2)
plt.title(f'Sampling Distribution of Mean (n=100)\nMean={np.mean(sample_means):.1f}, SE={np.std(sample_means):.1f}')
plt.tight_layout()
plt.show()
```

---

### Parameter vs. Statistic

| Term | Refers To | Example |
|------|-----------|---------|
| **Parameter** | Population value (usually unknown) | True average height of all humans = μ |
| **Statistic** | Sample value (we calculate this) | Average height of our 200 sampled people = x̄ |

**Simple Example**: We want the true average income of all Pakistanis (parameter μ). We survey 5,000 people and get average = $5,200 (statistic x̄). The statistic estimates the parameter, but they're rarely exactly equal.

---

### Probability Distributions

| Distribution | Simple Explanation | Shape | Example |
|-------------|-------------------|-------|---------|
| **Normal** | Bell-shaped, symmetric, most common natural phenomenon | Bell | Height, IQ, blood pressure |
| **Binomial** | Number of successes in fixed trials | Varies | 7 heads in 10 coin flips |
| **Poisson** | Count of rare events in fixed time/space | Right-skewed | Number of emails per hour |
| **Uniform** | All outcomes equally likely | Flat | Rolling a fair die |

### Expert Example
Fitting distributions and calculating probabilities:

```python
# Normal distribution
mean_height, std_height = 170, 10
# Probability a person is between 160-180 cm
p = stats.norm.cdf(180, mean_height, std_height) - stats.norm.cdf(160, mean_height, std_height)
print(f"P(160 < height < 180) = {p:.3f}")

# Binomial — probability of exactly 8/10 customers satisfied (p_satisfied=0.8)
p = stats.binom.pmf(k=8, n=10, p=0.8)
print(f"P(exactly 8 of 10 satisfied) = {p:.3f}")

# Poisson — probability of 5 or more customers arriving in next hour (λ=3/hr)
p = 1 - stats.poisson.cdf(4, mu=3)
print(f"P(≥5 arrivals next hour | λ=3) = {p:.3f}")

# QQ plot to check normality assumption
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
stats.probplot(df['feature'], dist='norm', plot=axes[0])
axes[0].set_title('Normal Q-Q Plot')

# Shapiro-Wilk test for normality
w_stat, p_value = stats.shapiro(df['feature'].sample(5000))  # Max 5000 samples
print(f"Shapiro-Wilk: W={w_stat:.4f}, p={p_value:.4f} — {'Normal' if p_value > 0.05 else 'Not normal'}")
```

---

### Central Limit Theorem (CLT)

**Simple Explanation**: The sample mean will be approximately normally distributed, regardless of the population distribution, as long as the sample size is large enough (usually n ≥ 30).

**Simple Example**: The population of individual incomes is heavily skewed (most people earn little, few earn a lot). But if you repeatedly take samples of 100 people and plot the average income of each sample, those averages will form a bell curve.

---

### Law of Large Numbers (LLN)

**Simple Explanation**: As sample size increases, the sample mean gets closer to the population mean.

**Simple Example**: 
- 1 coin flip: 100% heads (far from true 50%)
- 10 flips: 70% heads (closer)
- 100 flips: 52% heads (even closer)
- 1,000 flips: ~50.3% heads (very close to true probability)

---

### Confidence Intervals

**Simple Explanation**: A range of values that likely contains the true population parameter, with a stated level of confidence (usually 95%).

**Simple Example**: "Based on our survey, 62% of voters support Candidate A (95% CI: 59% to 65%)." This means: if we repeated the survey 100 times, 95 of the confidence intervals would contain the true population value.

### Expert Example

```python
# Calculate 95% CI for the mean
from scipy import stats

sample = df['revenue'].sample(1000)
mean = sample.mean()
se = sample.std() / np.sqrt(len(sample))
ci = stats.t.interval(0.95, df=len(sample)-1, loc=mean, scale=se)
print(f"95% CI for mean revenue: [{ci[0]:.2f}, {ci[1]:.2f}]")

# Bootstrap CI (non-parametric, no normality assumption)
np.random.seed(42)
bootstrap_means = [
    np.random.choice(sample, size=len(sample), replace=True).mean()
    for _ in range(10000)
]
bootstrap_ci = np.percentile(bootstrap_means, [2.5, 97.5])
print(f"Bootstrap 95% CI: [{bootstrap_ci[0]:.2f}, {bootstrap_ci[1]:.2f}]")

# CI for proportions
from statsmodels.stats.proportion import proportion_confint
n_success = (df['converted'] == 1).sum()
n_total = len(df)
ci_low, ci_upp = proportion_confint(
    n_success, n_total, 
    alpha=0.05, method='wilson'
)
print(f"Conversion rate: {n_success/n_total*100:.1f}% (95% CI: [{ci_low*100:.1f}%, {ci_upp*100:.1f}%])")
```

---

### Hypothesis Testing

**Simple Explanation**: A formal procedure to determine if an observed effect is statistically significant or just random chance.

| Term | Simple Explanation | Analogy |
|------|-------------------|---------|
| **Null Hypothesis (H₀)** | The default assumption — no effect, no difference | "Innocent until proven guilty" |
| **Alternative (H₁)** | What you want to prove — there IS an effect | "Guilty" |
| **p-value** | Probability of observing your results if H₀ were true | "How unlikely is this evidence if they're innocent?" |
| **Type I Error** | Rejecting H₀ when it's actually true (false positive) | Convicting an innocent person |
| **Type II Error** | Failing to reject H₀ when it's actually false (false negative) | Releasing a guilty person |

### Simple Example
Testing if a new drug lowers blood pressure:
- H₀: New drug = Placebo (no difference)
- H₁: New drug ≠ Placebo (different)
- p = 0.03 → only 3% chance of seeing this difference if the drug had no effect
- Since p < 0.05, we reject H₀ and conclude the drug works

### Expert Example

```python
# A/B Test Analysis
control = df[df['group'] == 'control']['conversion']
treatment = df[df['group'] == 'treatment']['conversion']

# 1. Two-sample t-test (for continuous metrics)
t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
print(f"Welch t-test: t={t_stat:.4f}, p={p_value:.4f}")

# 2. Effect size (Cohen's d)
pooled_std = np.sqrt((control.std()**2 + treatment.std()**2) / 2)
cohens_d = (treatment.mean() - control.mean()) / pooled_std
print(f"Cohen's d = {cohens_d:.3f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")

# 3. Chi-square test (for categorical outcomes)
from scipy.stats import chi2_contingency
ct = pd.crosstab(df['group'], df['converted'])
chi2, p_chi, dof, expected = chi2_contingency(ct)
print(f"Chi-square test: χ²={chi2:.2f}, p={p_chi:.4f}")

# 4. ANOVA (3+ groups)
from scipy.stats import f_oneway
group_a = df[df['variant'] == 'A']['revenue']
group_b = df[df['variant'] == 'B']['revenue']
group_c = df[df['variant'] == 'C']['revenue']
f_stat, p_anova = f_oneway(group_a, group_b, group_c)
print(f"ANOVA: F={f_stat:.2f}, p={p_anova:.4f}")

# 5. Post-hoc analysis (Tukey HSD)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(df['revenue'], df['variant'], alpha=0.05)
print(tukey)

# 6. Power analysis
from statsmodels.stats.power import TTestIndPower
power_analysis = TTestIndPower()
required_n = power_analysis.solve_power(
    effect_size=0.2,    # Small effect
    power=0.80,          # 80% power
    alpha=0.05,
    ratio=1.0,
    alternative='two-sided'
)
print(f"Required sample size per group (80% power, small effect): {required_n:.0f}")
```

---

> **Summary**: Part 3 covers the art and science of exploring data before modeling. EDA is where 80% of insights are found — don't skip it. Descriptive statistics summarizes data numerically, visualization makes patterns visible, and inferential statistics helps you determine if what you see is real or just noise. The golden rule of EDA: "The more you know your data, the better your models will be."

---

**← [Part 2 - Data Processing & Wrangling](Part%202%20-%20Data%20Processing%20%26%20Wrangling.md)** · **Next: [Part 4 - Programming & Tools](Part%204%20-%20Programming%20%26%20Tools.md)** · [Back to README](README.md)
