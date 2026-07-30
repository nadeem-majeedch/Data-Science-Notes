# Part 7: Data Ethics, Governance & Communication

> Comprehensive Lecture Notes for BS Data Science (4th Semester)

---

## 29. Data Ethics

### Simple Explanation
Data ethics is about doing the right thing with data. Just because you *can* collect, analyze, or use data in a certain way doesn't mean you *should*. It involves privacy, fairness, transparency, and accountability.

### Simple Example
A company collects location data from users' phones. Ethically:
- **Good**: Aggregate data to optimize traffic lights (benefits everyone, anonymous)
- **Bad**: Sell individual location data to advertisers without consent
- **Worse**: Use data to identify people visiting medical clinics and deny insurance

---

### Privacy and Consent

**Simple Explanation**: People should know what data is being collected about them, how it will be used, and have the choice to say no.

### Key Principles

| Principle | Simple Explanation | Example |
|-----------|-------------------|---------|
| **Informed Consent** | Tell people exactly what you'll do with their data | "We collect your email to send order updates" not "We may use your data for marketing" |
| **Purpose Limitation** | Only use data for the purpose it was collected | Don't use healthcare data for targeted advertising |
| **Data Minimization** | Collect only what you need | Don't ask for phone number if email is sufficient |
| **Right to be Forgotten** | Users can request data deletion | GDPR Article 17 |

### Simple Example
A fitness app asks for access to your contacts.
- **Unethical**: The app uploads all contacts to its servers without asking.
- **Ethical**: The app clearly says "We need contact access so you can share workout achievements with friends. We never store your contacts on our servers."

### Expert Example
Privacy-preserving data sharing framework:

```python
import hashlib
import secrets

class PrivacyPreservingPipeline:
    """Implement privacy by design in data pipelines"""
    
    def __init__(self, data):
        self.data = data
    
    def pseudonymize(self, identifier_cols):
        """Replace identifiers with irreversible hashes"""
        salt = secrets.token_hex(16)
        for col in identifier_cols:
            self.data[f'{col}_hash'] = self.data[col].apply(
                lambda x: hashlib.pbkdf2_hmac(
                    'sha256', 
                    str(x).encode(), 
                    salt.encode(), 
                    100000
                ).hex()
            )
        self.data = self.data.drop(columns=identifier_cols)
        return self
    
    def anonymize(self, k=5):
        """k-anonymity: ensure each combination of quasi-identifiers appears at least k times"""
        quasi_identifiers = ['age_group', 'zip_code', 'gender']
        
        # Generalize age to 10-year buckets
        self.data['age_group'] = (self.data['age'] // 10 * 10).astype(str) + '-' + \
                                  ((self.data['age'] // 10 * 10) + 9).astype(str)
        
        # Group by quasi-identifiers and suppress rare combos
        group_counts = self.data.groupby(quasi_identifiers).size()
        rare_groups = group_counts[group_counts < k].index
        
        for group in rare_groups:
            mask = pd.Series(True, index=self.data.index)
            for i, col in enumerate(quasi_identifiers):
                mask &= (self.data[col] == group[i])
            self.data.loc[mask, quasi_identifiers] = 'OTHER'
        
        return self
    
    def add_differential_privacy(self, column, epsilon=1.0):
        """Add Laplacian noise for differential privacy"""
        sensitivity = self.data[column].max() - self.data[column].min()
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, size=len(self.data))
        self.data[f'{column}_dp'] = self.data[column] + noise
        return self
```

---

### Anonymization and De-identification

**Simple Explanation**: Removing or altering personal information so individuals can't be identified from the data.

| Technique | Simple Explanation | Example |
|-----------|-------------------|---------|
| **Remove direct IDs** | Delete names, SSNs, emails | Remove "Name: John Smith" column |
| **Generalization** | Make data less precise | Age 32 → Age 30-40 |
| **Suppression** | Remove specific values | Remove zip codes for rare populations |
| **k-Anonymity** | Each combination appears ≥ k times | At least 5 people share same age+zip+gender |

### Simple Example
**Before**:
| Name | Age | ZIP | Diagnosis |
|------|-----|-----|-----------|
| Alice Smith | 32 | 10001 | Diabetes |
| Bob Jones | 45 | 10002 | Flu |

**After (anonymized)**:
| Age Range | ZIP Prefix | Diagnosis |
|-----------|-----------|-----------|
| 30-40 | 1000 | Diabetes |
| 40-50 | 1000 | Flu |

**Problem**: Even after removing names, if you know Bob is 45 and lives in 10002, you can identify his diagnosis (re-identification attack).

**Solution**: k-anonymity (k=2) ensures at least 2 people match each combination.

---

### Bias and Fairness in Data and Models

**Simple Explanation**: ML models can inherit and amplify human biases present in training data. If historical hiring data shows men were hired more, the model may learn to prefer male candidates.

### Types of Bias

| Bias Type | Simple Explanation | Example |
|-----------|-------------------|---------|
| **Historical Bias** | Existing societal bias reflected in data | Underrepresentation of women in tech resumes |
| **Sampling Bias** | Data not representative of population | Survey only on smartphones → excludes poor |
| **Label Bias** | Subjective labels contain bias | Two reviewers rate same essay differently |
| **Algorithmic Bias** | Model amplifies existing disparities | Risk assessment tool over-predicts recidivism for minorities |

### Simple Example
A hiring algorithm trained on 10 years of company data:
- **Biased data**: 80% of past hires were men (because of historical imbalance)
- **Biased model**: Learns to prefer male candidates
- **Result**: Female applicants with identical resumes get lower scores

### Expert Example
Fairness evaluation and mitigation:

```python
from sklearn.metrics import confusion_matrix
import pandas as pd

def evaluate_fairness(y_true, y_pred, sensitive_attr):
    """Compute fairness metrics for a sensitive attribute"""
    results = {}
    
    for group in y_true[sensitive_attr].unique():
        mask = sensitive_attr == group
        tn, fp, fn, tp = confusion_matrix(
            y_true[mask], y_pred[mask]
        ).ravel()
        
        results[group] = {
            'count': mask.sum(),
            'positive_rate': y_pred[mask].mean(),  # Demographic parity
            'true_positive_rate': tp / (tp + fn),  # Equal opportunity
            'false_positive_rate': fp / (fp + tn),  # Predictive equality
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0
        }
    
    df = pd.DataFrame(results).T
    
    # Disparate impact (80% rule): ratio of positive rates
    groups = list(df.index)
    if len(groups) == 2:
        pr_ratio = df.loc[groups[0], 'positive_rate'] / df.loc[groups[1], 'positive_rate']
        df['disparate_impact'] = min(pr_ratio, 1/pr_ratio)
        print(f"Disparate impact (80% rule): {min(pr_ratio, 1/pr_ratio):.2f}")
        print(f"  {'PASSES' if min(pr_ratio, 1/pr_ratio) >= 0.8 else 'FAILS'} fairness check")
    
    return df

# Fairness mitigation: reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

# Convert to AIF360 format
dataset = BinaryLabelDataset(
    df=dataset_df, 
    label_names=['prediction'],
    protected_attribute_names=['gender']
)

# Apply reweighing
rw = Reweighing(unprivileged_groups=[{'gender': 0}],
                privileged_groups=[{'gender': 1}])
dataset_transformed = rw.fit_transform(dataset)
```

---

### Transparency and Explainability

**Simple Explanation**: Models should be explainable — people affected by decisions have the right to understand why a decision was made.

### Simple Example
Loan denied. The applicant asks "Why?":
- **Black box**: "Our AI model determined you are high-risk." (unhelpful, possibly illegal)
- **Explainable**: "Your debt-to-income ratio is 55% (threshold: 50%), and you have 2 late payments in the last 12 months. These factors contributed to the decision."

### Expert Example
Model interpretability using SHAP and LIME:

```python
import shap
from lime import lime_tabular

# SHAP (SHapley Additive exPlanations) — game theory approach
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Force plot for individual prediction
shap.force_plot(
    explainer.expected_value, 
    shap_values[0], 
    X_test.iloc[0], 
    feature_names=feature_names,
    matplotlib=True
)
# Shows: baseline prediction, which features pushed it higher/lower, by how much

# LIME (Local Interpretable Model-agnostic Explanations)
explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['No Churn', 'Churn'],
    mode='classification'
)

# Explain a single prediction
exp = explainer.explain_instance(
    X_test.iloc[42].values,
    model.predict_proba,
    num_features=5
)
exp.show_in_notebook()
# Output: "This customer is predicted to churn because:
#   - Contract type = month-to-month (+0.23)
#   - Tenure = 3 months (+0.18)
#   - Support calls = 4 (+0.12)
#   - Monthly charges = $95 (+0.08)"
```

---

## 30. Data Governance

### Simple Explanation
Data governance is about managing data as a valuable organizational asset. It defines who can take what actions, with what data, under what circumstances, and using what methods.

### Simple Example
A bank's data governance policies:
- Only loan officers can view customer credit scores
- All access to personal data is logged and audited
- Customer data must be deleted 5 years after account closure
- All dashboards must be approved before sharing

---

### Data Quality

**Simple Explanation**: The degree to which data is accurate, complete, consistent, and fit for its intended use.

### Dimensions of Data Quality

| Dimension | Simple Explanation | How to Check |
|-----------|-------------------|--------------|
| **Accuracy** | Data reflects reality | Compare against source of truth |
| **Completeness** | No missing values | % of null values per column |
| **Consistency** | Same data across systems | Customer name matches in CRM and billing |
| **Timeliness** | Data is current | Last updated timestamp |
| **Uniqueness** | No duplicates | Count of duplicate records |
| **Validity** | Data conforms to rules | Age > 0 and < 120 |

### Expert Example
Automated data quality monitoring with Great Expectations:

```python
import great_expectations as ge
from great_expectations.dataset import PandasDataset

# Create expectation suite
def validate_data_quality(df):
    ds = PandasDataset(df)
    
    expectations = [
        ds.expect_column_values_to_not_be_null('customer_id'),
        ds.expect_column_values_to_be_unique('customer_id'),
        ds.expect_column_values_to_be_between('age', 0, 120),
        ds.expect_column_values_to_be_in_set('gender', ['M', 'F', 'Non-binary']),
        ds.expect_column_values_to_match_regex('email', r'^[^@]+@[^@]+\.[^@]+$'),
        ds.expect_column_values_to_be_between('salary', 0, 10000000),
        ds.expect_column_pair_values_to_be_equal('start_date', 'end_date', 
            mostly=lambda a, b: a < b, result_format='COMPLETE'),
        ds.expect_column_mean_to_be_between('tenure', 0, 50),
        ds.expect_table_row_count_to_be_between(1000, 10000000)
    ]
    
    results = ds.validate()
    
    # Generate quality report
    total_tests = len(results['results'])
    passed = sum(1 for r in results['results'] if r['success'])
    
    print(f"Data Quality Report:")
    print(f"  Tests Passed: {passed}/{total_tests} ({passed/total_tests*100:.1f}%)")
    
    for r in results['results']:
        if not r['success']:
            print(f"  FAILED: {r['expectation_config']['expectation_type']}")
            print(f"    Column: {r['expectation_config'].get('kwargs', {}).get('column', 'N/A')}")
    
    return results
```

---

### Data Lineage

**Simple Explanation**: Tracking data from its origin through all transformations to its final destination. Like a family tree for data showing where it came from and how it changed.

### Simple Example
```
Raw Logs (Server) → ETL (Python) → Staging (PostgreSQL) → Aggregation (Spark) → Dashboard (Tableau)
```

If a number in the dashboard looks wrong, data lineage tells you:
1. Which raw source it came from
2. Which transformations were applied
3. Which code ran when

### Expert Example
Data lineage tracking with SQL comments and dbt:

```sql
-- dbt model: customer_metrics.sql
-- {{ doc("customer_metrics") }}
-- Source: stg_customers, stg_orders
-- Transformations:
--   1. Filter active customers only
--   2. Join with orders to compute lifetime value
--   3. Segment based on RFM scores

WITH customer_base AS (
    SELECT * FROM {{ ref('stg_customers') }}
    WHERE is_active = TRUE
),

customer_orders AS (
    SELECT 
        customer_id,
        COUNT(order_id) AS order_count,
        SUM(amount) AS lifetime_value,
        MAX(order_date) AS last_order_date,
        MIN(order_date) AS first_order_date
    FROM {{ ref('stg_orders') }}
    GROUP BY customer_id
)

SELECT 
    cb.customer_id,
    cb.segment,
    co.order_count,
    co.lifetime_value,
    DATEDIFF('day', co.last_order_date, CURRENT_DATE) AS days_since_last_order,
    CURRENT_TIMESTAMP AS computed_at
FROM customer_base cb
LEFT JOIN customer_orders co ON cb.customer_id = co.customer_id
```

---

### Metadata Management

**Simple Explanation**: Data about data — descriptions of what data exists, what it means, where it came from, and how it should be used.

### Simple Example
A data catalog entry:
```
Dataset: customer_orders
Description: Daily customer orders aggregated by customer
Owner: Marketing Analytics Team
Update Frequency: Daily at 2:00 AM
Source: PostgreSQL → Order Service
Sensitive: Contains PII (email, phone)
Retention: 3 years
Size: 500 GB, 50M rows
```

---

### Data Security and Access Control

**Simple Explanation**: Ensuring only authorized people can access data, and only for approved purposes.

| Principle | Simple Explanation | Implementation |
|-----------|-------------------|---------------|
| **Authentication** | Verify who you are | Passwords, SSO, MFA |
| **Authorization** | What you're allowed to do | Role-based access (RBAC) |
| **Encryption** | Scramble data so only authorized can read | AES-256 for data at rest, TLS for data in transit |
| **Audit Logging** | Record who accessed what, when | CloudTrail, audit logs |
| **Data Masking** | Show partial data to non-privileged users | Agent shows last 4 digits: ****-****-****-1234 |

### Expert Example
Row-level security with attribute-based access control (ABAC):

```sql
-- PostgreSQL row-level security
-- Data scientists can only see their region's data
CREATE POLICY data_scientist_policy ON customer_data
    FOR ALL
    USING (region = current_setting('user.region'));

-- Dynamic data masking
CREATE OR REPLACE FUNCTION mask_email(email TEXT)
RETURNS TEXT
LANGUAGE SQL
AS $$
    SELECT CONCAT(LEFT(email, 2), '****', RIGHT(email, POSITION('@' IN email) - 1 > 2))
$$;

-- Access only through views with masking
CREATE VIEW customer_safe_view AS
SELECT 
    customer_id,
    mask_email(email) AS masked_email,
    age_group,
    region,
    lifetime_value
FROM customer_data;
```

---

## 31. Data Communication & Storytelling

### Simple Explanation
Data science is useless if nobody understands or acts on the findings. Data storytelling is the art of communicating insights in a way that drives decision-making. It combines three elements: **Data** (the numbers), **Story** (a compelling narrative), and **Visuals** (clear charts).

### The Data Storytelling Framework

```
     ┌─────────────────┐
     │     Context      │  "Why should I care?"
     ├─────────────────┤
     │   Conflict/Risk  │  "What's the problem/opportunity?"
     ├─────────────────┤
     │  Data Insight    │  "What does the data say?"
     ├─────────────────┤
     │  Recommendation  │  "What should we do?"
     └─────────────────┘
```

### Simple Example
Bad presentation: "Here's a table of churn rates by segment."
Good presentation: "Our premium customers are churning at 15% — that's 3x higher than last year. That's $2M in annual revenue at risk. The data shows these customers leave within 3 months of their first support call. I recommend we implement a proactive check-in call within 48 hours of any support interaction."

---

### Building Effective Dashboards

**Simple Explanation**: A dashboard is a visual display of the most important information needed to achieve one or more objectives, consolidated on a single screen.

### Dashboard Design Principles

| Principle | Simple Explanation | Bad Example | Good Example |
|-----------|-------------------|-------------|--------------|
| **Know your audience** | Design for who will use it | Technical jargon for executives | "Revenue: $12.4M" not "RMSE: 0.034" |
| **One screen** | No scrolling needed | 20 charts on 3 tabs | 5 key metrics + 3 charts |
| **Right chart type** | Use appropriate visualization | Pie chart with 15 slices | Bar chart showing top 5 |
| **Context matters** | Show comparison | "Sales: $1.2M" | "Sales: $1.2M (+15% vs last month)" |
| **Data-ink ratio** | Remove unnecessary decoration | 3D effects, excessive gridlines | Clean, minimal design |

### Expert Example
Dashboard with actionable metrics and alerts:

```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Modern dashboard layout
app = dash.Dash(__name__)

app.layout = html.Div([
    # KPI cards at top
    html.Div([
        html.Div([
            html.H3("Revenue YTD"),
            html.H2("$12.4M", style={'color': 'green'}),
            html.P("+15% vs target")
        ], className='kpi-card'),
        html.Div([
            html.H3("Churn Rate"),
            html.H2("3.2%", style={'color': 'red'}),
            html.P("Warning: Above 3% threshold")
        ], className='kpi-card'),
    ], style={'display': 'flex'}),
    
    # Main chart
    dcc.Graph(id='revenue-trend'),
    
    # Filters
    html.Div([
        dcc.Dropdown(id='region-filter', options=[
            {'label': 'All Regions', 'value': 'all'},
            {'label': 'North', 'value': 'north'},
            {'label': 'South', 'value': 'south'},
        ])
    ])
])

# Callbacks for interactivity
@app.callback(
    Output('revenue-trend', 'figure'),
    Input('region-filter', 'value')
)
def update_chart(region):
    df = load_data(region)
    fig = px.line(df, x='date', y='revenue', title=f'Revenue Trend - {region}')
    fig.add_hline(y=df['target'].mean(), line_dash="dash", line_color="red")
    return fig
```

---

### Audience-Aware Communication

**Simple Explanation**: Tailor your message based on who's listening. Executives want bottom-line insights. Technical teams want methodology. Business teams want actionable recommendations.

| Audience | What They Care About | How to Present |
|----------|---------------------|----------------|
| **CEO / Executives** | ROI, revenue impact, risk | 1-page summary, top-line metrics, recommendations |
| **Product Managers** | User behavior, feature performance | Funnel charts, cohort analysis, segmented insights |
| **Engineers** | Data pipeline, model architecture | System diagrams, performance metrics, code examples |
| **Business Analysts** | Methodology, assumptions | Statistical details, validation results, caveats |
| **External Clients** | Value proposition | Case studies, before/after comparisons, clear visuals |

### Expert Example

```markdown
# For Executives (1-page memo)

## Executive Summary: Customer Churn Analysis

**The Problem**: We're losing $2.4M/year to churn. Our churn rate (3.2%) is 50% above industry benchmark.

**Key Findings**:
1. Customers with 3+ support calls in 30 days churn at 8x the average
2. Month-to-month contracts churn at 5x annual contracts
3. We lose 70% of churned customers within the first 6 months

**Recommendations**:
1. Launch "VIP care" for customers with 2+ support calls → estimated $800K annual savings
2. Introduce 10% discount for annual commitment → estimated $500K annual savings
3. Early intervention campaign in months 3-5 → estimated $600K annual savings

**Total Projected Impact**: $1.9M annual savings, Investment needed: $200K, ROI: 9.5x
```

---

### Common Pitfalls in Data Presentation

| Pitfall | Why It's Bad | Fix |
|---------|-------------|-----|
| **Cherry-picking** | Only showing data that supports your conclusion | Present all relevant data, including counter-evidence |
| **Wrong chart type** | Misleads the audience | Use bar charts for comparisons, lines for trends |
| **Truncated y-axis** | Exaggerates differences | Start y-axis at 0 (or clearly label truncation) |
| **Overcomplication** | Too much information, no clear message | One clear insight per chart. "What is the ONE thing?" |
| **Correlation = Causation** | Implies relationship is causal | Use careful language: "associated with" not "causes" |
| **Survivorship bias** | Only looking at successes | Include failures in analysis |
| **Simpson's paradox** | Trend reverses when data is grouped | Always check segmented data |
| **No context** | Numbers without meaning | "500" vs "500 (+22% YoY)" |

### Expert Example

```python
# Simpson's Paradox demonstration
import pandas as pd
import matplotlib.pyplot as plt

# Data where each department hires more women, but overall company hires more men
data = pd.DataFrame({
    'department': ['A', 'B', 'C', 'A', 'B', 'C'],
    'gender': ['Male', 'Male', 'Male', 'Female', 'Female', 'Female'],
    'applied': [100, 200, 50, 80, 150, 40],
    'hired': [50, 80, 10, 44, 66, 9]
})

data['acceptance_rate'] = data['hired'] / data['applied']

# Department-wise: women have higher or equal acceptance rate
print(data.pivot_table(index='department', columns='gender', 
                       values='acceptance_rate'))

# But overall: men seem favored
overall = data.groupby('gender').agg({'hired': 'sum', 'applied': 'sum'})
overall['rate'] = overall['hired'] / overall['applied']
print(overall)
# Department A: Men 50%, Women 55% → Women higher
# Overall: Men 47%, Women 44%   → Men higher!
# Why? Women applied more to competitive departments (C: 22.5% vs 20%)
```

---

## 32. Reproducibility & Best Practices

### Simple Explanation
Your analysis should be reproducible — someone else (or you in 6 months) should be able to run your code and get the same results. This is the scientific foundation of data science.

### Simple Example
**Bad**: A report generated by clicking buttons in Excel with manual steps like "remove the outliers that look wrong" and "adjust the numbers until they make sense."

**Good**: A Jupyter notebook that:
1. Reads raw data
2. Runs every cleaning step programmatically
3. Documents each decision
4. Sets a random seed
5. Outputs the same results every time

---

### Literate Programming

**Simple Explanation**: Mix code, documentation, and results in a single document. The code is executable, the documentation explains it, and the output is shown inline.

| Tool | Environment | Best For |
|------|-------------|----------|
| **Jupyter Notebook** | Python | Exploration, tutorials, reports |
| **R Markdown** | R | Reproducible research, PDF/HTML reports |
| **Quarto** | Python, R, Julia | Multi-language, flexible output |
| **Observable** | JavaScript | Interactive data apps |

### Expert Example
Reproducible Jupyter notebook setup:

```python
# Cell 1: Setup and configuration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
from datetime import datetime

# Set reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Analysis started at {datetime.now()}")

# Load configuration (not hardcoded)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"Configuration loaded: {config['description']}")

# Cell 2: Load and validate data (never modify raw data)
INPUT_PATH = config['paths']['raw_data']
OUTPUT_PATH = config['paths']['processed_data']

raw_df = pd.read_csv(INPUT_PATH)
logger.info(f"Loaded raw data: {raw_df.shape}")

# Create a copy for processing
df = raw_df.copy()
```

---

### Documenting Code and Analysis

**Simple Explanation**: Write comments and documentation explaining WHY you made decisions, not just WHAT the code does.

### Simple Example
```python
# BAD COMMENT
# Multiply by 100
result = df['value'] * 100

# GOOD COMMENT
# Convert decimal to percentage for stakeholder readability
# Stakeholders confirmed they prefer "85%" over "0.85"
result = df['value'] * 100
```

### Expert Example
Documentation with pytest and docstrings:

```python
def calculate_rfm_scores(df, weights=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) scores for customer segmentation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'customer_id', 'order_date', 'order_id', 'amount'
    weights : dict, optional
        Weights for R, F, M scores. Default: {'recency': 1, 'frequency': 1, 'monetary': 1}
    
    Returns
    -------
    pd.DataFrame
        Customer-level RFM scores with segment assignment
    
    Notes
    -----
    - Recency is days since last purchase (lower = better)
    - Frequency is count of purchases (higher = better)
    - Monetary is total spend (higher = better)
    - Scores are 1-5 based on quintile ranking
    
    Examples
    --------
    >>> scores = calculate_rfm_scores(transactions)
    >>> scores.head()
    
    See Also
    --------
    calculate_customer_lifetime_value : For CLV-based segmentation
    """
    if weights is None:
        weights = {'recency': 1, 'frequency': 1, 'monetary': 1}
    
    required_cols = ['customer_id', 'order_date', 'order_id', 'amount']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    reference_date = df['order_date'].max()
    
    rfm = df.groupby('customer_id').agg(
        recency=('order_date', lambda x: (reference_date - x.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('amount', 'sum')
    ).reset_index()
    
    # Score each dimension (1-5, higher is better)
    for col, label in [('recency', 'R'), ('frequency', 'F'), ('monetary', 'M')]:
        if col == 'recency':  
            # Lower recency = better (reverse rank)
            rfm[f'{label}_score'] = pd.qcut(rfm[col], 5, labels=[5,4,3,2,1])
        else:
            rfm[f'{label}_score'] = pd.qcut(rfm[col], 5, labels=[1,2,3,4,5])
    
    # Composite score
    rfm['rfm_score'] = (
        weights['recency'] * rfm['R_score'].astype(int) +
        weights['frequency'] * rfm['F_score'].astype(int) +
        weights['monetary'] * rfm['M_score'].astype(int)
    )
    
    return rfm
```

---

### Random Seeds for Reproducibility

**Simple Explanation**: Random operations (train/test splits, weight initialization, sampling) give different results each time. Setting a "seed" makes them deterministic — same seed = same result.

### Expert Example

```python
import random
import numpy as np
import tensorflow as tf
import os

def set_all_seeds(seed=42):
    """Set seeds across all libraries for full reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For TensorFlow / Keras
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    
    # For some GPU operations
    tf.config.experimental.enable_op_determinism()
    
    # For sklearn
    from sklearn.utils import check_random_state
    check_random_state(seed)

set_all_seeds(42)

# Now this will produce identical results every time
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### Writing Clear Reports

**Simple Explanation**: A clear report answers three questions: (1) What did you find? (2) Why should I care? (3) What should I do about it?

### Report Structure

| Section | Purpose | Length |
|---------|---------|--------|
| **Executive Summary** | The bottom line — for busy executives | 3-5 bullet points |
| **Problem Statement** | What question are you answering? | 1 paragraph |
| **Methodology** | How did you do it? (keep brief) | 2-3 paragraphs |
| **Key Findings** | What did you discover? | Charts + 2-3 sentences each |
| **Limitations** | What should readers be careful about? | 1 paragraph |
| **Recommendations** | What should they do? | Actionable items |
| **Appendix** | Technical details, code, data sources | As needed |

### Expert Example

```markdown
# Churn Analysis Report: Q3 2024

## Executive Summary
- **Churn rate increased to 3.2% (from 2.1% in Q2)** — $480K revenue at risk
- **Primary driver**: Month-to-month customers with 3+ support calls
- **Quick win**: Proactive outreach after 2nd support call could save $200K/quarter
- **Recommendation**: Launch pilot program in North region within 2 weeks

## Problem Statement
Customer churn has increased 52% quarter-over-quarter, threatening our annual retention targets. This analysis identifies the root causes and recommends targeted interventions.

## Methodology
- Analyzed 500K customer records from Jan 2023-Sep 2024
- Built survival analysis (Kaplan-Meier) and random forest churn prediction
- Validated with 5-fold cross-validation (AUC = 0.87)

## Key Findings

### 1. Support calls are the strongest churn predictor
![Churn by Support Calls](chart.png)
Customers with 3+ support calls churn at 8.3x the rate of those with 0 calls.

### 2. Contract type matters
Month-to-month customers are 5.2x more likely to churn than annual contract holders.

### 3. The critical window is months 3-6
70% of churn happens in the first 6 months. Intervention in month 3-5 has highest impact.

## Limitations
- Analysis is correlational, not causal
- Some customer segments have small sample sizes
- Data only includes 18 months of history

## Recommendations
1. **Immediate**: Trigger VIP care protocol after 2nd support call in 30 days
2. **30-day**: Launch "3-month check-in" campaign for all new customers
3. **Quarterly**: Analyze annual contract conversion incentives
```

---

> **Summary**: Part 7 covers the "soft skills" that separate great data scientists from merely technical ones. Ethics ensures you do no harm. Governance ensures data is trustworthy. Communication ensures insights drive action. And reproducibility ensures your work stands the test of time. A data scientist who builds an excellent model but can't explain it, gets the data unethically, and can't reproduce the results — is not a good data scientist.
