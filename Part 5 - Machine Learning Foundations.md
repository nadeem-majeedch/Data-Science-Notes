# Part 5: Machine Learning Foundations

> Comprehensive Lecture Notes for BS Data Science (3rd Semester)

---

## 22. Supervised Learning

### Simple Explanation
Supervised learning is like a student learning with a teacher. You have a dataset with both inputs (features) and correct answers (labels). The model learns the relationship between inputs and outputs, so it can predict the output for new, unseen inputs.

**Two types**:
- **Regression**: Predict a continuous number (price, temperature)
- **Classification**: Predict a category (spam/not spam, dog/cat)

### Simple Example
**Regression**: Predict house price based on size, bedrooms, location.
- Input: [sq_ft=2000, bedrooms=3, city="Lahore"]
- Output: price = PKR 8.5M

**Classification**: Predict if an email is spam or not.
- Input: email text, sender, time
- Output: "Spam" or "Not Spam"

---

### Regression: Linear Regression

**Simple Explanation**: Draw a straight line through your data points that best predicts the output. The line equation: y = mx + b (or more formally: y = β₀ + β₁x)

**Simple Example**: Predict salary from years of experience:
```python
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]   # Years experience
y = [30, 35, 38, 42, 48]        # Salary in $1000s

model = LinearRegression()
model.fit(X, y)

# Predict salary for 6 years
print(model.predict([[6]]))  # ~51K
print(f"Coefficient: {model.coef_[0]:.1f}K/year")  # Each year = +4.3K
print(f"Intercept: {model.intercept_:.1f}K")        # Base salary
```

### Expert Example
Multiple linear regression with diagnostics:

```python
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load Boston Housing dataset
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Add interaction and polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# OLS with statsmodels for detailed statistics
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()
print(model_sm.summary())  # R-squared, F-stat, p-values, AIC, residual diagnostics

# Ridge regression (L2) — handles multicollinearity
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"Ridge R²: {r2_score(y_test, ridge.predict(X_test)):.3f}")

# Lasso regression (L1) — feature selection
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
selected = np.sum(lasso.coef_ != 0)
print(f"Lasso selected {selected} features out of {X_train.shape[1]}")

# Residual analysis
residuals = y_test - ridge.predict(X_test)
print(f"Residual std: {residuals.std():.3f}")
print(f"Residual skew: {pd.Series(residuals).skew():.3f}")
# Should be approximately normal, zero mean, homoscedastic
```

---

### Classification: Logistic Regression

**Simple Explanation**: Despite the name, this is used for classification. It calculates the probability that something belongs to a category. The "S-shaped" curve ensures the output is between 0 and 1 (a probability).

**Simple Example**: Predict if a student passes (1) or fails (0) based on study hours:
```python
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4], [5]]   # Hours studied
y = [0, 0, 1, 1, 1]             # Pass/Fail

model = LogisticRegression()
model.fit(X, y)

# Probability of passing with 2.5 hours
prob = model.predict_proba([[2.5]])[0][1]
print(f"P(pass | 2.5 hrs) = {prob:.1%}")

# Decision threshold at 0.5
print(f"Prediction: {'Pass' if prob >= 0.5 else 'Fail'}")
```

### Expert Example
Logistic regression with regularization, class imbalance handling, and evaluation:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_recall_curve)
import matplotlib.pyplot as plt

# Pipeline with preprocessing
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    ('classifier', LogisticRegression(
        penalty='elasticnet',
        solver='saga',  # Only solver supporting elasticnet
        l1_ratio=0.5,   # Mix of L1 and L2
        max_iter=1000,
        class_weight='balanced',  # Handle imbalance
        random_state=42
    ))
])

# Hyperparameter tuning
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__l1_ratio': [0.2, 0.5, 0.8],
    'poly__degree': [1, 2]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV AUC: {grid.best_score_:.3f}")

# Evaluation
y_pred = grid.predict(X_test)
y_prob = grid.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

---

### Decision Trees

**Simple Explanation**: A tree of if-then-else decisions. Each node asks a question about a feature, and each branch is an answer. You follow the path until you reach a leaf (prediction).

**Simple Example**: Decision tree for whether to play tennis:
```
          Outlook?
        /    |     \
    sunny   overcast  rainy
      |        |        |
   Humidity?  YES    Wind?
   /     \          /    \
High    Normal    Strong  Weak
  |        |         |      |
  NO      YES       NO     YES
```

### Expert Example
Decision tree with pruning and interpretability:

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# Train tree with pruning
tree = DecisionTreeClassifier(
    max_depth=4,            # Limit depth for interpretability
    min_samples_split=20,   # Minimum samples to split a node
    min_samples_leaf=10,    # Minimum samples per leaf
    max_features='sqrt',    # Random subset of features
    random_state=42
)
tree.fit(X_train, y_train)

# Print tree rules
text_rules = export_text(tree, feature_names=list(X.columns))
print(text_rules)

# Plot tree
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=list(X.columns), 
          class_names=['No', 'Yes'], filled=True, rounded=True)
plt.show()

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)

print(importances.head(10))
```

---

### k-Nearest Neighbors (k-NN)

**Simple Explanation**: To predict a new data point, look at the k closest data points in the training set. For classification, take a majority vote. For regression, take the average.

**Simple Example**: Predict if a fruit is apple or orange based on weight and color. A new fruit weighs 150g and is red. Its 3 nearest neighbors are: apple (140g, red), apple (160g, red), orange (145g, orange). Majority vote: apple.

### Expert Example

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Find optimal k using cross-validation
k_values = range(1, 51, 2)  # Odd k to avoid ties
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',       # Closer neighbors have more say
        metric='minkowski',       # Euclidean distance
        p=2,
        algorithm='kd_tree'       # Fast for low dimensions
    )
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k} (CV accuracy: {max(cv_scores):.3f})")

# Final model
knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
knn.fit(X_train, y_train)
print(f"Test accuracy: {knn.score(X_test, y_test):.3f}")
```

---

### Naive Bayes

**Simple Explanation**: A classifier based on Bayes' Theorem with the "naive" assumption that all features are independent. Despite this simplification, it works surprisingly well for text classification.

**Simple Example**: Spam detection. Word "FREE" appears in 80% of spam emails and 5% of non-spam. Word "WIN" appears in 60% of spam and 10% of non-spam. If an email contains both words, Naive Bayes combines these probabilities to classify it as spam.

### Expert Example

```python
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# For text classification
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        stop_words='english',
        sublinear_tf=True  # Use 1+log(tf)
    )),
    ('classifier', ComplementNB(alpha=0.1))  # Better for imbalanced text
])

pipeline.fit(X_train_text, y_train_text)
print(f"Text classification accuracy: {pipeline.score(X_test_text, y_test_text):.3f}")

# For continuous data: Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"Gaussian NB accuracy: {gnb.score(X_test, y_test):.3f}")
```

---

### Evaluation Metrics

#### Regression Metrics

| Metric | Simple Explanation | Formula | Range |
|--------|-------------------|---------|-------|
| **MSE** | Average squared error (penalizes large errors heavily) | Σ(y - ŷ)² / n | [0, ∞), lower is better |
| **RMSE** | √MSE — in original units, interpretable | √MSE | [0, ∞), lower is better |
| **MAE** | Average absolute error (less sensitive to outliers) | Σ|y - ŷ| / n | [0, ∞), lower is better |
| **R²** | Proportion of variance explained by model | 1 - SS_res / SS_tot | (-∞, 1], higher is better |

**Simple Example**: Actual prices: [100, 200, 300]. Predicted: [110, 190, 290]
- MSE = ((10)² + (-10)² + (-10)²) / 3 = 100
- RMSE = √100 = 10 (on average, off by ~$10)
- MAE = (|10| + |10| + |10|) / 3 = 10
- R² = 1 - (300 / 20000) = 0.985 (excellent fit)

#### Classification Metrics

| Metric | Simple Explanation | Formula |
|--------|-------------------|---------|
| **Accuracy** | Overall correct predictions | (TP + TN) / Total |
| **Precision** | Of predicted positives, how many were correct? | TP / (TP + FP) |
| **Recall** | Of actual positives, how many did we catch? | TP / (TP + FN) |
| **F1-score** | Harmonic mean of precision and recall | 2 × (P × R) / (P + R) |
| **ROC-AUC** | Model's ability to distinguish classes (0.5=random, 1.0=perfect) | Area under ROC curve |

**Simple Example**: 100 patients, model predicts disease. 10 have disease, 90 don't.
- Model: predicts 8 of 10 sick correctly (TP=8), says 2 are healthy (FN=2)
- Says 5 healthy are sick (FP=5), 85 healthy correctly (TN=85)
- Accuracy = (8+85)/100 = 93%
- Precision = 8/(8+5) = 61.5% — when model says sick, 61.5% chance they really are
- Recall = 8/(8+2) = 80% — model caught 80% of sick patients
- F1 = 2 × (0.615 × 0.80) / (0.615 + 0.80) = 69.6%

### Expert Example
Comprehensive model evaluation:

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, log_loss)
import matplotlib.pyplot as plt

# Calculate all metrics
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Classification Metrics ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"Log Loss:  {log_loss(y_test, y_prob):.4f}")
print(f"Avg Precision: {average_precision_score(y_test, y_prob):.4f}")

print("\n=== Detailed Report ===")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_score(y_test, y_prob):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Precision-Recall Curve (better for imbalanced data)
plt.subplot(1, 2, 2)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.plot(recall, precision, label=f'PR (AP = {average_precision_score(y_test, y_prob):.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.show()

# Find optimal threshold
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold (max F1): {optimal_threshold:.3f}")
```

---

## 23. Unsupervised Learning

### Simple Explanation
Unsupervised learning is like a student learning without a teacher. You only have inputs (features), no correct answers. The model finds hidden patterns, groups, or structures in the data on its own.

### Simple Example
You have customer purchase data but no labels. Unsupervised learning can group customers into segments based on their behavior — "budget shoppers," "luxury buyers," "impulse buyers" — without being told what these groups are.

---

### Clustering: k-Means

**Simple Explanation**: Partition data into k clusters. Each point belongs to the cluster with the nearest center (centroid). Simple, fast, and widely used.

**Simple Example**: Segment customers by income and spending score. k=3 gives:
- Cluster 0: High income, high spending — "Premium Customers"
- Cluster 1: Low income, high spending — "Careless Spenders"
- Cluster 2: Low income, low spending — "Budget Conscious"

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(df[['income', 'spending_score']])
```

### Expert Example
k-Means with optimal k selection and evaluation:

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale features (critical for k-means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k using elbow + silhouette
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Plot both
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].axvline(x=3, color='red', linestyle='--', label='Elbow')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].legend()

axes[1].plot(K_range, silhouettes, 'ro-')
axes[1].axvline(x=np.argmax(silhouettes) + 2, color='green', linestyle='--')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
plt.tight_layout()
plt.show()

# Final model with optimal k
optimal_k = np.argmax(silhouettes) + 2  # K_range starts at 2
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = final_kmeans.fit_predict(X_scaled)

# Profile clusters
df['cluster'] = clusters
profile = df.groupby('cluster').agg(['mean', 'std', 'count']).round(2)
print(profile.T)

# Visualize clusters in 2D (using PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(final_kmeans.cluster_centers_[:, 0], final_kmeans.cluster_centers_[:, 1], 
            marker='X', s=200, c='red', label='Centroids')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'K-Means Clustering (k={optimal_k})')
plt.legend()
plt.show()
```

---

### Hierarchical Clustering

**Simple Explanation**: Builds a tree of clusters. Start with each point as its own cluster, then repeatedly merge the closest clusters. The result is visualized as a dendrogram.

### Expert Example

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Compute linkage matrix
linkage_matrix = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or Cluster Size')
plt.ylabel('Distance (Ward)')
plt.axhline(y=30, color='red', linestyle='--', label='Cut at distance=30')
plt.legend()
plt.show()

# Cut tree to get flat clusters
hierarchical_labels = fcluster(linkage_matrix, t=5, criterion='maxclust')

# Compare with k-means
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(clusters, hierarchical_labels)
print(f"Adjusted Rand Index (k-means vs hierarchical): {ari:.3f}")
```

---

### DBSCAN

**Simple Explanation**: Density-based clustering — finds clusters as dense regions separated by sparse regions. Unlike k-means, it can find arbitrarily shaped clusters and identify outliers (noise points).

### Expert Example

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Find optimal eps using k-distance graph
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])  # 5th nearest neighbor distances

plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel('5th Nearest Neighbor Distance')
plt.title('K-Distance Graph (find elbow for eps)')
plt.axhline(y=0.5, color='red', linestyle='--')
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
dbscan_labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise} ({n_noise/len(dbscan_labels)*100:.1f}%)")

# Visualize
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, 
            cmap='Set1', alpha=0.6)
plt.title(f'DBSCAN Clustering (clusters={n_clusters}, noise={n_noise})')
plt.show()
```

---

### Association: Apriori Algorithm

**Simple Explanation**: Finds relationships between items in transaction data. "If a customer buys X, they are likely to also buy Y."

**Simple Example**: Market Basket Analysis at a grocery store:
- {Diapers} → {Beer}: Customers who buy diapers are 3x more likely to buy beer
- {Bread, Butter} → {Milk}: 70% of customers who buy bread and butter also buy milk

### Expert Example

```python
from mlxtend.frequent_patterns import apriori, association_rules

# Prepare transaction data (one-hot encoded)
# Each row = a transaction, each column = a product
df_encoded = df.groupby(['transaction_id', 'product'])['quantity'].sum().unstack().fillna(0)
df_encoded = df_encoded.applymap(lambda x: 1 if x > 0 else 0)

# Find frequent itemsets (support >= 0.05)
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
print(f"Found {len(frequent_itemsets)} frequent itemsets")
print(frequent_itemsets.sort_values('support', ascending=False).head(10))

# Generate association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.5)
rules = rules.sort_values('lift', ascending=False)

print(f"Generated {len(rules)} rules")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Key metrics explained:
# Support: How often the itemset appears in transactions
# Confidence: P(consequent | antecedent) — how reliable is the rule?
# Lift: How much more likely is consequent when antecedent is present?
#   Lift > 1 = positive association
#   Lift = 1 = independent
#   Lift < 1 = negative association
```

---

## 24. Model Evaluation & Validation

### Train / Test Split

**Simple Explanation**: Split data into training (teach the model) and testing (evaluate the model). Never evaluate on data the model has already seen — that's like giving a student the exam questions as homework and then testing them.

**Simple Example**: 80% of customers' data used to train a churn prediction model. 20% held back to test if the model actually works on new customers.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Expert Example
Multiple splitting strategies:

```python
from sklearn.model_selection import (train_test_split, TimeSeriesSplit, 
                                     StratifiedKFold, GroupKFold)

# 1. Standard random split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Time-series split (no future data leakage)
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold}: Train {train_idx[0]}-{train_idx[-1]}, Val {val_idx[0]}-{val_idx[-1]}")

# 3. Group split (same customer doesn't appear in both train and test)
group_kfold = GroupKFold(n_splits=5)
for train_idx, val_idx in group_kfold.split(X, y, groups=customer_ids):
    pass
```

---

### Cross-validation (k-fold, stratified)

**Simple Explanation**: Instead of one train/test split, split data into k folds. Train on k-1 folds, test on the remaining fold. Repeat k times, average the results. More reliable than a single split.

**Simple Example**: 5-fold CV — split data into 5 parts. Train on 4 parts, test on 1st part. Then train on different 4, test on 2nd part. Repeat 5 times. Report average performance.

### Expert Example

```python
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

# Multi-metric cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'roc_auc': 'roc_auc'
}

cv_results = cross_validate(
    model, X_train, y_train,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

# Summarize CV results
cv_summary = pd.DataFrame({
    'Metric': list(scoring.keys()),
    'Train Mean': [cv_results[f'train_{m}'].mean() for m in scoring.keys()],
    'Train Std': [cv_results[f'train_{m}'].std() for m in scoring.keys()],
    'Test Mean': [cv_results[f'test_{m}'].mean() for m in scoring.keys()],
    'Test Std': [cv_results[f'test_{m}'].std() for m in scoring.keys()],
})

# Check for overfitting (train >> test)
cv_summary['Overfit Gap'] = cv_summary['Train Mean'] - cv_summary['Test Mean']
print(cv_summary.round(4))
```

---

### Bias-Variance Tradeoff

**Simple Explanation**: 
- **Bias**: Error from assuming the model is simpler than reality (underfitting)
- **Variance**: Error from the model being too sensitive to training data (overfitting)

**The Tradeoff**: Simple models (linear regression) have high bias but low variance. Complex models (deep trees) have low bias but high variance. The goal is to find the sweet spot.

### Simple Example
- **High Bias (Underfitting)**: Using a straight line to predict housing prices — misses the complexity
  - Performance: Both train AND test error are high
- **High Variance (Overfitting)**: A deep decision tree that memorizes every house — perfect on training, terrible on new data
  - Performance: Train error near zero, test error very high
- **Balanced**: A moderate model that captures the general pattern without memorizing noise

### Expert Example

```python
# Demonstrate bias-variance tradeoff with polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

degrees = [1, 2, 3, 5, 10, 15]
train_errors = []
test_errors = []

for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, 'bo-', label='Train Error')
plt.plot(degrees, test_errors, 'ro-', label='Test Error')
plt.axvline(x=degrees[np.argmin(test_errors)], color='g', linestyle='--', 
            label=f'Optimal degree={degrees[np.argmin(test_errors)]}')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Mean Squared Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.yscale('log')
plt.show()

# Region 1 (left): High bias — both errors high
# Region 2 (middle): Optimal — test error minimized
# Region 3 (right): High variance — train low, test high
```

---

### Overfitting vs. Underfitting

| Problem | Simple Explanation | Training Performance | Test Performance | Fix |
|---------|-------------------|---------------------|------------------|-----|
| **Underfitting** | Model too simple, misses patterns | Poor | Poor | Add complexity, more features |
| **Overfitting** | Model too complex, memorizes noise | Excellent | Poor | Simplify, regularize, more data |

### Simple Example
**Underfitting**: You try to classify cats vs dogs using only weight. Many small dogs are lighter than large cats. The rule "weight > 15kg = dog" misses the true pattern.

**Overfitting**: You memorize 1,000 cat and dog photos including background details. A new photo of a cat on a blue background is classified as "dog" because the training dogs happened to be on blue backgrounds.

---

### Regularization (L1/Lasso, L2/Ridge)

**Simple Explanation**: Regularization adds a penalty for large coefficients, forcing the model to be simpler. It's like telling the model: "Keep it simple — don't rely too much on any single feature."

| Type | Penalty | Effect | Simple Analogy |
|------|---------|--------|---------------|
| **L1 (Lasso)** | Σ|coef| | Forces some coefficients to exactly 0 — feature selection | "If a feature isn't useful, ignore it entirely" |
| **L2 (Ridge)** | Σ(coef)² | Shrinks all coefficients toward 0 (but never to 0) | "Don't rely too heavily on anything" |
| **ElasticNet** | Mix of L1 + L2 | Combines both benefits | Balanced approach |

### Expert Example

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

# Ridge regression with cross-validated alpha
ridge = Ridge()
param_grid = {'alpha': np.logspace(-3, 3, 20)}
ridge_cv = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train, y_train)
print(f"Ridge optimal alpha: {ridge_cv.best_params_['alpha']:.4f}")

# Lasso (L1) — feature selection
lasso = Lasso(max_iter=10000)
lasso_cv = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_cv.fit(X_train, y_train)
print(f"Lasso optimal alpha: {lasso_cv.best_params_['alpha']:.4f}")
print(f"Features used: {np.sum(lasso_cv.best_estimator_.coef_ != 0)}/{X.shape[1]}")

# ElasticNet (hybrid)
elastic = ElasticNet(max_iter=10000)
param_grid = {
    'alpha': np.logspace(-3, 1, 10),
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}
elastic_cv = GridSearchCV(elastic, param_grid, cv=5, scoring='neg_mean_squared_error')
elastic_cv.fit(X_train, y_train)
print(f"ElasticNet best params: {elastic_cv.best_params_}")
```

---

### Hyperparameter Tuning (Grid Search, Random Search)

**Simple Explanation**: Every ML model has knobs (hyperparameters) you can turn. Grid search tries every combination of knob settings. Random search randomly samples combinations. Both find good settings automatically.

### Simple Example
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Grid Search — try every combination (exhaustive)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
# Total: 2 × 3 × 3 = 18 combinations × 5 CV folds = 90 fits

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Best params (grid): {grid.best_params_}")
```

### Expert Example
Randomized search with Bayesian optimization:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Random Search — sample from distributions (more efficient)
param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 50),
    'min_samples_leaf': randint(1, 20),
    'max_features': uniform(0.1, 0.9),  # Fraction of features
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,  # Only 50 combinations (vs thousands in grid)
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search.fit(X_train, y_train)

print(f"Best params (random): {random_search.best_params_}")
print(f"Best CV AUC: {random_search.best_score_:.4f}")

# Learning curve to verify tuning
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    random_search.best_estimator_, X_train, y_train,
    cv=5, scoring='roc_auc', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training')
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Cross-validation')
plt.fill_between(train_sizes, 
                 train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
plt.fill_between(train_sizes, 
                 test_scores.mean(axis=1) - test_scores.std(axis=1),
                 test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.1)
plt.xlabel('Training Examples')
plt.ylabel('ROC-AUC')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 25. Feature Engineering

### Simple Explanation
Feature engineering is creating new input variables from existing data to make it easier for models to learn patterns. This is where domain expertise matters most — knowing what predictive signals to extract from raw data.

### Feature Creation from Existing Data

**Simple Example**: From a date column, create:
- Is weekend? (binary)
- Month (categorical)
- Day of week (cyclical encoding)
- Quarter
- Days since last purchase
- Is holiday?

### Expert Example
```python
# Time-based features
def create_time_features(df, date_col):
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    
    # Basic time components
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['dayofweek'] = dt.dt.dayofweek     # 0=Monday
    df['quarter'] = dt.dt.quarter
    df['weekofyear'] = dt.dt.isocalendar().week.astype(int)
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclical encoding (preserves circular nature of time)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Relative features
    df['days_since_reference'] = (dt - dt.min()).dt.days
    
    # Rolling features (requires sorting by date first)
    df = df.sort_values('transaction_date')
    df['sales_lag_1'] = df['sales'].shift(1)
    df['sales_lag_7'] = df['sales'].shift(7)
    df['sales_lag_30'] = df['sales'].shift(30)
    df['rolling_mean_7'] = df['sales'].rolling(7).mean()
    df['rolling_std_7'] = df['sales'].rolling(7).std()
    df['rolling_mean_30'] = df['sales'].rolling(30).mean()
    df['expanding_mean'] = df['sales'].expanding().mean()
    
    return df

# Aggregation features (per customer)
customer_features = df.groupby('customer_id').agg(
    total_purchases=('amount', 'count'),
    avg_purchase=('amount', 'mean'),
    std_purchase=('amount', 'std'),
    min_purchase=('amount', 'min'),
    max_purchase=('amount', 'max'),
    total_spend=('amount', 'sum'),
    first_purchase_date=('date', 'min'),
    last_purchase_date=('date', 'max'),
    purchase_frequency=('date', lambda x: (x.max() - x.min()).days / max(len(x), 1)),
    unique_categories=('category', 'nunique')
).reset_index()

# Ratio features (often highly predictive)
customer_features['avg_purchase_to_spend_ratio'] = (
    customer_features['avg_purchase'] / customer_features['total_spend']
)
customer_features['purchase_recency'] = (
    reference_date - customer_features['last_purchase_date']
).dt.days
```

---

> **Summary**: Part 5 covers the fundamental machine learning concepts every data scientist must know. Supervised learning (regression + classification) is the most common task. Unsupervised learning finds hidden patterns. Model evaluation ensures you're not fooling yourself. And feature engineering is where the real value lies — garbage in, garbage out, but great features make great models.
