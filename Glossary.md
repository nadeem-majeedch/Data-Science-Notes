# Glossary of Key Terms

> Quick-reference definitions for every term introduced in the notes, organized by part. Terms are defined in plain English first, with the technical detail in parentheses.

---

## Part 1: Core Concepts & Foundations

| Term | Simple Definition |
|------|-------------------|
| **Data Science** | Extracting useful insights from data using statistics, programming, and domain knowledge |
| **Artificial Intelligence (AI)** | Machines performing tasks that normally require human intelligence |
| **Machine Learning (ML)** | A subset of AI where machines learn patterns from data without being explicitly programmed |
| **Big Data** | Datasets so large that traditional tools cannot handle them |
| **Structured data** | Data that fits neatly into rows and columns (tables) |
| **Unstructured data** | Data with no fixed format (text, images, audio, video) |
| **Semi-structured data** | Partly organized data with tags/markers (JSON, XML, HTML) |
| **Time-series data** | Observations indexed in time order |
| **Cross-sectional data** | Data on many subjects at one point in time |
| **Panel data** | Data on the same subjects observed over time |
| **Longitudinal data** | Following the same subjects over time (like panel, often health-focused) |
| **Discrete data** | Countable values (number of children, defect counts) |
| **Continuous data** | Any value in a range (height, income) |
| **Nominal scale** | Categories with no order (colors, countries) |
| **Ordinal scale** | Categories with a meaningful order (ratings, education level) |
| **Interval scale** | Ordered with equal gaps, no true zero (temperature in °C) |
| **Ratio scale** | Ordered with equal gaps and a true zero (weight, income) |
| **Primary data** | Data you collect yourself for your question |
| **Secondary data** | Data collected by someone else, reused |
| **API** | A programmed interface that lets you pull data from a service |
| **Web scraping** | Extracting data from websites with automated programs |
| **IoT sensors** | Physical devices that automatically collect and transmit measurements |
| **CRISP-DM** | Industry-standard 6-phase process: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment |
| **OSEMN** | 5-step process: Obtain → Scrub → Explore → Model → Interpret |

## Part 2: Data Processing & Wrangling

| Term | Simple Definition |
|------|-------------------|
| **Data wrangling (munging)** | Transforming raw data into a clean, usable form |
| **Missing value imputation** | Filling missing values (mean/median/mode or model-based) |
| **Outlier** | A value far outside the normal range of the data |
| **Z-score** | How many standard deviations a value is from the mean |
| **IQR (Interquartile Range)** | The middle 50% range of data (Q3 − Q1) |
| **Winsorization** | Capping extreme values at a threshold instead of removing them |
| **Deduplication** | Removing duplicate records |
| **Standardization** | Rescaling data to mean 0 and standard deviation 1 (Z-score) |
| **Normalization (Min-Max)** | Rescaling data to a fixed range, typically [0, 1] |
| **One-hot encoding** | Turning categories into binary 0/1 columns |
| **Label encoding** | Turning categories into integer labels |
| **Log / Box-Cox transformation** | Making skewed data more symmetric/normal |
| **Binning (discretization)** | Grouping continuous values into buckets |
| **Entity resolution** | Identifying records that refer to the same real-world entity |
| **Sampling** | Selecting a subset of data (random, stratified, systematic) |
| **PCA (Principal Component Analysis)** | Reducing dimensions by finding the directions of greatest variance |
| **t-SNE / UMAP** | Nonlinear dimensionality reduction for visualizing clusters |
| **Feature selection** | Choosing the most useful features (filter, wrapper, embedded) |

## Part 3: EDA & Visualization

| Term | Simple Definition |
|------|-------------------|
| **EDA** | Exploring data with statistics and plots to find patterns and anomalies |
| **Mean / Median / Mode** | Average / middle value / most frequent value |
| **Variance / Standard Deviation** | How spread out the data is around the mean |
| **Skewness** | Whether data has a long tail to the left or right |
| **Kurtosis** | How "heavy-tailed" or outlier-prone the distribution is |
| **Five-number summary** | Min, Q1, Median, Q3, Max |
| **Histogram** | Bars showing how many values fall in each range |
| **Box plot** | Visualizes median, quartiles, and outliers |
| **Scatter plot** | Points showing the relationship between two numeric variables |
| **Heatmap** | A grid colored by value (great for correlations) |
| **Pair plot** | Matrix of scatter plots between all feature pairs |
| **Facet grid** | Many small plots, one per category |
| **Frequency table** | Count of how often each category occurs |
| **Contingency table** | Counts of two categorical variables together |
| **Pivot table** | Re-arranging data into a summary grid |
| **Population** | The entire group you are studying |
| **Sample** | A subset of the population you actually measure |
| **Central Limit Theorem (CLT)** | Sample means become normally distributed as sample size grows |
| **Confidence interval** | A range likely to contain the true value (e.g., 95% CI) |
| **p-value** | Probability of seeing your result if the null hypothesis were true |
| **Type I error** | False positive — rejecting a true null |
| **Type II error** | False negative — failing to reject a false null |
| **Trend** | Long-term upward or downward movement in a series |
| **Seasonality** | Regular repeating pattern in a series |
| **Stationarity** | Constant mean and variance over time (required for ARIMA) |
| **ACF / PACF** | Autocorrelation and partial autocorrelation vs lag |
| **ARIMA(p, d, q)** | Autoregressive Integrated Moving Average forecasting model |

## Part 4: Programming & Tools

| Term | Simple Definition |
|------|-------------------|
| **NumPy** | Fast numerical array computing in Python |
| **Pandas** | DataFrame library for data manipulation |
| **Jupyter** | Interactive notebook for code + plots + text |
| **Vectorization** | Applying operations to whole arrays instead of Python loops (10–100x faster) |
| **dplyr** | R package for data manipulation |
| **ggplot2** | R package for layered, grammar-of-graphics plots |
| **SELECT / WHERE / GROUP BY** | SQL clauses to pick, filter, and aggregate rows |
| **JOIN** | Combining tables on a key (inner/left/right/full) |
| **CTE** | Named subquery for readable SQL |
| **Window function** | Computing over a moving window of rows (ROW_NUMBER, RANK, LAG, LEAD) |
| **Git** | Version control for tracking code changes |
| **Clone / commit / push / pull** | Core git operations |
| **Branch** | A separate line of development |
| **Pipe (|)** | Sending one command's output into the next command |
| **Redirection (>)** | Saving a command's output to a file |

## Part 5: Machine Learning Foundations

| Term | Simple Definition |
|------|-------------------|
| **Supervised learning** | Learning from labeled data (features → known answer) |
| **Regression** | Predicting a continuous number |
| **Classification** | Predicting a category |
| **Linear regression** | Fitting a straight line (y = β₀ + β₁x) to predict a number |
| **Logistic regression** | Predicting a probability, then a class |
| **Decision tree** | A sequence of if-then-else questions that ends in a prediction |
| **k-NN** | Predicting by majority vote of the k nearest neighbors |
| **Naive Bayes** | Probability-based classifier assuming independent features |
| **MSE / RMSE / MAE** | Regression error metrics (larger errors weighted differently) |
| **R-squared** | Proportion of variance explained by the model |
| **Accuracy** | Fraction of correct predictions |
| **Precision** | Of predicted positives, how many were correct |
| **Recall** | Of actual positives, how many were caught |
| **F1-score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Ability to separate classes (0.5 random, 1.0 perfect) |
| **Confusion matrix** | Table of TP / FP / TN / FN |
| **Unsupervised learning** | Finding structure in unlabeled data |
| **k-Means** | Partitioning data into k clusters around centroids |
| **DBSCAN** | Density-based clustering (finds arbitrary shapes and noise) |
| **Apriori** | Finds frequent itemsets and association rules |
| **Train/test split** | Holding out data the model never sees during training |
| **Cross-validation** | Repeatedly training on folds to estimate generalization |
| **Overfitting** | Memorizing training data; great train, poor test |
| **Underfitting** | Too simple to capture the pattern |
| **Bias-variance tradeoff** | Balance between too-simple (bias) and too-complex (variance) models |
| **Regularization (L1/L2)** | Penalizing large coefficients to prevent overfitting |
| **Hyperparameter tuning** | Searching for the best settings (Grid/Random Search) |
| **Bagging** | Parallel models on random samples; reduces variance |
| **Random Forest** | Bagging + random features per split |
| **Boosting** | Sequential models that fix earlier mistakes; reduces bias |
| **XGBoost / LightGBM** | Fast, popular gradient boosting implementations |
| **Feature engineering** | Creating predictive input features from raw data |

## Part 6: Big Data & Databases

| Term | Simple Definition |
|------|-------------------|
| **4 V's of Big Data** | Volume, Velocity, Variety, Veracity |
| **Batch processing** | Processing data in scheduled chunks |
| **Stream processing** | Processing data continuously as it arrives |
| **Relational (SQL) database** | Data in tables with strict schemas (MySQL, PostgreSQL) |
| **NoSQL database** | Flexible, non-tabular stores (document, wide-column, key-value) |
| **Data warehouse** | Cleaned, structured data for reporting |
| **Data lake** | Raw data in any format, cheap storage |
| **Lakehouse** | Warehouse + lake: SQL and ACID on raw object storage |
| **HDFS** | Hadoop's distributed file system |
| **MapReduce** | Old batch processing model (map then reduce) |
| **Apache Spark** | Fast in-memory distributed computing |
| **Apache Hive** | SQL interface over Hadoop/Spark data |
| **Partitioning** | Splitting tables/files by a column (e.g., date) for faster queries |

## Part 7: Data Ethics, Governance & Communication

| Term | Simple Definition |
|------|-------------------|
| **Anonymization** | Removing identifiers so individuals can't be identified |
| **k-anonymity** | Every combination of identifiers appears in at least k rows |
| **Differential privacy** | Adding calibrated noise so outputs reveal nothing about any individual |
| **Bias / fairness** | Models should not systematically favor or harm groups |
| **Disparate impact** | Ratio of acceptance rates between groups (below 0.80 = adverse impact) |
| **Explainability** | Understanding *why* a model made a prediction |
| **SHAP / LIME** | Tools that explain individual model predictions |
| **Data quality** | Accuracy, completeness, consistency, timeliness of data |
| **Data lineage** | Tracking where data came from and how it changed |
| **Metadata** | Data about data (schema, origin, ownership) |
| **RBAC / ABAC** | Role-based and attribute-based access control |
| **Data storytelling** | Communicating insights with a narrative, not just charts |
| **Reproducibility** | Other people can rerun your analysis and get the same result |
| **Simpson's Paradox** | A trend disappears or reverses when data is split into groups |

## Part 8: Emerging Topics

| Term | Simple Definition |
|------|-------------------|
| **Deep learning** | ML using multi-layer neural networks |
| **Neural network** | Layers of neurons that transform input to output |
| **Activation function** | Non-linear step that lets networks learn (ReLU, Sigmoid, Tanh, Softmax) |
| **ReLU** | Max(0, x) — avoids the vanishing gradient problem |
| **CNN** | Neural network for images (convolution + pooling layers) |
| **RNN / LSTM / GRU** | Networks for sequences; LSTM/GRU remember long-range context |
| **Tokenization** | Splitting text into words (tokens) |
| **Stopwords** | Common words (the, and, of) usually removed |
| **Stemming** | Crude word-root chopping |
| **Lemmatization** | Dictionary-based word-root reduction |
| **Bag of Words** | Counts of words ignoring order |
| **TF-IDF** | Weighs words by how distinctive they are |
| **Word embeddings** | Vector representations capturing word meaning |
| **Word2Vec / GloVe / BERT** | Popular embedding methods (BERT is contextual) |
| **Cloud computing** | Compute/storage rented from AWS, GCP, Azure |
| **Model deployment** | Putting a model into production (API, batch) |
| **MLflow** | Tool for experiment tracking and model registry |
| **MLOps** | Engineering practices for deploying and maintaining ML models |
| **Model monitoring** | Detecting when a deployed model drifts or decays |

---

> **Tip**: Use this file as a revision checklist — if you can define every term in this glossary from memory, you are ready for the exam.
