# Introduction to Data Science — Course Outlines

## BS Data Science (3rd Semester)

---

# Part 1: Core Concepts & Foundations

## 1. What is Data Science?
- Data Science definition and lifecycle
- Difference between Data Science, AI, ML, and Big Data
- The Data Science workflow: Ask → Acquire → Process → Analyze → Communicate → Deploy
- Roles: Data Scientist, Data Analyst, Data Engineer, ML Engineer
- Applications of Data Science across industries

## 2. Types of Data
- **Structured** — Tabular data (rows & columns)
- **Unstructured** — Text, images, audio, video
- **Semi-structured** — JSON, XML, HTML
- **Time-series** — Data indexed in time order
- **Cross-sectional vs. Panel vs. Longitudinal** data

## 3. Data Types & Measurement Scales
- **Numerical** — Discrete / Continuous
- **Categorical** — Nominal / Ordinal
- **Text** — Corpus, tokens, documents
- **Scales of Measurement**: Nominal, Ordinal, Interval, Ratio

## 4. Data Sources
- Primary vs. Secondary data
- APIs, web scraping, databases, surveys, sensors, logs
- Open data portals (e.g., Kaggle, UCI, Government open data)

## 5. Data Collection Methods
- Surveys and experiments
- Observational studies
- Web scraping and crawling
- Sensor/IoT data collection
- Log data from systems

---

# Part 2: Data Processing & Wrangling

## 6. Data Wrangling (Data Munging)
- Definition and importance
- Common operations: filtering, sorting, merging, reshaping
- Handling missing values (drop, impute, flag)
- Handling outliers (detection and treatment)

## 7. Data Cleaning
- Deduplication
- Standardization and normalization
- Handling inconsistent formats (dates, currencies, units)
- Dealing with noisy data

## 8. Data Transformation
- Scaling (Min-Max, Standardization/Z-score)
- Encoding categorical variables (One-hot, Label encoding)
- Log transformation and Box-Cox
- Binning / Discretization
- Feature extraction

## 9. Data Integration
- Combining data from multiple sources
- Joins (inner, outer, left, right)
- Union and concatenation
- Entity resolution / Record linkage

## 10. Data Reduction
- Sampling (random, stratified, systematic)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Feature selection (filter, wrapper, embedded methods)

---

# Part 3: Exploratory Data Analysis (EDA) & Visualization

## 11. Exploratory Data Analysis (EDA)
- Definition and goals
- Summary statistics (mean, median, mode, variance, standard deviation, IQR)
- Distribution analysis
- Correlation analysis
- Identifying patterns, trends, and anomalies

## 12. Descriptive Statistics
- Measures of Central Tendency: Mean, Median, Mode
- Measures of Dispersion: Range, Variance, Standard Deviation, IQR
- Skewness and Kurtosis
- Five-number summary

## 13. Data Visualization
- **Univariate plots**: Histogram, Box plot, Bar chart, Pie chart, Density plot
- **Bivariate plots**: Scatter plot, Line chart, Stacked bar, Heatmap
- **Multivariate plots**: Pair plot, Bubble chart, Facet grid
- Time-series visualization
- Correlation heatmaps

## 14. Visualization Tools & Libraries
- Matplotlib
- Seaborn
- Plotly / Plotly Express
- Tableau (basic introduction)
- ggplot2 (R)

## 15. Summary Tables
- Frequency tables
- Contingency / Cross-tabulation tables
- Pivot tables

## 16. Statistical Foundations
- Population vs. Sample
- Parameter vs. Statistic
- Probability distributions (Normal, Binomial, Poisson, Uniform)
- Central Limit Theorem
- Law of Large Numbers
- Confidence Intervals
- Hypothesis Testing (Null & Alternative, p-value, Type I & II errors)

---

# Part 4: Programming & Tools

## 17. Python for Data Science
- Core libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- Jupyter Notebook / JupyterLab
- Basic Python: data types, loops, functions, list comprehensions

## 18. R for Data Science (Overview)
- Key packages: dplyr, ggplot2, tidyr, readr
- When to use R vs. Python

## 19. SQL for Data Science
- SELECT, WHERE, GROUP BY, HAVING, ORDER BY
- Joins (INNER, LEFT, RIGHT, FULL)
- Subqueries and CTEs
- Aggregation functions (COUNT, SUM, AVG, MIN, MAX)
- Window functions (ROW_NUMBER, RANK, LAG, LEAD)

## 20. Version Control
- Git basics: clone, add, commit, push, pull, branch
- GitHub / GitLab for collaboration

## 21. Command Line / Shell Basics
- File navigation (ls, cd, pwd, mkdir)
- File operations (cp, mv, rm, cat, less)
- Pipes and redirection
- Basic shell scripting

---

# Part 5: Machine Learning Foundations

## 22. Supervised Learning
- Definition: labeled data, known target variable
- **Regression**: Linear Regression, Multiple Regression
- **Classification**: Logistic Regression, Decision Trees, k-NN, Naive Bayes
- Evaluation metrics:
  - Regression: MSE, RMSE, MAE, R-squared
  - Classification: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

## 23. Unsupervised Learning
- Definition: unlabeled data, finding hidden structure
- **Clustering**: k-Means, Hierarchical, DBSCAN
- **Association**: Apriori algorithm, Market Basket Analysis
- Evaluation: Silhouette Score, Elbow Method, Inertia

## 24. Model Evaluation & Validation
- Train / Test split
- Cross-validation (k-fold, stratified)
- Bias-Variance Tradeoff
- Overfitting vs. Underfitting
- Regularization (L1/Lasso, L2/Ridge)
- Hyperparameter tuning (Grid Search, Random Search)

## 25. Feature Engineering
- Feature creation from existing data
- Feature selection techniques
- Handling date/time features
- Text feature extraction (Bag of Words, TF-IDF)

---

# Part 6: Big Data & Databases

## 26. Big Data Concepts
- The 4 V's: Volume, Velocity, Variety, Veracity
- Batch vs. Stream processing
- Distributed computing concepts

## 27. Databases
- Relational (SQL): MySQL, PostgreSQL, SQLite
- NoSQL: MongoDB (document), Cassandra (wide-column), Redis (key-value)
- Data Warehousing vs. Data Lake

## 28. Big Data Tools (Conceptual Overview)
- Hadoop ecosystem (HDFS, MapReduce)
- Apache Spark (in-memory distributed computing)
- Apache Hive (SQL on Hadoop)

---

# Part 7: Data Ethics, Governance & Communication

## 29. Data Ethics
- Privacy and consent
- Anonymization and de-identification
- Bias and fairness in data and models
- Transparency and explainability

## 30. Data Governance
- Data quality
- Data lineage
- Metadata management
- Data security and access control

## 31. Data Communication & Storytelling
- Data storytelling principles
- Building effective dashboards
- Audience-aware communication
- Common pitfalls in data presentation

## 32. Reproducibility & Best Practices
- Literate programming (Jupyter, R Markdown)
- Documenting code and analysis
- Random seeds for reproducibility
- Writing clear reports

---

# Part 8: Emerging Topics (Conceptual Introduction)

## 33. Introduction to Artificial Intelligence
- AI vs. ML vs. Deep Learning
- Symbolic AI vs. Statistical AI

## 34. Deep Learning (Basic Ideas)
- Neural networks concept
- Activation functions (ReLU, Sigmoid, Tanh)
- Introduction to CNNs (images) and RNNs (sequences)

## 35. Natural Language Processing (NLP) Basics
- Tokenization, stopwords, stemming, lemmatization
- Bag of Words, TF-IDF
- Word embeddings (Word2Vec, GloVe — conceptual)

## 36. Cloud & MLOps (Conceptual)
- Cloud platforms: AWS, GCP, Azure for data science
- Model deployment basics (API, batch inference)
- ML pipelines

---

# Suggested Textbooks & Resources

1. **Python for Data Analysis** — Wes McKinney (O'Reilly)
2. **Introduction to Statistical Learning (ISLR)** — James, Witten, Hastie, Tibshirani
3. **Data Science from Scratch** — Joel Grus
4. **Naked Statistics** — Charles Wheelan (for conceptual statistics)
5. **Storytelling with Data** — Cole Nussbaumer Knaflic

---

> **Note to Instructor**: This outline is designed for a **14–16 week semester**. Each topic can be covered at a conceptual/introductory level (1–2 lectures per part). Students are expected to have basic programming knowledge (Python) as a prerequisite.
