# Data Science Notes — BS Data Science (3rd Semester)

A comprehensive lecture note repository for **Introduction to Data Science** — designed for 3rd semester BS Data Science students. Covers all fundamental concepts with **simple explanations**, **beginner examples**, and **expert-level examples** for each term.

---

## Repository Structure

```
├── Introduction_to_Data_Science_Outlines.md    # Full course outline + weekly schedule
├── Part 1 - Core Concepts & Foundations.md      # What is DS, data types, sources, collection
├── Part 2 - Data Processing & Wrangling.md      # Cleaning, transformation, integration, reduction
├── Part 3 - Exploratory Data Analysis (EDA) & Visualization.md
├── Part 4 - Programming & Tools.md              # Python, R, SQL, Git, command line
├── Part 5 - Machine Learning Foundations.md     # Supervised/Unsupervised, evaluation, feature eng
├── Part 6 - Big Data & Databases.md             # SQL/NoSQL, Hadoop, Spark, Hive
├── Part 7 - Data Ethics, Governance & Communication.md
├── Part 8 - Emerging Topics.md                  # AI/DL, NLP, Cloud, MLOps
├── Part 9 - Exercises & Solutions.md            # Practice problems with solutions
├── Glossary.md                                  # Quick-reference terms by part
└── README.md                                    # This file
```

---

## Quick Navigation

| File | What's Inside |
|------|---------------|
| `Introduction_to_Data_Science_Outlines.md` | Full course outline + suggested weekly schedule |
| Part 1 → Part 8 | Core lecture notes (simple explanation → simple example → expert example) |
| `Part 9 - Exercises & Solutions.md` | Practice problems + solutions (conceptual, coding, challenge) |
| `Glossary.md` | Every key term defined in one place |

**Suggested reading order**: Outline → Part 1 → … → Part 8, then practice with Part 9. Use `Glossary.md` for quick revision.

> **Note to students**: The parts build on each other. Part N+1 assumes you understand Part N.

---

## Part 1: Core Concepts & Foundations

| Topic | Key Terms |
|-------|-----------|
| What is Data Science? | DS definition, DS vs AI vs ML vs Big Data, workflow (Ask → Acquire → Process → Analyze → Communicate → Deploy), roles (Data Scientist/Analyst/Engineer/ML Engineer), process frameworks (CRISP-DM, OSEMN, KDD) |
| Types of Data | Structured, Unstructured, Semi-structured, Time-series, Cross-sectional vs Panel vs Longitudinal |
| Data Types & Scales | Numerical (Discrete/Continuous), Categorical (Nominal/Ordinal), Measurement Scales (Nominal/Ordinal/Interval/Ratio) |
| Data Sources | Primary vs Secondary, APIs, scraping, databases, surveys, sensors, logs, open data portals |
| Data Collection | Surveys, experiments, observational studies, web scraping, IoT sensors, system logs |

---

## Part 2: Data Processing & Wrangling

| Topic | Key Techniques |
|-------|----------------|
| Data Wrangling | Filtering, sorting, merging, reshaping, missing values (drop/impute/flag), outliers (Z-score/IQR/Isolation Forest) |
| Data Cleaning | Deduplication, standardization, normalization, inconsistent formats, noisy data |
| Data Transformation | Min-Max scaling, Z-score, one-hot/label encoding, log/Box-Cox, binning, feature extraction |
| Data Integration | Inner/Left/Right/Outer joins, union, concatenation, entity resolution, record linkage |
| Data Reduction | Random/stratified/systematic sampling, PCA, t-SNE, UMAP, filter/wrapper/embedded feature selection |

---

## Part 3: Exploratory Data Analysis (EDA) & Visualization

| Topic | Key Concepts |
|-------|--------------|
| EDA | Summary statistics, distribution analysis, correlation analysis, pattern/anomaly detection |
| Descriptive Statistics | Mean/Median/Mode, Variance/Std/IQR, Skewness/Kurtosis, Five-number summary |
| Visualization | Histogram, box plot, bar chart, scatter plot, heatmap, pair plot, bubble chart, facet grid |
| Time-series | Visualization, decomposition, stationarity, ACF/PACF, ARIMA/SARIMA forecasting, backtesting |
| Tools | Matplotlib, Seaborn, Plotly, Tableau, ggplot2 |
| Summary Tables | Frequency tables, contingency tables, pivot tables |
| Statistical Foundations | Population vs Sample, CLT, LLN, Confidence Intervals, Hypothesis Testing (p-value, Type I/II errors) |

---

## Part 4: Programming & Tools

| Topic | Libraries / Commands |
|-------|----------------------|
| Python | NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Jupyter |
| R | dplyr, ggplot2, tidyr, readr |
| SQL | SELECT, WHERE, GROUP BY, HAVING, JOINs, subqueries, CTEs, window functions (ROW_NUMBER, RANK, LAG, LEAD) |
| Git | clone, add, commit, push, pull, branch, merge |
| Command Line | ls, cd, pwd, mkdir, cp, mv, pipes, redirection, shell scripting |

---

## Part 5: Machine Learning Foundations

| Topic | Models / Methods |
|-------|------------------|
| Supervised Learning | Linear Regression, Logistic Regression, Decision Trees, k-NN, Naive Bayes |
| Ensemble Methods | Bagging, Random Forest, Boosting, Gradient Boosting, XGBoost, Stacking/Voting |
| Regression Metrics | MSE, RMSE, MAE, R-squared |
| Classification Metrics | Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix |
| Unsupervised Learning | k-Means, Hierarchical, DBSCAN, Apriori (association rules) |
| Evaluation | Train/Test split, Cross-validation, Bias-Variance Tradeoff, Overfitting/Underfitting, Regularization (L1/L2/ElasticNet) |
| Feature Engineering | Date features, aggregation features, ratio features, text features (BoW, TF-IDF) |

---

## Part 6: Big Data & Databases

| Topic | Key Concepts |
|-------|--------------|
| Big Data | 4 V's (Volume, Velocity, Variety, Veracity), Batch vs Stream processing, Distributed computing |
| Databases | MySQL, PostgreSQL (relational), MongoDB (document), Cassandra (wide-column), Redis (key-value) |
| Data Warehousing | Data Warehouse vs Data Lake, Lakehouse architecture (Bronze/Silver/Gold) |
| Big Data Tools | HDFS, MapReduce, Apache Spark, Apache Hive, Partitioning, Bucketing |

---

## Part 7: Data Ethics, Governance & Communication

| Topic | Key Concepts |
|-------|--------------|
| Data Ethics | Privacy, informed consent, anonymization (k-anonymity), differential privacy, bias & fairness, explainability (SHAP, LIME) |
| Data Governance | Data quality (accuracy/completeness/consistency), data lineage, metadata management, access control (RBAC/ABAC) |
| Data Storytelling | Context → Conflict → Insight → Recommendation, audience-aware communication, dashboard design principles |
| Reproducibility | Literate programming (Jupyter/R Markdown), random seeds, documentation, clear report structure |

---

## Part 8: Emerging Topics

| Topic | Key Concepts |
|-------|--------------|
| AI vs ML vs DL | Symbolic AI vs Statistical AI, rule-based vs learned approaches |
| Deep Learning | Neural networks, activation functions (ReLU/Sigmoid/Tanh/Softmax), CNNs (convolution/pooling), RNNs/LSTM/GRU |
| NLP | Tokenization, stopwords, stemming, lemmatization, Bag of Words, TF-IDF, Word Embeddings (Word2Vec/GloVe/BERT) |
| Cloud & MLOps | AWS/GCP/Azure, REST API deployment, batch inference, edge deployment, MLflow, experiment tracking, model monitoring |

---

## How to Use These Notes

1. **Follow the order**: Start with Part 1 and progress sequentially — each part builds on previous ones.
2. **Read the simple explanation first**: Each term starts with an intuitive, plain-English definition.
3. **Run the simple examples**: Copy-paste and execute them to verify understanding.
4. **Study the expert examples**: These show real-world, production-grade code and architecture.
5. **Use the outlines file**: `Introduction_to_Data_Science_Outlines.md` gives a bird's-eye view of all terms covered.

---

## Assessment Plan (Suggested)

| Component | Weight | Description |
|-----------|--------|-------------|
| Quizzes (×2) | 20% | Conceptual quizzes after every 3 weeks (Parts 1–3, then Parts 4–6) |
| Assignment 1 | 15% | Mid-semester EDA report on a real dataset (cleaning + visualization + findings) |
| Assignment 2 | 15% | Build and evaluate a supervised ML model end-to-end (train/test, CV, metrics) |
| Final Project | 30% | Group project: pick a dataset, run the full workflow (Ask → … → Deploy), present findings |
| Participation / Labs | 20% | Weekly lab submissions based on the exercises in `Part 9` |

**Example project prompts**:
- Predict house prices in your city from a scraped listing dataset.
- Churn prediction for a telecom or banking dataset.
- Sales forecasting for a retail chain using the time-series methods in Part 3.
- Fairness audit: evaluate whether a model's decisions differ across demographic groups (Part 7).

All projects must include: a clear business question, reproducible code (seeded), a short written report, and a dashboard or slide deck.

---

## Prerequisites

- Basic programming knowledge (Python recommended)
- High school-level mathematics (algebra, basic probability)
- No prior ML or data science experience required

---

## Suggested Textbooks

1. **Python for Data Analysis** — Wes McKinney (O'Reilly)
2. **Introduction to Statistical Learning (ISLR)** — James, Witten, Hastie, Tibshirani
3. **Data Science from Scratch** — Joel Grus
4. **Naked Statistics** — Charles Wheelan
5. **Storytelling with Data** — Cole Nussbaumer Knaflic

---

## Author

**Engr. Dr. Muhammad Nadeem Majeed** — Professor, Department of Data Science  
University of the Punjab, Lahore  
Created for BS Data Science (3rd Semester) — Introduction to Data Science course.

---

## License

This repository is released under the [MIT License](LICENSE). You are free to use, modify, and share these notes with attribution.

---

*Last updated: July 2026*
