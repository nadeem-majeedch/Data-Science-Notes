# Part 1: Core Concepts & Foundations

> Comprehensive Lecture Notes for BS/MS Data Science

---

## 1. What is Data Science?

### Simple Explanation
Data Science is the practice of extracting useful insights and knowledge from data. Think of it as being a detective — you collect clues (data), analyze them, and solve a mystery (business problem). It combines statistics, programming, and domain expertise.

### Simple Example
A supermarket wants to know which products to put on sale together. A data scientist looks at purchase history data, finds that people who buy diapers also often buy baby wipes, and recommends they be placed together. **Result**: More sales.

### Expert Example
A ride-sharing company like Uber uses data science for surge pricing. They collect real-time data on rider demand, driver availability, traffic conditions, weather, and historical patterns. A dynamic pricing model runs millions of calculations per minute to adjust prices in real-time, balancing supply and demand across a city. The data pipeline ingests streaming Kafka events, processes them with Apache Flink, and feeds an ML model that predicts the optimal price multiplier for each geohash zone every 60 seconds.

---

### Difference between Data Science, AI, ML, and Big Data

| Term | Simple Explanation | Analogy |
|------|-------------------|---------|
| **Data Science** | The overall field of extracting insights from data | The entire cooking process |
| **AI** | Machines that can perform tasks that normally require human intelligence | A chef who can invent new recipes |
| **ML** | A subset of AI where machines learn from data without being explicitly programmed | A chef who improves by tasting and adjusting |
| **Big Data** | Datasets so large that traditional tools can't handle them | A warehouse of ingredients too big to fit in one kitchen |

### Simple Example
You want to build a system that recommends movies.
- **Big Data**: You have 10 million users and 50,000 movies in your database.
- **Data Science**: You analyze what users watch and when they stop watching.
- **ML**: You train a model that learns: "Users who liked *Inception* also liked *Interstellar*."
- **AI**: The system automatically generates personalized recommendations for each user.

### Expert Example
In a self-driving car:
- **Big Data**: 10 PB of labeled driving footage from 1,000 vehicles across 50 cities.
- **Data Science**: Analyzing which road conditions cause the most disengagements.
- **ML**: Training a convolutional neural network to detect pedestrians with 99.97% accuracy.
- **AI**: The integrated system that perceives the environment, plans a path, and controls steering/brakes in real-time (30 Hz inference loop).

---

### The Data Science Workflow: Ask → Acquire → Process → Analyze → Communicate → Deploy

| Step | Simple Explanation | Example |
|------|-------------------|---------|
| **Ask** | Define the problem | "Why are customers canceling?" |
| **Acquire** | Get the data | Export from database, collect surveys |
| **Process** | Clean and prepare | Remove duplicates, fix missing values |
| **Analyze** | Find patterns and build models | Run statistics, train ML model |
| **Communicate** | Share results | Dashboard, presentation, report |
| **Deploy** | Put solution into production | Live recommendation system |

### Simple Example
**Ask**: A bank asks, "Which loan applicants are likely to default?"
**Acquire**: Pull 5 years of loan application data (100,000 records).
**Process**: Remove incomplete applications, standardize income formats, handle outliers.
**Analyze**: Build a logistic regression model that predicts default probability.
**Communicate**: Present findings: "Customers with debt-to-income ratio > 0.5 have 3x higher default risk."
**Deploy**: Integrate the model into the loan approval system as a real-time risk score.

### Expert Example
A fraud detection system at a payments company:
**Ask**: "Can we detect credit card fraud within 100ms of a transaction?"
**Acquire**: Stream 50,000 transactions/second from a Kafka cluster into a data lake (S3 + Parquet).
**Process**: Feature engineering pipeline (PySpark) computes 200+ features per transaction — velocity of spending, distance from home, device fingerprint, time since last transaction.
**Analyze**: Ensemble of XGBoost + deep learning autoencoder flags anomalous transactions.
**Communicate**: Real-time dashboard (Grafana) shows fraud rate, false positive rate, and model drift metrics.
**Deploy**: Model served via a TensorFlow Serving container in Kubernetes, with canary deployments and A/B testing against the previous rule-based system.

---

### Industry Process Frameworks (CRISP-DM, OSEMN, KDD)

**Simple Explanation**: The Ask → Acquire → Process → Analyze → Communicate → Deploy workflow is one way to describe data science. In industry, you will also hear about a few standard process frameworks that companies and teams use to organize their projects. They all describe the same loop — understand the problem, get and prepare data, build a model, and present results — just with different names and emphasis.

| Framework | Simple Explanation | Phases |
|-----------|-------------------|--------|
| **CRISP-DM** (Cross-Industry Standard Process for Data Mining) | The most widely used framework in industry; famous for the arrow going *backwards* from evaluation to understanding — you almost always loop back | Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment |
| **OSEMN** (Obtain, Scrub, Explore, Model, iNterpret) | A simple, Python-community-friendly 5-step name | Obtain → Scrub → Explore → Model → Interpret |
| **KDD** (Knowledge Discovery in Databases) | The classic academic process for finding knowledge in databases | Selection → Preprocessing → Transformation → Data Mining → Interpretation/Evaluation |

**Simple Example**: A telecom company wants to reduce customer churn.
- **CRISP-DM**: Business understanding ("each 1% churn = $2M lost") → collect call-log data → clean it → build a churn model → evaluate it against business goals → deploy a retention campaign.
- **OSEMN**: Obtain (export billing data) → Scrub (fix missing usage records) → Explore (visualize churn by plan) → Model (logistic regression) → Interpret (tell the marketing team who to call).

**Expert Example**: A bank's anti-money-laundering unit runs CRISP-DM with a formal governance structure. "Business understanding" produces a signed business requirements document and success criteria (e.g., detect 95% of flagged SAR patterns). "Data understanding" includes a data dictionary, lineage report, and quality scorecard (Great Expectations). Modeling runs in a versioned MLflow workspace with experiment tracking. "Evaluation" is a live shadow-deployment where the new model's alerts are compared against the current rules engine for 4 weeks before any business decision. "Deployment" is a Kubernetes service with drift monitoring and a quarterly model-refresh playbook. Every phase produces a formal deliverable so auditors can verify the process — which is exactly why regulated industries love CRISP-DM.

**Key takeaway**: These frameworks are just different words for the same core loop. Knowing at least CRISP-DM matters because it is the de-facto industry standard and often appears in job interviews and project documentation.

---

### Roles in Data Science

| Role | Simple Explanation | Key Skills |
|------|-------------------|------------|
| **Data Scientist** | Full-stack: asks questions, analyzes, builds models, tells stories | Stats, ML, coding, communication |
| **Data Analyst** | Focuses on reporting and visualization to answer business questions | SQL, Excel, Tableau, domain knowledge |
| **Data Engineer** | Builds the infrastructure to collect, store, and process data | ETL, Spark, databases, cloud |
| **ML Engineer** | Deploys and maintains ML models in production | MLOps, Docker, Kubernetes, API development |

### Simple Example
A hospital wants to predict patient readmission risk:
- **Data Analyst**: Creates weekly reports on readmission rates by department.
- **Data Engineer**: Builds a pipeline to consolidate data from EHR, lab systems, and insurance claims.
- **Data Scientist**: Develops an XGBoost model using patient demographics, vitals, and historical admissions.
- **ML Engineer**: Wraps the model in a REST API, deploys it on Kubernetes, sets up monitoring for data drift.

### Expert Example
At a FAANG company building a search engine:
- **Data Analyst**: Analyzes click-through rates for query suggestions, identifies that 12% of users reformulate their query within 5 seconds.
- **Data Engineer**: Maintains a Petabyte-scale data warehouse on BigQuery, manages streaming pipelines processing 1M+ events/sec, ensures 99.99% uptime for data ingestion.
- **Data Scientist**: Designs a query-document relevance model using BERT embeddings trained on 1B query-document pairs, achieves 15% improvement in NDCG@10.
- **ML Engineer**: Optimizes the model for sub-50ms inference latency using ONNX Runtime + INT8 quantization, implements feature store with Feast, manages model versioning with MLflow.

---

### Applications of Data Science Across Industries

| Industry | Application | Simple Example |
|----------|-------------|----------------|
| **Healthcare** | Disease diagnosis | AI reads X-rays to detect tumors |
| **Finance** | Fraud detection | Alert when your card is used in another country minutes after you used it locally |
| **E-commerce** | Recommendation systems | Amazon's "Customers who bought this also bought..." |
| **Transportation** | Route optimization | Google Maps finds the fastest route |
| **Education** | Personalized learning | Duolingo adapts lessons to your skill level |
| **Sports** | Player analytics | NBA teams decide player rotations based on shooting statistics |
| **Agriculture** | Crop yield prediction | Predict optimal planting time using weather data |
| **Energy** | Smart grid management | Predict electricity demand to balance supply |

### Expert Example
**Healthcare**: A deep learning model (EfficientNet-B7) trained on 112,000 chest X-rays from 30,000 patients detects 14 different pathologies, achieving AUC > 0.95 for pneumothorax and nodule detection, deployed via a HIPAA-compliant AWS environment with FHIR API integration.

---

## 2. Types of Data

### Structured Data

**Simple Explanation**: Data organized in rows and columns, like an Excel spreadsheet. Every row is a record, every column is a field.

**Simple Example**: An employee table:
| ID | Name | Department | Salary |
|----|------|------------|--------|
| 1 | Alice | Engineering | 80,000 |
| 2 | Bob | Marketing | 65,000 |

**Expert Example**: A financial trading database storing 500 million time-series rows per day in a columnar PostgreSQL database partitioned by date, with indexes on ticker symbol and timestamp for sub-millisecond query performance.

### Unstructured Data

**Simple Explanation**: Data with no predefined format or structure. It's messy and requires special processing.

**Simple Example**: A collection of customer support emails, social media posts, product images, and recorded phone calls.

**Expert Example**: A corpus of 50 million PubMed abstracts indexed in Elasticsearch, with full-text search and NLP pipelines (spaCy + BioBERT) extracting named entities (genes, diseases, drugs) and relations in batch mode on Spark.

### Semi-structured Data

**Simple Explanation**: Data that has some organizational structure but doesn't fit neatly into rows and columns. It uses tags or markers to separate elements.

**Simple Example**: A JSON object representing a customer:
```json
{
  "name": "Alice",
  "address": {
    "city": "New York",
    "zip": "10001"
  },
  "hobbies": ["reading", "hiking"]
}
```

**Expert Example**: Streaming IoT sensor data in Avro format, serialized by Kafka producers at 100K messages/sec, with a schema registry enforcing backward compatibility — read by a Flink job that joins this semi-structured data with structured lookup tables from a Cassandra cluster.

### Time-series Data

**Simple Explanation**: Data points collected or recorded at specific time intervals. The time component is critical.

**Simple Example**: Stock prices recorded every minute, daily temperatures, website traffic per hour.

**Expert Example**: 10,000 industrial IoT sensors emitting temperature, vibration, and pressure readings every 100ms to an InfluxDB time-series database. A streaming anomaly detection model (LSTM-autoencoder) processes 100M data points/day, triggering maintenance alerts when reconstruction error exceeds 3 standard deviations.

### Cross-sectional vs. Panel vs. Longitudinal Data

| Type | Simple Explanation | Example |
|------|-------------------|---------|
| **Cross-sectional** | Data collected at one point in time | A survey of 1,000 students' GPAs in 2024 |
| **Time-series** | One entity tracked over time | The price of IBM stock from 2010-2024 |
| **Panel (Longitudinal)** | Multiple entities tracked over time | GPAs of 1,000 students every semester from 2020-2024 |

**Expert Example**: A panel dataset from the World Bank containing GDP, education spending, and life expectancy for 195 countries measured annually from 1960-2023. This is panel data because it has both cross-sectional (countries) and time-series (years) dimensions. Econometric models like fixed-effects regression are used to control for unobserved country-specific heterogeneity.

---

## 3. Data Types & Measurement Scales

### Numerical Data

**Simple Explanation**: Data that is numeric and can be measured or counted.

| Subtype | Simple Explanation | Example |
|---------|-------------------|---------|
| **Discrete** | Countable, finite values | Number of students in a class (10, 25, 30) |
| **Continuous** | Any value within a range | Height of a person (5.8 ft, 6.1 ft, 5.95 ft) |

**Simple Example**: The number of emails you receive daily is discrete (0, 1, 2, ...). The exact temperature outside is continuous (72.3°F, 72.4°F).

**Expert Example**: In a credit risk model, continuous features (income, debt ratio, credit limit) are often discretized into bins using monotonic binning algorithms to satisfy regulatory requirements for scorecard interpretability. Each bin's WoE (Weight of Evidence) and IV (Information Value) are calculated to measure predictive power.

### Categorical Data

**Simple Explanation**: Data that represents categories or groups.

| Subtype | Simple Explanation | Example |
|---------|-------------------|---------|
| **Nominal** | Categories with no order | Colors: Red, Blue, Green. Gender: Male, Female |
| **Ordinal** | Categories with a meaningful order | Education: High School < Bachelor's < Master's < PhD |

**Simple Example**: 
- Nominal: Blood type (A, B, AB, O), City (Lahore, Karachi, Islamabad)
- Ordinal: Customer satisfaction (Very Unsatisfied < Unsatisfied < Neutral < Satisfied < Very Satisfied)

**Expert Example**: In natural language processing, text labels are categorical. A multi-class classifier for news articles uses ordinal encoding for category hierarchy (Sports > Cricket > International). An ordinal regression model (cumulative logit) preserves the ordering constraint, unlike a standard softmax classifier that treats categories as independent nominal classes.

### Scales of Measurement

| Scale | Simple Explanation | Mathematical Operations | Example |
|-------|-------------------|------------------------|---------|
| **Nominal** | Names/labels only | =, ≠ | Colors, genders |
| **Ordinal** | Ordered categories | =, ≠, <, > | Rankings (1st, 2nd, 3rd) |
| **Interval** | Equal intervals, no true zero | +, - (not ×, ÷) | Temperature in °C (20°C is not twice as hot as 10°C) |
| **Ratio** | Equal intervals, true zero exists | All operations including ×, ÷ | Height, weight, income |

**Simple Example**:
- **Nominal**: "What's your favorite pizza topping?" — Pepperoni, Mushroom, Onion
- **Ordinal**: "Rate your pain from 1-10" — 5 is worse than 3, but the difference isn't necessarily equal
- **Interval**: IQ scores — difference between 100 and 110 is same as 110 to 120
- **Ratio**: Salary — someone earning $100K makes twice as much as someone earning $50K

**Expert Example**: When building a regression model, understanding measurement scales is critical for interpreting coefficients. A one-unit increase in a ratio-scaled predictor (e.g., "years of experience") changes the log-odds of the outcome by β, and the ratio interpretation is meaningful. For interval-scaled predictors like IQ, the coefficient cannot be interpreted as a multiplier. This distinction matters in econometric modeling where elasticity (log-log model) is only valid for ratio-scale variables.

---

## 4. Data Sources

### Primary vs. Secondary Data

| Source | Simple Explanation | Example |
|--------|-------------------|---------|
| **Primary Data** | Collected by you for your specific purpose | You run a survey asking students about study habits |
| **Secondary Data** | Collected by someone else for another purpose | You use government census data to analyze population trends |

**Simple Example**: 
- Primary: A restaurant owner asks diners to fill out a feedback form.
- Secondary: The same owner looks at Yelp reviews that customers already wrote.

**Expert Example**: 
- Primary: A pharmaceutical company runs a double-blind RCT with 10,000 patients to test a new drug — collecting blood samples, vital signs, and patient-reported outcomes in a custom EDC (Electronic Data Capture) system.
- Secondary: The same company uses CMS Medicare claims data (already collected for billing) to study real-world effectiveness of existing treatments — adjusting for selection bias using propensity score matching.

### Data Collection Channels

| Channel | Simple Explanation | Example |
|---------|-------------------|---------|
| **APIs** | Programmatic interfaces to get data | Twitter API to get tweets about a hashtag |
| **Web Scraping** | Extract data from websites | Scrape product prices from an e-commerce site |
| **Databases** | Structured data storage | Query customer data from a MySQL database |
| **Surveys** | Questionnaires | Google Forms survey about brand preferences |
| **Sensors** | Physical devices that measure | Smartwatch tracks your heart rate |
| **Logs** | Automated system records | Server logs showing every website visit |

### Expert Example
A ride-sharing company's data pipeline ingests from 10+ sources simultaneously:
- **APIs**: Google Maps API for traffic data, Weather API for conditions
- **Databases**: PostgreSQL for driver profiles, MongoDB for ride history
- **Sensors**: GPS coordinates from driver phones (50M pings/day)
- **Logs**: Click-stream events from the mobile app (200K events/sec via Kafka)
- **Surveys**: Post-ride satisfaction ratings
- **Web scraping**: Competitor pricing from their public APIs

All streams are ingested into a data lake (S3 in Parquet format), cataloged via AWS Glue, and made available for both batch (Spark, 4-hour latency) and streaming (Flink, sub-second latency) processing.

### Open Data Portals

**Simple Explanation**: Websites that provide free access to datasets for anyone to use, analyze, and learn from.

**Simple Example**:
- **Kaggle**: Competitions and datasets for ML practice
- **UCI ML Repository**: Classic ML benchmark datasets (Iris, Wine, Titanic)
- **Data.gov**: US government open data (population, climate, education)
- **World Bank Open Data**: Economic indicators for all countries

**Expert Example**: A research team uses the UK Biobank dataset (500,000 participants, 40,000+ variables including genomics, imaging, and lifestyle data) accessed through a controlled research platform with IRB approval. They apply Mendelian randomization to study causal relationships between BMI and cardiovascular disease, using genetic variants as instrumental variables in a two-stage least squares regression.

---

## 5. Data Collection Methods

### Surveys and Experiments

| Method | Simple Explanation | Example |
|--------|-------------------|---------|
| **Survey** | Ask people questions | "How satisfied are you with our service?" |
| **Experiment** | Controlled test to determine cause-effect | Give Group A a drug, Group B a placebo, measure results |

**Simple Example**:
- **Survey**: A university surveys 500 students about how many hours they study per week.
- **Experiment**: A marketing team shows 50% of website visitors a red "Buy Now" button and 50% a green one to see which gets more clicks (A/B test).

**Expert Example**:
- **Survey**: A nationally representative survey (n=10,000) using stratified random sampling by age, gender, and region, with raking weights applied to match census demographics. Survey questions are randomized to avoid order bias, with attention-check questions embedded to filter low-quality responses.
- **Experiment**: An e-commerce platform runs a Bayesian A/B test with 95% power to detect a 0.5% lift in conversion rate. The experiment uses a multi-armed bandit algorithm (Thompson sampling) that automatically allocates more traffic to the winning variant, rather than a fixed 50/50 split. Results are analyzed with hierarchical Bayesian models to account for day-of-week effects and user segment heterogeneity.

### Observational Studies

**Simple Explanation**: Researchers observe subjects without intervening or manipulating anything. They just watch and record.

**Simple Example**: A researcher records the study habits of students by sitting in the library and noting what they do — no one is asked to change their behavior.

**Expert Example**: An epidemiologist uses the Nurses' Health Study (prospective cohort, n=121,700 women followed since 1976) to study the relationship between red meat consumption and colorectal cancer. Because this is observational (not an RCT), sophisticated causal inference methods are needed:
- Propensity score matching to balance confounders (age, BMI, smoking, physical activity)
- Inverse probability weighting (IPW) to handle loss to follow-up
- Sensitivity analysis (E-value calculation) to assess how strong an unmeasured confounder would need to be to explain away the observed effect

### Web Scraping and Crawling

**Simple Explanation**: Using automated programs to extract data from websites. Crawling discovers pages; scraping extracts data from them.

**Simple Example**: You write a Python script using BeautifulSoup and Requests to extract all product names and prices from an e-commerce website's catalog pages.

**Expert Example**: A price intelligence startup scrapes 500M product pages daily from 10,000 retail websites. The architecture:
- **Scrapy** framework with 200 concurrent spiders managed by Scrapyd
- **Rotating proxy pool** (50K residential IPs via BrightData) to avoid IP blocking
- **Playwright** for JavaScript-heavy sites (SPA apps that load content dynamically)
- **Kafka** for streaming scraped HTML to a processing pipeline
- **BeautifulSoup + lxml** parsing on Spark workers (500-node cluster)
- **Deduplication** via MD5 hashing of normalized content
- **Rate limiting** with token bucket algorithm to respect robots.txt
- Legal compliance: abides by CCPA opt-out requests and Terms of Service

### Sensor / IoT Data Collection

**Simple Explanation**: Physical devices (sensors) automatically collect and transmit data about the physical world.

**Simple Example**: A smart thermostat like Nest collects temperature, humidity, and occupancy data every 5 minutes to learn your schedule and optimize heating/cooling.

**Expert Example**: An oil refinery deploys 50,000 industrial IoT sensors from Siemens monitoring temperature, pressure, vibration, and flow rate at 100Hz. The data architecture:
- **Edge computing**: Raspberry Pi + AWS Greengrass runs initial anomaly detection locally, sends only alerts and aggregated statistics to the cloud (saving 95% bandwidth)
- **Protocol**: OPC-UA and MQTT with TLS encryption
- **Ingestion**: 5M data points/sec into a time-series database (TimescaleDB)
- **Processing**: Streaming analytics with Apache Flink detects equipment degradation patterns
- **Storage**: Hot data in InfluxDB (7 days), warm in Parquet on S3 (1 year), cold in Glacier (7 years for compliance)
- **ML model**: An autoencoder trained on normal operating conditions detects bearing failures 72 hours before they happen (predictive maintenance, saving $2M/year in unplanned downtime)

### Log Data from Systems

**Simple Explanation**: Automatic records created by computer systems every time something happens — a web request, an error, a login attempt.

**Simple Example**: Every time you visit a website, the server creates a log entry: `192.168.1.1 - - [10/Oct/2024:13:55:36 -0500] "GET /index.html HTTP/1.1" 200 2326`

**Expert Example**: A SaaS platform serving 10M users collects application logs structured as JSON:
```json
{
  "timestamp": "2024-10-10T13:55:36.123Z",
  "user_id": "u_abc123",
  "session_id": "sess_xyz789",
  "event": "api_request",
  "endpoint": "/api/v2/recommendations",
  "latency_ms": 342,
  "status_code": 200,
  "error": null,
  "feature_flags": ["new_algo_v3", "dark_mode"],
  "device": {"type": "mobile", "os": "iOS 17.4"}
}
```
Pipeline: Filebeat > Kafka > Logstash (parsing/enrichment) > Elasticsearch > Kibana dashboards. 10 TB of logs ingested daily with a 30-day retention period. Alerting via ElastAlert triggers on 5xx error rate > 1% in any 5-minute window. User behavior analytics pipelines in Spark process these logs to compute funnel analysis, retention cohorts, and feature adoption metrics.

---

> **Summary**: Part 1 establishes the foundational vocabulary and concepts of data science — what data is, where it comes from, how it's categorized, and who works with it. Students should be able to identify data types, measurement scales, and appropriate collection methods for any given problem after this module.

---

**Next: [Part 2 - Data Processing & Wrangling](Part%202%20-%20Data%20Processing%20%26%20Wrangling.md)** · [Back to README](README.md)
