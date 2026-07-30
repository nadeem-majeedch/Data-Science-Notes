# Part 6: Big Data & Databases

> Comprehensive Lecture Notes for BS Data Science (3rd Semester)

---

## 26. Big Data Concepts

### Simple Explanation
Big Data refers to datasets so large or complex that traditional data processing tools cannot handle them. Think of it as the difference between a recipe for one cake (Excel can handle that) and a recipe for 10 million cakes (you need an industrial kitchen).

### The 4 V's of Big Data

| V | Simple Explanation | Example |
|---|-------------------|---------|
| **Volume** | The amount of data | Facebook generates 4 PB of new data daily |
| **Velocity** | Speed of data generation | Twitter processes 500M tweets/day = 6,000/sec |
| **Variety** | Different types of data | Text, images, video, sensor data, logs — all combined |
| **Veracity** | Quality and trustworthiness of data | 95% of CSV files have errors, social media is noisy |

### Simple Example
A retail chain with 500 stores:
- **Volume**: 50M transactions per day = 50 GB of new data daily
- **Velocity**: 500 transactions per second during peak hours
- **Variety**: Sales (structured), security footage (unstructured), customer reviews (text), delivery GPS (spatial)
- **Veracity**: 12% of transactions have missing store IDs, 3% have incorrect totals

### Expert Example
A global ride-sharing company's real-time data architecture:

| V | Scale | Technology |
|---|-------|-----------|
| **Volume** | 5 PB data lake, 50 TB new data/day | S3 + HDFS, Parquet format |
| **Velocity** | 2M GPS pings/sec, 100K trip events/sec | Kafka (100+ partitions), Flink |
| **Variety** | Structured (trips), semi-structured (app logs), unstructured (images) | Delta Lake + schema-on-read |
| **Veracity** | 0.5% data loss tolerance, 99.9% uptime | Data quality checks with Great Expectations, dedup with Spark |

---

### Batch vs. Stream Processing

| Type | Simple Explanation | Analogy | Latency | Tools |
|------|-------------------|---------|---------|-------|
| **Batch** | Process data in large chunks at scheduled intervals | A bus that leaves every hour with all waiting passengers | Minutes to hours | Hadoop, Spark batch, Hive |
| **Stream** | Process data as it arrives, in real-time | A taxi that leaves immediately with each passenger | Milliseconds to seconds | Kafka Streams, Flink, Spark Streaming |

### Simple Example
**Batch**: Every midnight, process all of today's sales to generate daily reports.
**Stream**: As each credit card transaction arrives, check for fraud in under 100ms.

### Expert Example
Lambda architecture (batch + stream combined):

```
        Real-time Stream (seconds)
Data → Kafka → Flink → Redis (hot data) → Real-time Dashboard
  |
  └──→ Batch Layer (hours)    
       S3 → Spark → Hive → Presto → Historical Reports
                ↓
            Merge Layer → Combined View
```

---

### Distributed Computing Concepts

**Simple Explanation**: Instead of one computer doing all the work, distribute the data and computation across hundreds or thousands of machines that work together. Each machine processes its portion of data.

### Simple Example
You need to count words in a million books.
- One computer: Process books one by one — would take months
- 1,000 computers (distributed): Each reads 1,000 books, counts words locally, sends results to a coordinator who sums them up — done in hours

### Key Concepts

| Concept | Simple Explanation | Analogy |
|---------|-------------------|---------|
| **Sharding** | Splitting data across machines | Each bookshelf holds different letters (A-F, G-M, N-Z) |
| **Replication** | Copying data to multiple machines | Three people have the same book — if one loses it, others have copies |
| **Fault Tolerance** | System keeps working if machines fail | If one chef gets sick, the kitchen keeps running |
| **Data Locality** | Bring computation to data, not vice versa | Instead of carrying ingredients to the chef, chef moves to ingredients |
| **MapReduce** | Two-step: Map (process locally) → Reduce (aggregate results) | Vote: each precinct counts locally (Map), then reports totals (Reduce) |

---

## 27. Databases

### Relational Databases (SQL)

**Simple Explanation**: Data organized in tables with rows and columns, connected through relationships. Like an Excel workbook where sheets can reference each other.

| Database | Simple Description | Best For |
|----------|-------------------|----------|
| **MySQL** | Most popular open-source RDBMS | Web applications |
| **PostgreSQL** | Advanced open-source, supports JSON, spatial, etc. | Complex queries, analytics |
| **SQLite** | File-based, no server needed | Mobile apps, prototypes |

### Simple Example
```sql
-- Relational database schema for an e-commerce site
-- Customers table
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT UNIQUE,
    signup_date DATE
);

-- Orders table (references customers)
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date DATE,
    total_amount DECIMAL(10,2)
);

-- Find all orders by customer
SELECT c.name, COUNT(o.order_id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name;
```

### Expert Example
PostgreSQL with performance optimization:

```sql
-- Index strategy for analytical queries
CREATE INDEX idx_orders_customer_date 
ON orders(customer_id, order_date DESC);

CREATE INDEX idx_orders_date_status 
ON orders(order_date, status) 
WHERE status = 'pending';  -- Partial index

CREATE INDEX idx_products_search 
ON products USING GIN(to_tsvector('english', product_name || ' ' || description));

-- Explain plan to check query performance
EXPLAIN ANALYZE
SELECT c.segment, 
       DATE_TRUNC('month', o.order_date) AS month,
       SUM(o.total_amount) AS revenue,
       COUNT(DISTINCT o.customer_id) AS active_customers,
       COUNT(o.order_id) AS order_count
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= NOW() - INTERVAL '6 months'
  AND c.is_active = TRUE
GROUP BY c.segment, DATE_TRUNC('month', o.order_date);

-- Materialized view for pre-computed aggregations
CREATE MATERIALIZED VIEW monthly_segment_sales AS
SELECT c.segment, 
       DATE_TRUNC('month', o.order_date) AS month,
       SUM(o.total_amount) AS revenue,
       COUNT(DISTINCT o.customer_id) AS customers
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.segment, DATE_TRUNC('month', o.order_date);

-- Refresh materialized view
REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_segment_sales;
```

---

### NoSQL Databases

**Simple Explanation**: Not Only SQL — databases designed for data that doesn't fit neatly into tables. They sacrifice ACID transactions for scalability and flexibility.

| Type | Simple Explanation | Database | Example Use Case |
|------|-------------------|----------|-----------------|
| **Document** | Store JSON-like documents | MongoDB | Product catalog (each product has different attributes) |
| **Wide-column** | Store rows with many (possibly different) columns | Cassandra | Time-series sensor data, user activity logs |
| **Key-Value** | Simple key → value pairs | Redis | Session cache, real-time leaderboard |

### Simple Example

**MongoDB (Document)**:
```json
// Each product can have different fields — no rigid schema
{
  "_id": "prod_123",
  "name": "iPhone 15",
  "price": 999,
  "specs": {
    "storage": "256GB",
    "color": "Black"
  },
  "reviews": [
    {"user": "alice", "rating": 5, "text": "Great phone!"}
  ]
}
```

**Redis (Key-Value)**:
```
SET session:user_456 "{'cart': ['item_1', 'item_2'], 'logged_in': true}"
GET session:user_456
```

### Expert Example
Cassandra data model for IoT sensor data:

```sql
-- Cassandra: designed for high write throughput
-- Partition key determines data distribution

CREATE TABLE sensor_data (
    sensor_id TEXT,
    timestamp TIMESTAMP,
    temperature DOUBLE,
    humidity DOUBLE,
    pressure DOUBLE,
    battery_level DOUBLE,
    PRIMARY KEY ((sensor_id, month_bucket), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Query: latest readings for a sensor
SELECT * FROM sensor_data 
WHERE sensor_id = 'sensor_001' 
  AND month_bucket = '2024-10'
LIMIT 100;

-- This query is efficient because:
-- 1. Partition key (sensor_id, month_bucket) routes to exactly one node
-- 2. Clustering key (timestamp) is already sorted DESC
-- No table scans, no cross-node queries!
```

---

### Data Warehousing vs. Data Lake

| Feature | Data Warehouse | Data Lake |
|---------|---------------|-----------|
| **Simple Explanation** | Clean, structured data ready for analysis | Raw data in native formats, schema applied on read |
| **Data Quality** | High — cleaned, transformed | Raw — may be messy |
| **Schema** | Schema-on-write (defined before loading) | Schema-on-read (applied when querying) |
| **Best For** | Business reporting, BI dashboards | Data science exploration, ML |
| **Storage** | Expensive, high-performance | Cheap (S3, HDFS) |
| **Tools** | Snowflake, Redshift, BigQuery | S3, ADLS, HDFS |

### Simple Example
- **Data Warehouse**: A retail dashboard showing "Sales this month vs last month by region" — needs clean, aggregated data
- **Data Lake**: A data scientist exploring raw clickstream logs to find new patterns — needs all the data, messiness included

### Expert Example
Modern Lakehouse architecture (combines both):

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Lakehouse                             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Bronze Layer │  │  Silver Layer │  │   Gold Layer │      │
│  │  (Raw Data)   │ →│  (Cleaned)   │ →│(Aggregated)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  Format: Delta Lake / Iceberg / Hudi                        │
│  Engine: Spark, Presto, Athena                              │
│  Catalog: Apache Hive Metastore, AWS Glue                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 28. Big Data Tools (Conceptual Overview)

### Hadoop Ecosystem

| Component | Simple Explanation | Analogy |
|-----------|-------------------|---------|
| **HDFS** | Hadoop Distributed File System — store huge files across many machines | A giant filing cabinet spread across 1,000 desks |
| **MapReduce** | Process data in parallel across machines | 1,000 people each counting words in their book, then summing results |
| **YARN** | Resource manager — decides which jobs run where | A dispatcher who assigns tasks to available workers |

### Simple Example
**HDFS**: A 10 GB file is too big for one machine. HDFS splits it into 128 MB blocks and stores each block on 3 different machines (replication). If one machine fails, the data is still available.

**MapReduce**: Count word frequency in Wikipedia (60 GB of text):
- **Map phase**: 500 machines each read a chunk and count words locally
- **Shuffle**: Same words are sent to the same reducer
- **Reduce phase**: Each reducer sums counts for its words

### Expert Example

```python
# PySpark (modern alternative to MapReduce)
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .appName("WordCount") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Read distributed data
df = spark.read.text("hdfs://cluster/data/books/*.txt")

# Word count (MapReduce in Spark)
word_counts = (df
    .select(explode(split(col("value"), "\\s+")).alias("word"))
    .filter(col("word") != "")
    .groupBy("word")
    .count()
    .orderBy(col("count").desc())
)

word_counts.show(10)
print(f"Total unique words: {word_counts.count()}")
```

---

### Apache Spark

**Simple Explanation**: A unified analytics engine for large-scale data processing, running 10-100x faster than traditional MapReduce because it keeps data in memory.

### Simple Example
```python
# Spark DataFrame API (similar to pandas but distributed)
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SalesAnalysis").getOrCreate()

# Read 10 GB of sales data
sales = spark.read.parquet("s3://data-lake/sales/")

# Analysis (executed in parallel across 100 nodes)
result = (sales
    .filter(col("date") >= "2024-01-01")
    .groupBy("region", "product_category")
    .agg(
        sum("amount").alias("total_sales"),
        count("*").alias("transaction_count"),
        avg("amount").alias("avg_ticket_size")
    )
    .orderBy(col("total_sales").desc())
)

result.show(20)
```

### Expert Example
Spark optimization techniques:

```python
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel

spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# 1. Read only needed columns and rows (predicate pushdown)
df = spark.read.parquet("s3://data/transactions/") \
    .select("customer_id", "amount", "date", "category") \
    .filter(F.col("date") >= "2024-01-01")

# 2. Cache frequently used data
df.cache()
df.count()  # Force cache

# 3. Broadcast join (small table to all executors)
customers = spark.read.parquet("s3://data/customers/")
result = df.join(F.broadcast(customers), "customer_id", "left")

# 4. Window functions for ranking
window_spec = Window.partitionBy("customer_id") \
    .orderBy(F.desc("amount")) \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

result = df.withColumn("rank", F.row_number().over(window_spec)) \
    .filter(F.col("rank") <= 5)  # Top 5 transactions per customer

# 5. Delta Lake for ACID transactions
result.write \
    .format("delta") \
    .mode("overwrite") \
    .option("replaceWhere", "date >= '2024-01-01'") \
    .save("s3://data/processed/top_customers/")

# 6. UDF (User Defined Function) — use pandas UDF for performance
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf("double")
def calculate_bmi(weight_kg: pd.Series, height_m: pd.Series) -> pd.Series:
    return weight_kg / (height_m ** 2)

df = df.withColumn("bmi", calculate_bmi(F.col("weight"), F.col("height")))
```

---

### Apache Hive

**Simple Explanation**: A data warehouse system built on top of Hadoop that lets you query data using SQL. Translates SQL into MapReduce/Spark jobs.

### Simple Example
```sql
-- Create table on data stored in HDFS
CREATE EXTERNAL TABLE sales (
    order_id INT,
    customer_id INT,
    amount DECIMAL(10,2),
    order_date STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/data/sales/';

-- Query with HiveQL (translated to MapReduce)
SELECT 
    year(order_date) AS year,
    month(order_date) AS month,
    SUM(amount) AS total_sales,
    COUNT(DISTINCT customer_id) AS unique_customers
FROM sales
WHERE year(order_date) >= 2023
GROUP BY year(order_date), month(order_date)
ORDER BY year DESC, month DESC;
```

### Expert Example
Hive with partitioning and bucketing for performance:

```sql
-- Partitioned and bucketed table
CREATE EXTERNAL TABLE sales_partitioned (
    order_id INT,
    customer_id INT,
    amount DECIMAL(10,2),
    product_category STRING
)
PARTITIONED BY (order_year INT, order_month INT)
CLUSTERED BY (customer_id) INTO 64 BUCKETS
STORED AS PARQUET
LOCATION '/data/sales_partitioned/';

-- Dynamic partition insert
INSERT OVERWRITE TABLE sales_partitioned
PARTITION (order_year, order_month)
SELECT 
    order_id, customer_id, amount, product_category,
    YEAR(order_date) AS order_year,
    MONTH(order_date) AS order_month
FROM raw_sales;

-- Query with partition pruning (only reads relevant partitions)
SELECT product_category, SUM(amount) AS total
FROM sales_partitioned
WHERE order_year = 2024 AND order_month = 10
GROUP BY product_category;

-- Use ORC format + bloom filters for fast filtering
CREATE TABLE optimized_sales (
    order_id INT,
    customer_id INT,
    amount DECIMAL(10,2)
)
STORED AS ORC
TBLPROPERTIES (
    'orc.bloom.filter.columns' = 'customer_id',
    'orc.compress' = 'SNAPPY'
);
```

---

### Big Data Processing: Key Optimization Principles

| Principle | Simple Explanation | Technique |
|-----------|-------------------|-----------|
| **Data Locality** | Process data where it lives | Spark runs executors on the same nodes that store data blocks |
| **Predicate Pushdown** | Filter data before reading it | Parquet + partition pruning skip irrelevant data |
| **Columnar Storage** | Store column data together, not rows | Parquet/ORC compress better, read only needed columns |
| **Caching** | Keep frequently used data in memory | Spark `.cache()`, Redis for hot data |
| **Vectorization** | Process multiple rows at once | Apache Arrow, Spark Tungsten engine |
| **Shuffle Minimization** | Reduce data movement across network | Co-located joins, bucketing |

### Simple Example
**Without optimization**: Read a 100 GB CSV file, parse it, filter, group — reads all 100 GB every time.
**With optimization**: Store as Parquet, partition by date, filter by date — read only 2 GB (the relevant partition). 50x faster.

---

> **Summary**: Part 6 covers the world of big data — when data is too large for traditional tools, you need distributed systems. Hadoop pioneered the concept, Spark made it fast (in-memory), and modern data architectures (lakehouse, streaming) continue to evolve. The key idea is always the same: distribute data across machines, bring computation to data, and plan for failures. For BS data scientists, understanding these concepts conceptually (and knowing when to use Spark vs. pandas vs. SQL) is more important than mastering every tool.
