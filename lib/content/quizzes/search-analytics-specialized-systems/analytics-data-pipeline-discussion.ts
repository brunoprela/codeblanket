import { Quiz } from '@/lib/types';

const analyticsDataPipelineDiscussionQuiz: Quiz = {
  id: 'analytics-data-pipeline-discussion',
  title: 'Analytics Data Pipeline - Discussion Questions',
  questions: [
    {
      id: 'adp-discussion-1',
      type: 'discussion',
      question:
        'Your e-commerce company needs to build an analytics pipeline that supports: (1) real-time fraud detection on transactions (latency < 100ms), (2) daily sales reports for executives, (3) ML model training on 6 months of historical data weekly. The system processes 10,000 transactions/sec and stores 2TB of data daily. Should you use Lambda architecture, Kappa architecture, or a hybrid approach? Justify your decision with specific technology choices and discuss the trade-offs.',
      sampleAnswer: `This is a classic scenario requiring careful architecture design to balance real-time, batch, and ML workloads. Let me analyze each approach:

**Requirements Analysis:**

1. **Real-time fraud detection**: <100ms latency, critical path
2. **Daily sales reports**: Batch, accuracy critical, T+1 acceptable  
3. **ML training**: Batch, compute-intensive, weekly cadence
4. **Scale**: 10k TPS = 864M transactions/day, 2TB/day, 60TB/month

**Option A: Lambda Architecture**

\`\`\`
Batch Layer:          Speed Layer:           Serving Layer:
Kafka → S3         Kafka → Flink       Druid + Snowflake
  ↓                    ↓                      ↓
Spark (nightly)   Real-time views      Combined queries
  ↓                    ↓
Snowflake        Redis/Druid
(complete)       (last 24h)
\`\`\`

**Implementation:**

**Speed Layer** (Real-time fraud detection):
\`\`\`java
// Flink job: Real-time fraud detection
DataStream<Transaction> transactions = kafka.source("transactions");

transactions
  .keyBy(t -> t.userId)
  .process(new FraudDetectionFunction())  // Stateful processing
  .sinkTo(redis);  // <100ms end-to-end

class FraudDetectionFunction extends KeyedProcessFunction<String, Transaction, Alert> {
    // Check velocity, location, amount patterns
    // Maintain state per user
    // Emit alerts for suspicious transactions
}
\`\`\`

**Batch Layer** (Historical analytics + ML training):
\`\`\`python
# Airflow DAG: Daily batch processing
@dag(schedule_interval='0 2 * * *')
def daily_analytics():
    # Extract from Kafka → S3 (raw data lake)
    extract = extract_kafka_to_s3(lookback_days=1)
    
    # Transform with Spark
    transform = spark_transform_sales(extract)
    
    # Load to Snowflake (data warehouse)
    load = load_to_snowflake(transform, table='sales_daily')
    
    # ML feature engineering
    features = compute_ml_features(lookback_days=180)
    
    extract >> transform >> load >> features

# Weekly ML training
@dag(schedule_interval='0 3 * * 0')  # Sunday 3 AM
def weekly_ml_training():
    train_fraud_model(data_days=180)  # 6 months
\`\`\`

**Serving Layer**:
\`\`\`python
def get_user_transaction_summary(user_id, days=30):
    # Recent data from speed layer (Redis/Druid)
    recent = redis.get(f"user:{user_id}:last_24h")
    
    # Historical data from batch layer (Snowflake)
    historical = snowflake.query(f"""
        SELECT * FROM sales_daily 
        WHERE user_id = {user_id} 
        AND date BETWEEN CURRENT_DATE - {days} AND CURRENT_DATE - 1
    """)
    
    # Merge results
    return merge(recent, historical)
\`\`\`

**Pros:**
- ✅ Real-time fraud detection (Flink <50ms)
- ✅ Accurate historical analytics (Spark + Snowflake)
- ✅ Separate ML training (doesn't impact real-time)
- ✅ Can reprocess batch layer if errors

**Cons:**
- ❌ Duplicate logic (Flink for real-time, Spark for batch)
- ❌ Operational complexity (2 processing systems)
- ❌ Reconciliation needed (speed vs batch views)
- ❌ Higher cost (both systems running)

**Option B: Kappa Architecture**

\`\`\`
                Kafka (all data)
                     ↓
          ┌──────────┴──────────┐
          ↓                     ↓
    Flink (real-time)    Flink (batch replay)
          ↓                     ↓
    Druid (live)          Snowflake (historical)
\`\`\`

**Implementation:**

Single Flink codebase for both:
\`\`\`java
// Same Flink code for both real-time and batch
public class UnifiedTransactionProcessor extends KeyedProcessFunction<String, Transaction, Output> {
    
    @Override
    public void processElement(Transaction txn, Context ctx, Collector<Output> out) {
        // Fraud detection (real-time)
        if (isFraudulent(txn)) {
            out.collect(new Alert(txn));
        }
        
        // Analytics aggregation (batch)
        updateAggregates(txn);
        
        // ML features (batch)
        updateFeatures(txn);
    }
}

// Deployment 1: Real-time (tail of Kafka)
FlinkJob realtime = new FlinkJob(source=kafka.tail(), sink=druid);

// Deployment 2: Batch (replay from beginning)
FlinkJob batch = new FlinkJob(source=kafka.replay(days=180), sink=snowflake);
\`\`\`

**Pros:**
- ✅ Single codebase (no duplication!)
- ✅ Easy reprocessing (replay Kafka)
- ✅ Simpler operations (one system)
- ✅ Consistent logic everywhere

**Cons:**
- ❌ Kafka retention (must keep 6 months = 360TB!)
- ❌ Flink must handle both real-time and batch scale
- ❌ Batch replay may slow down real-time
- ❌ Less mature for batch processing than Spark

**My Recommendation: Hybrid Architecture**

\`\`\`
┌─────────────────────────────────────────┐
│              Kafka (7 days)              │
│           (Real-time + buffer)           │
└────────┬────────────────┬────────────────┘
         ↓                ↓
    ┌────────┐      ┌──────────┐
    │ Flink  │      │   S3     │
    │(RT)    │      │(Archive) │
    └───┬────┘      └────┬─────┘
        ↓                ↓
    ┌────────┐      ┌──────────┐
    │ Druid  │      │  Spark   │
    │(Live)  │      │(Batch)   │
    └────────┘      └────┬─────┘
                         ↓
                    ┌──────────┐
                    │Snowflake │
                    │(DW + ML) │
                    └──────────┘
\`\`\`

**Why Hybrid:**

1. **Real-time** (Flink + Druid):
   - Fraud detection: <100ms ✅
   - Live dashboards
   - 7-day retention in Kafka (manageable: ~14TB)

2. **Batch** (Spark + Snowflake):
   - Daily reports
   - ML training
   - Historical analytics
   - Read from S3 (cheap long-term storage)

3. **Separation of Concerns**:
   - Flink optimized for real-time
   - Spark optimized for batch
   - Use right tool for right job

**Detailed Implementation:**

**1. Data Ingestion:**
\`\`\`python
# Producer: Write all transactions to Kafka
producer.send('transactions', transaction_json)

# Consumer 1: Flink (real-time)
# Consumer 2: S3 Sink (archival) via Kafka Connect
kafka_connect.s3_sink(
    topic='transactions',
    s3_path='s3://data-lake/transactions/{year}/{month}/{day}/',
    format='parquet'
)
\`\`\`

**2. Real-time Layer:**
\`\`\`java
// Flink: Real-time fraud detection + live metrics
env.addSource(kafka)
  .keyBy(t -> t.userId)
  .process(new FraudDetector())  // Stateful processing
  .addSink(druid);  // Query in <100ms

// Druid: Real-time analytics (last 7 days)
// Automatically archives to deep storage after 7 days
\`\`\`

**3. Batch Layer:**
\`\`\`python
# Airflow: Daily batch job
@dag(schedule_interval='0 2 * * *')
def daily_batch():
    # Spark reads from S3 (not Kafka)
    df = spark.read.parquet('s3://data-lake/transactions/2024/01/15/')
    
    # Transform
    sales = df.groupBy('product_id').agg(
        sum('amount').alias('daily_sales'),
        count('*').alias('transaction_count')
    )
    
    # Load to Snowflake
    sales.write.format('snowflake').save('sales_daily')

# Weekly ML training
@dag(schedule_interval='0 3 * * 0')
def weekly_ml():
    # Load 6 months from Snowflake (fast!)
    df = snowflake.query("SELECT * FROM transactions WHERE date >= DATEADD(month, -6, CURRENT_DATE)")
    
    # Train model
    model = train_fraud_model(df)
    model.save('s3://models/fraud_v2024_01_15')
\`\`\`

**4. Serving Layer:**
\`\`\`python
# API: Smart query routing
def get_transaction_analytics(user_id, start_date, end_date):
    days_ago = (today - end_date).days
    
    if days_ago <= 7:
        # Recent: Query Druid (fast!)
        return druid.query(f"""
            SELECT SUM(amount) as total
            FROM transactions
            WHERE user_id = {user_id}
            AND __time >= '{start_date}'
        """)
    else:
        # Historical: Query Snowflake
        return snowflake.query(f"""
            SELECT SUM(amount) as total
            FROM transactions
            WHERE user_id = {user_id}
            AND date BETWEEN '{start_date}' AND '{end_date}'
        """)
\`\`\`

**Technology Stack:**

- **Message Queue**: Kafka (7-day retention, ~14TB)
- **Stream Processing**: Apache Flink (real-time fraud)
- **Batch Processing**: Apache Spark (historical analytics)
- **Real-time Store**: Apache Druid (live dashboards, <100ms queries)
- **Data Warehouse**: Snowflake (reports, ML training)
- **Data Lake**: S3 (long-term archival, Parquet format)
- **Orchestration**: Apache Airflow (batch jobs)
- **Monitoring**: Datadog, Grafana

**Cost Analysis:**

**Lambda (full)**: $25k/month
- Kafka (7 days): $3k
- Flink: $5k
- Spark: $8k
- Druid: $4k
- Snowflake: $5k

**Kappa (stream-only)**: $30k/month
- Kafka (6 months): $15k ⚠️ (expensive!)
- Flink (larger): $10k
- Druid: $5k

**Hybrid (recommended)**: $22k/month
- Kafka (7 days): $3k
- Flink: $4k
- Spark: $6k
- Druid: $3k
- Snowflake: $4k
- S3: $2k

**Trade-offs:**

| Aspect | Lambda | Kappa | Hybrid (Recommended) |
|--------|--------|-------|---------------------|
| Complexity | High | Medium | Medium-High |
| Cost | Medium | High | Low-Medium |
| Real-time | ✅ | ✅ | ✅ |
| Batch | ✅ | ⚠️ | ✅ |
| Reprocessing | Easy | Easy | Medium |
| Code Duplication | Yes | No | Some |
| Operations | Complex | Simple | Medium |

**Decision: Hybrid Architecture**

**Why:**
1. **Performance**: Flink for <100ms fraud detection ✅
2. **Cost**: 7-day Kafka retention vs 6-month (save $12k/month)
3. **Scalability**: Spark better for large batch ML jobs
4. **Pragmatic**: Use best tool for each job
5. **Proven**: Most companies use this pattern at scale

**Monitoring:**

\`\`\`python
metrics = {
    'fraud_detection_latency_p99': 85ms,  # < 100ms target ✅
    'batch_job_duration': 45min,  # Daily job
    'ml_training_duration': 3hr,  # Weekly
    'kafka_lag': 0,  # Real-time processing keeping up
    'data_quality_score': 99.9%
}
\`\`\`

This hybrid approach provides the performance, cost-effectiveness, and operational simplicity needed for the described requirements.`,
      keyPoints: [
        'Lambda architecture duplicates logic but uses optimal tools for each workload',
        'Kappa architecture simplifies code but requires expensive long-term Kafka retention',
        'Hybrid approach: stream processing (Flink) for real-time + batch processing (Spark) for historical',
        'Cost consideration: 7-day Kafka retention vs 6-month retention saves significant money',
        'Separate data stores: Druid for real-time queries (<100ms), Snowflake for batch analytics',
        'S3 data lake for cheap long-term storage, avoiding expensive Kafka retention',
      ],
    },
    {
      id: 'adp-discussion-2',
      type: 'discussion',
      question:
        'Your data team is debating between ETL and ELT for ingesting e-commerce data into your analytics warehouse. The CTO argues for ETL (transform before loading) to ensure data quality and reduce storage costs, while the Data Science lead argues for ELT (load raw data first) to enable flexibility and exploration. The system ingests from 50+ sources including databases, APIs, and third-party services. What would you recommend and why? Discuss the technical implications, costs, and organizational considerations.',
      sampleAnswer: `This is a strategic architecture decision with far-reaching implications. Let me analyze both approaches comprehensively:

**ETL (Extract, Transform, Load) - CTO's Argument**

\`\`\`
Sources → Extract → Transform (Spark/Airflow) → Load → Warehouse (clean)
                         ↓
                   Staging (S3)
\`\`\`

**ETL Implementation:**
\`\`\`python
# Airflow DAG: ETL Pipeline
@dag(schedule_interval='0 2 * * *')
def etl_user_data():
    # Extract from PostgreSQL
    raw_users = extract_postgres("SELECT * FROM users WHERE updated_at >= CURRENT_DATE")
    
    # Transform: Clean, validate, enrich
    cleaned = transform_users(raw_users)
    # - PII masking (email → hashed)
    # - Data validation (age 18-100)
    # - Denormalization (join with accounts table)
    # - Aggregation (user lifetime value)
    
    # Load only transformed data
    load_to_redshift(cleaned, table='analytics.users')
    
def transform_users(raw):
    df = pd.DataFrame(raw)
    
    # Data quality
    df = df[df['age'].between(18, 100)]
    df = df[df['email'].str.contains('@')]
    
    # PII handling
    df['email_hash'] = df['email'].apply(hash_email)
    df = df.drop('email', axis=1)
    
    # Enrichment
    df['country'] = df['ip'].apply(geolocate)
    df['lifetime_value'] = df['user_id'].apply(calculate_ltv)
    
    return df[['user_id', 'email_hash', 'age', 'country', 'lifetime_value']]  # Only needed columns
\`\`\`

**CTO's Arguments (ETL Pros):**

1. **Data Quality Before Storage:**
   - Invalid data rejected upfront
   - Consistent transformations applied
   - No bad data in warehouse

2. **Storage Efficiency:**
   - Only transformed data stored (smaller)
   - Example: 1TB raw → 200GB transformed
   - Warehouse costs: $50/TB/month → Save $40k/month

3. **PII/Compliance:**
   - Sensitive data never touches warehouse
   - GDPR/CCPA compliant by design
   - Email hashed before storage

4. **Performance:**
   - Queries run on pre-aggregated data
   - No transformation overhead at query time
   - Predictable query performance

5. **Traditional/Proven:**
   - Decades of best practices
   - Well-understood patterns
   - Less risk

**ETL Cons:**

1. **Transformation Rigidity:**
\`\`\`python
# Problem: New analysis needs different transformation
# Original: Aggregated daily sales per user
load_to_warehouse(df.groupby(['user_id', 'date']).agg({'amount': 'sum'}))

# New need: Hourly sales per user per product
# IMPOSSIBLE: Raw transaction data was thrown away!
# Must reprocess from source (slow, may not be available)
\`\`\`

2. **Cannot Answer New Questions:**
   - Raw data discarded
   - Can't reprocess with new logic
   - Limited to predefined transformations

3. **Slow Time to Value:**
   - Must define transformations upfront
   - Complex transformation logic
   - Long development cycles

4. **Debugging Difficulty:**
   - Can't inspect raw data
   - Hard to trace issues back to source
   - No audit trail

**ELT (Extract, Load, Transform) - Data Science Lead's Argument**

\`\`\`
Sources → Extract → Load (raw) → Warehouse → Transform (SQL/dbt)
                         ↓           ↓               ↓
                       None      Raw tables    Transformed views
\`\`\`

**ELT Implementation:**
\`\`\`sql
-- Step 1: Load raw data (fast, no transformation)
COPY raw.users 
FROM 's3://bucket/users_20240115.csv'
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftLoadRole';

-- Step 2: Transform in warehouse using dbt
-- models/staging/stg_users.sql
{{
  config(
    materialized='view'
  )
}}

SELECT
  user_id,
  SHA2(email, 256) as email_hash,
  age,
  CASE
    WHEN age BETWEEN 18 AND 100 THEN age
    ELSE NULL
  END as validated_age,
  country,
  created_at
FROM {{ source('raw', 'users') }}
WHERE age BETWEEN 18 AND 100
  AND email LIKE '%@%'

-- models/marts/mart_user_analytics.sql
{{
  config(
    materialized='table'
  )
}}

SELECT
  u.user_id,
  u.email_hash,
  u.country,
  SUM(o.amount) as lifetime_value,
  COUNT(o.order_id) as order_count,
  MAX(o.created_at) as last_order_date
FROM {{ ref('stg_users') }} u
LEFT JOIN {{ ref('stg_orders') }} o ON u.user_id = o.user_id
GROUP BY u.user_id, u.email_hash, u.country
\`\`\`

**Data Science Lead's Arguments (ELT Pros):**

1. **Flexibility:**
\`\`\`sql
-- New analysis? Just write SQL!
-- Hourly sales per product
CREATE VIEW hourly_product_sales AS
SELECT 
  DATE_TRUNC('hour', created_at) as hour,
  product_id,
  SUM(amount) as sales
FROM raw.transactions  -- Raw data available!
GROUP BY 1, 2;
\`\`\`

2. **Exploration:**
   - Data scientists query raw data
   - Discover new patterns
   - Prototype quickly

3. **Reprocessing:**
\`\`\`sql
-- Found bug in transformation? Just rerun!
DROP VIEW mart_user_analytics;
CREATE VIEW mart_user_analytics AS
  SELECT ... -- New logic
  FROM raw.users;  -- Raw data still there
\`\`\`

4. **Fast Time to Value:**
   - Load data immediately
   - Transform as needed
   - Iterative development

5. **Audit Trail:**
   - Raw data preserved
   - Can trace transformations
   - Debugging easier

**ELT Cons:**

1. **Storage Costs:**
\`\`\`
Raw data: 1TB/day × 365 days = 365TB
Transformed: 200GB/day × 365 days = 73TB
Total: 438TB vs 73TB (ETL)

Cost: 438TB × $50/TB/month = $21,900/month
ETL cost: 73TB × $50/TB/month = $3,650/month
Difference: $18,250/month extra!
\`\`\`

2. **PII in Warehouse:**
   - Raw data contains emails, phone numbers
   - Must implement access controls
   - Compliance risk if breached

3. **Query Performance:**
   - Transformations at query time
   - Can be slow for complex logic
   - Unpredictable performance

4. **Requires Powerful Warehouse:**
   - Needs compute to transform
   - Not all warehouses support complex transforms
   - Requires Snowflake/BigQuery

**My Recommendation: Hybrid EL(T) with Layered Architecture**

\`\`\`
Sources → Extract → Load → Data Warehouse (Snowflake/BigQuery)
                     ↓
            ┌────────┴────────┬────────────┬──────────┐
            ↓                 ↓            ↓          ↓
        Raw Layer       Staging Layer  Mart Layer  Metrics
       (7-30 days)      (cleaned)      (business)  (aggregated)
            ↓                 ↓            ↓          ↓
        Minimal          dbt models   dbt models   dbt models
        access           (views)      (tables)     (tables)
\`\`\`

**Implementation:**

\`\`\`sql
-- Layer 1: RAW (7-30 day retention)
-- Load all data, minimal transformation
CREATE SCHEMA raw;
COPY raw.users FROM 's3://...';  -- Raw load
ALTER TABLE raw.users SET DATA_RETENTION_TIME_IN_DAYS = 7;  -- Auto-delete after 7 days

-- Layer 2: STAGING (cleaned, validated)
-- dbt models
CREATE VIEW staging.users AS
SELECT
  user_id,
  CASE WHEN email LIKE '%@%' THEN SHA2(email, 256) END as email_hash,  -- PII handled here
  CASE WHEN age BETWEEN 18 AND 100 THEN age END as age,
  country,
  created_at
FROM raw.users
WHERE email IS NOT NULL;

-- Layer 3: MARTS (business logic)
CREATE TABLE mart.user_analytics AS
SELECT
  u.user_id,
  u.country,
  SUM(o.amount) as ltv,
  COUNT(o.order_id) as orders
FROM staging.users u
LEFT JOIN staging.orders o ON u.user_id = o.user_id
GROUP BY 1, 2;

-- Layer 4: METRICS (aggregated, fast)
CREATE TABLE metrics.daily_sales AS
SELECT
  DATE(created_at) as date,
  country,
  SUM(amount) as sales
FROM mart.user_analytics
GROUP BY 1, 2;
\`\`\`

**Benefits of Hybrid:**

1. **Flexibility + Cost Control:**
   - Raw data for 7-30 days (exploration)
   - Auto-delete old raw data (cost control)
   - Staging/marts retained long-term

2. **PII Handled Early:**
   - PII masked in staging layer
   - Analysts query staging/marts (no PII)
   - Raw layer restricted access

3. **Performance:**
   - Materialized marts for common queries (fast)
   - Raw data for ad-hoc analysis
   - Best of both worlds

4. **Governance:**
\`\`\`sql
-- Access controls
GRANT SELECT ON raw.* TO ROLE data_engineer;  -- Only engineers
GRANT SELECT ON staging.* TO ROLE analyst;     -- Analysts
GRANT SELECT ON mart.* TO ROLE all_users;      -- Everyone
\`\`\`

**Cost Analysis:**

\`\`\`
Storage breakdown (per month):
- Raw (30 days): 30TB × $23/TB = $690
- Staging (1 year): 73TB × $23/TB = $1,679
- Marts (1 year): 50TB × $23/TB = $1,150
- Metrics (1 year): 10TB × $23/TB = $230

Total: $3,749/month (vs $18,250 full ELT, vs $3,650 ETL)

Savings: 80% cheaper than full ELT, similar to ETL!
\`\`\`

**Organizational Considerations:**

**Team Skills:**
- ELT requires SQL expertise (dbt)
- ETL requires Python/Spark expertise
- **Hybrid**: Both, but SQL for most work

**Development Speed:**
- ETL: 2-4 weeks per pipeline
- ELT: 2-3 days per model
- **Hybrid**: Fast iteration (ELT), controlled costs

**Compliance:**
- ETL: PII never in warehouse (best)
- ELT: PII in warehouse (risky)
- **Hybrid**: PII in raw layer (restricted), masked in staging (safe)

**Data Quality:**
- ETL: Enforced before load
- ELT: Enforced via dbt tests
- **Hybrid**: Tests in staging layer

\`\`\`yaml
# dbt tests
models:
  - name: staging_users
    columns:
      - name: email_hash
        tests:
          - not_null
          - unique
      - name: age
        tests:
          - accepted_values:
              values: [18, 19, ..., 100]
\`\`\`

**Technology Stack:**

- **Warehouse**: Snowflake or BigQuery (ELT-capable)
- **Orchestration**: Airflow (loading) + dbt Cloud (transformations)
- **Data Lake**: S3 (backup, long-term archival)
- **Catalog**: dbt docs + Snowflake metadata

**Decision Matrix:**

| Factor | ETL | ELT | Hybrid |
|--------|-----|-----|--------|
| Storage Cost | $$$ Low | $$$$$ High | $$$ Low |
| Flexibility | ❌ Low | ✅ High | ✅ High |
| Development Speed | 🐌 Slow | 🚀 Fast | 🚀 Fast |
| PII Handling | ✅ Best | ❌ Risky | ⚠️ Controlled |
| Query Performance | ✅ Fast | ⚠️ Variable | ✅ Fast (marts) |
| Reprocessing | ❌ Hard | ✅ Easy | ✅ Easy |
| **Recommendation** | Old school | Data science | **Best for most** |

**Final Recommendation: Hybrid EL(T) with Layered Architecture**

**Why:**
1. Flexibility of ELT (raw data for exploration)
2. Cost control of ETL (auto-delete old raw data)
3. Performance of ETL (materialized marts)
4. PII handling (masked in staging, restricted raw access)
5. Industry best practice (dbt + modern warehouse)

This approach gives data scientists the flexibility they need while addressing the CTO's concerns about cost and data quality. It's become the de facto standard at data-driven companies.`,
      keyPoints: [
        'ETL optimizes for storage cost and data quality but sacrifices flexibility and reprocessing',
        'ELT enables exploration and flexibility but dramatically increases storage costs',
        'Hybrid approach: short-term raw data retention (7-30 days) + long-term transformed data',
        'Layered architecture: Raw → Staging → Marts → Metrics with appropriate retention policies',
        'PII handled in staging layer with restricted access to raw layer for compliance',
        'Modern warehouses (Snowflake, BigQuery) make ELT viable with compute-storage separation',
      ],
    },
    {
      id: 'adp-discussion-3',
      type: 'discussion',
      question:
        'Your analytics pipeline has been running smoothly for 6 months, processing 500GB daily. Suddenly, data quality issues emerge: duplicate records, missing timestamps, and incorrect aggregations. The executive dashboard shows sales dropped 40% overnight, triggering panic. Your CEO asks: "How did this happen, and how do we prevent it?" Design a comprehensive data quality framework including validation, monitoring, alerting, and incident response. Discuss specific tools, metrics, and processes.',
      sampleAnswer: `This scenario represents a critical failure in data quality management. Let me design a comprehensive framework to prevent and quickly detect such issues:

**Root Cause Analysis (What Went Wrong):**

Likely causes:
1. **Schema change**: Source system changed field names/types
2. **Upstream bug**: Application bug created duplicates/nulls
3. **Pipeline bug**: Transformation logic error
4. **Infrastructure issue**: Partial data load, network timeout

**Without proper data quality checks, issues propagate undetected!**

**Comprehensive Data Quality Framework:**

**Layer 1: Source Data Validation (At Ingestion)**

\`\`\`python
# Airflow DAG: Data quality checks at ingestion
from great_expectations import DataContext

@task
def validate_raw_data(date_partition):
    # Load raw data
    df = load_from_source(f"sales_{date_partition}")
    
    # Great Expectations suite
    context = DataContext()
    
    suite = context.get_expectation_suite("sales_raw")
    results = context.run_validation_operator(
        "action_list_operator",
        assets_to_validate=[df],
        run_name=f"sales_{date_partition}"
    )
    
    if not results['success']:
        raise DataQualityException(f"Validation failed: {results}")
    
    return df

# Define expectations
class SalesExpectations:
    def __init__(self, df):
        # Row count expectations
        df.expect_table_row_count_to_be_between(
            min_value=10000,  # At least 10k sales/day
            max_value=1000000  # Max 1M sales/day
        )
        
        # Column expectations
        df.expect_column_values_to_not_be_null("transaction_id")
        df.expect_column_values_to_be_unique("transaction_id")
        df.expect_column_values_to_not_be_null("timestamp")
        
        # Timestamp validity
        df.expect_column_values_to_be_between(
            "timestamp",
            min_value=f"{date_partition} 00:00:00",
            max_value=f"{date_partition} 23:59:59"
        )
        
        # Amount validity
        df.expect_column_values_to_be_between(
            "amount",
            min_value=0,
            max_value=100000  # No transaction > $100k
        )
        
        # Schema stability
        df.expect_table_columns_to_match_ordered_list([
            "transaction_id", "user_id", "amount", "timestamp", "product_id"
        ])
        
        # Statistical expectations
        df.expect_column_mean_to_be_between(
            "amount",
            min_value=50,   # Average order $50-200
            max_value=200
        )
\`\`\`

**Layer 2: Transformation Validation**

\`\`\`sql
-- dbt tests (SQL-based validation)

-- models/staging/stg_sales.sql
{{
  config(
    materialized='table'
  )
}}

SELECT
  transaction_id,
  user_id,
  amount,
  timestamp,
  product_id
FROM {{ source('raw', 'sales') }}

-- tests/stg_sales.yml
version: 2

models:
  - name: stg_sales
    description: "Staging sales data"
    tests:
      - dbt_utils.recency:
          datepart: day
          field: timestamp
          interval: 1  # Data must be within 1 day
      - dbt_expectations.expect_table_row_count_to_be_between:
          min_value: 10000
          max_value: 1000000
    
    columns:
      - name: transaction_id
        tests:
          - unique
          - not_null
      
      - name: amount
        tests:
          - not_null
          - dbt_expectations.expect_column_values_to_be_between:
              min_value: 0
              max_value: 100000
      
      - name: timestamp
        tests:
          - not_null
          - dbt_expectations.expect_column_values_to_be_of_type:
              column_type: timestamp

-- Custom test: Check for duplicates
-- tests/generic/test_no_duplicates_by_date.sql
{% test no_duplicates_by_date(model, column_name, date_column) %}

WITH duplicates AS (
  SELECT
    {{ column_name }},
    {{ date_column }},
    COUNT(*) as count
  FROM {{ model }}
  GROUP BY {{ column_name }}, {{ date_column }}
  HAVING COUNT(*) > 1
)

SELECT * FROM duplicates

{% endtest %}
\`\`\`

**Layer 3: Anomaly Detection**

\`\`\`python
# Airflow task: Statistical anomaly detection
from scipy import stats
import numpy as np

@task
def detect_anomalies(date_partition):
    # Get historical baseline (30 days)
    historical = get_metrics(days=30, end_date=date_partition - 1)
    current = get_metrics(days=1, date=date_partition)
    
    anomalies = []
    
    # Check row count
    if current['row_count'] < historical['row_count_p5']:  # Below 5th percentile
        anomalies.append({
            'metric': 'row_count',
            'expected': historical['row_count_mean'],
            'actual': current['row_count'],
            'severity': 'HIGH'
        })
    
    # Check average amount
    if not (historical['amount_mean'] - 3*historical['amount_std'] 
            < current['amount_mean'] 
            < historical['amount_mean'] + 3*historical['amount_std']):
        anomalies.append({
            'metric': 'average_amount',
            'expected': historical['amount_mean'],
            'actual': current['amount_mean'],
            'severity': 'MEDIUM'
        })
    
    # Check null percentage
    if current['null_pct'] > historical['null_pct'] * 1.5:
        anomalies.append({
            'metric': 'null_percentage',
            'expected': historical['null_pct'],
            'actual': current['null_pct'],
            'severity': 'HIGH'
        })
    
    if anomalies:
        send_alert(anomalies)
        raise AnomalyDetectedException(anomalies)
    
    return current
\`\`\`

**Layer 4: Business Logic Validation**

\`\`\`sql
-- Materialized view: Daily metrics for monitoring
CREATE MATERIALIZED VIEW data_quality_metrics AS
SELECT
  DATE(timestamp) as date,
  COUNT(*) as total_records,
  COUNT(DISTINCT transaction_id) as unique_transactions,
  COUNT(*) - COUNT(DISTINCT transaction_id) as duplicate_count,
  SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) as null_amounts,
  SUM(CASE WHEN timestamp IS NULL THEN 1 ELSE 0 END) as null_timestamps,
  AVG(amount) as avg_amount,
  STDDEV(amount) as stddev_amount,
  MIN(timestamp) as min_timestamp,
  MAX(timestamp) as max_timestamp,
  COUNT(DISTINCT user_id) as unique_users,
  SUM(amount) as total_sales
FROM staging.sales
GROUP BY DATE(timestamp);

-- Automated quality check query
WITH 
  today AS (
    SELECT * FROM data_quality_metrics WHERE date = CURRENT_DATE
  ),
  yesterday AS (
    SELECT * FROM data_quality_metrics WHERE date = CURRENT_DATE - 1
  ),
  avg_30d AS (
    SELECT
      AVG(total_records) as avg_records,
      AVG(total_sales) as avg_sales,
      STDDEV(total_sales) as stddev_sales
    FROM data_quality_metrics
    WHERE date >= CURRENT_DATE - 30
  )

SELECT
  -- Row count drop > 20%
  CASE 
    WHEN t.total_records < y.total_records * 0.8 
    THEN 'ALERT: Row count dropped 20%+' 
  END as row_count_alert,
  
  -- Sales anomaly (> 3 std devs from mean)
  CASE
    WHEN ABS(t.total_sales - a.avg_sales) > 3 * a.stddev_sales
    THEN 'ALERT: Sales anomaly detected'
  END as sales_alert,
  
  -- Duplicates
  CASE
    WHEN t.duplicate_count > 0
    THEN 'ALERT: Duplicates detected'
  END as duplicate_alert,
  
  -- Missing timestamps
  CASE
    WHEN t.null_timestamps > t.total_records * 0.01
    THEN 'ALERT: >1% missing timestamps'
  END as timestamp_alert

FROM today t
CROSS JOIN yesterday y
CROSS JOIN avg_30d a;
\`\`\`

**Layer 5: End-to-End Validation**

\`\`\`python
# Reconciliation test: Source vs Destination
@task
def reconcile_data(date_partition):
    # Count in source
    source_count = count_in_source_db(f"WHERE DATE(created_at) = '{date_partition}'")
    
    # Count in warehouse
    warehouse_count = count_in_warehouse(f"WHERE DATE(timestamp) = '{date_partition}'")
    
    # Amounts match
    source_sum = sum_in_source_db(f"WHERE DATE(created_at) = '{date_partition}'")
    warehouse_sum = sum_in_warehouse(f"WHERE DATE(timestamp) = '{date_partition}'")
    
    # Tolerance: 0.1% difference allowed
    if abs(source_count - warehouse_count) > source_count * 0.001:
        raise ReconciliationError(
            f"Count mismatch: source={source_count}, warehouse={warehouse_count}"
        )
    
    if abs(source_sum - warehouse_sum) > source_sum * 0.001:
        raise ReconciliationError(
            f"Sum mismatch: source={source_sum}, warehouse={warehouse_sum}"
        )
\`\`\`

**Monitoring & Alerting:**

\`\`\`python
# Datadog monitoring
from datadog import statsd

# Emit metrics
statsd.gauge('pipeline.row_count', row_count, tags=['pipeline:sales'])
statsd.gauge('pipeline.null_percentage', null_pct, tags=['pipeline:sales'])
statsd.gauge('pipeline.duplicate_count', dup_count, tags=['pipeline:sales'])
statsd.histogram('pipeline.amount', amount, tags=['pipeline:sales'])
statsd.gauge('pipeline.total_sales', total_sales, tags=['pipeline:sales'])

# Alerts in Datadog
alerts = [
    {
        'name': 'Sales Pipeline - Row Count Drop',
        'query': 'avg(last_1h):avg:pipeline.row_count{pipeline:sales} < 10000',
        'message': '@slack-data-eng @pagerduty Row count dropped below threshold',
        'priority': 'P1'
    },
    {
        'name': 'Sales Pipeline - High Null Rate',
        'query': 'avg(last_1h):avg:pipeline.null_percentage{pipeline:sales} > 0.01',
        'message': '@slack-data-eng >1% null values detected',
        'priority': 'P2'
    },
    {
        'name': 'Sales Pipeline - Duplicates Detected',
        'query': 'avg(last_1h):avg:pipeline.duplicate_count{pipeline:sales} > 0',
        'message': '@slack-data-eng @oncall Duplicates in sales data!',
        'priority': 'P1'
    },
    {
        'name': 'Sales Pipeline - Total Sales Anomaly',
        'query': 'anomalies(avg:pipeline.total_sales{pipeline:sales}, "agile", 3)',
        'message': '@slack-executives @data-eng Sales anomaly detected!',
        'priority': 'P0'
    }
]
\`\`\`

**Dashboard: Real-Time Data Quality**

\`\`\`yaml
# Grafana/Superset Dashboard

Panels:
  1. "Pipeline Health Score"
     - Composite score: row count, null %, duplicates, schema stability
     - Green: 95-100%, Yellow: 80-95%, Red: <80%
  
  2. "Row Count Over Time"
     - Line chart: Daily row count with 30-day average band
     - Alerts visible on chart
  
  3. "Null Percentage Heatmap"
     - Per-column null % over time
     - Red if > threshold
  
  4. "Sales Trend"
     - Total sales with anomaly detection bands
     - Yesterday's 40% drop would be BRIGHT RED
  
  5. "Data Freshness"
     - Time since last successful load
     - Alert if > 4 hours
  
  6. "Test Results"
     - dbt test pass rate
     - Great Expectations validation results
\`\`\`

**Incident Response Playbook:**

\`\`\`markdown
# Data Quality Incident Response

## Severity Levels
- **P0**: Sales/revenue metrics wrong (dashboard shows 40% drop)
- **P1**: High null rate, duplicates, row count anomaly
- **P2**: Schema changes, minor anomalies

## Response Steps:

### 1. Detection (0-5 minutes)
- Automated alert fires
- On-call engineer paged (P0/P1)
- Slack notification (P2)

### 2. Assessment (5-15 minutes)
- Check data quality dashboard
- Query metrics table: \`SELECT * FROM data_quality_metrics WHERE date = CURRENT_DATE\`
- Compare to yesterday: What changed?
- Check source system: Is issue upstream?

### 3. Mitigation (15-60 minutes)
- **Stop pipeline**: Prevent bad data propagation
- **Rollback dashboard**: Show yesterday's data with banner
- **Notify stakeholders**: "Data quality issue, investigating"

### 4. Root Cause (1-4 hours)
- Schema change? Check source system changes
- Upstream bug? Check application logs
- Pipeline bug? Check Airflow logs, code changes
- Infrastructure? Check AWS/GCP alerts

### 5. Fix (4-24 hours)
- Fix root cause
- Reprocess data for affected date
- Validate fix with quality checks
- Update dashboard

### 6. Postmortem (1 week)
- What happened?
- Why did it happen?
- How did we detect it? (Too slow?)
- How do we prevent recurrence?
- Action items with owners
\`\`\`

**Prevention: Shift-Left Testing**

\`\`\`python
# Pre-merge pipeline testing (CI/CD)
# .github/workflows/data_quality.yml

name: Data Quality Tests

on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run dbt tests on sample data
        run: |
          dbt test --select staging
          dbt test --select marts
      
      - name: Run Great Expectations
        run: |
          great_expectations checkpoint run sales_validation
      
      - name: Check for schema changes
        run: |
          python scripts/detect_schema_drift.py
\`\`\`

**Metrics to Track:**

\`\`\`python
data_quality_kpis = {
    # Detection
    'mean_time_to_detect': 15,  # Minutes (target: < 30 min)
    'false_positive_rate': 0.05,  # 5% (target: < 10%)
    
    # Response
    'mean_time_to_acknowledge': 5,  # Minutes (target: < 10 min)
    'mean_time_to_mitigate': 30,  # Minutes (target: < 1 hour)
    'mean_time_to_resolve': 4,  # Hours (target: < 24 hours)
    
    # Quality
    'test_pass_rate': 0.998,  # 99.8% (target: > 99%)
    'data_sla_compliance': 0.999,  # 99.9% (target: 99.9%)
    'incidents_per_month': 2  # (target: < 5)
}
\`\`\`

**Technology Stack:**

- **Validation**: Great Expectations, dbt tests
- **Monitoring**: Datadog, Prometheus
- **Alerting**: PagerDuty, Slack
- **Dashboard**: Grafana, Superset
- **Orchestration**: Airflow (with quality checks)
- **Incident Management**: PagerDuty, Jira

**Cost-Benefit:**

\`\`\`
Cost of framework: $5k/month
- Great Expectations: $2k
- Datadog: $2k
- PagerDuty: $500
- Engineering time: $500

Cost of 40% sales drop incident:
- Executive panic: Priceless
- Lost trust: Months to recover
- Emergency response: 20 engineer-hours
- Opportunity cost: Wrong decisions based on bad data

ROI: Framework pays for itself after ONE prevented incident.
\`\`\`

**Summary:**

A comprehensive data quality framework requires:
1. **Validation**: At every layer (source, staging, marts)
2. **Anomaly detection**: Statistical monitoring
3. **Reconciliation**: Source-to-dest validation
4. **Real-time monitoring**: Dashboards and alerts
5. **Incident response**: Clear playbooks
6. **Prevention**: CI/CD testing, shift-left quality

The 40% sales drop should have been detected within minutes, not overnight. With this framework, such incidents become rare and quickly resolved.`,
      keyPoints: [
        'Multi-layer validation: source validation, transformation tests, anomaly detection, reconciliation',
        'Great Expectations and dbt tests provide declarative data quality checks at each pipeline stage',
        'Statistical anomaly detection catches issues that static thresholds miss (e.g., 40% sales drop)',
        'Real-time monitoring with alerts (PagerDuty, Datadog) enables quick detection (<30 min)',
        'Incident response playbook with severity levels ensures coordinated response',
        'Shift-left testing: validate data quality in CI/CD before deploying pipeline changes',
      ],
    },
  ],
};

export default analyticsDataPipelineDiscussionQuiz;
