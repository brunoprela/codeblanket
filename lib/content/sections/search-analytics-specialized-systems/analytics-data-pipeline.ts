import { ModuleSection } from '@/lib/types';

const analyticsDataPipelineSection: ModuleSection = {
  id: 'analytics-data-pipeline',
  title: 'Analytics Data Pipeline',
  content: `
# Analytics Data Pipeline

## Introduction

Modern organizations make data-driven decisions, and analytics data pipelines are the backbone that transforms raw data into actionable insights. Whether you're analyzing user behavior, monitoring system performance, or running business intelligence queries, you need a robust pipeline to ingest, process, transform, and serve data at scale.

This section covers the architectural patterns, technologies, and trade-offs involved in building analytics pipelines that can handle everything from batch processing of historical data to real-time stream analytics.

## What is an Analytics Data Pipeline?

An **analytics data pipeline** is a set of processes that:
1. **Ingests** data from various sources
2. **Transforms** and cleanses the data
3. **Stores** the data in queryable format
4. **Serves** the data to analytics tools and applications

**Key Requirements:**
- **Scalability**: Handle terabytes to petabytes of data
- **Reliability**: No data loss, consistent results
- **Performance**: Low latency for queries
- **Cost-effectiveness**: Balance storage and compute costs
- **Flexibility**: Support various data formats and sources

## Pipeline Architectures

### Batch Processing

**Description**: Process data in large,scheduled chunks (hourly, daily).

**Architecture:**
\`\`\`
Data Sources → Ingestion → Staging → Transformation → Data Warehouse
                  ↓           ↓            ↓              ↓
               S3/HDFS    Spark/Hadoop   dbt/Airflow   Redshift/BigQuery
\`\`\`

**Example Use Cases:**
- Daily sales reports
- Monthly financial analytics
- Historical trend analysis
- Model training on large datasets

**Pros:**
- Simple to implement and reason about
- Cost-effective (process once, use many times)
- Can handle very large datasets
- Well-established tooling

**Cons:**
- High latency (hours to days)
- Not suitable for real-time decisions
- Resource-intensive when running

**Technologies:**
- **Ingestion**: Apache Sqoop, AWS Data Pipeline, Fivetran
- **Processing**: Apache Spark, Hadoop MapReduce, AWS Glue
- **Orchestration**: Apache Airflow, Luigi, Dagster
- **Storage**: HDFS, S3, Azure Data Lake

### Stream Processing

**Description**: Process data continuously as it arrives (milliseconds to seconds).

**Architecture:**
\`\`\`
Data Sources → Message Queue → Stream Processing → Data Store → Real-time Dashboard
                     ↓               ↓                  ↓              ↓
                  Kafka          Flink/Storm      Redis/Druid    Grafana/Superset
\`\`\`

**Example Use Cases:**
- Real-time fraud detection
- Live dashboards and monitoring
- Personalized recommendations
- Alerting and anomaly detection

**Pros:**
- Low latency (milliseconds to seconds)
- Enables real-time decision making
- Continuous processing (no gaps)
- Event-driven architecture

**Cons:**
- More complex to implement
- Higher operational overhead
- Handling late data and out-of-order events
- More expensive (always running)

**Technologies:**
- **Message Queue**: Apache Kafka, AWS Kinesis, Pulsar
- **Processing**: Apache Flink, Spark Streaming, Storm, ksqlDB
- **Storage**: Apache Druid, ClickHouse, TimescaleDB

### Lambda Architecture

**Description**: Combine batch and stream processing for comprehensive analytics.

**Architecture:**
\`\`\`
                    Data Sources
                         ↓
            ┌────────────┴────────────┐
            ↓                         ↓
     Batch Layer                Stream Layer
    (Historical)              (Real-time)
            ↓                         ↓
   Batch Views                Speed Views
    (Complete)                (Approximate)
            └──────────┬──────────────┘
                       ↓
                 Serving Layer
                 (Query both)
\`\`\`

**Components:**

1. **Batch Layer**: Process complete historical data
   - High latency but accurate
   - Recompute views from scratch
   - Handles all data

2. **Speed Layer**: Process recent data in real-time
   - Low latency but approximate
   - Incremental updates
   - Only handles new data

3. **Serving Layer**: Merge batch and speed views
   - Query both layers
   - Recent data from speed layer
   - Historical data from batch layer

**Example:**
\`\`\`
Batch Layer (daily): Full user activity analysis over 2 years
Speed Layer (real-time): User activity in last 24 hours
Serving Layer: Query combines both for complete view
\`\`\`

**Pros:**
- Best of both worlds (completeness + freshness)
- Fault-tolerant (batch layer can recompute)
- Accurate historical data + real-time updates

**Cons:**
- High complexity (maintain two pipelines)
- Duplicate logic in batch and stream
- Higher operational cost
- Reconciling batch and stream results

**When to Use:**
- Need both real-time and historical analytics
- Can afford operational complexity
- Accuracy is critical (recomputation possible)

### Kappa Architecture

**Description**: Simplified Lambda using only stream processing.

**Architecture:**
\`\`\`
         Data Sources
              ↓
       Message Queue (immutable log)
              ↓
      Stream Processing
       (same code path)
              ↓
      Serving Layer
\`\`\`

**Key Idea**: Everything is a stream!
- Batch = replay of stream from beginning
- Real-time = processing stream tail
- Single code path for both

**Example:**
\`\`\`python
# Same code processes both historical and real-time
def process_event (event):
    user = extract_user (event)
    update_metrics (user)
    save_to_store (user)

# Historical: Replay Kafka from offset 0
# Real-time: Process Kafka tail
\`\`\`

**Pros:**
- Simpler than Lambda (one pipeline)
- Single codebase (no duplication)
- Easy to reprocess data (replay stream)
- Lower operational overhead

**Cons:**
- Message queue must retain all historical data
- Stream processing must handle batch-sized replays
- May be slower than specialized batch processing
- Requires mature stream processing (Kafka + Flink)

**When to Use:**
- Primarily real-time use cases
- Can retain full history in message queue
- Team experienced with stream processing
- Want to avoid code duplication

## Data Ingestion Patterns

### ETL (Extract, Transform, Load)

**Order**: Transform data BEFORE loading into warehouse.

\`\`\`
Source → Extract → Transform (Spark/Airflow) → Load → Warehouse
                         ↓
                   Staging Area
\`\`\`

**Process:**
1. **Extract**: Pull data from sources (APIs, databases, files)
2. **Transform**: Clean, filter, aggregate, join data
3. **Load**: Insert transformed data into warehouse

**Example:**
\`\`\`python
# ETL Pipeline
def etl_daily_sales():
    # Extract
    raw_sales = extract_from_db("SELECT * FROM sales WHERE date = '2024-01-15'")
    raw_products = extract_from_db("SELECT * FROM products")
    
    # Transform
    cleaned = clean_data (raw_sales)
    enriched = join_with_products (cleaned, raw_products)
    aggregated = aggregate_by_category (enriched)
    
    # Load
    load_to_warehouse (aggregated, table="sales_daily_summary")
\`\`\`

**Pros:**
- Data cleaned before storage (consistent quality)
- Only transformed data loaded (smaller storage)
- Privacy/compliance (sensitive data filtered)
- Traditional approach (well understood)

**Cons:**
- Transformation logic rigid (hard to change)
- Can't query raw data (if needed later)
- Slow for large datasets (transform before load)
- Transformation bottleneck

**When to Use:**
- Data quality critical
- Storage expensive
- Clear transformation requirements
- Traditional data warehouse model

### ELT (Extract, Load, Transform)

**Order**: Load raw data FIRST, transform IN the warehouse.

\`\`\`
Source → Extract → Load → Warehouse → Transform (SQL/dbt)
                     ↓                      ↓
                 Raw tables           Transformed views
\`\`\`

**Process:**
1. **Extract**: Pull data from sources
2. **Load**: Insert raw data into warehouse (fast!)
3. **Transform**: Use SQL/dbt to transform within warehouse

**Example:**
\`\`\`sql
-- ELT Pipeline (using dbt)

-- Step 1: Load raw data (fast, no transformation)
COPY raw.sales FROM 's3://bucket/sales_20240115.csv';
COPY raw.products FROM 's3://bucket/products.csv';

-- Step 2: Transform using SQL
CREATE VIEW analytics.sales_daily AS
SELECT 
  s.date,
  p.category,
  SUM(s.amount) as total_sales,
  COUNT(*) as transaction_count
FROM raw.sales s
JOIN raw.products p ON s.product_id = p.id
WHERE s.date = '2024-01-15'
GROUP BY s.date, p.category;
\`\`\`

**Pros:**
- Fast ingestion (no transformation)
- Flexible transformations (change anytime with SQL)
- Raw data available (audit, reprocess)
- Leverages warehouse compute power

**Cons:**
- Requires powerful warehouse (Snowflake, BigQuery)
- Raw data stored (higher storage costs)
- Transformation happens at query time (latency)
- Sensitive data loaded (must handle carefully)

**When to Use:**
- Modern cloud data warehouse
- Compute cheap, storage cheap
- Transformation requirements change often
- Need raw data for exploration

### Change Data Capture (CDC)

**Description**: Capture and stream database changes in real-time.

\`\`\`
Operational DB → CDC (Debezium/Maxwell) → Kafka → Analytics DB
   (OLTP)          (captures changes)     (stream)    (OLAP)
\`\`\`

**How it Works:**
- Monitor database transaction log (binlog, WAL)
- Capture INSERT, UPDATE, DELETE events
- Stream changes to message queue
- Apply changes to analytics store

**Example Event:**
\`\`\`json
{
  "operation": "UPDATE",
  "timestamp": "2024-01-15T10:30:00Z",
  "table": "users",
  "before": {
    "id": 123,
    "email": "old@example.com",
    "status": "active"
  },
  "after": {
    "id": 123,
    "email": "new@example.com",
    "status": "active"
  }
}
\`\`\`

**Pros:**
- Real-time data synchronization
- No impact on source database (reads transaction log)
- Captures all changes (no missed data)
- Enables event-driven architectures

**Cons:**
- Complex setup (database-specific)
- Requires access to transaction log
- Schema changes need handling
- Backfilling historical data challenging

**Technologies:**
- **Debezium**: Open-source CDC (MySQL, Postgres, MongoDB)
- **Maxwell**: MySQL-specific CDC
- **AWS DMS**: Managed CDC service
- **Airbyte**: CDC + data integration platform

## Data Lake vs Data Warehouse

### Data Lake

**Description**: Store raw data in native format (schema-on-read).

\`\`\`
Data Lake (S3, HDFS):
├── raw/
│   ├── logs/          (JSON)
│   ├── events/        (Avro)
│   ├── databases/     (Parquet)
│   └── files/         (CSV, images, videos)
├── processed/
│   └── analytics/
└── curated/
    └── models/
\`\`\`

**Characteristics:**
- **Schema-on-read**: Define schema when querying
- **Unstructured/semi-structured**: Any format
- **Cheap storage**: Object storage (S3)
- **Flexible**: Add data without planning
- **Exploration**: Data scientists analyze raw data

**Technologies:**
- **Storage**: AWS S3, Azure Data Lake, HDFS
- **Compute**: AWS Athena, Presto, Spark
- **Metadata**: AWS Glue Catalog, Apache Hive

**Use Cases:**
- Machine learning (raw features)
- Exploratory data analysis
- Long-term archival
- Unstructured data (logs, images)

### Data Warehouse

**Description**: Store structured data optimized for analytics (schema-on-write).

\`\`\`
Data Warehouse (Redshift, Snowflake):
├── Fact Tables
│   ├── sales_transactions
│   ├── user_events
│   └── inventory_changes
└── Dimension Tables
    ├── users
    ├── products
    └── time_dimensions
\`\`\`

**Characteristics:**
- **Schema-on-write**: Define schema when loading
- **Structured**: Tables, columns, types
- **Optimized for queries**: Columnar storage, indexes
- **Business intelligence**: SQL-based reporting
- **Governed**: Data quality, access control

**Technologies:**
- **Cloud**: AWS Redshift, Snowflake, Google BigQuery
- **On-premise**: Teradata, Oracle Exadata

**Use Cases:**
- Business intelligence dashboards
- SQL-based reporting
- Executive analytics
- Regulatory reporting

### Lake house (Best of Both)

**Description**: Combine flexibility of data lake with performance of warehouse.

\`\`\`
Data Lakehouse (Delta Lake, Apache Iceberg):
├── Raw data in S3 (data lake)
├── ACID transactions (data warehouse)
├── Schema enforcement (data warehouse)
└── Direct SQL queries (data warehouse)
\`\`\`

**Key Technologies:**
- **Delta Lake** (Databricks): ACID on data lake
- **Apache Iceberg**: Table format for data lakes
- **Apache Hudi**: Incremental data processing

**Benefits:**
- Single platform (reduce complexity)
- Flexible storage (data lake)
- Fast queries (data warehouse)
- ACID transactions
- Time travel (versioning)

## Data Pipeline Orchestration

### Apache Airflow

**Description**: Programmatic workflow orchestration using Python DAGs.

\`\`\`python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Define DAG
dag = DAG(
    'daily_sales_pipeline',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False
)

# Task 1: Extract data
def extract():
    # Pull from databases, APIs
    pass

# Task 2: Transform data
def transform():
    # Clean, join, aggregate
    pass

# Task 3: Load to warehouse
def load():
    # Insert into Redshift
    pass

# Task 4: Data quality checks
def validate():
    # Check row counts, nulls, etc.
    pass

# Define task dependencies
extract_task = PythonOperator (task_id='extract', python_callable=extract, dag=dag)
transform_task = PythonOperator (task_id='transform', python_callable=transform, dag=dag)
load_task = PythonOperator (task_id='load', python_callable=load, dag=dag)
validate_task = PythonOperator (task_id='validate', python_callable=validate, dag=dag)

extract_task >> transform_task >> load_task >> validate_task
\`\`\`

**Features:**
- Visual DAG interface
- Retry logic and alerting
- Backfilling historical runs
- Extensible (custom operators)
- Monitoring and logging

## Best Practices

### 1. Data Quality Checks

Always validate data at ingestion:

\`\`\`python
def validate_data (df):
    checks = {
        'row_count': len (df) > 0,
        'no_nulls_in_id': df['id'].isnull().sum() == 0,
        'email_format': df['email'].str.contains('@').all(),
        'date_range': (df['date'] >= '2024-01-01').all()
    }
    
    failures = [k for k, v in checks.items() if not v]
    if failures:
        raise ValueError (f"Data quality checks failed: {failures}")
\`\`\`

### 2. Idempotency

Pipelines should be rerunnable:

\`\`\`sql
-- Bad: INSERT without checking
INSERT INTO sales_daily SELECT * FROM raw_sales WHERE date = '2024-01-15';

-- Good: MERGE (upsert)
MERGE INTO sales_daily AS target
USING raw_sales AS source
ON target.date = source.date AND target.product_id = source.product_id
WHEN MATCHED THEN UPDATE SET target.amount = source.amount
WHEN NOT MATCHED THEN INSERT VALUES (source.date, source.product_id, source.amount);
\`\`\`

### 3. Partitioning

Partition data by time for efficient queries:

\`\`\`sql
-- Partition by date
CREATE TABLE events (
    event_id STRING,
    user_id STRING,
    timestamp TIMESTAMP,
    event_type STRING
)
PARTITIONED BY (date STRING);

-- Queries only scan relevant partitions
SELECT COUNT(*) 
FROM events 
WHERE date BETWEEN '2024-01-01' AND '2024-01-07';  -- Scans 7 partitions, not all data
\`\`\`

### 4. Monitoring and Alerting

Track pipeline health:

\`\`\`python
metrics = {
    'pipeline_duration_seconds': 3600,
    'rows_processed': 1_000_000,
    'rows_failed': 5,
    'data_quality_score': 0.9995
}

if metrics['pipeline_duration_seconds'] > 7200:
    alert("Pipeline running slow")
    
if metrics['rows_failed'] > 100:
    alert("High failure rate")
\`\`\`

## Interview Tips

When discussing analytics pipelines:

1. **Clarify requirements**: Batch or real-time? Latency needs?
2. **Discuss trade-offs**: Lambda vs Kappa, ETL vs ELT
3. **Mention specific technologies**: Kafka, Spark, Airflow, Snowflake
4. **Consider scale**: TB or PB? QPS?
5. **Data quality**: Validation, monitoring, alerting
6. **Cost**: Storage vs compute, cloud pricing

**Example question**: "Design an analytics pipeline for an e-commerce company."

**Strong answer**: "I'd use a hybrid architecture. For real-time needs (fraud detection, inventory), I'd implement stream processing with Kafka + Flink feeding Druid for real-time dashboards. For batch analytics (daily reports, ML training), I'd use Airflow to orchestrate Spark jobs that process data from S3 and load into Snowflake. I'd implement CDC with Debezium to stream database changes to Kafka. Data quality checks at ingestion, partitioning by date for efficient queries, and monitoring with Datadog. This provides both real-time capabilities and comprehensive batch analytics while balancing cost and complexity."

## Summary

Analytics pipelines transform raw data into insights through:
- **Batch processing**: High latency, cost-effective, handles large datasets
- **Stream processing**: Low latency, real-time, continuous processing
- **Lambda**: Combines batch + stream for completeness + freshness
- **Kappa**: Simplified stream-only approach
- **ETL vs ELT**: Transform before vs after loading
- **Data Lake vs Warehouse**: Flexibility vs performance
- **Orchestration**: Airflow, dbt for workflow management

Choose architecture based on latency requirements, scale, cost, and complexity tolerance.
`,
};

export default analyticsDataPipelineSection;
