import { ModuleSection } from '@/lib/types';

const columnOrientedDatabasesSection: ModuleSection = {
  id: 'column-oriented-databases',
  title: 'Column-Oriented Databases',
  content: `
# Column-Oriented Databases

## Introduction

Traditional databases store data **row by row**, optimizing for transactional workloads where you frequently read entire records. But what if you primarily analyze data by **columns**—calculating sums, averages, or filtering billions of rows by specific attributes?

**Column-oriented databases** revolutionize analytics by storing data **column by column**, enabling 10-100x faster analytical queries while dramatically reducing storage through compression. This section covers how columnar storage works, when to use it, and the technologies that power modern analytics at scale.

## Row-Oriented vs Column-Oriented Storage

### Row-Oriented Storage (Traditional)

**How data is stored:**

\`\`\`
Table: Users
| user_id | name    | age | country | signup_date |
|---------|---------|-----|---------|-------------|
| 1       | Alice   | 28  | US      | 2024-01-15  |
| 2       | Bob     | 35  | UK      | 2024-01-16  |
| 3       | Charlie | 42  | US      | 2024-01-17  |

Disk layout (row-oriented):
Block 1: [1, Alice, 28, US, 2024-01-15]
Block 2: [2, Bob, 35, UK, 2024-01-16]
Block 3: [3, Charlie, 42, US, 2024-01-17]
\`\`\`

**Query: "What\'s the average age?"**

\`\`\`sql
SELECT AVG(age) FROM users;
\`\`\`

**Row-oriented execution:**
- Read ALL blocks (entire rows)
- Extract only the "age" column from each row
- Discard user_id, name, country, signup_date (wasted I/O!)
- Calculate average

**Problem**: Read 100% of data to use 20% (one column out of five).

### Column-Oriented Storage

**How data is stored:**

\`\`\`
Same table, different layout:

user_id column: [1, 2, 3]
name column: [Alice, Bob, Charlie]
age column: [28, 35, 42]
country column: [US, UK, US]
signup_date column: [2024-01-15, 2024-01-16, 2024-01-17]
\`\`\`

**Query: "What's the average age?"**

\`\`\`sql
SELECT AVG(age) FROM users;
\`\`\`

**Column-oriented execution:**
- Read ONLY the age column
- No wasted I/O
- Calculate average

**Result**: Read 20% of data (80% savings!)

## Why Columnar Storage is Faster for Analytics

### 1. I/O Efficiency

**Example Query: "Total sales by country"**

\`\`\`sql
SELECT country, SUM(amount) 
FROM transactions 
WHERE date >= '2024-01-01'
GROUP BY country;
\`\`\`

**Table**: 1 billion rows, 20 columns, 10TB

**Row-oriented**: 
- Must read all 20 columns for 1 billion rows
- Total I/O: 10TB
- Time: ~10 minutes

**Column-oriented**:
- Read only date (filtering), country (grouping), amount (aggregation)
- Total I/O: 3/20 × 10TB = 1.5TB
- **Time: ~90 seconds** (6.7x faster!)

### 2. Compression

Columnar data compresses extremely well:

**Row-oriented data:**
\`\`\`
Row 1: [1, US, 2024-01-15, 100.50]
Row 2: [2, US, 2024-01-15, 200.75]
Row 3: [3, UK, 2024-01-15, 150.00]

Compression: Limited (mixed data types, low redundancy)
Ratio: 2-3x
\`\`\`

**Column-oriented data:**
\`\`\`
country: [US, US, US, US, UK, UK, UK, US, US, US, ...]

Compression techniques:
1. Run-length encoding: "US" appears 1000 times → store once + count
2. Dictionary encoding: US=1, UK=2, JP=3 → [1,1,1,1,2,2,2,1,1,1]
3. Bit packing: Values 0-255 → 1 byte instead of 4-byte int

Compression ratio: 10-50x!
\`\`\`

**Real-world example:**
- Raw data: 10TB
- Row-oriented (compressed): 4TB (2.5x)
- Column-oriented (compressed): 500GB (20x!)

### 3. Vectorization (SIMD)

Modern CPUs process multiple values in one instruction (SIMD):

\`\`\`cpp
// Row-oriented: Can't vectorize (different types)
for (int i = 0; i < rows; i++) {
    sum += row[i].amount;  // One value at a time
}

// Column-oriented: Vectorizable (same type)
// Process 8 values simultaneously with SIMD
for (int i = 0; i < amounts.size(); i += 8) {
    sum += simd_sum (amounts[i:i+8]);  // 8 values at once!
}
\`\`\`

**Performance**: 8x speedup from SIMD + CPU cache efficiency

### 4. Late Materialization

**Row-oriented**: Construct full rows early (memory intensive)

**Column-oriented**: Keep columns separate until final result

\`\`\`sql
SELECT name, age 
FROM users 
WHERE country = 'US' AND age > 25;
\`\`\`

**Column-oriented execution:**
\`\`\`
1. Scan country column → [1, 1, 0, 1, 0, ...] (1 = match, 0 = no match)
2. Scan age column → [1, 0, 1, 1, 1, ...]
3. AND bitmaps → [1, 0, 0, 1, 0, ...]
4. Only NOW fetch name, age for matching rows
\`\`\`

**Benefit**: Process bitmaps (fast), fetch data only for results

## Popular Column-Oriented Databases

### ClickHouse

**Description**: Open-source OLAP database by Yandex.

**Strengths:**
- Extremely fast for analytical queries
- SQL interface
- Real-time ingestion
- Horizontal scaling

**Use Cases:**
- Real-time analytics dashboards
- Time-series analytics
- Log analysis

**Example:**
\`\`\`sql
-- Create table
CREATE TABLE events (
    date Date,
    user_id UInt32,
    event_type String,
    country String,
    value Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id);

-- Query (billions of rows in seconds)
SELECT 
    country,
    COUNT() as events,
    AVG(value) as avg_value
FROM events
WHERE date >= '2024-01-01'
GROUP BY country
ORDER BY events DESC;
\`\`\`

**Performance**: Processes 1 billion rows/second on modern hardware.

### Apache Druid

**Description**: Real-time analytics database for event data.

**Strengths:**
- Sub-second query latency
- Real-time and batch ingestion
- Built-in rollup and aggregation
- Time-series optimization

**Use Cases:**
- Real-time dashboards
- User analytics
- Network telemetry
- Business intelligence

**Architecture:**
\`\`\`
Ingestion → Real-time nodes (recent data) → Historical nodes (older data)
                ↓                                    ↓
            Query both simultaneously
\`\`\`

**Example:**
\`\`\`json
{
  "datasource": "pageviews",
  "intervals": ["2024-01-01/2024-01-31"],
  "granularity": "day",
  "dimensions": ["country", "device"],
  "metrics": ["page_views", "unique_users"],
  "filter": {"country": {"type": "equals", "value": "US"}}
}
\`\`\`

### Google BigQuery

**Description**: Serverless, cloud-native data warehouse.

**Strengths:**
- Petabyte scale
- Serverless (no infrastructure management)
- Blazing fast with Dremel engine
- Pay per query (compute/storage separation)

**Use Cases:**
- Enterprise data warehouse
- Big data analytics
- Machine learning at scale

**Example:**
\`\`\`sql
-- Query 1TB in seconds
SELECT 
  country,
  COUNT(*) as request_count,
  AVG(response_time) as avg_response_time
FROM \`project.dataset.logs\`
WHERE date >= '2024-01-01'
  AND status_code = 200
GROUP BY country
ORDER BY request_count DESC
LIMIT 100;

-- Cost: $5/TB scanned
\`\`\`

**Performance**: Scans terabytes in seconds using thousands of workers.

### Amazon Redshift

**Description**: Cloud data warehouse based on ParAccel (columnar).

**Strengths:**
- Familiar SQL interface
- Integration with AWS ecosystem
- Mature and widely adopted
- Spectrum for querying S3

**Use Cases:**
- Enterprise data warehouse
- Business intelligence
- Data lake analytics

**Example:**
\`\`\`sql
-- Create columnar table
CREATE TABLE sales (
    sale_id BIGINT,
    product_id INTEGER,
    amount DECIMAL(10,2),
    sale_date DATE
)
DISTKEY(product_id)
SORTKEY(sale_date)
ENCODE AUTO;  -- Automatic compression

-- Query
SELECT product_id, SUM(amount)
FROM sales
WHERE sale_date >= '2024-01-01'
GROUP BY product_id;
\`\`\`

### Apache Parquet

**Description**: Columnar file format (not a database).

**Strengths:**
- Open standard
- Works with Spark, Hive, Presto, Athena
- Excellent compression
- Schema evolution support

**Use Cases:**
- Data lake storage format
- Big data processing
- Analytics on S3

**File Structure:**
\`\`\`
Parquet File:
├── Row Group 1 (64MB)
│   ├── Column chunk: user_id (compressed)
│   ├── Column chunk: name (compressed)
│   └── Column chunk: age (compressed)
├── Row Group 2 (64MB)
│   └── ...
└── Footer (metadata, schema, statistics)
\`\`\`

**Example:**
\`\`\`python
# Write Parquet
import pandas as pd
df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
df.to_parquet('data.parquet', compression='snappy', engine='pyarrow')

# Read specific columns only
df = pd.read_parquet('data.parquet', columns=['id'])  # Only read ID column!
\`\`\`

## Compression Techniques

### 1. Dictionary Encoding

**Problem**: Repeated string values waste space.

\`\`\`
country: [US, US, US, UK, UK, US, US, US]
Storage: 8 strings × 2 bytes = 16 bytes
\`\`\`

**Solution**: Create dictionary, store indices.

\`\`\`
Dictionary: {0: US, 1: UK}
Encoded: [0, 0, 0, 1, 1, 0, 0, 0]
Storage: 2 strings (dictionary) + 8 bytes (indices) = 10 bytes
Savings: 37.5%
\`\`\`

For 1 billion rows with 100 unique countries:
- Raw: 2GB
- Dictionary: 200 bytes (dict) + 1GB (indices) = **1GB (50% savings)**

### 2. Run-Length Encoding (RLE)

**Problem**: Consecutive identical values.

\`\`\`
status: [active, active, active, active, inactive, inactive]
Storage: 6 strings
\`\`\`

**Solution**: Store value + count.

\`\`\`
Encoded: [(active, 4), (inactive, 2)]
Storage: 2 pairs instead of 6 strings
\`\`\`

**Best for**: Sorted data (all "active" together).

### 3. Bit Packing

**Problem**: Integers use more bits than needed.

\`\`\`
age: [28, 35, 42, 31, 29]
Storage: 5 × 32 bits = 160 bits
\`\`\`

**Observation**: All values fit in 6 bits (max 63).

\`\`\`
Encoded: Pack into 6 bits each
Storage: 5 × 6 bits = 30 bits
Savings: 81%!
\`\`\`

### 4. Delta Encoding

**Problem**: Timestamps have small differences.

\`\`\`
timestamps: [1705276800, 1705276801, 1705276802, 1705276803]
Storage: 4 × 32 bits = 128 bits
\`\`\`

**Solution**: Store first value + deltas.

\`\`\`
Base: 1705276800
Deltas: [0, 1, 1, 1]
Storage: 32 bits (base) + 4 × 2 bits (deltas) = 40 bits
Savings: 69%
\`\`\`

## When to Use Column-Oriented Databases

### Use Columnar Storage When:

✅ **Analytical queries** (aggregations, GROUP BY)
✅ **Read-heavy workload** (few writes, many reads)
✅ **Wide tables** (many columns, query few)
✅ **Large datasets** (billions of rows)
✅ **Historical data** (append-only or infrequent updates)
✅ **Aggregations** (SUM, AVG, COUNT)

**Example Use Cases:**
- Business intelligence dashboards
- Data warehousing
- Log analytics
- Time-series analytics
- Clickstream analysis
- Financial reporting

### Use Row-Oriented Storage When:

✅ **Transactional queries** (INSERT, UPDATE, DELETE)
✅ **Write-heavy workload**
✅ **Need full rows** (SELECT * common)
✅ **Small datasets** (<1TB)
✅ **Real-time updates** (OLTP)
✅ **Point lookups** by primary key

**Example Use Cases:**
- E-commerce transactions
- User authentication
- Inventory management
- Real-time gaming
- Social media posts

## Performance Comparison

### Query: "Average order amount by country"

**Dataset**: 1 billion orders, 10 columns, 500GB

**PostgreSQL (row-oriented):**
\`\`\`
Execution time: 12 minutes
I/O: 500GB (all columns)
Compression: 2x
\`\`\`

**ClickHouse (column-oriented):**
\`\`\`
Execution time: 8 seconds
I/O: 10GB (2 columns: country, amount)
Compression: 50x

Speed: 90x faster!
\`\`\`

### Insert Performance

**Row-oriented (PostgreSQL):**
\`\`\`
INSERT INTO orders VALUES (...);
10,000 inserts/second
\`\`\`

**Column-oriented (ClickHouse):**
\`\`\`
INSERT INTO orders VALUES (...);
5,000 inserts/second

Bulk insert (batch):
INSERT INTO orders SELECT * FROM staging;
500,000 inserts/second (100x faster!)
\`\`\`

**Lesson**: Columnar excels at bulk inserts, not individual INSERTs.

## Hybrid Approaches

### Columnar for Analytics, Row for OLTP

\`\`\`
Application → PostgreSQL (row-oriented, OLTP)
                     ↓
               CDC (Debezium)
                     ↓
        ClickHouse (column-oriented, OLAP)
                     ↓
              Dashboards
\`\`\`

### In-Memory Columnar: Apache Arrow

**Description**: Columnar memory format for fast data exchange.

**Use Case**: Zero-copy data sharing between Spark, Pandas, TensorFlow.

\`\`\`python
import pyarrow as pa
import pandas as pd

# Convert Pandas DataFrame to Arrow (zero-copy)
df = pd.DataFrame({'a': [1, 2, 3]})
arrow_table = pa.Table.from_pandas (df)

# Now Spark can read this without copying!
spark.createDataFrame (arrow_table.to_pandas())
\`\`\`

## Best Practices

### 1. Partition Data

\`\`\`sql
-- ClickHouse: Partition by month
CREATE TABLE events (
    date Date,
    user_id UInt32,
    event String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id);

-- Queries only scan relevant partitions
SELECT COUNT(*) FROM events WHERE date = '2024-01-15';
-- Only scans January 2024 partition!
\`\`\`

### 2. Order Data by Query Patterns

\`\`\`sql
-- Order by most-filtered columns
ORDER BY (country, date, user_id)

-- Query: WHERE country = 'US' AND date >= '2024-01-01'
-- Reads contiguous blocks (efficient!)
\`\`\`

### 3. Use Appropriate Compression

\`\`\`sql
-- ClickHouse: Automatic compression codec selection
CREATE TABLE events (
    country String CODEC(Dictionary),  -- Dictionary for low cardinality
    timestamp DateTime CODEC(Delta, LZ4),  -- Delta + LZ4 for timestamps
    value Float64 CODEC(Gorilla)  -- Gorilla for float compression
) ENGINE = MergeTree();
\`\`\`

### 4. Materialize Common Aggregations

\`\`\`sql
-- Druid: Rollup at ingestion
{
  "type": "index_hadoop",
  "spec": {
    "dataSchema": {
      "metricsSpec": [
        {"type": "count", "name": "count"},
        {"type": "doubleSum", "name": "revenue", "fieldName": "amount"}
      ],
      "granularitySpec": {"rollup": true, "queryGranularity": "hour"}
    }
  }
}

-- Pre-aggregated to hourly! Queries 24x faster
\`\`\`

## Common Mistakes

### 1. Using Columnar for OLTP

\`\`\`sql
-- BAD: Individual INSERTs to ClickHouse
for order in orders:
    INSERT INTO orders VALUES (order.id, order.amount)  -- Slow!

-- GOOD: Batch INSERT
INSERT INTO orders SELECT * FROM staging_orders;  -- 100x faster
\`\`\`

### 2. Not Sorting Data

\`\`\`sql
-- BAD: Random order (no ORDER BY)
-- Queries scan entire dataset

-- GOOD: Sorted by common filters
ORDER BY (country, date)
-- Queries skip irrelevant data
\`\`\`

### 3. Selecting All Columns

\`\`\`sql
-- BAD: Wastes columnar advantage
SELECT * FROM events WHERE date = '2024-01-15';

-- GOOD: Select only needed columns
SELECT user_id, event_type FROM events WHERE date = '2024-01-15';
\`\`\`

## Interview Tips

When discussing columnar databases:

1. **Explain the core advantage**: I/O efficiency (read only needed columns)
2. **Mention compression**: 10-50x better than row-oriented
3. **Discuss use cases**: Analytics (YES), OLTP (NO)
4. **Know the technologies**: ClickHouse, Druid, BigQuery, Redshift
5. **Understand trade-offs**: Fast reads, slower writes
6. **Consider hybrid approaches**: OLTP + OLAP with CDC

**Example question**: "Why is BigQuery fast for analytics?"

**Strong answer**: "BigQuery uses columnar storage, which provides massive performance benefits for analytics. First, it only reads the columns referenced in the query—for 'SELECT country, AVG(amount) FROM sales', it reads just 2 columns instead of all 20, reducing I/O by 90%. Second, columnar data compresses 10-50x better because similar data is stored together, enabling techniques like dictionary encoding and run-length encoding. Third, BigQuery's Dremel engine uses massive parallelism—distributing queries across thousands of workers. Finally, separation of compute and storage means you only pay for what you query. A query scanning 1TB might only read 50GB after compression and projection pushdown, completing in seconds."

## Summary

Column-oriented databases revolutionize analytics through:
- **I/O efficiency**: Read only needed columns (10-100x less I/O)
- **Compression**: 10-50x better than row-oriented (dictionary, RLE, bit packing)
- **Vectorization**: SIMD processing for speed
- **Technologies**: ClickHouse, Druid, BigQuery, Redshift, Parquet
- **Use cases**: Analytics, not OLTP
- **Trade-offs**: Optimized for reads and batch writes, not individual updates

Choose columnar storage for analytical workloads over billions of rows where you aggregate, filter, and group by dimensions—exactly what data warehouses and BI tools do.
`,
};

export default columnOrientedDatabasesSection;
