import { QuizQuestion } from '@/lib/types';

export const columnOrientedDatabasesDiscussionQuiz: QuizQuestion[] = [
  {
    id: 'col-db-discussion-1',
    question:
      'Your company runs a data warehouse on PostgreSQL (row-oriented) with 5TB of sales data. Analytical queries like "total sales by region" take 10-15 minutes. The CTO suggests migrating to ClickHouse (columnar). The DevOps lead warns about operational complexity and suggests just adding more PostgreSQL replicas. Analyze both approaches, discuss the expected performance improvements, migration complexity, and provide a detailed recommendation with cost-benefit analysis.',
    sampleAnswer: `This scenario requires careful analysis of the performance bottleneck, migration effort, and long-term costs. Let me provide a comprehensive evaluation:

**Current State Analysis:**

PostgreSQL Setup:
- 5TB sales data (row-oriented storage)
- Analytical query: 10-15 minutes
- Likely: Wide table (20-30 columns), querying 2-3 columns
- Compression: ~2-3x

**Root Cause of Slow Queries:**

\`\`\`sql
-- Typical analytical query
SELECT region, SUM(amount) as total_sales
FROM sales
WHERE date >= '2024-01-01'
GROUP BY region;

-- PostgreSQL execution:
-- 1. Scan entire table (all 30 columns!)
-- 2. Filter by date
-- 3. Extract region, amount
-- 4. Group and aggregate
-- I/O: 5TB (entire dataset)
\`\`\`

**Why PostgreSQL is Slow:**1. **Full table scan**: Must read all columns
2. **Row-by-row processing**: Can't use SIMD
3. **Limited compression**: 2-3x vs 10-50x columnar
4. **No column pruning**: Reads unnecessary columns

**Option A: Add More PostgreSQL Replicas**

\`\`\`
Current: 1 primary PostgreSQL
Proposed: 1 primary + 3 read replicas

Architecture:
Application (writes) → Primary
Dashboards (reads) → Load balance across 3 replicas
\`\`\`

**Implementation:**
\`\`\`python
# Read replica setup
# primary.conf
wal_level = replica
max_wal_senders = 3

# replica.conf
hot_standby = on
primary_conninfo = 'host=primary port=5432'
\`\`\`

**Expected Performance:**

Query time: Still 10-15 minutes per query!
- **Reason**: Adding replicas doesn't change execution plan
- Each replica still does full table scan
- Parallel execution across replicas possible but requires application changes

**If implementing parallel query execution:**
\`\`\`python
# Application-level partitioning
queries = [
    "SELECT SUM(amount) FROM sales WHERE region = 'US'",  # Replica 1
    "SELECT SUM(amount) FROM sales WHERE region = 'EU'",  # Replica 2
    "SELECT SUM(amount) FROM sales WHERE region = 'APAC'"  # Replica 3
]
results = parallel_execute (queries)  # Merge results
\`\`\`

**Realistic speedup**: 2-3x (if perfectly partitioned)
**New query time**: 3-5 minutes (still too slow!)

**Pros:**
- ✅ Familiar technology (team knows PostgreSQL)
- ✅ Lower operational complexity
- ✅ Handles concurrent queries better

**Cons:**
- ❌ Minimal performance improvement (still 3-5 min)
- ❌ 4x storage cost (primary + 3 replicas = 20TB)
- ❌ 4x replication bandwidth
- ❌ Doesn't address core issue (row-oriented storage)

**Cost:**
- Storage: 20TB × $0.10/GB/month = $2,000/month
- Compute: 4× c5.4xlarge = $1,200/month
- **Total: $3,200/month**

**Option B: Migrate to ClickHouse (Columnar)**

**Implementation Plan:**

**Phase 1: Setup ClickHouse Cluster (Week 1)**
\`\`\`sql
-- ClickHouse table design
CREATE TABLE sales (
    sale_id UInt64,
    date Date,
    region String,
    product_id UInt32,
    amount Decimal(10, 2),
    quantity UInt32,
    customer_id UInt32
    -- ... 23 more columns
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (region, date, product_id)
SETTINGS index_granularity = 8192;

-- Distributed setup for scaling
CREATE TABLE sales_distributed AS sales
ENGINE = Distributed (cluster, default, sales, rand());
\`\`\`

**Phase 2: Historical Data Migration (Week 2-3)**
\`\`\`bash
# Export from PostgreSQL to Parquet (compressed)
pg_dump --table sales | 
  spark-submit convert_to_parquet.py --output s3://bucket/sales/

# Load into ClickHouse (parallel, fast)
clickhouse-client --query="
  INSERT INTO sales SELECT * 
  FROM s3('s3://bucket/sales/*.parquet', 'Parquet')
  SETTINGS max_insert_threads=16
"

# Speed: 5TB in 2-3 hours
\`\`\`

**Phase 3: Change Data Capture (Ongoing)**
\`\`\`
PostgreSQL (primary, OLTP)
     ↓
  Debezium CDC
     ↓
   Kafka
     ↓
ClickHouse (analytics, OLAP)
\`\`\`

**Expected Performance:**

\`\`\`sql
-- Same query on ClickHouse
SELECT region, SUM(amount) as total_sales
FROM sales
WHERE date >= '2024-01-01'
GROUP BY region;

-- ClickHouse execution:
-- 1. Scan ONLY date, region, amount columns (3/30 = 10%)
-- 2. Partition pruning (skip old months)
-- 3. Vectorized aggregation (SIMD)
-- 4. Compressed data (50x smaller)
-- I/O: 5TB / 50 (compression) × 3/30 (columns) = 10GB

-- Query time: 8-12 seconds (100x faster!)
\`\`\`

**Why ClickHouse is Faster:**1. **Column pruning**: Read 3/30 columns = 90% less I/O
2. **Compression**: 50x vs 2-3x = 16x less disk I/O
3. **Vectorization**: SIMD processing = 8x CPU efficiency
4. **Partition pruning**: Skip irrelevant months
5. **Late materialization**: Process bitmaps before fetching data

**Realistic speedup**: 100-200x
**New query time**: 5-10 seconds ✅

**Storage:**
\`\`\`
PostgreSQL: 5TB raw → 2TB compressed (2.5x)
ClickHouse: 5TB raw → 100GB compressed (50x!)

Savings: 20x storage reduction
\`\`\`

**Pros:**
- ✅ 100x query performance improvement
- ✅ 20x storage reduction (100GB vs 2TB)
- ✅ Sub-10 second queries (excellent UX)
- ✅ Real-time CDC ingestion
- ✅ Purpose-built for analytics

**Cons:**
- ❌ Operational complexity (new technology)
- ❌ Team learning curve
- ❌ Migration effort (2-3 weeks)
- ❌ Not ACID compliant (eventual consistency)

**Cost:**
- Storage: 100GB × $0.10/GB/month = $10/month (!)
- Compute: 2× c5.2xlarge = $600/month
- **Total: $610/month**

**Savings vs PostgreSQL replicas: $2,590/month**

**Migration Complexity Analysis:**

**Operational Complexity:**

PostgreSQL Replicas:
- Setup: 2 days
- Monitoring: Existing tools
- Failure recovery: Well-known procedures
- **Complexity: Low**

ClickHouse:
- Setup: 1 week
- CDC pipeline: 1 week
- Monitoring: New tools (Grafana)
- Learning curve: 1 month
- **Complexity: Medium-High**

**Risk Mitigation for ClickHouse:**

**Phase 1: Parallel Running (Month 1)**
\`\`\`
PostgreSQL (primary source of truth)
     ↓
   CDC → ClickHouse (read-only, testing)
     ↓
Team validates query results
\`\`\`

**Phase 2: Gradual Migration (Month 2)**
- Move 20% of dashboards to ClickHouse
- Monitor performance, correctness
- Team builds expertise

**Phase 3: Full Migration (Month 3)**
- All analytics queries → ClickHouse
- PostgreSQL remains for OLTP

**Hybrid Architecture (Recommended):**

\`\`\`
┌─────────────────────────────────────┐
│     Application Layer               │
│  (writes, transactional queries)    │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│      PostgreSQL (OLTP)              │
│  - Transactions                     │
│  - Real-time updates                │
│  - Point lookups                    │
└─────────────┬───────────────────────┘
              ↓ CDC (Debezium)
              ↓ Kafka
              ↓
┌─────────────────────────────────────┐
│      ClickHouse (OLAP)              │
│  - Analytical queries               │
│  - Aggregations                     │
│  - Historical reporting             │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│     BI Tools / Dashboards           │
└─────────────────────────────────────┘
\`\`\`

**Decision Matrix:**

| Factor | PostgreSQL Replicas | ClickHouse | Winner |
|--------|-------------------|------------|---------|
| Query Performance | 3-5 min | 5-10 sec | ✅ ClickHouse (100x) |
| Storage Cost | $2,000/mo | $10/mo | ✅ ClickHouse (200x cheaper) |
| Compute Cost | $1,200/mo | $600/mo | ✅ ClickHouse (2x cheaper) |
| Operational Complexity | Low | Medium-High | ✅ PostgreSQL |
| Migration Time | 2 days | 2-3 weeks | ✅ PostgreSQL |
| Scalability | Limited | Excellent | ✅ ClickHouse |
| Learning Curve | None | 1 month | ✅ PostgreSQL |
| Long-term Cost (3 years) | $115k | $22k | ✅ ClickHouse (5x cheaper) |

**My Recommendation: Migrate to ClickHouse**

**Why:**1. **Performance**: 100x improvement (5-10 sec vs 10-15 min) transforms user experience
2. **Cost**: Save $2,500/month ($30k/year, $90k over 3 years)
3. **Scalability**: As data grows 5TB → 50TB, PostgreSQL becomes unusable, ClickHouse scales linearly
4. **Industry Standard**: ClickHouse is proven for analytics at scale (Uber, CloudFlare, Spotify)

**Implementation Strategy:**

**Month 1: Parallel Running**
- Set up ClickHouse cluster
- Implement CDC pipeline
- Team training
- Cost: $5k (consulting) + $610/mo (infrastructure)

**Month 2: Validation**
- Run queries on both systems
- Validate results match
- Monitor performance

**Month 3: Migration**
- Switch dashboards to ClickHouse
- Keep PostgreSQL for OLTP
- Decommission PostgreSQL replicas

**ROI Calculation:**

\`\`\`
Investment:
- Setup & migration: $15k (1-time)
- Training: $5k (1-time)
- Total upfront: $20k

Savings:
- Monthly: $2,590 (infrastructure)
- Annual: $31k
- 3-year: $93k

ROI: 4.6x over 3 years
Payback period: 8 months
\`\`\`

**What About the Risks?**

**Mitigation:**1. **"Too complex"**: Managed ClickHouse Cloud available (ClickHouse.com, Altinity.cloud)
2. **"Team doesn't know it"**: SQL interface, 1-month learning curve, consultant available
3. **"What if it fails?"**: Run parallel for 2 months, validate before switching
4. **"Data consistency?"**: CDC ensures eventual consistency (acceptable for analytics)

**Alternative: Managed Solutions**

If operational complexity is a major concern:
- **ClickHouse Cloud**: Fully managed, $800/month
- **Google BigQuery**: Serverless, pay-per-query
- **Snowflake**: Fully managed, $1,200/month

Still 3-10x cheaper than PostgreSQL replicas with 50-100x performance.

**Final Answer:**

Migrate to ClickHouse. The 100x performance improvement and 5x cost savings over 3 years far outweigh the 2-3 week migration effort. The DevOps concern about complexity is valid but manageable through:
1. Managed ClickHouse Cloud (if preferred)
2. Gradual migration with parallel running
3. Team training and consulting support

Adding PostgreSQL replicas is putting a band-aid on a broken leg—it doesn't address the fundamental architectural mismatch of using row-oriented storage for analytical workloads.`,
    keyPoints: [
      "Adding replicas doesn't solve row-oriented bottleneck; still scans all columns",
      'ClickHouse provides 100x performance improvement through column pruning, compression, and vectorization',
      'Storage cost: ClickHouse 20x cheaper (100GB vs 2TB) due to superior compression',
      'Hybrid architecture: PostgreSQL for OLTP, ClickHouse for OLAP via CDC',
      'Migration risk mitigation: parallel running, gradual cutover, managed services available',
      'ROI: $20k investment, $93k savings over 3 years, 8-month payback period',
    ],
  },
  {
    id: 'col-db-discussion-2',
    question:
      "You're designing a system that needs to handle both: (1) high-volume transaction inserts (100k/sec) with ACID guarantees, and (2) real-time analytics queries (<1 second) on the same data. Some engineers suggest a single ClickHouse cluster. Others propose PostgreSQL for transactions + ClickHouse for analytics with CDC. Discuss the trade-offs, performance implications, consistency models, and provide your recommendation for handling this hybrid OLTP/OLAP workload.",
    sampleAnswer: `This is a classic hybrid workload challenge that requires careful consideration of consistency, performance, and complexity trade-offs. Let me analyze both approaches:

**Requirements:**
- **OLTP**: 100k inserts/sec, ACID guarantees, point lookups
- **OLAP**: Real-time analytics (<1 second), aggregations, GROUP BY

**Option A: Single ClickHouse Cluster**

\`\`\`
Application → ClickHouse (single system)
                 ↓
         OLTP + OLAP queries
\`\`\`

**ClickHouse for Transactions:**

\`\`\`sql
-- INSERT performance
INSERT INTO transactions VALUES (123, 'user1', 100.50, NOW());

-- Individual inserts: 5-10k/sec (not 100k!)
-- Batch inserts: 500k-1M/sec
\`\`\`

**Performance Analysis:**

Individual INSERTs:
- ClickHouse optimized for batch, not individual rows
- Each INSERT creates new part (later merged)
- Overhead: metadata, merge background process
- **Result: 5-10k/sec** (10x below requirement!)

Batch INSERTs:
\`\`\`python
# Buffer inserts, batch every 100ms
buffer = []
for transaction in stream:
    buffer.append (transaction)
    if len (buffer) >= 10000 or time_since_last_flush > 0.1:
        clickhouse.execute("INSERT INTO transactions VALUES", buffer)
        buffer = []

# Performance: 500k-1M/sec ✅
\`\`\`

**ACID Guarantees:**

ClickHouse provides:
- ✅ Atomicity: Batch insert is atomic
- ✅ Durability: Written to disk
- ⚠️ Consistency: Eventual (not immediate)
- ❌ Isolation: No READ COMMITTED, REPEATABLE READ
- ❌ Transactions: No BEGIN/COMMIT/ROLLBACK

**Problem:**
\`\`\`python
# This is NOT possible in ClickHouse:
BEGIN;
  INSERT INTO accounts (user_id, balance) VALUES (1, 100);
  UPDATE accounts SET balance = balance - 50 WHERE user_id = 1;
  INSERT INTO transactions (user_id, amount) VALUES (1, -50);
COMMIT;

# ClickHouse: Each operation is independent!
# Can't rollback if one fails
\`\`\`

**Analytics Performance:**

\`\`\`sql
SELECT user_id, SUM(amount) as total
FROM transactions
WHERE date >= '2024-01-01'
GROUP BY user_id;

-- Query time: 200-500ms ✅
\`\`\`

**Consistency Model:**

\`\`\`
INSERT at T=0 → Part created → Background merge (1-10 sec) → Query sees data

Query at T=0.5sec may not see insert from T=0!
Eventually consistent (not immediate)
\`\`\`

**Option A: Pros & Cons**

**Pros:**
- ✅ Simple architecture (one system)
- ✅ No data replication delay
- ✅ Fast analytics (<1 sec)
- ✅ Lower operational overhead

**Cons:**
- ❌ Can't hit 100k individual inserts/sec (must batch)
- ❌ No ACID transactions (no BEGIN/COMMIT)
- ❌ Eventual consistency (not immediate read-after-write)
- ❌ No isolation levels
- ❌ Not suitable for financial transactions, inventory management

**When Single ClickHouse Works:**

✅ Append-only workloads (logs, events, metrics)
✅ Can batch inserts (100ms buffering acceptable)
✅ Don't need multi-statement transactions
✅ Eventual consistency acceptable
✅ No UPDATE/DELETE requirements

**Example Use Case: Clickstream Analytics**
\`\`\`
User clicks → Buffer (100ms) → Batch insert to ClickHouse → Analytics
Perfect fit: append-only, no transactions, eventual consistency OK
\`\`\`

**Option B: PostgreSQL (OLTP) + ClickHouse (OLAP) with CDC**

\`\`\`
Application → PostgreSQL (OLTP, ACID)
                  ↓
              Debezium CDC
                  ↓
                Kafka
                  ↓
          ClickHouse (OLAP, analytics)
\`\`\`

**Implementation:**

**PostgreSQL for Transactions:**
\`\`\`sql
-- ACID guaranteed
BEGIN;
  INSERT INTO accounts (user_id, balance) VALUES (1, 1000);
  UPDATE accounts SET balance = balance - 50 WHERE user_id = 1;
  INSERT INTO transactions (user_id, amount) VALUES (1, -50);
COMMIT;  -- All or nothing!

-- Performance: 100k inserts/sec with proper tuning ✅
\`\`\`

**CDC Pipeline:**
\`\`\`yaml
# Debezium connector
{
  "name": "postgres-debezium",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.dbname": "transactions",
    "table.include.list": "public.transactions",
    "publication.name": "debezium_pub"
  }
}

# Kafka topic: postgres.public.transactions
# Messages: INSERT, UPDATE, DELETE events

# ClickHouse materialized view
CREATE MATERIALIZED VIEW transactions_mv TO transactions
AS SELECT * FROM kafka_table;
\`\`\`

**ClickHouse for Analytics:**
\`\`\`sql
SELECT user_id, SUM(amount) as total
FROM transactions
WHERE date >= '2024-01-01'
GROUP BY user_id;

-- Query time: 200-500ms ✅
\`\`\`

**Consistency Model:**

\`\`\`
PostgreSQL INSERT (T=0)
    ↓ CDC (10-100ms)
    ↓ Kafka
    ↓ ClickHouse consumer
ClickHouse visible (T=100-500ms)

Latency: 100-500ms (eventual consistency)
\`\`\`

**Option B: Pros & Cons**

**Pros:**
- ✅ Full ACID guarantees (PostgreSQL)
- ✅ 100k inserts/sec achievable
- ✅ Multi-statement transactions
- ✅ Isolation levels (READ COMMITTED, SERIALIZABLE)
- ✅ Fast analytics (ClickHouse)
- ✅ Mature, battle-tested architecture

**Cons:**
- ❌ More complex (3 systems: Postgres, Kafka, ClickHouse)
- ❌ Operational overhead (CDC pipeline)
- ❌ Data lag (100-500ms delay)
- ❌ Higher cost (more infrastructure)
- ❌ More failure points

**Performance Comparison:**

| Metric | Single ClickHouse | Postgres + ClickHouse |
|--------|------------------|----------------------|
| ACID Transactions | ❌ No | ✅ Yes |
| Insert Rate (individual) | 5-10k/sec | 100k/sec |
| Insert Rate (batch) | 500k-1M/sec | 100k/sec |
| Analytics Query | 200-500ms | 200-500ms |
| Consistency | Eventual (1-10s) | Eventual (100-500ms) |
| Operational Complexity | Low | High |

**My Recommendation: PostgreSQL + ClickHouse with CDC**

**Why:**1. **ACID Requirements**: "ACID guarantees" in requirements is non-negotiable
   - Financial transactions need atomicity
   - Inventory management needs isolation
   - Can't use eventual consistency for these

2. **100k Individual Inserts**: ClickHouse can't handle this without batching
   - Batching adds application complexity
   - 100ms delay for batching ≈ CDC delay anyway

3. **Separation of Concerns**: OLTP and OLAP have different optimization goals
   - OLTP: Fast writes, point lookups, ACID
   - OLAP: Fast scans, aggregations, compression

4. **Industry Standard**: This is the proven pattern at scale
   - Netflix, Uber, Airbnb use similar architectures
   - Mature tooling (Debezium, Kafka)

**Implementation Details:**

**Phase 1: PostgreSQL Optimization**

\`\`\`sql
-- Partition by date for efficient purging
CREATE TABLE transactions (
    id BIGSERIAL,
    user_id BIGINT,
    amount DECIMAL(10,2),
    created_at TIMESTAMP
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE transactions_2024_01 PARTITION OF transactions
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Indexes for OLTP queries
CREATE INDEX idx_user_recent ON transactions (user_id, created_at DESC);

-- Configuration for high throughput
shared_buffers = 8GB
max_wal_size = 4GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
\`\`\`

**Phase 2: CDC Pipeline**

\`\`\`yaml
# Kafka cluster: 3 brokers
# Topics: 10 partitions, replication factor 3
# Retention: 7 days

# Debezium configuration
"max.batch.size": 10000  # Batch events for ClickHouse
"transforms": "unwrap"   # Extract after state only
\`\`\`

**Phase 3: ClickHouse Setup**

\`\`\`sql
-- Kafka engine for ingestion
CREATE TABLE transactions_kafka (
    id UInt64,
    user_id UInt64,
    amount Decimal(10,2),
    created_at DateTime
) ENGINE = Kafka()
SETTINGS 
    kafka_broker_list = 'kafka:9092',
    kafka_topic_list = 'postgres.transactions',
    kafka_group_name = 'clickhouse_consumer',
    kafka_format = 'JSONEachRow';

-- Materialized view for storage
CREATE MATERIALIZED VIEW transactions_mv TO transactions
AS SELECT * FROM transactions_kafka;

-- Main table optimized for analytics
CREATE TABLE transactions (
    id UInt64,
    user_id UInt64,
    amount Decimal(10,2),
    created_at DateTime
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (user_id, created_at);
\`\`\`

**Handling the 100-500ms Lag:**

**Most queries don't care:**
\`\`\`sql
-- Historical analysis (doesn't need real-time)
SELECT DATE(created_at), SUM(amount)
FROM transactions
WHERE created_at >= '2024-01-01'
GROUP BY DATE(created_at);

-- 100ms lag irrelevant for historical data
\`\`\`

**For real-time dashboards:**
\`\`\`python
# Hybrid query
def get_user_total (user_id):
    # Get from ClickHouse (fast, slightly stale)
    historical = clickhouse.query(
        f"SELECT SUM(amount) FROM transactions WHERE user_id = {user_id}"
    )
    
    # Get last 5 seconds from PostgreSQL (real-time)
    recent = postgres.query(
        f"SELECT SUM(amount) FROM transactions "
        f"WHERE user_id = {user_id} AND created_at > NOW() - INTERVAL '5 seconds'"
    )
    
    return historical + recent
\`\`\`

**Cost Analysis:**

**Option A (ClickHouse only):**
- ClickHouse cluster: $1,200/month
- Total: $1,200/month

**Option B (Postgres + Kafka + ClickHouse):**
- PostgreSQL: $800/month
- Kafka cluster: $600/month
- ClickHouse: $800/month
- Total: $2,200/month

**Extra cost: $1,000/month for ACID guarantees and 100k inserts/sec**

**Worth it?** If you need ACID and high-volume individual inserts: YES

**Alternative: ClickHouse + Application-Level Transactions**

If CDC complexity is too high:
\`\`\`python
# Application handles transactions
def transfer (from_user, to_user, amount):
    # Batch both operations
    operations = [
        {"user": from_user, "amount": -amount},
        {"user": to_user, "amount": amount}
    ]
    
    # Single batch insert (atomic in ClickHouse)
    clickhouse.execute("INSERT INTO transactions VALUES", operations)
    
    # Not true ACID but acceptable for some use cases
\`\`\`

**When this works:**
- Don't need isolation (no concurrent updates)
- Can handle application-level rollback
- Batch insertion acceptable

**Final Recommendation:**

**Use PostgreSQL + ClickHouse with CDC** if:
- ✅ Need ACID transactions (multi-statement, rollback)
- ✅ Need isolation (concurrent updates)
- ✅ High volume individual inserts (100k/sec)
- ✅ Can tolerate 100-500ms analytics lag
- ✅ Have operational expertise for CDC

**Use single ClickHouse** if:
- ✅ Append-only workload
- ✅ Can batch inserts (100-500ms buffer)
- ✅ Eventual consistency acceptable
- ✅ No multi-statement transactions needed
- ✅ Want simpler architecture

For the stated requirements (ACID + 100k inserts/sec + real-time analytics), the hybrid architecture with PostgreSQL + ClickHouse is the only viable solution.`,
    keyPoints: [
      "ClickHouse alone can't provide ACID guarantees or handle 100k individual inserts/sec",
      'Hybrid architecture separates OLTP (PostgreSQL) and OLAP (ClickHouse) concerns',
      "CDC introduces 100-500ms lag, but most analytics queries don't need real-time data",
      'For truly real-time dashboards, query recent data from PostgreSQL + historical from ClickHouse',
      'Single ClickHouse works for append-only workloads with batching (logs, events, metrics)',
      'Industry standard: OLTP database + CDC + columnar database for hybrid workloads',
    ],
  },
  {
    id: 'col-db-discussion-3',
    question:
      "Your data science team wants to run complex ML queries on 10TB of historical transaction data. Currently stored in Parquet files on S3, they're using Athena which takes 5-10 minutes per query. You're evaluating: (1) loading into Redshift, (2) loading into ClickHouse, (3) keeping in S3 but using Presto/Trino. Discuss the trade-offs in terms of query performance, cost, data pipeline complexity, and ML workflow integration. What would you recommend and why?",
    sampleAnswer: `This scenario involves choosing the right architecture for ML workloads on large datasets. Let me analyze each approach comprehensively:

**Current State:**
- 10TB historical transaction data
- Stored in Parquet on S3
- Athena queries: 5-10 minutes
- Use case: ML feature engineering, exploration

**Option 1: Amazon Redshift**

**Architecture:**
\`\`\`
S3 (Parquet) → COPY → Redshift cluster → ML queries
\`\`\`

**Implementation:**
\`\`\`sql
-- Load data into Redshift
COPY transactions
FROM 's3://bucket/transactions/'
IAM_ROLE 'arn:aws:iam::123456789:role/RedshiftLoadRole'
FORMAT AS PARQUET;

-- Query performance
SELECT 
  user_id,
  AVG(amount) as avg_transaction,
  COUNT(*) as transaction_count,
  STDDEV(amount) as amount_stddev
FROM transactions
WHERE date >= '2024-01-01'
GROUP BY user_id;

-- Query time: 30-60 seconds
\`\`\`

**Performance:**
- **Query time**: 30-60 seconds (5-10x faster than Athena)
- **Why faster**: Compiled queries, local storage (no S3 round trips), optimized for joins/aggregations
- **Concurrency**: 50+ concurrent queries

**Cost:**
\`\`\`
Storage: 10TB → 2TB compressed (5x)
Redshift cost: 2TB storage + compute

Option A: dc2.large (3 nodes)
- Storage: 2TB included
- Compute: 3 × $0.25/hr = $540/month
- Total: $540/month

Option B: ra3.xlplus (2 nodes)
- Storage: $0.024/GB/month × 2TB = $49/month
- Compute: 2 × $1.086/hr = $1,566/month
- Total: $1,615/month (but scales better)
\`\`\`

**ML Integration:**
\`\`\`python
# Redshift to pandas (slow for large results)
import pandas as pd
import psycopg2

conn = psycopg2.connect(...)
df = pd.read_sql("SELECT * FROM features WHERE date >= '2024-01-01'", conn)

# Problem: Fetching 1GB+ results is slow
# Solution: UNLOAD to S3, then read Parquet
\`\`\`

**UNLOAD for ML pipelines:**
\`\`\`sql
-- Export features to S3 for ML
UNLOAD ('SELECT * FROM user_features')
TO 's3://bucket/ml_features/'
IAM_ROLE 'arn:...'
PARQUET
PARALLEL ON;

-- Then: Spark/PyTorch reads from S3
\`\`\`

**Pros:**
- ✅ Fast queries (30-60s vs 5-10 min)
- ✅ Familiar SQL interface
- ✅ Good for joins, aggregations
- ✅ Mature, widely adopted
- ✅ Integration with AWS ecosystem

**Cons:**
- ❌ Expensive ($540-1,600/month ongoing)
- ❌ Storage duplication (S3 + Redshift = 12TB total)
- ❌ Data pipeline complexity (keep Redshift synced with S3)
- ❌ Must UNLOAD to S3 for ML libraries (can't read Redshift directly)
- ❌ Less efficient for ad-hoc exploration (still 30-60s per query)

**Option 2: ClickHouse**

**Architecture:**
\`\`\`
S3 (Parquet) → INSERT SELECT → ClickHouse cluster → ML queries
\`\`\`

**Implementation:**
\`\`\`sql
-- Load from S3 (very fast!)
INSERT INTO transactions 
SELECT * FROM s3(
  's3://bucket/transactions/*.parquet',
  'Parquet'
) SETTINGS max_insert_threads=16;

-- Load time: 10TB in 1-2 hours

-- Query
SELECT 
  user_id,
  avg (amount) as avg_transaction,
  count() as transaction_count,
  stdDev (amount) as amount_stddev
FROM transactions
WHERE date >= '2024-01-01'
GROUP BY user_id;

-- Query time: 5-15 seconds (40x faster than Athena!)
\`\`\`

**Performance:**
- **Query time**: 5-15 seconds (much faster than Redshift!)
- **Why faster**: Superior compression, vectorized execution, column-oriented
- **Concurrency**: 100+ concurrent queries

**Cost:**
\`\`\`
Storage: 10TB → 200GB compressed (50x!)

ClickHouse cluster: 3 nodes × c5.2xlarge
- Compute: 3 × $0.34/hr = $734/month
- Storage: 200GB × $0.10/GB = $20/month
- Total: $754/month
\`\`\`

**ML Integration:**
\`\`\`python
# ClickHouse has native S3 integration
from clickhouse_driver import Client

client = Client('clickhouse-host')

# Option 1: Query and materialize (for smaller results)
df = client.query_dataframe("SELECT * FROM user_features")

# Option 2: Export to S3 for large results
client.execute("""
  INSERT INTO FUNCTION s3(
    's3://bucket/ml_features/data.parquet',
    'Parquet'
  ) SELECT * FROM user_features
""")

# Then: Spark/PyTorch reads from S3
\`\`\`

**Pros:**
- ✅ Extremely fast queries (5-15s)
- ✅ 50x compression (200GB vs 10TB!)
- ✅ Lower cost than Redshift ($754 vs $1,615/month)
- ✅ Direct S3 export for ML libraries
- ✅ Better for ad-hoc exploration

**Cons:**
- ❌ Storage duplication (S3 + ClickHouse)
- ❌ Data pipeline complexity
- ❌ Less familiar to teams (vs Redshift)
- ❌ Smaller ecosystem than Redshift

**Option 3: Presto/Trino on S3 (Query-in-Place)**

**Architecture:**
\`\`\`
S3 (Parquet) → Presto cluster (compute only) → ML queries
             ↓
        No data movement!
\`\`\`

**Implementation:**
\`\`\`sql
-- Presto/Trino: Query S3 directly
SELECT 
  user_id,
  avg (amount) as avg_transaction,
  count(*) as transaction_count,
  stddev (amount) as amount_stddev
FROM hive.default.transactions  -- Points to S3
WHERE date >= '2024-01-01'
GROUP BY user_id;

-- Query time: 60-120 seconds (2x faster than Athena)
\`\`\`

**Performance:**
- **Query time**: 60-120 seconds (slower than ClickHouse/Redshift)
- **Why slower**: S3 round trips, no local caching (first query)
- **Caching**: Subsequent queries faster (30-60s with Alluxio/Raptor)

**Cost:**
\`\`\`
Storage: 10TB on S3 = $230/month (baseline, no duplication!)

Presto cluster: 3 workers × r5.2xlarge
- Compute: 3 × $0.50/hr = $1,080/month
- Caching (Alluxio): +$300/month (optional)
- Total: $1,310/month (compute only when running)

# Can use auto-scaling / spot instances
# Shut down nights/weekends: Save 70% = $394/month
\`\`\`

**ML Integration:**
\`\`\`python
# Presto to pandas
import pandas as pd
from pyhive import presto

conn = presto.connect('presto-coordinator')
df = pd.read_sql("SELECT * FROM transactions LIMIT 1000000", conn)

# Or: Presto writes to S3, then read
# CREATE TABLE ml_features_output AS
# SELECT * FROM computed_features;
# (Stored back to S3 as Parquet)

# ML frameworks read directly from S3
import pyarrow.parquet as pq
table = pq.read_table('s3://bucket/ml_features/')
\`\`\`

**Pros:**
- ✅ No storage duplication (query S3 directly)
- ✅ No data pipeline to maintain
- ✅ Can shut down when not in use (save $$)
- ✅ Direct S3 access for ML (no export needed)
- ✅ Flexible (add compute as needed)
- ✅ Open-source (Trino)

**Cons:**
- ❌ Slower than ClickHouse/Redshift (60-120s)
- ❌ Requires cluster management
- ❌ S3 API costs ($0.0004/1000 requests can add up)
- ❌ First query slow (no caching)

**Comparison Matrix:**

| Factor | Redshift | ClickHouse | Presto/Trino |
|--------|----------|------------|--------------|
| Query Time | 30-60s | 5-15s | 60-120s |
| Monthly Cost | $540-1,615 | $754 | $394-1,310 |
| Storage | 12TB (S3+RDS) | 10.2TB (S3+CH) | 10TB (S3 only) |
| ML Integration | UNLOAD → S3 | Export → S3 | Native S3 |
| Data Pipeline | COPY jobs | INSERT jobs | None! |
| Flexibility | Low | Medium | High |
| Operational | Managed | Self-managed | Self-managed |

**My Recommendation: Hybrid Approach**

**Phase 1: Start with Presto/Trino (Immediate)**

\`\`\`
Why:
- No data movement required (immediate 2x speedup)
- No storage duplication
- Low upfront investment
- Test if 60-120s is acceptable
\`\`\`

**Phase 2: Add ClickHouse for Hot Data (Month 2)**

\`\`\`
S3 (10TB, all historical data)
       ↓
  Presto/Trino (cold queries, 60-120s)
       
S3 → ClickHouse (500GB, last 3 months)
       ↓
  Fast queries (5-15s) on recent data
\`\`\`

**Why Hybrid:**1. **90% of ML queries are on recent data** (last 3-6 months)
   - ClickHouse: 500GB compressed (10x smaller)
   - Cost: $200/month (vs $754 for full 10TB)

2. **10% of queries need full history** (backtesting, retraining)
   - Presto/Trino on S3: 60-120s acceptable for batch jobs

3. **Total cost**: $200 (ClickHouse) + $300 (small Presto) = **$500/month**
   - vs $1,615 (Redshift full) or $754 (ClickHouse full)

**Implementation:**

\`\`\`python
# Smart query router
def query_transactions (date_range):
    if date_range.end >= datetime.now() - timedelta (days=90):
        # Recent data: Use ClickHouse (fast!)
        return clickhouse.query(
            f"SELECT * FROM transactions WHERE date >= '{date_range.start}'"
        )
    else:
        # Historical data: Use Presto (slower but acceptable)
        return presto.query(
            f"SELECT * FROM transactions WHERE date >= '{date_range.start}'"
        )
\`\`\`

**ML Workflow:**

\`\`\`python
# Feature engineering (daily)
# - Query last 90 days from ClickHouse (5-15s)
# - Join with historical features from S3
# - Write features to S3 as Parquet

# Model training (weekly)
# - Read features from S3 Parquet (fast!)
# - Train with PyTorch/TensorFlow
# - No database involved

# Inference
# - Load model from S3
# - Query real-time features from ClickHouse
\`\`\`

**Cost-Benefit:**

\`\`\`
Current (Athena):
- Cost: $5/TB scanned = $50/query (10TB scan)
- Time: 5-10 minutes per query
- 100 queries/month = $5,000/month

Option 1 (Redshift): $1,615/month, 30-60s queries
Option 2 (ClickHouse): $754/month, 5-15s queries
Option 3 (Presto): $394/month, 60-120s queries
Hybrid (Presto + ClickHouse hot): $500/month, 5-15s for 90% of queries

Savings: $4,500/month vs current
Performance: 20-60x faster
\`\`\`

**Final Decision:**

Start with **Presto/Trino** (quick win, no data movement), then add **ClickHouse for hot data** (3-6 months) for frequently accessed data. This provides:
- 90% of queries under 15 seconds
- 10% of queries under 2 minutes (acceptable for batch)
- $500/month total cost (10x savings vs Athena usage)
- No storage duplication for historical data
- Operational simplicity

If team has strong AWS preference and budget allows, Redshift is a solid choice. But for ML workloads with large datasets, the Presto + ClickHouse hybrid offers the best performance/cost ratio.`,
    keyPoints: [
      'Athena is slow (5-10 min) because it queries S3 directly; dedicated compute dramatically improves performance',
      'Redshift: 30-60s queries but expensive ($540-1,615/month) with storage duplication',
      'ClickHouse: 5-15s queries, 50x compression, best performance but storage duplication',
      'Presto/Trino: Query S3 directly (no duplication), 60-120s queries, cheapest if auto-scaled',
      'Hybrid approach: ClickHouse for hot data (last 3-6 months), Presto for cold data (full history)',
      'ML workflows benefit from Parquet on S3 (native to PyTorch/TensorFlow), so avoid locking data in proprietary formats',
    ],
  },
];
