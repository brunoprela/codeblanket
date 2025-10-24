import { Quiz } from '@/lib/types';

const dataWarehousingDiscussionQuiz: Quiz = {
  id: 'data-warehousing-discussion',
  title: 'Data Warehousing - Discussion Questions',
  questions: [
    {
      id: 'dw-discussion-1',
      type: 'discussion',
      question:
        "Your e-commerce company stores customer addresses in the data warehouse. Customers frequently move, and business analysts need to report on: (1) where customers currently live, and (2) historical sales by the customer's address at the time of purchase. Which Slowly Changing Dimension (SCD) type should you implement and how would you design the schema to support both requirements efficiently?",
      sampleAnswer: `This scenario requires **SCD Type 2** (add new row for each change) to maintain complete history while supporting both current and historical queries.

**Schema Design:**

\`\`\`sql
CREATE TABLE dim_customer (
    customer_key BIGINT PRIMARY KEY,  -- Surrogate key
    customer_id VARCHAR(50),           -- Business key (natural)
    name VARCHAR(200),
    email VARCHAR(200),
    address VARCHAR(500),
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(20),
    country VARCHAR(50),
    
    -- SCD Type 2 metadata
    effective_date DATE NOT NULL,
    expiration_date DATE NOT NULL,
    is_current BOOLEAN NOT NULL,
    
    -- Optional: version number for easy tracking
    version INT NOT NULL
);

-- Index for current records
CREATE INDEX idx_customer_current ON dim_customer(customer_id, is_current);

-- Index for historical lookups
CREATE INDEX idx_customer_effective ON dim_customer(customer_id, effective_date, expiration_date);
\`\`\`

**Implementation Logic:**

\`\`\`python
def update_customer_address(customer_id, new_address, effective_date):
    # Step 1: Expire current record
    conn.execute("""
        UPDATE dim_customer
        SET expiration_date = %s - INTERVAL '1 day',
            is_current = FALSE
        WHERE customer_id = %s 
          AND is_current = TRUE
    """, (effective_date, customer_id))
    
    # Step 2: Insert new record
    conn.execute("""
        INSERT INTO dim_customer (
            customer_key, customer_id, name, email,
            address, city, state, zip_code, country,
            effective_date, expiration_date, is_current, version
        )
        SELECT 
            nextval('customer_key_seq'),
            customer_id, name, email,
            %s, %s, %s, %s, %s,  -- New address fields
            %s,                   -- effective_date
            '9999-12-31',         -- expiration_date (far future)
            TRUE,                 -- is_current
            COALESCE(MAX(version), 0) + 1  -- Increment version
        FROM dim_customer
        WHERE customer_id = %s
          AND is_current = FALSE
        GROUP BY customer_id, name, email
    """, (new_address, new_city, new_state, new_zip, new_country,
          effective_date, customer_id))
\`\`\`

**Query Patterns:**

**Pattern 1: Current Customer Information**
\`\`\`sql
-- Simple: Just current customers
SELECT 
    customer_id,
    name,
    city,
    state
FROM dim_customer
WHERE is_current = TRUE;

-- Performance: Index on (customer_id, is_current) makes this O(1)
\`\`\`

**Pattern 2: Historical Sales by Customer's Address at Time of Sale**
\`\`\`sql
SELECT 
    f.sale_date,
    c.customer_id,
    c.name,
    c.city,
    c.state,
    SUM(f.total_amount) as sales
FROM fact_sales f
JOIN dim_customer c 
    ON f.customer_key = c.customer_key
WHERE f.sale_date >= '2024-01-01'
GROUP BY f.sale_date, c.customer_id, c.name, c.city, c.state;

-- KEY: fact_sales.customer_key points to the SPECIFIC version
-- of dim_customer that was current at sale time
\`\`\`

**Pattern 3: Time-Travel Query (Customer Info at Specific Date)**
\`\`\`sql
SELECT 
    customer_id,
    name,
    city,
    state
FROM dim_customer
WHERE customer_id = 'CUST12345'
  AND '2024-03-15' BETWEEN effective_date AND expiration_date;
\`\`\`

**Fact Table Design:**

\`\`\`sql
CREATE TABLE fact_sales (
    sale_id BIGINT PRIMARY KEY,
    sale_date DATE NOT NULL,
    customer_key BIGINT NOT NULL,  -- Points to specific version!
    product_key BIGINT NOT NULL,
    
    quantity INT,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2)
);

-- When inserting sales, lookup current customer_key:
INSERT INTO fact_sales (sale_id, sale_date, customer_key, ...)
SELECT 
    %s,  -- sale_id
    %s,  -- sale_date
    c.customer_key,  -- Current customer_key at sale time
    ...
FROM dim_customer c
WHERE c.customer_id = %s
  AND c.is_current = TRUE;
\`\`\`

**Why This Works:**

1. **Historical Accuracy**: Fact table stores customer_key (not customer_id), which points to the exact version of customer dimension at sale time
2. **Current Data**: is_current flag enables instant lookup of latest info
3. **Efficient Queries**: Indexed properly for both access patterns

**Trade-offs:**

**Pros:**
- ✅ Complete history preserved
- ✅ Historical accuracy guaranteed
- ✅ Simple to query (standard JOIN)
- ✅ Industry standard (most BI tools understand this)

**Cons:**
- ❌ Dimension table grows over time
- ❌ Customer with 10 address changes → 10 rows
- ❌ More storage than SCD Type 1

**Optimization for Large Dimensions:**

If dimension grows too large (millions of customers × average 3 moves = tens of millions of rows):

\`\`\`sql
-- Partition by is_current for query performance
CREATE TABLE dim_customer (
    ...
) PARTITION BY LIST (is_current);

CREATE TABLE dim_customer_current PARTITION OF dim_customer
    FOR VALUES IN (TRUE);
    
CREATE TABLE dim_customer_historical PARTITION OF dim_customer
    FOR VALUES IN (FALSE);
\`\`\`

This answer demonstrates understanding of: dimensional modeling, SCD Type 2 implementation, historical accuracy requirements, query patterns, and optimization strategies.`,
      keyPoints: [
        'SCD Type 2 maintains complete history by inserting new row for each change',
        'Fact table stores surrogate key (customer_key) pointing to specific dimension version',
        'is_current flag enables efficient lookup of current records',
        'effective_date and expiration_date enable time-travel queries',
        "Historical accuracy: sales are attributed to customer's address at time of sale",
        'Trade-off: More storage vs complete historical accuracy',
      ],
    },
    {
      id: 'dw-discussion-2',
      type: 'discussion',
      question:
        "You're designing a data warehouse for a retail company with 100M daily transactions. The business wants reports that aggregate sales by: time (hour/day/month/year), location (store/city/state/country), and product (SKU/category/department). Should you use a star schema or snowflake schema, and how would you handle the time dimension to optimize for common date-based queries?",
      sampleAnswer: `**Recommendation: Star Schema with Specialized Date Dimension**

For this high-volume retail scenario, star schema is optimal for performance, and the date dimension should be pre-computed with all common date attributes.

**Why Star Schema:**

1. **Query Performance**: 100M daily transactions means 36B transactions/year. Complex joins (snowflake) would be too slow.
2. **BI Tool Compatibility**: Most tools assume star schema
3. **Simplicity**: Business users understand star schema more easily

**Fact Table Design:**

\`\`\`sql
CREATE TABLE fact_sales (
    sale_id BIGINT PRIMARY KEY,
    
    -- Foreign keys to dimensions (star: direct references)
    date_key INTEGER NOT NULL,
    time_key INTEGER NOT NULL,  -- Separate for hour-level detail
    store_key INTEGER NOT NULL,
    product_key INTEGER NOT NULL,
    payment_method_key INTEGER NOT NULL,
    
    -- Measures (additive facts)
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    discount_amount DECIMAL(10,2),
    tax_amount DECIMAL(10,2),
    
    -- Derived measure
    net_amount AS (total_amount - discount_amount) PERSISTED
)
PARTITION BY RANGE (date_key);  -- Partition by date for performance

-- Partitions: One per month
CREATE TABLE fact_sales_202401 PARTITION OF fact_sales
    FOR VALUES FROM (20240101) TO (20240201);
\`\`\`

**Date Dimension (Critical for Performance):**

\`\`\`sql
CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,  -- 20240115 (YYYYMMDD)
    
    -- Full date
    full_date DATE NOT NULL UNIQUE,
    
    -- Year attributes
    year INTEGER NOT NULL,
    year_name VARCHAR(10),  -- "2024"
    
    -- Quarter attributes  
    quarter INTEGER NOT NULL,  -- 1,2,3,4
    quarter_name VARCHAR(10),  -- "Q1 2024"
    
    -- Month attributes
    month INTEGER NOT NULL,  -- 1-12
    month_name VARCHAR(20),  -- "January"
    month_abbr VARCHAR(3),   -- "Jan"
    year_month VARCHAR(10),  -- "2024-01"
    
    -- Week attributes
    week_of_year INTEGER NOT NULL,  -- 1-53
    week_start_date DATE NOT NULL,
    week_end_date DATE NOT NULL,
    
    -- Day attributes
    day_of_month INTEGER NOT NULL,  -- 1-31
    day_of_week INTEGER NOT NULL,   -- 1-7 (Monday=1)
    day_name VARCHAR(20),  -- "Monday"
    day_abbr VARCHAR(3),   -- "Mon"
    day_of_year INTEGER NOT NULL,  -- 1-366
    
    -- Business attributes
    is_weekend BOOLEAN NOT NULL,
    is_holiday BOOLEAN NOT NULL,
    holiday_name VARCHAR(100),
    is_business_day BOOLEAN NOT NULL,
    
    -- Fiscal calendar (if different from calendar year)
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    fiscal_month INTEGER,
    
    -- Relative attributes (useful for "last 30 days" queries)
    days_from_today INTEGER  -- Computed daily: CURRENT_DATE - full_date
);

-- Indexes for common queries
CREATE INDEX idx_date_year_month ON dim_date(year, month);
CREATE INDEX idx_date_quarter ON dim_date(year, quarter);
CREATE INDEX idx_date_week ON dim_date(week_of_year, year);
CREATE INDEX idx_date_is_weekend ON dim_date(is_weekend);
CREATE INDEX idx_date_is_business_day ON dim_date(is_business_day);
\`\`\`

**Why Pre-compute Date Attributes:**

Instead of calculating at query time:
\`\`\`sql
-- SLOW (calculate every time)
SELECT 
    EXTRACT(YEAR FROM sale_date),
    EXTRACT(MONTH FROM sale_date),
    SUM(total_amount)
FROM fact_sales
GROUP BY EXTRACT(YEAR FROM sale_date), EXTRACT(MONTH FROM sale_date);
\`\`\`

Use pre-computed attributes:
\`\`\`sql
-- FAST (pre-computed, indexed)
SELECT 
    d.year,
    d.month_name,
    SUM(f.total_amount)
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.year, d.month_name;
\`\`\`

**Time Dimension (Separate for Hour-Level Detail):**

\`\`\`sql
CREATE TABLE dim_time (
    time_key INTEGER PRIMARY KEY,  -- 143052 (HHMMSS)
    
    hour INTEGER NOT NULL,          -- 0-23
    hour_12 INTEGER NOT NULL,       -- 1-12
    am_pm VARCHAR(2) NOT NULL,      -- "AM" or "PM"
    hour_name VARCHAR(20),          -- "2:00 PM"
    
    minute INTEGER NOT NULL,        -- 0-59
    second INTEGER NOT NULL,        -- 0-59
    
    -- Business segments
    time_of_day VARCHAR(20),        -- "Morning", "Afternoon", "Evening"
    is_business_hours BOOLEAN,
    
    -- For rollups
    hour_minute VARCHAR(5)          -- "14:30"
);
\`\`\`

**Product Dimension (Denormalized Star Schema):**

\`\`\`sql
CREATE TABLE dim_product (
    product_key INTEGER PRIMARY KEY,
    
    -- Product details
    sku VARCHAR(50) NOT NULL UNIQUE,
    product_name VARCHAR(200),
    description TEXT,
    
    -- Hierarchy (denormalized in star schema!)
    department_id VARCHAR(10),
    department_name VARCHAR(100),
    category_id VARCHAR(10),
    category_name VARCHAR(100),
    subcategory_id VARCHAR(10),
    subcategory_name VARCHAR(100),
    
    -- Product attributes
    brand VARCHAR(100),
    manufacturer VARCHAR(100),
    unit_cost DECIMAL(10,2),
    list_price DECIMAL(10,2),
    size VARCHAR(50),
    color VARCHAR(50),
    
    -- SCD Type 2 fields
    effective_date DATE,
    expiration_date DATE,
    is_current BOOLEAN
);

-- Why denormalized?
-- In snowflake schema, you'd have separate tables:
-- dim_product → dim_category → dim_department
-- Requires 3 JOINs vs 1 JOIN in star schema!
\`\`\`

**Store/Location Dimension (Also Denormalized):**

\`\`\`sql
CREATE TABLE dim_store (
    store_key INTEGER PRIMARY KEY,
    
    store_id VARCHAR(50) NOT NULL,
    store_name VARCHAR(200),
    
    -- Hierarchy (denormalized!)
    city VARCHAR(100),
    state VARCHAR(100),
    state_abbr VARCHAR(2),
    country VARCHAR(100),
    region VARCHAR(100),  -- "Northeast", "West"
    
    -- Store attributes
    store_type VARCHAR(50),  -- "Mall", "Standalone"
    store_size INTEGER,      -- Square feet
    opening_date DATE,
    is_active BOOLEAN
);
\`\`\`

**Query Examples:**

**1. Sales by Month and Category (Common Report):**
\`\`\`sql
SELECT 
    d.year,
    d.month_name,
    p.category_name,
    COUNT(*) as transaction_count,
    SUM(f.total_amount) as total_sales,
    AVG(f.total_amount) as avg_transaction
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_product p ON f.product_key = p.product_key
WHERE d.year = 2024
  AND d.quarter = 1
GROUP BY d.year, d.month_name, p.category_name
ORDER BY d.year, d.month, total_sales DESC;

-- Performance: 
-- - Partition pruning (only scans Q1 2024 partition)
-- - Single join to each dimension (star schema advantage)
-- - No date functions (pre-computed month_name)
-- - Query time: < 5 seconds for 100M rows
\`\`\`

**2. Weekend vs Weekday Sales:**
\`\`\`sql
SELECT 
    CASE WHEN d.is_weekend THEN 'Weekend' ELSE 'Weekday' END as day_type,
    SUM(f.total_amount) as total_sales,
    COUNT(*) as transaction_count
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
WHERE d.year = 2024
GROUP BY d.is_weekend;

-- Fast because is_weekend is indexed and pre-computed
\`\`\`

**3. Sales by Hour of Day (Staffing Optimization):**
\`\`\`sql
SELECT 
    t.hour,
    t.hour_name,
    t.time_of_day,
    SUM(f.total_amount) as hourly_sales
FROM fact_sales f
JOIN dim_time t ON f.time_key = t.time_key
WHERE f.date_key BETWEEN 20240101 AND 20240131
GROUP BY t.hour, t.hour_name, t.time_of_day
ORDER BY t.hour;
\`\`\`

**4. Top Stores by Region:**
\`\`\`sql
SELECT 
    s.region,
    s.state,
    s.store_name,
    SUM(f.total_amount) as store_sales,
    RANK() OVER (PARTITION BY s.region ORDER BY SUM(f.total_amount) DESC) as rank_in_region
FROM fact_sales f
JOIN dim_store s ON f.store_key = s.store_key
JOIN dim_date d ON f.date_key = d.date_key
WHERE d.year = 2024
GROUP BY s.region, s.state, s.store_name
HAVING RANK() <= 10;  -- Top 10 per region
\`\`\`

**Performance Optimizations:**

1. **Partitioning**: Fact table partitioned by date_key (monthly partitions)
2. **Indexes**: Dimension foreign keys indexed on fact table
3. **Materialized Views**: For common aggregations
4. **Columnar Storage**: Use columnar format (Redshift, BigQuery)
5. **Distribution Keys**: Distribute fact table by store_key for co-located joins

**Storage Estimates:**

\`\`\`
Fact table (100M transactions/day):
- 36.5B rows/year
- 100 bytes/row (10 columns)
- Raw: 3.65TB/year
- Compressed (columnar): ~400GB/year

Dimension tables:
- dim_date: 20 years × 365 days = 7,300 rows (tiny!)
- dim_time: 86,400 seconds = 86,400 rows (tiny!)
- dim_product: 100K products × 3 versions avg = 300K rows
- dim_store: 5,000 stores

Total: ~450GB/year (very manageable!)
\`\`\`

**Why Not Snowflake Schema:**

If using snowflake schema with normalized dimensions:
\`\`\`sql
-- Snowflake would require:
fact_sales 
  → dim_product → dim_category → dim_department (3 joins!)
  → dim_store → dim_city → dim_state → dim_country (4 joins!)

-- Total: 7 joins vs 3 joins in star schema
-- Query time: 3-5x slower
\`\`\`

**When Snowflake Might Be Better:**

- Storage costs dominate (unlikely with compression)
- Dimensions change frequently (need normalized updates)
- Dimension tables are huge (rare in practice)

**For this scenario (100M transactions/day), star schema is clearly superior.**`,
      keyPoints: [
        'Star schema provides better query performance for high-volume fact tables',
        'Pre-computed date dimension eliminates expensive date functions in queries',
        'Denormalized dimensions reduce joins from 7 to 3 for complex hierarchies',
        'Partition fact table by date for query performance (only scan relevant partitions)',
        'Separate time dimension enables hour-level analysis without bloating date dimension',
        'Materialized views cache common aggregations for instant dashboard performance',
      ],
    },
    {
      id: 'dw-discussion-3',
      type: 'discussion',
      question:
        "Your company is migrating from an on-premise Teradata data warehouse (10TB, $500k/year) to the cloud. You're evaluating Snowflake, Redshift, and BigQuery. The workload includes: 50 analysts running ad-hoc queries (8am-6pm weekdays), nightly ETL jobs (2am-5am), and ML model training (weekends). Discuss the architectural differences, pricing models, and provide a recommendation with cost analysis.",
      sampleAnswer: `**Comprehensive Cloud Data Warehouse Evaluation:**

Let me analyze each option considering architecture, pricing, and workload patterns.

**Current State:**
- Teradata: 10TB data warehouse
- Cost: $500k/year
- Workload: Mixed (ad-hoc queries + ETL + ML)
- Users: 50 analysts (business hours)

**Option 1: Snowflake**

**Architecture:**
- Separation of compute and storage
- Multiple virtual warehouses (independent compute clusters)
- Auto-suspend and auto-resume
- Multi-cloud (AWS, Azure, GCP)

**Workload Design:**
\`\`\`
Storage Layer: 10TB (always on, single cost)
    ↓
Virtual Warehouse "ANALYST_WH"
- Size: Large (16 nodes)
- Usage: 8am-6pm weekdays = 50 hrs/week
- Concurrent users: 50 analysts

Virtual Warehouse "ETL_WH"
- Size: X-Large (32 nodes)
- Usage: 2am-5am daily = 21 hrs/week
- Auto-suspend after 5 min idle

Virtual Warehouse "ML_WH"
- Size: 2X-Large (64 nodes)
- Usage: Weekends = 48 hrs/weekend
- Auto-suspend when not in use
\`\`\`

**Pricing Calculation:**

Storage:
\`\`\`
10TB × $23/TB/month = $230/month = $2,760/year
\`\`\`

Compute:
\`\`\`
ANALYST_WH (Large = $4/hr):
- 50 hrs/week × 52 weeks = 2,600 hrs/year
- Cost: 2,600 × $4 = $10,400/year

ETL_WH (X-Large = $8/hr):
- 21 hrs/week × 52 weeks = 1,092 hrs/year
- Cost: 1,092 × $8 = $8,736/year

ML_WH (2X-Large = $16/hr):
- 48 hrs/week × 52 weeks = 2,496 hrs/year
- Cost: 2,496 × $16 = $39,936/year

Total Compute: $59,072/year
\`\`\`

**Total Snowflake Cost: $61,832/year**

**Pros:**
- ✅ True compute-storage separation (best isolation)
- ✅ Zero management (fully managed)
- ✅ Instant scaling (add warehouse in seconds)
- ✅ Time travel (90 days query history)
- ✅ Zero-copy cloning (dev/test environments free)
- ✅ No data movement (query directly on storage)

**Cons:**
- ❌ Can get expensive if warehouses not auto-suspended
- ❌ More expensive than Redshift for always-on workloads

**Option 2: Amazon Redshift**

**Architecture:**
- MPP cluster (fixed nodes)
- Concurrency scaling (auto-scale for queries)
- Redshift Spectrum (query S3)

**Workload Design:**
\`\`\`
Base Cluster: 4× ra3.4xlarge nodes
- Storage: 10TB (included in node price)
- Compute: Always running
- Handles all workloads

Concurrency Scaling:
- Auto-activates during peak (8am-6pm)
- Adds temporary clusters for queries
- First hour/day free
\`\`\`

**Pricing Calculation:**

Base Cluster (On-Demand):
\`\`\`
4× ra3.4xlarge = 4 × $3.26/hr = $13.04/hr
24/7 operation: $13.04 × 8,760 hrs/year = $114,230/year
\`\`\`

Base Cluster (Reserved - 1 year, All Upfront):
\`\`\`
4× ra3.4xlarge reserved: $3.26/hr × 8,760 × 0.25 = $28,558/year
Savings: 75% discount!
\`\`\`

Concurrency Scaling:
\`\`\`
Peak hours: 50 hrs/week - 1 hr/day free = 45 hrs/week chargeable
45 hrs/week × 52 weeks × $13.04/hr = $30,513/year
\`\`\`

**Total Redshift Cost (Reserved + Concurrency): $59,071/year**

**Pros:**
- ✅ Predictable pricing (reserved instances)
- ✅ AWS ecosystem integration
- ✅ Mature (10+ years in market)
- ✅ Redshift Spectrum (query S3 without loading)

**Cons:**
- ❌ Cluster always running (pay 24/7 even if not using)
- ❌ Scaling requires cluster resize (slower than Snowflake)
- ❌ Concurrency scaling adds significant cost

**Option 3: Google BigQuery**

**Architecture:**
- Fully serverless (no clusters)
- Automatic scaling (1000s of workers)
- Pay per query (charge by TB scanned)

**Pricing Model:**
\`\`\`
Storage: $20/TB/month (first 10GB free)
Queries: $5/TB scanned (first 1TB/month free)
\`\`\`

**Workload Analysis:**

Analysts (50 users, business hours):
\`\`\`
- Avg query scans: 100GB (after compression, partition pruning)
- Queries/day: 50 users × 10 queries = 500 queries/day
- Queries/month: 500 × 22 business days = 11,000 queries/month
- Data scanned: 11,000 × 100GB = 1,100TB/month

Cost: 1,100TB × $5/TB = $5,500/month = $66,000/year
\`\`\`

ETL Jobs:
\`\`\`
- Nightly full scan: 10TB raw → 2TB after compression
- 30 nights/month: 30 × 2TB = 60TB/month

Cost: 60TB × $5/TB = $300/month = $3,600/year
\`\`\`

ML Training (Weekends):
\`\`\`
- Full dataset scans: 10TB → 2TB compressed
- 8 runs/month: 8 × 2TB = 16TB/month

Cost: 16TB × $5/TB = $80/month = $960/year
\`\`\`

**Total BigQuery Cost:**
\`\`\`
Storage: 10TB × $20/TB/month = $200/month = $2,400/year
Queries: $66,000 + $3,600 + $960 = $70,560/year

Total: $72,960/year
\`\`\`

**Pros:**
- ✅ Zero administration (fully serverless)
- ✅ Instant scaling (no waiting)
- ✅ Pay only for what you query
- ✅ BigQuery ML (train models in SQL)
- ✅ Real-time ingestion

**Cons:**
- ❌ Expensive for high query volume
- ❌ Query cost can be unpredictable
- ❌ Charges by data scanned (encourages optimization but adds complexity)

**Cost Comparison:**

| Provider | Annual Cost | vs Teradata | Notes |
|----------|-------------|-------------|-------|
| Teradata (current) | $500,000 | Baseline | On-premise |
| **Snowflake** | **$61,832** | **88% savings** | Best flexibility |
| Redshift (reserved) | $59,071 | 88% savings | Best AWS integration |
| BigQuery | $72,960 | 85% savings | Best for serverless |

**My Recommendation: Snowflake**

**Why Snowflake:**

1. **Workload Isolation**: Separate warehouses for analysts, ETL, ML
   - ETL doesn't impact analyst queries
   - ML training doesn't slow down dashboards
   - Each workload independently scalable

2. **Cost Optimization**:
   - Auto-suspend warehouses (only pay when running)
   - Analysts: 50 hrs/week (not 168 hrs)
   - ETL: 21 hrs/week (not 168 hrs)
   - Savings: ~70% vs always-on cluster

3. **Future-Proof**:
   - Multi-cloud (can move AWS → Azure if needed)
   - Zero-copy cloning (dev/test/staging free)
   - Time travel (recover from errors easily)

4. **Developer Experience**:
   - Instant warehouse creation
   - No cluster management
   - No vacuum/analyze maintenance

**Implementation Plan:**

**Phase 1: Setup (Week 1)**
\`\`\`sql
-- Create virtual warehouses
CREATE WAREHOUSE ANALYST_WH
    WAREHOUSE_SIZE = LARGE
    AUTO_SUSPEND = 300  -- 5 minutes
    AUTO_RESUME = TRUE;

CREATE WAREHOUSE ETL_WH
    WAREHOUSE_SIZE = XLARGE
    AUTO_SUSPEND = 60   -- 1 minute
    AUTO_RESUME = TRUE;

CREATE WAREHOUSE ML_WH
    WAREHOUSE_SIZE = 2XLARGE
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE;
\`\`\`

**Phase 2: Migration (Weeks 2-4)**
\`\`\`bash
# Export from Teradata
bteq < export_script.sql > data.csv

# Load to S3
aws s3 sync ./exports s3://migration-bucket/

# Load to Snowflake (parallel, fast!)
COPY INTO customers
FROM 's3://migration-bucket/customers/'
FILE_FORMAT = (TYPE = CSV)
ON_ERROR = CONTINUE;
\`\`\`

**Phase 3: Optimization (Week 5)**
\`\`\`sql
-- Cluster tables for common queries
ALTER TABLE fact_sales 
CLUSTER BY (date_key, customer_key);

-- Create materialized views for dashboards
CREATE MATERIALIZED VIEW mv_daily_sales AS
SELECT date_key, SUM(amount) as daily_sales
FROM fact_sales
GROUP BY date_key;
\`\`\`

**Cost Optimization Strategies:**

1. **Resource Monitors**:
\`\`\`sql
CREATE RESOURCE MONITOR analyst_monitor
WITH CREDIT_QUOTA = 100  -- $100/month limit
TRIGGERS ON 90 PERCENT DO NOTIFY
         ON 100 PERCENT DO SUSPEND;
\`\`\`

2. **Query Optimization**:
   - Partition pruning (query only necessary partitions)
   - Cluster keys (sort data for common filters)
   - Result caching (repeated queries instant)

3. **Right-Sizing**:
   - Start with LARGE warehouse
   - Monitor query performance
   - Scale up only if needed

**Expected Savings: $438,168/year (88%) vs Teradata**

**Alternative: Cost-Optimized Hybrid**

For even lower cost:
\`\`\`
Data Lake (S3): 10TB × $23/TB/year = $230/year
Snowflake (compute only): $20,000/year (smaller warehouses)
BigQuery (ad-hoc only): $10,000/year

Total: $30,230/year (94% savings!)
\`\`\`

Use S3 as primary storage, Snowflake for regular analytics, BigQuery for ad-hoc exploration.

**Final Recommendation: Snowflake at $61,832/year provides best balance of cost, performance, and flexibility for this workload.**`,
      keyPoints: [
        'Compute-storage separation (Snowflake, BigQuery) enables paying only for active compute',
        'Redshift reserved instances provide predictable costs but pay 24/7 even when idle',
        'BigQuery pay-per-query model expensive for high query volume (11k queries/month)',
        'Workload isolation (separate warehouses) prevents ETL/ML from impacting analyst queries',
        'Auto-suspend in Snowflake saves ~70% vs always-on clusters (50 hrs vs 168 hrs/week)',
        'Cloud migration saves 85-88% vs on-premise Teradata ($60k vs $500k annually)',
      ],
    },
  ],
};

export default dataWarehousingDiscussionQuiz;
