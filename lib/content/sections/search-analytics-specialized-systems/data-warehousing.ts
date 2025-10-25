import { ModuleSection } from '@/lib/types';

const dataWarehousingSection: ModuleSection = {
  id: 'data-warehousing',
  title: 'Data Warehousing',
  content: `
# Data Warehousing

## Introduction

A **data warehouse** is a centralized repository optimized for analytics and reporting, designed to support business intelligence (BI) and decision-making. Unlike operational databases (OLTP) that handle transactions, data warehouses (OLAP) are built for complex queries across massive datasets.

This section covers dimensional modeling, star/snowflake schemas, slowly changing dimensions, MPP architecture, and modern cloud data warehouses.

## What is a Data Warehouse?

**Definition**: Subject-oriented, integrated, time-variant, non-volatile collection of data for decision support.

**Key Characteristics:**
- **Subject-oriented**: Organized around subjects (customers, products, sales)
- **Integrated**: Data from multiple sources consolidated
- **Time-variant**: Historical data retained (snapshots over time)
- **Non-volatile**: Data loaded and read, rarely updated
- **Optimized for queries**: Complex aggregations, not transactions

## OLTP vs OLAP

| Aspect | OLTP (Operational) | OLAP (Analytical) |
|--------|-------------------|-------------------|
| Purpose | Run business | Analyze business |
| Queries | Simple, fast | Complex, slow |
| Data | Current | Historical |
| Volume | GB to TB | TB to PB |
| Updates | Frequent INSERT/UPDATE | Batch load |
| Schema | Normalized (3NF) | Denormalized (star) |
| Users | 1000s concurrent | 10-100 analysts |
| Example | PostgreSQL, MySQL | Snowflake, Redshift |

## Dimensional Modeling

### Fact Tables

**Description**: Contain measurable business events (facts).

**Example: Sales Fact Table**
\`\`\`sql
CREATE TABLE fact_sales (
    sale_id BIGINT PRIMARY KEY,
    date_key INTEGER REFERENCES dim_date(date_key),
    product_key INTEGER REFERENCES dim_product(product_key),
    customer_key INTEGER REFERENCES dim_customer(customer_key),
    store_key INTEGER REFERENCES dim_store(store_key),
    
    -- Measures (facts)
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    discount_amount DECIMAL(10,2),
    tax_amount DECIMAL(10,2)
);
\`\`\`

**Characteristics:**
- **Foreign keys** to dimension tables
- **Measures** (numeric facts to aggregate)
- **Grain**: Level of detail (one row per transaction)
- **Large**: Billions of rows

### Dimension Tables

**Description**: Provide context (who, what, where, when, why).

**Example: Product Dimension**
\`\`\`sql
CREATE TABLE dim_product (
    product_key INTEGER PRIMARY KEY,
    product_id VARCHAR(50),  -- Business key
    product_name VARCHAR(200),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    brand VARCHAR(100),
    manufacturer VARCHAR(100),
    unit_cost DECIMAL(10,2),
    list_price DECIMAL(10,2),
    -- Metadata
    effective_date DATE,
    expiration_date DATE,
    is_current BOOLEAN
);
\`\`\`

**Characteristics:**
- **Descriptive attributes**
- **Relatively small** (thousands to millions of rows)
- **Slowly changing** (updated periodically)
- **Denormalized** (category and subcategory in same table)

## Star Schema

**Description**: Fact table in center, dimension tables radiate out (like a star).

\`\`\`
           dim_date
               |
dim_customer - fact_sales - dim_product
               |
           dim_store
\`\`\`

**Example Query:**
\`\`\`sql
SELECT 
    d.year,
    d.quarter,
    p.category,
    s.region,
    SUM(f.total_amount) as total_sales,
    COUNT(*) as transaction_count
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_product p ON f.product_key = p.product_key
JOIN dim_store s ON f.store_key = s.store_key
WHERE d.year = 2024
  AND s.country = 'US'
GROUP BY d.year, d.quarter, p.category, s.region;
\`\`\`

**Advantages:**
- Simple to understand
- Fast queries (fewer joins)
- Denormalized dimensions (no sub-joins)
- Optimized for BI tools

**Disadvantages:**
- Data redundancy in dimensions
- Larger dimension table size
- Update anomalies if not careful

## Snowflake Schema

**Description**: Normalized dimensions (subcategory table separate from category).

\`\`\`
                dim_date
                    |
    dim_customer - fact_sales - dim_product - dim_category - dim_manufacturer
                    |
                dim_store - dim_region - dim_country
\`\`\`

**Example:**
\`\`\`sql
-- Product dimension (normalized)
CREATE TABLE dim_product (
    product_key INTEGER PRIMARY KEY,
    product_name VARCHAR(200),
    category_key INTEGER REFERENCES dim_category(category_key)
);

CREATE TABLE dim_category (
    category_key INTEGER PRIMARY KEY,
    category_name VARCHAR(100),
    department_key INTEGER REFERENCES dim_department(department_key)
);

CREATE TABLE dim_department (
    department_key INTEGER PRIMARY KEY,
    department_name VARCHAR(100)
);
\`\`\`

**Advantages:**
- Less redundancy
- Smaller dimension tables
- Easier to maintain hierarchies

**Disadvantages:**
- More complex queries (more joins)
- Slower performance
- Less intuitive for business users

## Star vs Snowflake: When to Use

**Use Star Schema:**
- Performance critical
- Simpler is better
- BI tool compatibility
- **Most common in practice**

**Use Snowflake Schema:**
- Storage costs critical
- Complex hierarchies
- Need to update dimensions frequently

**Reality**: Most data warehouses use **star schema** for performance.

## Slowly Changing Dimensions (SCD)

Dimensions change over time. How do we handle historical changes?

### Type 0: Retain Original

**Rule**: Never change. Keep original value.

\`\`\`sql
-- Customer moves, but we keep original city
customer_key | customer_id | name  | city
1            | C001        | Alice | NYC   -- Never changes
\`\`\`

**Use case**: Birth date, original signup location.

### Type 1: Overwrite

**Rule**: Update in place, no history.

\`\`\`sql
-- Customer moves, overwrite city
UPDATE dim_customer 
SET city = 'Boston' 
WHERE customer_id = 'C001';

-- Result: No history of NYC
customer_key | customer_id | name  | city
1            | C001        | Alice | Boston
\`\`\`

**Use case**: Corrections, data quality fixes.

### Type 2: Add New Row (Most Common)

**Rule**: Insert new row with new values, mark old row as expired.

\`\`\`sql
-- Before (Alice in NYC)
customer_key | customer_id | name  | city   | effective_date | expiration_date | is_current
1            | C001        | Alice | NYC    | 2023-01-01     | 9999-12-31      | TRUE

-- Alice moves to Boston on 2024-06-15
-- Update old row
UPDATE dim_customer 
SET expiration_date = '2024-06-14', is_current = FALSE
WHERE customer_key = 1;

-- Insert new row
INSERT INTO dim_customer VALUES
(2, 'C001', 'Alice', 'Boston', '2024-06-15', '9999-12-31', TRUE);

-- Result: Full history!
customer_key | customer_id | name  | city   | effective_date | expiration_date | is_current
1            | C001        | Alice | NYC    | 2023-01-01     | 2024-06-14      | FALSE
2            | C001        | Alice | Boston | 2024-06-15     | 9999-12-31      | TRUE
\`\`\`

**Query for historical accuracy:**
\`\`\`sql
-- Sales in Q1 2024 (Alice was in NYC)
SELECT c.city, SUM(f.total_amount)
FROM fact_sales f
JOIN dim_customer c ON f.customer_key = c.customer_key
WHERE f.date BETWEEN '2024-01-01' AND '2024-03-31'
  AND c.customer_id = 'C001'
  AND f.date BETWEEN c.effective_date AND c.expiration_date
GROUP BY c.city;

-- Result: NYC (correct!)
\`\`\`

**Use case**: When history matters (customer address, product price, employee department).

### Type 3: Add New Column

**Rule**: Keep old and new values in separate columns.

\`\`\`sql
customer_key | customer_id | name  | current_city | previous_city
1            | C001        | Alice | Boston       | NYC
\`\`\`

**Use case**: Need to compare old vs new (price increase analysis).

### Type 4: Mini-Dimension

**Rule**: Split rapidly changing attributes into separate table.

\`\`\`sql
-- Stable attributes
dim_customer: customer_key, name, birthdate

-- Rapidly changing attributes
dim_customer_demographics: demo_key, income_range, credit_score
\`\`\`

**Use case**: Attributes that change daily/weekly (credit score, loyalty tier).

## Massively Parallel Processing (MPP)

**Description**: Distribute data and queries across many nodes.

### Shared-Nothing Architecture

\`\`\`
Query Coordinator
       ↓
   ┌───┴───┬───────┬───────┐
   ↓       ↓       ↓       ↓
 Node1   Node2   Node3   Node4
 (CPU)   (CPU)   (CPU)   (CPU)
 (RAM)   (RAM)   (RAM)   (RAM)
 (Disk)  (Disk)  (Disk)  (Disk)
\`\`\`

**Each node:**
- Independent CPU, RAM, disk
- Processes its data partition
- No contention

**Query Execution:**
\`\`\`sql
SELECT region, SUM(sales) FROM orders GROUP BY region;

-- Execution plan:
-- 1. Each node sums sales for its partition
-- 2. Coordinator aggregates results
-- 3. Returns final result
\`\`\`

**Benefits:**
- Linear scalability (add more nodes = more performance)
- Parallel processing (10 nodes = 10x throughput)
- Fault tolerance (node failure doesn't lose data)

**Technologies:**
- Teradata
- Greenplum
- Snowflake
- Redshift
- BigQuery

## Modern Cloud Data Warehouses

### Snowflake

**Architecture**: Compute-storage separation.

\`\`\`
Storage Layer (S3) → Always available
       ↑
Virtual Warehouses (compute) → Scale independently
       ↑
Cloud Services (metadata, optimization)
\`\`\`

**Key Features:**
- Auto-scaling compute
- Pay per second
- Zero-copy cloning
- Time travel (query historical data)
- Multi-cloud (AWS, Azure, GCP)

**Pricing:**
- Storage: $23/TB/month (compressed)
- Compute: $2-4/hour per warehouse

### Amazon Redshift

**Architecture**: MPP columnar storage.

**Key Features:**
- Redshift Spectrum (query S3 directly)
- Concurrency scaling (auto-scale for queries)
- Materialized views
- ML integration (Redshift ML)

**Pricing:**
- On-demand: $0.25/hour per node
- Reserved: 75% discount (1-3 year)

### Google BigQuery

**Architecture**: Serverless, auto-scaling.

**Key Features:**
- Serverless (no cluster management)
- Real-time analytics
- ML built-in (BigQuery ML)
- Pay per query

**Pricing:**
- Storage: $20/TB/month
- Queries: $5/TB scanned (first 1TB free/month)

## ETL for Data Warehouses

\`\`\`
Source Systems → Extract → Transform → Load → Data Warehouse
                    ↓         ↓          ↓
                  Fivetran  dbt     Airflow
\`\`\`

**dbt (data build tool):**
\`\`\`sql
-- models/staging/stg_orders.sql
SELECT
    order_id,
    customer_id,
    order_date,
    total_amount
FROM {{ source('ecommerce', 'orders') }}
WHERE order_date >= '2024-01-01'

-- models/marts/fct_daily_sales.sql
SELECT
    DATE(order_date) as date_key,
    SUM(total_amount) as total_sales
FROM {{ ref('stg_orders') }}
GROUP BY DATE(order_date)
\`\`\`

## Best Practices

1. **Design for queries, not loads**: Denormalize for performance
2. **Use surrogate keys**: Auto-increment integers as primary keys
3. **Implement SCD Type 2**: Preserve history
4. **Partition fact tables**: By date for efficient queries
5. **Create aggregate tables**: Pre-compute common metrics
6. **Use materialized views**: Cache expensive computations

## Summary

Data warehousing enables business intelligence through:
- **Dimensional modeling**: Fact and dimension tables
- **Star schema**: Simple, fast, denormalized
- **SCD Type 2**: Historical accuracy
- **MPP architecture**: Parallel processing
- **Cloud warehouses**: Snowflake, Redshift, BigQuery

Modern warehouses separate compute and storage, enabling flexible scaling and pay-per-use pricing.
`,
};

export default dataWarehousingSection;
