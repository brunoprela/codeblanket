import { Content } from '@/lib/types';

export const databasesFinancialData: Content = {
  title: 'Databases for Financial Data',
  subtitle: 'Efficient storage and retrieval of market data at scale',
  description:
    'Master database technologies for storing and querying financial time series data. Learn to design schemas, optimize queries, and build high-performance data pipelines for quantitative trading.',
  sections: [
    {
      title: 'Why Databases for Financial Data?',
      content: `
# The Case for Databases in Trading

## Problems with File-Based Storage

\`\`\`plaintext
CSV/Parquet Files Approach:

data/
├── AAPL_2020.csv
├── AAPL_2021.csv
├── AAPL_2022.csv
├── AAPL_2023.csv
├── MSFT_2020.csv
...
└── [5000 files later]

Challenges:
├── File Organization Nightmare
│   └── Finding data across thousands of files
├── No Concurrent Access
│   └── Multiple researchers overwriting files
├── Slow Queries
│   └── Must load entire file to filter
├── Data Duplication
│   └── Same date range across multiple analyses
├── No Data Integrity
│   └── Corrupted files, missing data, duplicates
└── Difficult Updates
    └── Append-only, hard to correct errors
\`\`\`

## What Databases Provide

### 1. **Structured Queries**
\`\`\`sql
-- Get AAPL prices for specific date range
SELECT date, close, volume
FROM prices
WHERE ticker = 'AAPL'
  AND date BETWEEN '2023-01-01' AND '2023-12-31'
ORDER BY date;

-- Instant results from billions of rows
\`\`\`

### 2. **Concurrent Access**
Multiple users and processes can read/write simultaneously without conflicts.

### 3. **Data Integrity**
- Constraints prevent bad data (e.g., negative prices)
- Transactions ensure consistency
- Indexes speed up queries

### 4. **Efficient Updates**
\`\`\`sql
-- Fix incorrect price
UPDATE prices
SET close = 150.25
WHERE ticker = 'AAPL' AND date = '2023-10-15';

-- Append new data
INSERT INTO prices (date, ticker, open, high, low, close, volume)
VALUES ('2024-01-15', 'AAPL', 185.50, 187.25, 184.75, 186.80, 52341567);
\`\`\`

### 5. **Scalability**
Handle terabytes of data efficiently with proper indexing.

## Database Types for Finance

\`\`\`plaintext
Financial Data Storage Options:

├── Relational Databases (SQL)
│   ├── PostgreSQL
│   │   ├── General purpose, very reliable
│   │   ├── Great for structured data
│   │   └── Excellent for complex queries
│   ├── TimescaleDB (PostgreSQL extension)
│   │   ├── Optimized for time series
│   │   ├── Automatic partitioning
│   │   └── Time-based aggregations
│   └── MySQL / MariaDB
│       └── Alternative to PostgreSQL
├── Time Series Databases
│   ├── InfluxDB
│   │   ├── Purpose-built for time series
│   │   ├── High write throughput
│   │   └── Automatic downsampling
│   └── Questd
│       ├── Ultra-fast columnar database
│       └── Excellent for financial data
├── NoSQL Databases
│   ├── MongoDB
│   │   ├── Flexible schema
│   │   └── Good for unstructured data (news, filings)
│   └── Cassandra
│       └── Massive scale distributed database
└── Specialized
    ├── KDB+/q
    │   ├── Industry standard (investment banks)
    │   └── Extremely fast, expensive
    └── Arctic (MongoDB-based)
        └── Designed for financial time series
\`\`\`
      `,
    },
    {
      title: 'PostgreSQL for Financial Data',
      content: `
# PostgreSQL: The Workhorse Database

## Installation and Setup

\`\`\`bash
# Install PostgreSQL (macOS)
brew install postgresql
brew services start postgresql

# Install PostgreSQL (Ubuntu)
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql

# Install PostgreSQL (Docker - recommended)
docker run --name postgres-trading \\
  -e POSTGRES_PASSWORD=mypassword \\
  -e POSTGRES_DB=marketdata \\
  -p 5432:5432 \\
  -v pgdata:/var/lib/postgresql/data \\
  -d postgres:15

# Connect to database
psql -U postgres -d marketdata
\`\`\`

## Schema Design for OHLCV Data

\`\`\`sql
-- Create database
CREATE DATABASE marketdata;

-- Connect to database
\\c marketdata

-- Create prices table
CREATE TABLE prices (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open NUMERIC(12, 4) NOT NULL,
    high NUMERIC(12, 4) NOT NULL,
    low NUMERIC(12, 4) NOT NULL,
    close NUMERIC(12, 4) NOT NULL,
    volume BIGINT NOT NULL,
    adj_close NUMERIC(12, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Prevent duplicate entries
    UNIQUE(date, ticker)
);

-- Create indexes for fast queries
CREATE INDEX idx_prices_ticker ON prices (ticker);
CREATE INDEX idx_prices_date ON prices (date);
CREATE INDEX idx_prices_ticker_date ON prices (ticker, date);

-- Explain indexes:
-- Single column: idx_prices_ticker
--   Fast when: WHERE ticker = 'AAPL'
-- 
-- Single column: idx_prices_date
--   Fast when: WHERE date = '2023-10-15'
--
-- Composite: idx_prices_ticker_date
--   Fast when: WHERE ticker = 'AAPL' AND date BETWEEN '2023-01-01' AND '2023-12-31'
\`\`\`

## Loading Data into PostgreSQL

### From Python

\`\`\`python
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import yfinance as yf

# Create connection
engine = create_engine('postgresql://postgres:mypassword@localhost:5432/marketdata')

# Download data
df = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
df = df.reset_index()

# Prepare dataframe
df['ticker'] = 'AAPL'
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# Insert to database
df.to_sql('prices', engine, if_exists='append', index=False, method='multi')

print(f"Inserted {len (df)} rows")
\`\`\`

### Bulk Insert (Faster)

\`\`\`python
from psycopg2 import sql
from psycopg2.extras import execute_values

# Download multiple tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
dfs = []

for ticker in tickers:
    df = yf.download (ticker, start='2020-01-01', end='2024-01-01', progress=False)
    df = df.reset_index()
    df['ticker'] = ticker
    dfs.append (df)

combined = pd.concat (dfs, ignore_index=True)
combined.columns = [col.lower().replace(' ', '_') for col in combined.columns]

# Bulk insert using execute_values (much faster)
conn = psycopg2.connect(
    host='localhost',
    database='marketdata',
    user='postgres',
    password='mypassword'
)

cur = conn.cursor()

# Prepare data as list of tuples
data_tuples = [
    (
        row['date'],
        row['ticker'],
        float (row['open']),
        float (row['high']),
        float (row['low']),
        float (row['close']),
        int (row['volume']),
        float (row['adj close']) if 'adj close' in row else None
    )
    for _, row in combined.iterrows()
]

# Bulk insert
insert_query = \"\"\"
    INSERT INTO prices (date, ticker, open, high, low, close, volume, adj_close)
    VALUES %s
    ON CONFLICT (date, ticker) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        adj_close = EXCLUDED.adj_close
\"\"\"

execute_values (cur, insert_query, data_tuples, page_size=1000)
conn.commit()

cur.close()
conn.close()

print(f"Inserted {len (data_tuples)} rows")
\`\`\`

## Querying Financial Data

\`\`\`python
# Query data back
query = \"\"\"
    SELECT date, ticker, close, volume
    FROM prices
    WHERE ticker IN ('AAPL', 'MSFT')
      AND date >= '2023-01-01'
    ORDER BY ticker, date
\"\"\"

df = pd.read_sql (query, engine)
print(df.head())

# Calculate returns
df['returns'] = df.groupby('ticker')['close'].pct_change()

# More complex query: Moving averages
query = \"\"\"
    SELECT 
        date,
        ticker,
        close,
        AVG(close) OVER (
            PARTITION BY ticker 
            ORDER BY date 
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) as sma_20,
        AVG(close) OVER (
            PARTITION BY ticker 
            ORDER BY date 
            ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
        ) as sma_50
    FROM prices
    WHERE ticker = 'AAPL'
      AND date >= '2023-01-01'
    ORDER BY date
\"\"\"

df = pd.read_sql (query, engine)
print(df.tail())
\`\`\`

## Advanced PostgreSQL Features

### Partitioning for Large Tables

\`\`\`sql
-- Partition by year for very large datasets
CREATE TABLE prices (
    id BIGSERIAL,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open NUMERIC(12, 4),
    high NUMERIC(12, 4),
    low NUMERIC(12, 4),
    close NUMERIC(12, 4),
    volume BIGINT,
    UNIQUE(date, ticker, id)
) PARTITION BY RANGE (date);

-- Create partitions for each year
CREATE TABLE prices_2020 PARTITION OF prices
    FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');

CREATE TABLE prices_2021 PARTITION OF prices
    FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');

CREATE TABLE prices_2022 PARTITION OF prices
    FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');

CREATE TABLE prices_2023 PARTITION OF prices
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

-- PostgreSQL automatically routes queries to correct partition
-- Queries for 2023 data only scan prices_2023 table (much faster)
\`\`\`

### Materialized Views for Performance

\`\`\`sql
-- Create materialized view for daily stats
CREATE MATERIALIZED VIEW daily_statistics AS
SELECT 
    date,
    ticker,
    close,
    volume,
    (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) / LAG(close) OVER (PARTITION BY ticker ORDER BY date) as daily_return,
    AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as sma_20
FROM prices
ORDER BY ticker, date;

-- Create index on materialized view
CREATE INDEX idx_daily_stats_ticker_date ON daily_statistics (ticker, date);

-- Refresh materialized view (after new data loaded)
REFRESH MATERIALIZED VIEW CONCURRENTLY daily_statistics;

-- Query materialized view (very fast)
SELECT * FROM daily_statistics
WHERE ticker = 'AAPL' AND date >= '2023-01-01';
\`\`\`
      `,
    },
    {
      title: 'TimescaleDB for Time Series',
      content: `
# TimescaleDB: PostgreSQL for Time Series

## Why TimescaleDB?

TimescaleDB is a PostgreSQL extension optimized for time-series data:

\`\`\`plaintext
Benefits for Financial Data:
├── Automatic Partitioning (Hypertables)
│   └── Data automatically split by time
├── Time-Based Queries Optimized
│   └── 10-100x faster than regular PostgreSQL
├── Compression
│   └── 90%+ compression ratio
├── Continuous Aggregates
│   └── Real-time pre-computed aggregations
├── Data Retention Policies
│   └── Automatic old data deletion
└── All PostgreSQL Features
    └── Full SQL, indexes, constraints, etc.
\`\`\`

## Installation

\`\`\`bash
# Docker (easiest)
docker run -d --name timescaledb \\
  -p 5432:5432 \\
  -e POSTGRES_PASSWORD=password \\
  timescale/timescaledb:latest-pg15

# Or install extension on existing PostgreSQL
# See: https://docs.timescale.com/install/
\`\`\`

## Creating Hypertables

\`\`\`sql
-- Connect to database
\\c marketdata

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create regular table
CREATE TABLE prices (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open NUMERIC(12, 4),
    high NUMERIC(12, 4),
    low NUMERIC(12, 4),
    close NUMERIC(12, 4),
    volume BIGINT
);

-- Convert to hypertable (this is the magic!)
SELECT create_hypertable('prices', 'time');

-- Create indexes
CREATE INDEX idx_prices_ticker_time ON prices (ticker, time DESC);
CREATE INDEX idx_prices_time ON prices (time DESC);

-- TimescaleDB automatically partitions data by time
-- No manual partition management needed!
\`\`\`

## Loading Data

\`\`\`python
import pandas as pd
from sqlalchemy import create_engine
import yfinance as yf

# Connect to TimescaleDB
engine = create_engine('postgresql://postgres:password@localhost:5432/marketdata')

# Download data with timezone-aware timestamps
df = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
df = df.reset_index()

# Prepare data
df['ticker'] = 'AAPL'
df['time'] = pd.to_datetime (df['Date']).dt.tz_localize('America/New_York')
df = df.drop('Date', axis=1)
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# Insert
df.to_sql('prices', engine, if_exists='append', index=False, method='multi')
\`\`\`

## Time-Bucket Aggregations

TimescaleDB's killer feature for financial data:

\`\`\`sql
-- Convert minute data to hourly bars
SELECT 
    time_bucket('1 hour', time) AS hour,
    ticker,
    first (open, time) AS open,
    max (high) AS high,
    min (low) AS low,
    last (close, time) AS close,
    sum (volume) AS volume
FROM prices
WHERE ticker = 'AAPL'
  AND time >= NOW() - INTERVAL '1 week'
GROUP BY hour, ticker
ORDER BY hour DESC;

-- Daily bars from minute data
SELECT 
    time_bucket('1 day', time) AS day,
    ticker,
    first (open, time) AS open,
    max (high) AS high,
    min (low) AS low,
    last (close, time) AS close,
    sum (volume) AS volume
FROM prices
WHERE ticker = 'AAPL'
  AND time >= '2023-01-01'
GROUP BY day, ticker
ORDER BY day;

-- Weekly bars
SELECT 
    time_bucket('1 week', time) AS week,
    ticker,
    first (open, time) AS open,
    max (high) AS high,
    min (low) AS low,
    last (close, time) AS close,
    sum (volume) AS volume
FROM prices
WHERE ticker = 'AAPL'
GROUP BY week, ticker
ORDER BY week DESC
LIMIT 52;  -- Last year of weekly data
\`\`\`

## Continuous Aggregates (Real-Time Views)

\`\`\`sql
-- Create continuous aggregate for daily OHLCV
-- Automatically updated as new data arrives!
CREATE MATERIALIZED VIEW prices_daily
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS day,
    ticker,
    first (open, time) AS open,
    max (high) AS high,
    min (low) AS low,
    last (close, time) AS close,
    sum (volume) AS volume,
    count(*) AS num_bars
FROM prices
GROUP BY day, ticker;

-- Add refresh policy (update every hour)
SELECT add_continuous_aggregate_policy('prices_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- Query is instant (pre-aggregated)
SELECT * FROM prices_daily
WHERE ticker = 'AAPL'
  AND day >= '2023-01-01'
ORDER BY day DESC;

-- Create weekly aggregate from daily aggregate
CREATE MATERIALIZED VIEW prices_weekly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 week', day) AS week,
    ticker,
    first (open, day) AS open,
    max (high) AS high,
    min (low) AS low,
    last (close, day) AS close,
    sum (volume) AS volume
FROM prices_daily
GROUP BY week, ticker;
\`\`\`

## Compression

\`\`\`sql
-- Enable compression (90%+ space savings!)
ALTER TABLE prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby = 'time DESC'
);

-- Add compression policy (compress data older than 7 days)
SELECT add_compression_policy('prices', INTERVAL '7 days');

-- Manually compress older data
SELECT compress_chunk (chunk)
FROM show_chunks('prices', older_than => INTERVAL '30 days') AS chunk;

-- Check compression stats
SELECT 
    pg_size_pretty (before_compression_total_bytes) as before,
    pg_size_pretty (after_compression_total_bytes) as after,
    round((1 - after_compression_total_bytes::numeric / before_compression_total_bytes::numeric) * 100, 2) as compression_ratio
FROM timescaledb_information.hypertable_compression_stats
WHERE hypertable_name = 'prices';

-- Typical results:
-- before: 150 GB
-- after: 12 GB
-- compression_ratio: 92%
\`\`\`

## Data Retention Policies

\`\`\`sql
-- Automatically drop data older than 2 years
SELECT add_retention_policy('prices', INTERVAL '2 years');

-- Manually drop old chunks
SELECT drop_chunks('prices', INTERVAL '3 years');

-- Useful for:
-- - Minute data: Keep 90 days, drop older
-- - Daily data: Keep 10 years
-- - Tick data: Keep 30 days
\`\`\`

## Query Performance Examples

\`\`\`python
import time
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:password@localhost:5432/marketdata')

# Query 1 billion rows of minute data
# Regular PostgreSQL: 45 seconds
# TimescaleDB hypertable: 2 seconds (22x faster)

query = \"\"\"
    SELECT 
        time_bucket('1 hour', time) AS hour,
        ticker,
        first (open, time) AS open,
        max (high) AS high,
        min (low) AS low,
        last (close, time) AS close,
        sum (volume) AS volume
    FROM prices
    WHERE ticker = 'AAPL'
      AND time >= NOW() - INTERVAL '1 year'
    GROUP BY hour, ticker
    ORDER BY hour DESC
\"\"\"

start = time.time()
df = pd.read_sql (query, engine)
print(f"Query time: {time.time() - start:.2f}s")
print(f"Rows returned: {len (df)}")

# Result: 2.1 seconds for 1 year of hourly data from 1 billion rows
\`\`\`
      `,
    },
    {
      title: 'Database Design Patterns',
      content: `
# Schema Design for Trading Systems

## Multi-Asset Schema

\`\`\`sql
-- Tickers table (reference data)
CREATE TABLE tickers (
    ticker_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(255),
    exchange VARCHAR(50),
    sector VARCHAR(100),
    industry VARCHAR(100),
    currency VARCHAR(3) DEFAULT 'USD',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prices table (OHLCV)
CREATE TABLE prices (
    time TIMESTAMPTZ NOT NULL,
    ticker_id INTEGER NOT NULL REFERENCES tickers (ticker_id),
    open NUMERIC(12, 4),
    high NUMERIC(12, 4),
    low NUMERIC(12, 4),
    close NUMERIC(12, 4),
    volume BIGINT,
    PRIMARY KEY (time, ticker_id)
);

-- Convert to hypertable
SELECT create_hypertable('prices', 'time');
CREATE INDEX idx_prices_ticker_time ON prices (ticker_id, time DESC);

-- Fundamentals table
CREATE TABLE fundamentals (
    ticker_id INTEGER REFERENCES tickers (ticker_id),
    date DATE NOT NULL,
    revenue NUMERIC(20, 2),
    earnings NUMERIC(20, 2),
    eps NUMERIC(10, 4),
    pe_ratio NUMERIC(10, 2),
    market_cap BIGINT,
    PRIMARY KEY (ticker_id, date)
);

-- Splits and dividends
CREATE TABLE corporate_actions (
    action_id SERIAL PRIMARY KEY,
    ticker_id INTEGER REFERENCES tickers (ticker_id),
    date DATE NOT NULL,
    action_type VARCHAR(20) NOT NULL,  -- 'split', 'dividend', 'merger'
    split_ratio NUMERIC(10, 6),  -- e.g., 2.0 for 2:1 split
    dividend_amount NUMERIC(10, 4),
    currency VARCHAR(3),
    notes TEXT
);

-- Metadata tracking
CREATE TABLE data_updates (
    update_id SERIAL PRIMARY KEY,
    ticker_id INTEGER REFERENCES tickers (ticker_id),
    update_type VARCHAR(50),  -- 'price', 'fundamental', 'action'
    start_date DATE,
    end_date DATE,
    records_added INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
\`\`\`

## Handling Different Data Frequencies

\`\`\`sql
-- Separate tables for different frequencies
CREATE TABLE prices_tick (
    time TIMESTAMPTZ NOT NULL,
    ticker_id INTEGER NOT NULL,
    price NUMERIC(12, 4),
    size INTEGER,  -- Trade size
    exchange VARCHAR(10)
);
SELECT create_hypertable('prices_tick', 'time');

CREATE TABLE prices_minute (
    time TIMESTAMPTZ NOT NULL,
    ticker_id INTEGER NOT NULL,
    open NUMERIC(12, 4),
    high NUMERIC(12, 4),
    low NUMERIC(12, 4),
    close NUMERIC(12, 4),
    volume BIGINT,
    num_trades INTEGER,
    vwap NUMERIC(12, 4),
    PRIMARY KEY (time, ticker_id)
);
SELECT create_hypertable('prices_minute', 'time');

CREATE TABLE prices_daily (
    date DATE NOT NULL,
    ticker_id INTEGER NOT NULL,
    open NUMERIC(12, 4),
    high NUMERIC(12, 4),
    low NUMERIC(12, 4),
    close NUMERIC(12, 4),
    volume BIGINT,
    adj_close NUMERIC(12, 4),
    PRIMARY KEY (date, ticker_id)
);
-- Note: Daily data doesn't need hypertable (smaller volume)
\`\`\`

## Alternative Data Schema

\`\`\`sql
-- News and sentiment
CREATE TABLE news (
    news_id SERIAL PRIMARY KEY,
    published_at TIMESTAMPTZ NOT NULL,
    ticker_id INTEGER REFERENCES tickers (ticker_id),
    headline TEXT,
    content TEXT,
    source VARCHAR(100),
    url TEXT,
    sentiment_score NUMERIC(5, 4),  -- -1 to 1
    relevance_score NUMERIC(5, 4)   -- 0 to 1
);
CREATE INDEX idx_news_ticker_time ON news (ticker_id, published_at DESC);

-- Social media mentions
CREATE TABLE social_mentions (
    time TIMESTAMPTZ NOT NULL,
    ticker_id INTEGER NOT NULL,
    platform VARCHAR(50),  -- 'twitter', 'reddit', 'stocktwits'
    mention_count INTEGER,
    sentiment_avg NUMERIC(5, 4),
    PRIMARY KEY (time, ticker_id, platform)
);
SELECT create_hypertable('social_mentions', 'time');

-- Economic indicators
CREATE TABLE economic_indicators (
    date DATE NOT NULL,
    indicator VARCHAR(50) NOT NULL,  -- 'GDP', 'CPI', 'unemployment', etc.
    value NUMERIC(20, 6),
    country VARCHAR(3) DEFAULT 'USA',
    PRIMARY KEY (date, indicator, country)
);
\`\`\`

## Trading Activity Tracking

\`\`\`sql
-- Strategy performance tracking
CREATE TABLE strategies (
    strategy_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trades
CREATE TABLE trades (
    trade_id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies (strategy_id),
    ticker_id INTEGER REFERENCES tickers (ticker_id),
    entry_time TIMESTAMPTZ NOT NULL,
    entry_price NUMERIC(12, 4) NOT NULL,
    position_size INTEGER NOT NULL,  -- Number of shares
    direction VARCHAR(10) NOT NULL,  -- 'long' or 'short'
    exit_time TIMESTAMPTZ,
    exit_price NUMERIC(12, 4),
    pnl NUMERIC(15, 2),
    pnl_pct NUMERIC(10, 4),
    commission NUMERIC(10, 2),
    slippage NUMERIC(10, 2),
    notes TEXT
);
CREATE INDEX idx_trades_strategy ON trades (strategy_id, entry_time DESC);
CREATE INDEX idx_trades_ticker ON trades (ticker_id, entry_time DESC);

-- Daily strategy performance
CREATE TABLE strategy_performance (
    date DATE NOT NULL,
    strategy_id INTEGER REFERENCES strategies (strategy_id),
    pnl NUMERIC(15, 2),
    trades_count INTEGER,
    win_rate NUMERIC(5, 4),
    sharpe_ratio NUMERIC(10, 4),
    max_drawdown NUMERIC(10, 4),
    portfolio_value NUMERIC(20, 2),
    PRIMARY KEY (date, strategy_id)
);
\`\`\`

## Query Optimization

### Explain Analyze

\`\`\`sql
-- Check query performance
EXPLAIN ANALYZE
SELECT 
    t.symbol,
    p.time,
    p.close,
    p.volume
FROM prices p
JOIN tickers t ON p.ticker_id = t.ticker_id
WHERE t.symbol = 'AAPL'
  AND p.time >= '2023-01-01'
ORDER BY p.time DESC
LIMIT 100;

-- Output shows:
-- - Query plan
-- - Index usage
-- - Execution time
-- - Rows scanned vs returned

-- Look for:
-- - "Seq Scan" = BAD (full table scan, no index used)
-- - "Index Scan" = GOOD (using index)
-- - High "rows" numbers = potential problem
\`\`\`

### Index Strategies

\`\`\`sql
-- Composite index for common query pattern
CREATE INDEX idx_prices_ticker_time_close 
ON prices (ticker_id, time DESC, close);

-- Covering index (includes all columns in query)
-- Query never needs to access table, just index
CREATE INDEX idx_prices_covering 
ON prices (ticker_id, time DESC) 
INCLUDE (open, high, low, close, volume);

-- Partial index (only index subset of data)
CREATE INDEX idx_prices_recent 
ON prices (ticker_id, time DESC, close)
WHERE time >= CURRENT_DATE - INTERVAL '30 days';

-- Expression index
CREATE INDEX idx_tickers_upper_symbol 
ON tickers(UPPER(symbol));
-- Fast: WHERE UPPER(symbol) = 'AAPL'
\`\`\`

### Query Tuning

\`\`\`sql
-- Slow query (no index on time range)
SELECT * FROM prices
WHERE time >= '2023-01-01' AND time < '2024-01-01';
-- Scans entire table

-- Fast query (index on time)
SELECT * FROM prices
WHERE ticker_id = 123
  AND time >= '2023-01-01' AND time < '2024-01-01';
-- Uses idx_prices_ticker_time index

-- Even faster (limit results)
SELECT * FROM prices
WHERE ticker_id = 123
  AND time >= '2023-01-01'
ORDER BY time DESC
LIMIT 1000;
\`\`\`

## Backup and Recovery

\`\`\`bash
# Backup database
pg_dump -U postgres -d marketdata > marketdata_backup.sql

# Backup with compression
pg_dump -U postgres -d marketdata | gzip > marketdata_backup.sql.gz

# Restore database
psql -U postgres -d marketdata < marketdata_backup.sql

# Automated daily backups (cron)
0 2 * * * pg_dump -U postgres -d marketdata | gzip > /backups/marketdata_$(date +\%Y\%m\%d).sql.gz

# Backup specific table
pg_dump -U postgres -d marketdata -t prices > prices_backup.sql

# Copy table to CSV
COPY (SELECT * FROM prices WHERE ticker_id = 123) TO '/tmp/aapl_prices.csv' CSV HEADER;
\`\`\`
      `,
    },
    {
      title: 'Production Best Practices',
      content: `
# Database Operations for Trading Systems

## Connection Management

### Connection Pooling

\`\`\`python
from sqlalchemy import create_engine, pool

# Without pooling (inefficient - creates new connection each query)
engine = create_engine('postgresql://postgres:password@localhost/marketdata')

# With connection pooling (efficient - reuses connections)
engine = create_engine(
    'postgresql://postgres:password@localhost/marketdata',
    poolclass=pool.QueuePool,
    pool_size=10,          # Number of connections to maintain
    max_overflow=20,       # Additional connections when pool exhausted
    pool_timeout=30,       # Wait time for connection
    pool_recycle=3600,     # Recycle connections after 1 hour
    pool_pre_ping=True     # Verify connection health before use
)

# Usage (connection automatically managed)
import pandas as pd

df = pd.read_sql("SELECT * FROM prices WHERE ticker_id = 1 LIMIT 1000", engine)
# Connection returned to pool after query
\`\`\`

### Context Managers

\`\`\`python
from sqlalchemy import create_engine
from contextlib import contextmanager

engine = create_engine('postgresql://postgres:password@localhost/marketdata')

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = engine.connect()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()

# Usage
with get_db_connection() as conn:
    result = conn.execute("SELECT * FROM prices WHERE ticker_id = 1")
    rows = result.fetchall()
# Connection automatically closed
\`\`\`

## Data Validation and Quality

\`\`\`sql
-- Add constraints to ensure data quality
ALTER TABLE prices ADD CONSTRAINT check_positive_price 
    CHECK (close > 0 AND open > 0 AND high > 0 AND low > 0);

ALTER TABLE prices ADD CONSTRAINT check_high_low 
    CHECK (high >= low);

ALTER TABLE prices ADD CONSTRAINT check_ohlc_bounds
    CHECK (high >= open AND high >= close AND low <= open AND low <= close);

ALTER TABLE prices ADD CONSTRAINT check_positive_volume 
    CHECK (volume >= 0);

-- These prevent bad data from entering database
-- INSERT with negative price will fail
\`\`\`

### Data Quality Checks

\`\`\`python
def validate_price_data (df):
    \"\"\"Validate price data before database insert\"\"\"
    issues = []
    
    # Check for missing values
    if df.isnull().any().any():
        issues.append("Missing values detected")
    
    # Check for negative prices
    if (df[['open', 'high', 'low', 'close']] < 0).any().any():
        issues.append("Negative prices detected")
    
    # Check OHLC relationships
    if (df['high'] < df['low']).any():
        issues.append("High < Low detected")
    
    if ((df['high'] < df['open']) | (df['high'] < df['close'])).any():
        issues.append("High not highest")
    
    if ((df['low'] > df['open']) | (df['low'] > df['close'])).any():
        issues.append("Low not lowest")
    
    # Check for duplicate dates
    if df.duplicated (subset=['date', 'ticker']).any():
        issues.append("Duplicate date/ticker combinations")
    
    # Check for unrealistic price movements (> 50% daily change)
    df['pct_change'] = df.groupby('ticker')['close'].pct_change()
    if (df['pct_change'].abs() > 0.5).any():
        issues.append("Extreme price movements (>50% daily change)")
    
    if issues:
        raise ValueError (f"Data validation failed: {', '.join (issues)}")
    
    return True

# Usage
try:
    validate_price_data (df)
    df.to_sql('prices', engine, if_exists='append', index=False)
    print("Data inserted successfully")
except ValueError as e:
    print(f"Validation error: {e}")
\`\`\`

## Monitoring and Maintenance

### Database Statistics

\`\`\`sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty (pg_total_relation_size (schemaname||'.'||tablename)) AS size,
    pg_total_relation_size (schemaname||'.'||tablename) AS size_bytes
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY size_bytes DESC;

-- Check index usage
SELECT 
    indexrelname AS index_name,
    relname AS table_name,
    idx_scan AS index_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched,
    pg_size_pretty (pg_relation_size (indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;  -- Unused indexes at top

-- Find slow queries
SELECT 
    query,
    calls,
    total_exec_time / 1000 AS total_seconds,
    mean_exec_time / 1000 AS mean_seconds,
    max_exec_time / 1000 AS max_seconds
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
-- Requires: CREATE EXTENSION pg_stat_statements;

-- Check for bloat
SELECT 
    schemaname,
    tablename,
    pg_size_pretty (pg_total_relation_size (schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size (schemaname||'.'||tablename) DESC;
\`\`\`

### Vacuum and Analyze

\`\`\`sql
-- Update statistics (improves query planner decisions)
ANALYZE prices;

-- Reclaim space from deleted/updated rows
VACUUM prices;

-- Full vacuum (locks table, more thorough)
VACUUM FULL prices;

-- Automate maintenance
ALTER TABLE prices SET (autovacuum_enabled = true);
ALTER TABLE prices SET (autovacuum_vacuum_scale_factor = 0.05);
ALTER TABLE prices SET (autovacuum_analyze_scale_factor = 0.02);
\`\`\`

## High Availability Setup

\`\`\`yaml
# docker-compose.yml for HA PostgreSQL
version: '3.8'

services:
  postgres-primary:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: marketdata
    volumes:
      - pg-primary-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    command: |
      postgres
      -c wal_level=replica
      -c max_wal_senders=3
      -c max_replication_slots=3
  
  postgres-replica:
    image: postgres:15
    environment:
      PGPASSWORD: password
    depends_on:
      - postgres-primary
    volumes:
      - pg-replica-data:/var/lib/postgresql/data
    ports:
      - "5433:5432"
    command: |
      bash -c "
      until pg_basebackup -h postgres-primary -D /var/lib/postgresql/data -U postgres -Fp -Xs -P -R
      do
        echo 'Waiting for primary to be ready...'
        sleep 5
      done
      postgres
      "

volumes:
  pg-primary-data:
  pg-replica-data:
\`\`\`

## Error Handling

\`\`\`python
import psycopg2
from psycopg2 import OperationalError, IntegrityError
import time
import logging

logging.basicConfig (level=logging.INFO)
logger = logging.getLogger(__name__)

def insert_with_retry (data, max_retries=3):
    \"\"\"Insert data with automatic retry on failure\"\"\"
    for attempt in range (max_retries):
        try:
            conn = psycopg2.connect(
                host='localhost',
                database='marketdata',
                user='postgres',
                password='password'
            )
            
            cur = conn.cursor()
            
            # Insert data
            insert_query = \"\"\"
                INSERT INTO prices (date, ticker_id, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date, ticker_id) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            \"\"\"
            
            cur.executemany (insert_query, data)
            conn.commit()
            
            logger.info (f"Inserted {len (data)} rows successfully")
            
            cur.close()
            conn.close()
            
            return True
            
        except IntegrityError as e:
            # Data constraint violation - don't retry
            logger.error (f"Data integrity error: {e}")
            conn.rollback()
            raise
            
        except OperationalError as e:
            # Connection/network error - retry
            logger.warning (f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                logger.error("Max retries exceeded")
                raise
        
        finally:
            if 'conn' in locals():
                conn.close()
    
    return False

# Usage
try:
    data = [
        ('2024-01-15', 1, 185.50, 187.25, 184.75, 186.80, 52341567),
        # ... more rows
    ]
    insert_with_retry (data)
except Exception as e:
    logger.error (f"Failed to insert data: {e}")
\`\`\`

## Security Best Practices

\`\`\`sql
-- Create read-only user for analysts
CREATE USER analyst WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE marketdata TO analyst;
GRANT USAGE ON SCHEMA public TO analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst;

-- Create read-write user for data pipeline
CREATE USER data_pipeline WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE marketdata TO data_pipeline;
GRANT USAGE ON SCHEMA public TO data_pipeline;
GRANT SELECT, INSERT, UPDATE ON prices TO data_pipeline;
GRANT USAGE ON SEQUENCE prices_id_seq TO data_pipeline;

-- Never use postgres superuser in application code!

-- Revoke permissions
REVOKE INSERT ON prices FROM analyst;

-- Use SSL for connections
-- In connection string: sslmode=require
\`\`\`

\`\`\`python
# Secure connection from Python
from sqlalchemy import create_engine

# Production connection with SSL
engine = create_engine(
    'postgresql://data_pipeline:secure_password@prod-db.company.com:5432/marketdata',
    connect_args={
        'sslmode': 'require',
        'sslcert': '/path/to/client-cert.pem',
        'sslkey': '/path/to/client-key.pem',
        'sslrootcert': '/path/to/ca-cert.pem'
    }
)

# Never hardcode credentials - use environment variables
import os
DB_PASSWORD = os.getenv('DB_PASSWORD')
\`\`\`
      `,
    },
  ],
  exercises: [
    {
      title: 'Design Financial Database Schema',
      description:
        'Design a complete database schema for storing multi-asset price data, including tickers, prices at multiple frequencies, corporate actions, and fundamental data.',
      difficulty: 'intermediate',
      hints: [
        'Create separate tables for reference data (tickers) and time-series data (prices)',
        'Use foreign keys to maintain referential integrity',
        'Consider partitioning strategy for large tables',
        'Add appropriate indexes for common query patterns',
        'Include constraints to ensure data quality',
      ],
    },
    {
      id: 'databases-2',
      question: 'Build ETL Pipeline',
      description:
        'Create a Python script that downloads stock data from Yahoo Finance and loads it into PostgreSQL/TimescaleDB with proper error handling, validation, and logging.',
      difficulty: 'intermediate',
      hints: [
        'Use yfinance for data download',
        'Implement data validation before insert',
        'Handle duplicates with ON CONFLICT',
        'Add retry logic for network errors',
        'Log all operations for debugging',
        'Use connection pooling for efficiency',
      ],
    },
    {
      title: 'Query Optimization Challenge',
      description:
        'Optimize slow queries on a large price database. Use EXPLAIN ANALYZE, create appropriate indexes, and benchmark improvements.',
      difficulty: 'advanced',
      hints: [
        'Start with EXPLAIN ANALYZE to understand query plan',
        'Look for sequential scans that should be index scans',
        'Consider composite indexes for multi-column filters',
        'Use covering indexes to avoid table access',
        'Test with realistic data volumes (millions of rows)',
      ],
    },
    {
      title: 'TimescaleDB Implementation',
      description:
        'Set up TimescaleDB with hypertables, continuous aggregates, and compression for efficient storage of minute-bar data for 500 stocks over 5 years.',
      difficulty: 'advanced',
      hints: [
        'Create hypertable with appropriate chunk interval',
        'Set up continuous aggregates for daily/weekly bars',
        'Configure compression policy for old data',
        'Add retention policy to drop ancient data',
        'Benchmark query performance vs regular PostgreSQL',
      ],
    },
  ],
};
