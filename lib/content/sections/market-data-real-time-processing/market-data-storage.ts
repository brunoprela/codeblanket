export const marketDataStorage = {
  title: 'Market Data Storage (Tick Databases)',
  id: 'market-data-storage',
  content: `
# Market Data Storage (Tick Databases)

## Introduction

Storing billions of ticks efficiently requires specialized time-series databases. Tick databases optimize for high-write throughput, compression, and fast time-range queries.

**Storage Requirements:**
- **Volume**: 100K ticks/sec × 86400 sec/day = 8.6B ticks/day
- **Retention**: 1-10 years for backtesting and compliance
- **Query speed**: Sub-second queries for backtest replay
- **Compression**: 10-20× to reduce storage costs

**Time-Series Databases:**
- **TimescaleDB**: PostgreSQL extension, 20× compression
- **QuestDB**: Fastest ingestion (4M rows/sec), columnar
- **InfluxDB**: Popular, good for metrics, moderate performance
- **ClickHouse**: Excellent compression, analytics-focused
- **Arctic (MongoDB)**: Pandas DataFrame storage

This section covers database selection, schema design, compression strategies, and production deployment.

---

## TimescaleDB Implementation

\`\`\`python
import psycopg2
from datetime import datetime
from decimal import Decimal

class TimescaleTickDB:
    """TimescaleDB for tick storage"""
    
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
        self.create_schema()
    
    def create_schema(self):
        """Create hypertables for tick data"""
        with self.conn.cursor() as cur:
            # Create tick table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ticks (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT,
                    price DECIMAL(12, 4),
                    size INTEGER,
                    side TEXT,
                    conditions TEXT
                );
            """)
            
            # Convert to hypertable (time-series optimized)
            cur.execute("""
                SELECT create_hypertable('ticks', 'time', 
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Create index on symbol + time
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_time 
                ON ticks (symbol, time DESC);
            """)
            
            # Enable compression (10-20× reduction)
            cur.execute("""
                ALTER TABLE ticks SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol'
                );
            """)
            
            # Compress chunks older than 7 days
            cur.execute("""
                SELECT add_compression_policy('ticks', 
                    INTERVAL '7 days');
            """)
            
            self.conn.commit()
    
    def insert_tick(self, symbol: str, timestamp: datetime, 
                   price: Decimal, size: int, side: str):
        """Insert single tick"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ticks (time, symbol, price, size, side)
                VALUES (%s, %s, %s, %s, %s)
            """, (timestamp, symbol, price, size, side))
        self.conn.commit()
    
    def query_ticks(self, symbol: str, start: datetime, end: datetime):
        """Query ticks for symbol in time range"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT time, price, size, side
                FROM ticks
                WHERE symbol = %s
                  AND time >= %s
                  AND time < %s
                ORDER BY time
            """, (symbol, start, end))
            
            return cur.fetchall()

# Performance: 100K inserts/sec, 1M queries/sec on 1B row table
\`\`\`

---

## QuestDB (Fastest Ingestion)

\`\`\`python
# QuestDB: 4M rows/sec ingestion, columnar storage

CREATE TABLE ticks (
    symbol SYMBOL,  -- Dictionary-encoded (save space)
    timestamp TIMESTAMP,
    price DOUBLE,
    size INT,
    side SYMBOL
) TIMESTAMP(timestamp) PARTITION BY DAY;

# Insert (ILP protocol - fastest)
ticks,symbol=AAPL,side=B price=150.25,size=100 1234567890000000

# Query
SELECT * FROM ticks 
WHERE symbol = 'AAPL' 
  AND timestamp BETWEEN '2024-01-01' AND '2024-01-31';
\`\`\`

---

## Best Practices

1. **Partition by day** - Efficient compression and deletion
2. **Index on (symbol, time)** - Fast symbol lookups
3. **Compress old data** - 10-20× reduction after 7 days
4. **Batch inserts** - 1000 rows at once (10× faster)
5. **Use SYMBOL type** - Dictionary encoding for repeated strings
6. **Separate hot/cold storage** - Recent data on SSD, old on HDD

Now you can store billions of ticks efficiently!
`,
};
