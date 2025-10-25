import { QuizQuestion } from '@/lib/types';

export const timeSeriesDatabasesDiscussionQuiz: QuizQuestion[] = [
  {
    id: 'tsdb-discussion-1',
    question:
      "Your IoT platform collects temperature readings from 100,000 sensors every 10 seconds. That's 864 million datapoints per day. A year of data in PostgreSQL is 5TB and queries are slow. How would you redesign this using TimescaleDB with compression and downsampling?",
    sampleAnswer: `**TimescaleDB Architecture:**

\`\`\`sql
CREATE TABLE sensor_readings (
  time TIMESTAMPTZ NOT NULL,
  sensor_id TEXT NOT NULL,
  temperature DOUBLE PRECISION,
  humidity DOUBLE PRECISION
);

SELECT create_hypertable('sensor_readings', 'time', 
  chunk_time_interval => INTERVAL '1 day');
\`\`\`

**Automatic Partitioning**: 365 chunks (one per day), fast time-range queries.

**Compression (90% savings):**
\`\`\`sql
ALTER TABLE sensor_readings SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'sensor_id'
);

SELECT add_compression_policy('sensor_readings', 
  INTERVAL '7 days');
\`\`\`

Result: 5TB → 500GB

**Downsampling:**
\`\`\`sql
CREATE MATERIALIZED VIEW sensor_readings_hourly
WITH (timescaledb.continuous) AS
SELECT 
  time_bucket('1 hour', time) AS hour,
  sensor_id,
  AVG(temperature) as avg_temp,
  MAX(temperature) as max_temp
FROM sensor_readings
GROUP BY hour, sensor_id;
\`\`\`

**Query Strategy:**
- Last 7 days: Query raw data (compressed)
- 7-30 days: Query hourly aggregates
- >30 days: Daily aggregates

**Result**: 5TB → 50GB total, queries 100x faster.`,
    keyPoints: [
      'Hypertables automatically partition by time for fast time-range queries',
      'Compression after 7 days reduces storage by 90% (5TB → 500GB)',
      'Continuous aggregates (materialized views) pre-compute hourly/daily summaries',
      'Progressive downsampling: raw (7d) → hourly (30d) → daily (forever)',
      'Storage reduction: 100x (5TB → 50GB) with 100x faster queries',
    ],
  },
  {
    id: 'tsdb-discussion-2',
    question:
      'Your application monitoring system uses Prometheus to collect metrics from 1,000 services. Prometheus local storage is limited to 15 days due to disk constraints. Design a long-term storage solution using Thanos or Cortex.',
    sampleAnswer: `**Thanos Architecture (Recommended):**

\`\`\`
Prometheus (15 days local) → Thanos Sidecar → S3 (unlimited)
                                    ↓
                            Thanos Query (unified view)
                                    ↓
                              Grafana
\`\`\`

**Benefits:**
- Prometheus: 15 days hot data (fast queries)
- S3: Years of historical data ($23/TB/month)
- Thanos Query: Seamless queries across both
- Downsampling: 5m resolution → 1h resolution after 30 days (10x storage savings)

**Cost:**
- Prometheus local: 1TB × 1,000 services = 1TB (fixed)
- S3 (1 year): 365TB × $0.023/GB = $8,400/year
vs keeping 1 year in Prometheus local: $50k+ in SSDs

**Query:**
\`\`\`
rate(http_requests_total[5m])  # Works across 1 year!
\`\`\``,
    keyPoints: [
      'Prometheus local storage limited; Thanos extends to unlimited S3 storage',
      'Thanos Sidecar uploads blocks to object storage (S3)',
      'Thanos Query provides unified interface across local + S3',
      'Downsampling reduces storage for old data (5m → 1h resolution)',
      'Cost: $8k/year (S3) vs $50k (local SSD) for 1-year retention',
    ],
  },
  {
    id: 'tsdb-discussion-3',
    question:
      "You need to store stock market data (OHLCV) for 10,000 stocks with 1-second resolution. That's 86,400 datapoints per stock per day. Design a time-series database schema optimized for common queries: (1) latest price, (2) 1-minute candles, (3) moving averages.",
    sampleAnswer: `**InfluxDB Schema:**

\`\`\`
Measurement: stock_ticks
Tags: {symbol: "AAPL", exchange: "NASDAQ"}
Fields: {price: 150.25, volume: 1000}
Timestamp: nanosecond precision
\`\`\`

**Write (1-second ticks):**
\`\`\`
stock_ticks,symbol=AAPL,exchange=NASDAQ price=150.25,volume=1000 1705327800000000000
\`\`\`

**Query 1: Latest Price**
\`\`\`sql
SELECT LAST(price) 
FROM stock_ticks 
WHERE symbol='AAPL' 
AND time > now() - 1m
\`\`\`

**Continuous Query: 1-Minute Candles**
\`\`\`sql
CREATE CONTINUOUS QUERY "candles_1m" ON "trading"
BEGIN
  SELECT 
    FIRST(price) as open,
    MAX(price) as high,
    MIN(price) as low,
    LAST(price) as close,
    SUM(volume) as volume
  INTO "candles_1m"
  FROM "stock_ticks"
  GROUP BY time(1m), symbol, exchange
END
\`\`\`

**Query 2: 1-Minute Candles**
\`\`\`sql
SELECT * FROM candles_1m 
WHERE symbol='AAPL' 
AND time > now() - 1h
\`\`\`

**Query 3: Moving Average**
\`\`\`sql
SELECT MOVING_AVERAGE(MEAN(close), 20) 
FROM candles_1m 
WHERE symbol='AAPL' 
GROUP BY time(1m)
\`\`\`

**Storage:**
- Raw ticks: 10k stocks × 86,400 ticks/day × 365 days = 315B points/year (12TB uncompressed)
- With compression: 1TB
- 1-minute candles: 10k × 1,440 candles/day = 14.4M candles/day (50GB/year)

**Retention Policy:**
\`\`\`sql
CREATE RETENTION POLICY "ticks_1d" 
ON "trading" 
DURATION 1d  -- Keep raw ticks 1 day

CREATE RETENTION POLICY "candles_1y"
ON "trading"
DURATION 365d  -- Keep 1-min candles 1 year
\`\`\`

**Result**: Sub-millisecond latest price queries, pre-computed candles, 1TB storage.`,
    keyPoints: [
      'Tags (symbol, exchange) are indexed for fast filtering',
      'Fields (price, volume) are measurements, not indexed',
      'Continuous queries auto-compute 1-minute candles (real-time aggregation)',
      'Retention policies: raw ticks (1 day), candles (1 year)',
      'Compression reduces 12TB → 1TB for raw ticks',
    ],
  },
];
