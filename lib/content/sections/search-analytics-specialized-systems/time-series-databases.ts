import { ModuleSection } from '@/lib/types';

const timeSeriesDatabasesSection: ModuleSection = {
  id: 'time-series-databases',
  title: 'Time-Series Databases',
  content: `
# Time-Series Databases

## Introduction

**Time-series data** is a sequence of data points indexed by time, where each point represents a measurement at a specific timestamp. Time-series databases are specialized systems optimized for efficiently storing, querying, and analyzing this temporal data at scale.

Time-series data is everywhere:
- **Monitoring**: CPU usage, memory, requests/second, latency
- **IoT sensors**: Temperature, pressure, humidity, GPS coordinates
- **Financial markets**: Stock prices, trades, order book depths
- **DevOps metrics**: Application performance, infrastructure health
- **Business metrics**: Sales, user signups, conversion rates

Unlike traditional databases designed for transactional workloads (CRUD operations), time-series databases are purpose-built for:
- **High write throughput**: Millions of datapoints per second
- **Append-only data**: Rarely update or delete historical data
- **Time-range queries**: "Show me CPU usage for the last hour"
- **Aggregations**: Average, sum, percentiles over time windows
- **Retention and downsampling**: Automatically age out old data or reduce resolution

This section covers the characteristics of time-series data, specialized databases (InfluxDB, TimescaleDB, Prometheus), compression techniques, and real-world use cases.

## Characteristics of Time-Series Data

### 1. Timestamp-Indexed

Every data point has a timestamp—the primary axis for queries.

\`\`\`
Temperature sensor readings:
2024-01-15 14:00:00 → 72.5°F
2024-01-15 14:01:00 → 72.7°F
2024-01-15 14:02:00 → 72.4°F
\`\`\`

### 2. Append-Only

New measurements are added continuously. Historical data rarely changes.

**Write pattern**: INSERT, INSERT, INSERT... (no UPDATE/DELETE)

**Implication**: Optimize for writes, not updates.

### 3. Time-Ordered

Data arrives roughly in time order (with some out-of-order tolerance for network delays).

**Query pattern**: Almost always include time range:
- "Last 5 minutes"
- "Yesterday"
- "Between 2024-01-01 and 2024-01-31"

### 4. High Write Volume

Monitoring 10,000 servers:
- 10,000 servers × 100 metrics each × 10-second interval = **100,000 writes/second**

IoT fleet:
- 1M sensors × 1 reading/minute = **16,666 writes/second**

### 5. Aggregation-Heavy

Queries typically aggregate:
- Average CPU over last hour
- 95th percentile latency per minute
- Total sales per day

**Not typical**: "Show me the temperature reading at exactly 14:32:17.384"

### 6. Recent Data Accessed Most

**Query frequency**:
- Last hour: 80% of queries
- Last day: 15% of queries
- Last week: 4% of queries
- Older: 1% of queries

**Implication**: Optimize for recent data access, age out old data.

## Why Specialized Time-Series Databases?

Traditional relational databases struggle with time-series workloads.

### PostgreSQL/MySQL Challenges

**Example: Metrics table**
\`\`\`sql
CREATE TABLE metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100),
    host VARCHAR(100),
    value DOUBLE PRECISION,
    PRIMARY KEY (timestamp, metric_name, host)
);

-- Index for queries
CREATE INDEX idx_time ON metrics (timestamp);
CREATE INDEX idx_metric_time ON metrics (metric_name, timestamp);
\`\`\`

**Problems**:

**1. Index bloat**:
- 100K writes/sec × 86,400 sec/day = 8.6B rows/day
- B-tree index grows huge (10s of GBs per index)
- INSERT performance degrades over time

**2. No native compression**:
- Each timestamp stored as 8-byte integer (wasteful for sequential timestamps)
- Each value stored as 8-byte float (no compression)
- Storage grows linearly with data volume

**3. Slow time-range queries**:
- Even with index, scanning millions of rows for aggregations is slow
- "AVG(value) WHERE timestamp > now() - 1 hour" scans ~360K rows

**4. Manual retention management**:
- Must write DELETE statements to remove old data
- DELETE is expensive (bloats table, requires VACUUM)

**5. No downsampling**:
- Want to reduce resolution of old data (1-second → 1-minute)
- Requires manual ETL jobs

### Time-Series Database Advantages

| Feature | PostgreSQL | InfluxDB / TimescaleDB |
|---------|-----------|------------------------|
| **Compression** | Generic (2-3×) | Timestamp-aware (10-20×) |
| **Write throughput** | 10K writes/sec | 1M+ writes/sec |
| **Time-range queries** | Index scan | Optimized chunks |
| **Retention policies** | Manual DELETE | Automatic |
| **Downsampling** | Manual ETL | Built-in continuous aggregates |
| **Storage (1 year of metrics)** | 5TB | 250GB |

## InfluxDB

InfluxDB is a purpose-built time-series database with its own query language (InfluxQL and Flux).

### Data Model

\`\`\`
Measurement: cpu_usage (like a table)
Tags: {host: "server1", region: "us-west", env: "prod"}
Fields: {usage_percent: 65.3, user_percent: 45.2, system_percent: 20.1}
Timestamp: 2024-01-15T14:30:00Z
\`\`\`

**Tags** = indexed (for filtering)
**Fields** = not indexed (for aggregation)

**Example data point**:
\`\`\`
cpu_usage,host=server1,region=us-west,env=prod usage_percent=65.3,user_percent=45.2,system_percent=20.1 1705327800000000000
\`\`\`

### Writing Data

**Line Protocol** (text format):
\`\`\`
<measurement>[,<tag_key>=<tag_value>]* <field_key>=<field_value>[,<field_key>=<field_value>]* [timestamp]
\`\`\`

**Example**:
\`\`\`bash
curl -XPOST 'http://localhost:8086/write?db=metrics' \\
  --data-binary 'cpu_usage,host=server1,region=us-west usage_percent=65.3 1705327800000000000'
\`\`\`

**Batch writes** (recommended for performance):
\`\`\`python
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

client = InfluxDBClient (url="http://localhost:8086", token="my-token", org="my-org")
write_api = client.write_api (write_options=SYNCHRONOUS)

# Batch write
points = [
    Point("cpu_usage").tag("host", "server1").field("value", 65.3).time (timestamp1),
    Point("cpu_usage").tag("host", "server2").field("value", 72.1).time (timestamp2),
    # ... 1000 more points
]

write_api.write (bucket="metrics", record=points)
\`\`\`

**Best practice**: Batch 1,000-10,000 points per write for optimal throughput.

### Querying Data

**InfluxQL** (SQL-like):
\`\`\`sql
SELECT MEAN("usage_percent") 
FROM "cpu_usage" 
WHERE time > now() - 1h 
  AND "host" = 'server1'
GROUP BY time(5m), "host"
\`\`\`

**Result**:
\`\`\`
time                  host     mean
2024-01-15T14:00:00Z  server1  64.2
2024-01-15T14:05:00Z  server1  67.8
2024-01-15T14:10:00Z  server1  65.1
...
\`\`\`

**Flux** (functional query language):
\`\`\`
from (bucket: "metrics")
  |> range (start: -1h)
  |> filter (fn: (r) => r["_measurement"] == "cpu_usage")
  |> filter (fn: (r) => r["host"] == "server1")
  |> aggregateWindow (every: 5m, fn: mean)
\`\`\`

### Retention Policies

Automatically delete old data:

\`\`\`sql
-- Create retention policy: keep 7 days
CREATE RETENTION POLICY "one_week"
ON "metrics"
DURATION 7d
REPLICATION 1
DEFAULT

-- Create retention policy: keep 1 year (for aggregated data)
CREATE RETENTION POLICY "one_year"
ON "metrics"
DURATION 365d
REPLICATION 1
\`\`\`

**Usage**:
- Write raw metrics to "one_week" retention policy (auto-deleted after 7 days)
- Write aggregated metrics (hourly averages) to "one_year" (kept for 1 year)

### Continuous Queries (Downsampling)

Automatically aggregate high-resolution data into lower-resolution:

\`\`\`sql
CREATE CONTINUOUS QUERY "downsample_cpu_1h"
ON "metrics"
BEGIN
  SELECT MEAN("usage_percent") AS "mean_usage"
  INTO "metrics"."one_year"."cpu_usage_hourly"
  FROM "metrics"."one_week"."cpu_usage"
  GROUP BY time(1h), *
END
\`\`\`

**What this does**:
1. Every hour, automatically runs
2. Reads raw CPU data (1-second resolution) from last hour
3. Computes hourly average
4. Writes to "cpu_usage_hourly" measurement (stored for 1 year)

**Result**:
- Raw data: 1-second resolution, 7-day retention (604,800 points per metric)
- Downsampled: 1-hour resolution, 1-year retention (8,760 points per metric)
- **Storage reduction: 69× fewer points!**

## TimescaleDB

TimescaleDB is a PostgreSQL extension that adds time-series capabilities while keeping full SQL compatibility.

### Hypertables

**Problem**: Storing billions of time-series points in a single PostgreSQL table is slow.

**Solution**: Automatically partition table into "chunks" by time.

\`\`\`sql
-- Create normal table
CREATE TABLE metrics (
  time TIMESTAMPTZ NOT NULL,
  device_id TEXT,
  temperature DOUBLE PRECISION,
  humidity DOUBLE PRECISION
);

-- Convert to hypertable (automatic partitioning)
SELECT create_hypertable('metrics', 'time', chunk_time_interval => INTERVAL '1 day');
\`\`\`

**What happens under the hood**:

\`\`\`
metrics (hypertable)
├── _hyper_1_1_chunk (2024-01-15 data)
├── _hyper_1_2_chunk (2024-01-16 data)
├── _hyper_1_3_chunk (2024-01-17 data)
└── ...
\`\`\`

Each chunk is a separate PostgreSQL table, but queries work on the hypertable seamlessly:

\`\`\`sql
-- Query the hypertable (automatically queries relevant chunks)
SELECT device_id, AVG(temperature)
FROM metrics
WHERE time > NOW() - INTERVAL '1 day'
GROUP BY device_id;

-- Only scans _hyper_1_1_chunk (today's data), skips other chunks!
\`\`\`

**Benefits**:
- **Partition pruning**: Only scans relevant chunks (10× faster queries)
- **Fast deletes**: Drop old chunk (instant) vs DELETE millions of rows (slow)
- **Parallel processing**: Query chunks in parallel

### Compression

TimescaleDB compression reduces storage by 10-20×.

\`\`\`sql
-- Enable compression
ALTER TABLE metrics SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'device_id'
);

-- Compress chunks older than 7 days
SELECT add_compression_policy('metrics', INTERVAL '7 days');
\`\`\`

**What happens**:
- After 7 days, chunk is compressed using columnar storage + delta encoding
- Compressed chunks are read-only (no new writes)
- Queries still work transparently

**Example**:
- Raw chunk: 10GB
- Compressed chunk: 500MB (20× compression!)

### Continuous Aggregates

Pre-compute aggregations for faster queries.

\`\`\`sql
-- Create continuous aggregate (like a materialized view that auto-updates)
CREATE MATERIALIZED VIEW metrics_hourly
WITH (timescaledb.continuous) AS
SELECT 
  time_bucket('1 hour', time) AS hour,
  device_id,
  AVG(temperature) AS avg_temp,
  MAX(temperature) AS max_temp,
  MIN(temperature) AS min_temp
FROM metrics
GROUP BY hour, device_id;

-- Add refresh policy (keep up to date)
SELECT add_continuous_aggregate_policy('metrics_hourly',
  start_offset => INTERVAL '3 hours',
  end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '1 hour'
);
\`\`\`

**Benefit**:
- Query hourly aggregates from \`metrics_hourly\` (instant, pre-computed)
- Instead of scanning billions of rows in \`metrics\` and computing AVG at query time

**Query**:
\`\`\`sql
-- Fast: queries pre-computed aggregates
SELECT hour, device_id, avg_temp
FROM metrics_hourly
WHERE hour > NOW() - INTERVAL '7 days'
  AND device_id = 'sensor_123';

-- Slow: computes aggregation on every query
SELECT time_bucket('1 hour', time) AS hour, device_id, AVG(temperature)
FROM metrics
WHERE time > NOW() - INTERVAL '7 days'
  AND device_id = 'sensor_123'
GROUP BY hour, device_id;
\`\`\`

### Retention Policies

\`\`\`sql
-- Drop chunks older than 90 days
SELECT add_retention_policy('metrics', INTERVAL '90 days');
\`\`\`

**How it works**:
- Every day, job runs and drops chunks older than 90 days
- Instant (drops table) vs slow DELETE

## Prometheus

Prometheus is a monitoring system with a built-in time-series database, optimized for operational metrics.

### Data Model

\`\`\`
metric_name{label1="value1", label2="value2"} value timestamp
\`\`\`

**Example**:
\`\`\`
http_requests_total{method="GET", status="200", endpoint="/api/users"} 15420 1705327800
\`\`\`

**Labels** = dimensions (like InfluxDB tags)

### Scraping Model (Pull)

Unlike InfluxDB (push model where apps send data), Prometheus **pulls** metrics from targets.

\`\`\`
Prometheus Server
  ↓ (scrapes every 15 seconds)
Application /metrics endpoint
  ↓ (exposes current metrics)
\`\`\`

**Application exposes /metrics**:
\`\`\`
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",status="200"} 15420
http_requests_total{method="POST",status="201"} 3201
http_requests_total{method="GET",status="500"} 23

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 12000
http_request_duration_seconds_bucket{le="0.5"} 14500
http_request_duration_seconds_bucket{le="1.0"} 15200
http_request_duration_seconds_sum 8920.5
http_request_duration_seconds_count 15420
\`\`\`

**Prometheus configuration**:
\`\`\`yaml
scrape_configs:
  - job_name: 'api-server'
    scrape_interval: 15s
    static_configs:
      - targets: ['api-server1:8080', 'api-server2:8080']
\`\`\`

### PromQL (Query Language)

**Instant vector** (latest value):
\`\`\`
http_requests_total{method="GET"}
\`\`\`

**Rate** (requests per second):
\`\`\`
rate (http_requests_total[5m])
\`\`\`

**Aggregation**:
\`\`\`
sum (rate (http_requests_total[5m])) by (method)
\`\`\`

**Percentile** (P95 latency):
\`\`\`
histogram_quantile(0.95, 
  rate (http_request_duration_seconds_bucket[5m])
)
\`\`\`

**Alert rule**:
\`\`\`yaml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: rate (http_requests_total{status=~"5.."}[5m]) > 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on {{ $labels.instance }}"
\`\`\`

### Metric Types

**1. Counter**: Monotonically increasing value
\`\`\`python
from prometheus_client import Counter

requests = Counter('http_requests_total', 'Total requests', ['method', 'status'])

requests.labels (method='GET', status='200').inc()  # Increment by 1
\`\`\`

**Use case**: Total requests, total errors, total bytes sent

**2. Gauge**: Value that can go up or down
\`\`\`python
from prometheus_client import Gauge

cpu_usage = Gauge('cpu_usage_percent', 'CPU usage', ['host'])

cpu_usage.labels (host='server1').set(65.3)
\`\`\`

**Use case**: CPU %, memory usage, queue length, active connections

**3. Histogram**: Distribution of values
\`\`\`python
from prometheus_client import Histogram

latency = Histogram('http_request_duration_seconds', 'Request latency', ['endpoint'])

latency.labels (endpoint='/api/users').observe(0.234)  # 234ms
\`\`\`

**Use case**: Request latencies, response sizes

**Query**:
\`\`\`
# P50, P95, P99 latencies
histogram_quantile(0.50, rate (http_request_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate (http_request_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate (http_request_duration_seconds_bucket[5m]))
\`\`\`

**4. Summary**: Similar to histogram but client-side quantiles
\`\`\`python
from prometheus_client import Summary

latency = Summary('request_latency_seconds', 'Request latency')

with latency.time():
    # Code to measure
    process_request()
\`\`\`

## Compression Techniques

Time-series data is highly compressible due to temporal patterns.

### 1. Delta Encoding

**Problem**: Timestamps are sequential.

\`\`\`
Raw timestamps (8 bytes each):
1705327800, 1705327801, 1705327802, 1705327803
Total: 4 × 8 = 32 bytes
\`\`\`

**Delta encoding**:
\`\`\`
Base: 1705327800 (8 bytes)
Deltas: [0, +1, +1, +1] (4 × 1 byte = 4 bytes)
Total: 8 + 4 = 12 bytes
Compression: 62% savings
\`\`\`

**Implementation**: Store first timestamp, then store differences.

### 2. Run-Length Encoding (RLE)

**Problem**: Repeated values (common in metrics that don't change often).

\`\`\`
Raw values:
[100, 100, 100, 100, 200, 200, 150]
7 values × 8 bytes = 56 bytes
\`\`\`

**RLE**:
\`\`\`
[(value=100, count=4), (value=200, count=2), (value=150, count=1)]
3 pairs × 12 bytes = 36 bytes
Compression: 36% savings
\`\`\`

### 3. Gorilla Compression (Facebook)

Optimized for floating-point time-series values.

**Key insight**: Consecutive values often similar (CPU goes from 65.2% to 65.4%, not 65.2% to 92.7%).

**How it works**:
1. XOR current value with previous value
2. Most bits are 0 (similar values)
3. Store only non-zero bits

**Example**:
\`\`\`
Value 1: 65.234
Value 2: 65.237
XOR: Most bits are 0, only last few bits differ
Store: 3 bits instead of 64 bits
\`\`\`

**Compression**:
- Raw: 8 bytes per value (64-bit float)
- Gorilla: 1.37 bytes per value average
- **Compression: 83% savings!**

**Usage**: InfluxDB, Prometheus, TimescaleDB all use Gorilla-like compression.

### 4. Dictionary Encoding

**Problem**: Repeated string values (host names, device IDs).

\`\`\`
Raw data:
server1, server1, server1, server2, server2, server1
6 × 7 bytes = 42 bytes
\`\`\`

**Dictionary encoding**:
\`\`\`
Dictionary:
0 → "server1"
1 → "server2"

Encoded:
[0, 0, 0, 1, 1, 0]
6 × 1 byte = 6 bytes (plus 14-byte dictionary = 20 bytes total)
Compression: 52% savings
\`\`\`

### Combined Compression

\`\`\`
Example: 1 year of CPU metrics (1-second resolution)
- 31,536,000 datapoints
- Raw: 31.5M × 16 bytes (timestamp + value) = 504MB
- Compressed:
  - Delta encoding timestamps: 504MB → 320MB (37% savings)
  - Gorilla compression values: 320MB → 60MB (81% total savings)
  - Dictionary encoding tags: 60MB → 25MB (95% total savings)

Final: 504MB → 25MB (20× compression!)
\`\`\`

## Downsampling and Retention

**Problem**: 1-second resolution for 1 year = 31.5M datapoints per metric.

### Progressive Downsampling

\`\`\`
Resolution          Retention      Datapoints/Metric
1-second (raw)      7 days         604,800
1-minute (avg)      30 days        43,200
1-hour (avg)        1 year         8,760
1-day (avg)         Forever        365/year

Total: ~660,000 points vs 31.5M (95% reduction!)
\`\`\`

### Implementation (InfluxDB)

\`\`\`sql
-- Raw data: 1-second resolution, 7-day retention
CREATE RETENTION POLICY "raw_7d" ON "metrics" DURATION 7d REPLICATION 1

-- 1-minute aggregates, 30-day retention
CREATE RETENTION POLICY "1m_30d" ON "metrics" DURATION 30d REPLICATION 1
CREATE CONTINUOUS QUERY "downsample_1m" ON "metrics"
BEGIN
  SELECT MEAN("value") INTO "metrics"."1m_30d"."cpu_avg"
  FROM "metrics"."raw_7d"."cpu_usage"
  GROUP BY time(1m), *
END

-- 1-hour aggregates, 1-year retention
CREATE RETENTION POLICY "1h_1y" ON "metrics" DURATION 365d REPLICATION 1
CREATE CONTINUOUS QUERY "downsample_1h" ON "metrics"
BEGIN
  SELECT MEAN("value") INTO "metrics"."1h_1y"."cpu_avg"
  FROM "metrics"."1m_30d"."cpu_avg"
  GROUP BY time(1h), *
END
\`\`\`

**Query strategy**:
- Last hour: Query raw data (1-second resolution)
- Last day: Query 1-minute aggregates
- Last month: Query 1-hour aggregates
- Older: Query 1-day aggregates

## Use Cases

### Application Monitoring

**Metrics**:
\`\`\`
http_requests_total{method, status, endpoint}
http_request_duration_seconds{endpoint}
database_query_duration_seconds{query_type}
memory_usage_bytes
cpu_usage_percent
\`\`\`

**Queries**:
\`\`\`
# Request rate per endpoint
rate (http_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate (http_request_duration_seconds_bucket[5m]))

# Error rate
rate (http_requests_total{status=~"5.."}[5m])
\`\`\`

**Dashboard**: Grafana showing request rate, latency, error rate over time.

### IoT Sensor Fleet

**Scenario**: 100,000 temperature sensors, 1 reading per minute

**Volume**:
- 100,000 sensors × 1,440 readings/day = 144M readings/day
- 1 year = 52.6B readings

**Storage (TimescaleDB)**:
\`\`\`sql
CREATE TABLE sensor_readings (
  time TIMESTAMPTZ NOT NULL,
  sensor_id TEXT NOT NULL,
  temperature DOUBLE PRECISION,
  humidity DOUBLE PRECISION,
  battery_percent DOUBLE PRECISION
);

SELECT create_hypertable('sensor_readings', 'time', 
  chunk_time_interval => INTERVAL '1 day');

-- Enable compression after 7 days
ALTER TABLE sensor_readings SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'sensor_id'
);
SELECT add_compression_policy('sensor_readings', INTERVAL '7 days');

-- Continuous aggregate: hourly averages
CREATE MATERIALIZED VIEW sensor_readings_hourly
WITH (timescaledb.continuous) AS
SELECT 
  time_bucket('1 hour', time) AS hour,
  sensor_id,
  AVG(temperature) AS avg_temp,
  MAX(temperature) AS max_temp,
  MIN(temperature) AS min_temp
FROM sensor_readings
GROUP BY hour, sensor_id;

-- Delete raw data after 30 days
SELECT add_retention_policy('sensor_readings', INTERVAL '30 days');
\`\`\`

**Result**:
- Raw data: 100KB × 144M = 14.4TB/year
- Compressed: 14.4TB → 720GB (20× compression)
- With downsampling (hourly after 30 days): 720GB → 50GB

### Financial Market Data

**Scenario**: Stock market tick data (OHLCV: Open, High, Low, Close, Volume)

**Volume**:
- 10,000 stocks
- 1-second tick data during trading hours (6.5 hours/day, 250 trading days/year)
- 10,000 × 23,400 seconds/day × 250 days = 58.5B ticks/year

**Schema (InfluxDB)**:
\`\`\`
Measurement: stock_ticks
Tags: {symbol, exchange}
Fields: {price, volume, bid, ask}
Timestamp: nanosecond precision
\`\`\`

**Continuous query: 1-minute candles**:
\`\`\`sql
CREATE CONTINUOUS QUERY "candles_1m" ON "trading"
BEGIN
  SELECT 
    FIRST(price) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price) AS close,
    SUM(volume) AS volume
  INTO "trading"."autogen"."candles_1m"
  FROM "trading"."autogen"."stock_ticks"
  GROUP BY time(1m), symbol, exchange
END
\`\`\`

**Query**:
\`\`\`sql
-- Get 1-minute candles for AAPL today
SELECT open, high, low, close, volume
FROM candles_1m
WHERE symbol = 'AAPL'
  AND time > now() - 1d
\`\`\`

**Storage**:
- Raw ticks (1-second): Keep 1 day (58.5B ticks × 1/250 days = 234M ticks = 9.4GB compressed)
- 1-minute candles: Keep 1 year (10,000 stocks × 1,440 candles/day × 250 days = 3.6B candles = 144GB)
- 1-hour candles: Keep forever (10,000 × 24 × 250 = 60M candles = 2.4GB)

## Best Practices

### 1. Use Tags for Dimensions, Fields for Measurements

**Tags**: Low cardinality (host, region, service)—used for filtering
**Fields**: High cardinality (actual measurements)—used for aggregation

**Good**:
\`\`\`
cpu_usage,host=server1,region=us-west usage_percent=65.3
\`\`\`

**Bad** (high-cardinality tag):
\`\`\`
cpu_usage,value=65.3 host="server1"
\`\`\`

### 2. Set Retention Policies

Don't keep data forever. Define how long you need each resolution:
- Raw: 7-30 days
- Minute aggregates: 30-90 days
- Hour aggregates: 1 year
- Day aggregates: Forever

### 3. Downsample Aggressively

Reduce resolution of old data:
- Last day: 1-second resolution
- Last week: 1-minute resolution
- Last month: 1-hour resolution
- Older: 1-day resolution

### 4. Partition by Time

Use date-based partitions/chunks for:
- Fast queries (only scan relevant partitions)
- Fast deletes (drop partition instead of DELETE)

### 5. Batch Writes

Write in batches of 1,000-10,000 points:
\`\`\`python
points = []
for metric in metrics:
    points.append (create_point (metric))
    if len (points) >= 1000:
        write_api.write (bucket="metrics", record=points)
        points = []
\`\`\`

### 6. Monitor Cardinality

**High cardinality kills performance**:
\`\`\`
# Good: 1,000 hosts × 10 regions = 10,000 unique tag combinations
cpu_usage,host=server1,region=us-west

# Bad: 1M user IDs × 1M request IDs = 1 trillion combinations (index explosion!)
requests,user_id=12345,request_id=req-abc-123
\`\`\`

**Rule**: Keep tag cardinality < 1 million unique combinations.

### 7. Use Continuous Aggregates / Materialized Views

Pre-compute common aggregations:
\`\`\`sql
-- Instead of computing AVG every query
SELECT time_bucket('1 hour', time), AVG(temperature)
FROM sensors
GROUP BY 1

-- Pre-compute and query materialized view
SELECT hour, avg_temp
FROM sensors_hourly
\`\`\`

## Trade-Offs

### Time-Series DB vs Relational DB

| Aspect | Time-Series DB | Relational DB |
|--------|---------------|---------------|
| **Write throughput** | 1M+ writes/sec | 10K writes/sec |
| **Compression** | 10-20× (timestamp-aware) | 2-3× (generic) |
| **Time-range queries** | Optimized (chunks/partitions) | Slow (B-tree index) |
| **Flexibility** | Time-series only | General purpose |
| **Learning curve** | New query language | Standard SQL |
| **Retention** | Automatic policies | Manual DELETE |

### InfluxDB vs TimescaleDB

| Aspect | InfluxDB | TimescaleDB |
|--------|----------|-------------|
| **SQL compatibility** | No (InfluxQL/Flux) | Yes (PostgreSQL) |
| **Ease of use** | Simple | Moderate (need to understand hypertables) |
| **Ecosystem** | Purpose-built | PostgreSQL ecosystem |
| **Advanced features** | Flux (functional queries) | Full SQL, JOINs, CTEs |
| **Best for** | Pure time-series | Mixed time-series + relational |

### Prometheus vs InfluxDB

| Aspect | Prometheus | InfluxDB |
|--------|-----------|----------|
| **Model** | Pull (scrape) | Push (write) |
| **Use case** | Monitoring alerts | General time-series |
| **Storage** | Local (15 days typical) | Distributed, long-term |
| **Query language** | PromQL (powerful) | InfluxQL/Flux |
| **Best for** | Kubernetes monitoring | Long-term analytics |

## Summary

Time-series databases optimize for:

**High write throughput**: Millions of points/second
**Time-range queries**: "Last hour", "Yesterday"
**Compression**: 10-20× reduction (delta encoding, Gorilla, RLE)
**Automatic downsampling**: Reduce resolution of old data
**Retention policies**: Auto-delete old data
**Built-in aggregations**: MEAN, MAX, percentiles optimized

**Popular choices**:
- **InfluxDB**: Purpose-built, InfluxQL/Flux, retention policies
- **TimescaleDB**: PostgreSQL extension, full SQL, hypertables
- **Prometheus**: Monitoring-focused, pull model, PromQL
- **Druid**: Real-time OLAP, sub-second queries

**When to use**:
- Monitoring and observability
- IoT sensor data
- Financial market data
- Application metrics

**When NOT to use**:
- Relational data with complex relationships
- Frequent updates/deletes
- Not time-indexed data

Time-series databases are essential for modern observability, IoT, and analytics use cases where data is indexed by time and queried by time ranges.
`,
};

export default timeSeriesDatabasesSection;
