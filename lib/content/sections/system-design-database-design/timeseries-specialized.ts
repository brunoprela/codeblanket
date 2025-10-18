/**
 * Time-Series and Specialized Databases Section
 */

export const timeseriesspecializedSection = {
  id: 'timeseries-specialized',
  title: 'Time-Series and Specialized Databases',
  content: `Time-series databases and other specialized databases are optimized for specific use cases that general-purpose databases handle poorly. Understanding when and how to use them is crucial for building efficient, scalable systems.

## What is a Time-Series Database?

A **time-series database (TSDB)** is optimized for storing and querying data points indexed by time.

### Characteristics of Time-Series Data:

1. **Time-stamped:** Every data point has a timestamp
2. **Append-only:** Data is mostly inserted, rarely updated
3. **High write throughput:** Millions of data points per second
4. **Range queries:** Query data within time ranges
5. **Aggregations:** Compute statistics over time windows
6. **Data retention:** Old data is often downsampled or deleted

### Examples of Time-Series Data:

- **Monitoring:** Server CPU, memory, disk usage
- **IoT sensors:** Temperature, pressure, location
- **Financial:** Stock prices, trades, market data
- **Application metrics:** Request latency, error rates, throughput
- **Logs:** Application logs, access logs

### Why Not Use PostgreSQL/MySQL?

**Problem with traditional RDBMS for time-series:**

\`\`\`sql
-- Store metrics in PostgreSQL
CREATE TABLE metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100),
    value DOUBLE PRECISION,
    tags JSONB,
    timestamp TIMESTAMP
);

CREATE INDEX idx_metrics_time ON metrics(timestamp);
CREATE INDEX idx_metrics_name_time ON metrics(metric_name, timestamp);

-- Insert 1 million metrics per minute
-- After 1 day: 1.44 billion rows
-- After 1 month: 43 billion rows
\`\`\`

**Problems:**
1. **Storage explosion:** Indexes grow massive (often 2-3x data size)
2. **Write performance degrades:** B-tree index updates are expensive at scale
3. **Query performance suffers:** Scanning billions of rows is slow
4. **No compression:** Traditional databases don't compress time-series well
5. **No downsampling:** Must manually implement data retention
6. **No time-based features:** No native support for time windows, interpolation

**With Time-Series Database:**
- **100x better compression:** Store 100x more data in same space
- **10-100x faster writes:** Optimized for append-only workload
- **10-100x faster queries:** Time-based indexing and partitioning
- **Built-in downsampling:** Automatic data aggregation and retention
- **Time-series functions:** Native support for time operations

## Popular Time-Series Databases

### 1. InfluxDB

**Best for:** General-purpose time-series, monitoring, IoT

**Architecture:**
- Written in Go
- Schemaless (tags and fields)
- Built-in HTTP API
- InfluxQL (SQL-like query language)

**Data Model:**
\`\`\`
measurement,tag1=value1,tag2=value2 field1=value1,field2=value2 timestamp

Example:
cpu,host=server01,region=us-west usage_idle=90.5,usage_user=5.2 1609459200000000000
\`\`\`

**Example:**
\`\`\`python
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

client = InfluxDBClient(url="http://localhost:8086", token="my-token", org="my-org")
write_api = client.write_api(write_options=SYNCHRONOUS)

# Write data
point = Point("cpu") \\
    .tag("host", "server01") \\
    .tag("region", "us-west") \\
    .field("usage_idle", 90.5) \\
    .field("usage_user", 5.2) \\
    .time(datetime.utcnow())

write_api.write(bucket="metrics", record=point)

# Query data
query = '''
from(bucket: "metrics")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "cpu")
  |> filter(fn: (r) => r.host == "server01")
  |> aggregateWindow(every: 1m, fn: mean)
'''

result = client.query_api().query(query=query)
\`\`\`

**Use Cases:**
- Application monitoring
- IoT sensor data
- Real-time analytics

**Pros:**
- Easy to use
- Good compression
- Built-in downsampling

**Cons:**
- Less mature than Prometheus for monitoring
- Clustering requires Enterprise license

### 2. Prometheus

**Best for:** Monitoring and alerting

**Architecture:**
- Written in Go
- Pull-based (scrapes metrics from targets)
- PromQL query language
- Built-in alerting (Alertmanager)

**Data Model:**
\`\`\`
metric_name{label1="value1", label2="value2"} value timestamp

Example:
http_requests_total{method="GET", endpoint="/api/users"} 1027 1609459200
\`\`\`

**Example:**
\`\`\`python
# Expose metrics (application side)
from prometheus_client import Counter, Histogram, start_http_server
import time

# Define metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')

@request_duration.time()
def handle_request(method, endpoint):
    request_count.labels(method=method, endpoint=endpoint).inc()
    # ... handle request

# Start metrics server
start_http_server(8000)

# Prometheus scrapes http://localhost:8000/metrics
\`\`\`

**PromQL Queries:**
\`\`\`promql
# Request rate over 5 minutes
rate(http_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))
\`\`\`

**Use Cases:**
- Infrastructure monitoring
- Application metrics
- Alerting

**Pros:**
- Industry standard for monitoring
- Powerful query language
- Great ecosystem (Grafana, Alertmanager)
- Pull-based (service discovery)

**Cons:**
- Local storage only (no clustering without Thanos/Cortex)
- Limited long-term storage (recommend 15-30 days)
- Pull model not suitable for all scenarios

### 3. TimescaleDB

**Best for:** Time-series data in PostgreSQL

**Architecture:**
- Extension for PostgreSQL
- Automatic partitioning (hypertables)
- SQL interface
- All PostgreSQL features available

**Example:**
\`\`\`sql
-- Create hypertable (automatically partitioned by time)
CREATE TABLE conditions (
    time TIMESTAMPTZ NOT NULL,
    location TEXT NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION
);

SELECT create_hypertable('conditions', 'time');

-- Insert data (same as regular PostgreSQL)
INSERT INTO conditions VALUES
    ('2024-01-01 00:00:00', 'office', 22.5, 45.2),
    ('2024-01-01 00:01:00', 'office', 22.6, 45.3);

-- Time-based queries
SELECT time_bucket('1 hour', time) AS hour,
       location,
       AVG(temperature) AS avg_temp
FROM conditions
WHERE time > NOW() - INTERVAL '24 hours'
GROUP BY hour, location
ORDER BY hour DESC;

-- Automatic downsampling (continuous aggregates)
CREATE MATERIALIZED VIEW conditions_hourly
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', time) AS hour,
       location,
       AVG(temperature) AS avg_temp,
       AVG(humidity) AS avg_humidity
FROM conditions
GROUP BY hour, location;
\`\`\`

**Use Cases:**
- When you need time-series + relational data
- Existing PostgreSQL infrastructure
- Complex queries with JOINs

**Pros:**
- Full SQL support
- Can join with relational tables
- All PostgreSQL features (ACID, transactions, etc.)
- Good compression

**Cons:**
- Requires PostgreSQL knowledge
- Not as specialized as pure TSDBs
- More complex to optimize

### 4. Apache Druid

**Best for:** Real-time analytics, OLAP on time-series

**Architecture:**
- Columnar storage
- Distributed architecture
- Real-time and batch ingestion
- Fast aggregations

**Use Cases:**
- Real-time analytics dashboards
- Clickstream analysis
- Network telemetry

**Pros:**
- Sub-second query latency
- High availability
- Real-time data ingestion

**Cons:**
- Complex to operate
- High resource requirements
- Limited UPDATE support

## Time-Series Database Features

### 1. Compression

Time-series databases use specialized compression:

**Delta encoding:**
\`\`\`
Timestamps: 1000, 1001, 1002, 1003
→ Store: 1000, +1, +1, +1 (much smaller)
\`\`\`

**Delta-of-delta encoding:**
\`\`\`
Values: 100, 102, 104, 106
→ Deltas: +2, +2, +2
→ Store: 100, +2, 0, 0 (pattern detected)
\`\`\`

**Run-length encoding:**
\`\`\`
Values: 5, 5, 5, 5, 5
→ Store: 5 (count=5)
\`\`\`

**Compression ratios:**
- PostgreSQL: 1:1 to 2:1
- InfluxDB: 10:1 to 20:1
- Prometheus: 10:1 to 30:1
- TimescaleDB: 10:1 to 20:1

### 2. Downsampling

Reduce data resolution over time:

\`\`\`sql
-- Raw data (1-second granularity)
time: 2024-01-01 00:00:00, cpu: 45.2
time: 2024-01-01 00:00:01, cpu: 45.4
time: 2024-01-01 00:00:02, cpu: 45.1
...

-- After 7 days: downsample to 1-minute aggregates
time: 2024-01-01 00:00:00, cpu_min: 45.1, cpu_avg: 45.3, cpu_max: 45.9

-- After 30 days: downsample to 1-hour aggregates
time: 2024-01-01 00:00:00, cpu_min: 42.1, cpu_avg: 48.5, cpu_max: 67.2
\`\`\`

**Benefits:**
- Massive storage savings
- Faster queries over long time ranges
- Still retain important patterns

**InfluxDB retention policies:**
\`\`\`sql
-- Keep raw data for 7 days
CREATE RETENTION POLICY "raw" ON "metrics" DURATION 7d REPLICATION 1 DEFAULT

-- Downsample to 1-minute after 7 days
SELECT mean("value") 
INTO "metrics"."monthly".:MEASUREMENT
FROM "metrics"."raw".:MEASUREMENT
GROUP BY time(1m), *

-- Keep 1-minute data for 90 days
CREATE RETENTION POLICY "monthly" ON "metrics" DURATION 90d REPLICATION 1
\`\`\`

### 3. Time-Series Functions

**Aggregation over time windows:**
\`\`\`sql
-- Moving average
SELECT time_bucket('5 minutes', time) AS bucket,
       AVG(cpu_usage) OVER (ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS moving_avg
FROM metrics;

-- Rate of change
SELECT time,
       (value - LAG(value) OVER (ORDER BY time)) / EXTRACT(EPOCH FROM (time - LAG(time) OVER (ORDER BY time))) AS rate
FROM metrics;
\`\`\`

**Interpolation:**
\`\`\`sql
-- Fill missing data points
SELECT time_bucket_gapfill('1 minute', time) AS bucket,
       location,
       AVG(temperature) AS avg_temp
FROM conditions
WHERE time > NOW() - INTERVAL '1 hour'
GROUP BY bucket, location;
\`\`\`

## When to Use Time-Series Databases

### ✅ Use TSDB When:

1. **High write throughput:** Millions of data points per second
2. **Time-based queries:** Most queries filter by time range
3. **Retention policies needed:** Auto-delete or downsample old data
4. **Monitoring/metrics:** Application or infrastructure monitoring
5. **IoT/sensor data:** High-frequency measurements
6. **Financial data:** Stock prices, trades

### ❌ Don't Use TSDB When:

1. **Low data volume:** < 1000 writes/second (PostgreSQL is fine)
2. **Complex relationships:** Need JOINs with many tables (use RDBMS)
3. **ACID transactions critical:** Need strong consistency
4. **Frequent updates:** Data changes after insertion
5. **Ad-hoc queries:** Unpredictable query patterns

## Other Specialized Databases

### 1. Graph Databases (Neo4j, Amazon Neptune)

**Use case:** Data with complex relationships

**Example: Social Network**
\`\`\`cypher
-- Neo4j Cypher query
// Find friends of friends
MATCH (user:Person {name: "Alice"})-[:FRIENDS_WITH]->(friend)-[:FRIENDS_WITH]->(fof)
WHERE NOT (user)-[:FRIENDS_WITH]->(fof) AND user <> fof
RETURN fof.name, COUNT(*) AS mutual_friends
ORDER BY mutual_friends DESC
LIMIT 10;
\`\`\`

**When to use:**
- Social networks (friends, followers)
- Recommendation engines
- Fraud detection (transaction networks)
- Knowledge graphs

### 2. Search Databases (Elasticsearch, Solr)

**Use case:** Full-text search, logging

**Example: Product Search**
\`\`\`json
POST /products/_search
{
  "query": {
    "multi_match": {
      "query": "wireless headphones",
      "fields": ["name^2", "description", "tags"]
    }
  },
  "aggs": {
    "price_ranges": {
      "range": {
        "field": "price",
        "ranges": [
          { "to": 50 },
          { "from": 50, "to": 100 },
          { "from": 100 }
        ]
      }
    }
  }
}
\`\`\`

**When to use:**
- E-commerce product search
- Log aggregation and analysis
- Text-heavy search (documents, articles)

### 3. Columnar Databases (Apache Cassandra, HBase)

**Use case:** Wide tables, high write throughput

**Example: Analytics**
\`\`\`sql
-- Cassandra
CREATE TABLE events (
    user_id UUID,
    event_date DATE,
    event_time TIMESTAMP,
    event_type TEXT,
    properties MAP<TEXT, TEXT>,
    PRIMARY KEY ((user_id, event_date), event_time)
) WITH CLUSTERING ORDER BY (event_time DESC);

-- Fast queries by user and date
SELECT * FROM events 
WHERE user_id = ? AND event_date = ?
ORDER BY event_time DESC
LIMIT 100;
\`\`\`

**When to use:**
- Write-heavy workloads
- Wide tables (many columns)
- Time-series with high cardinality

### 4. In-Memory Databases (Redis, Memcached)

**Use case:** Caching, session storage

**Example: Caching**
\`\`\`python
import redis

r = redis.Redis(host='localhost', port=6379)

# Cache user profile
r.setex(f"user:{user_id}", 3600, json.dumps(user_profile))

# Get from cache
cached = r.get(f"user:{user_id}")
if cached:
    return json.loads(cached)
\`\`\`

**When to use:**
- Session storage
- Caching (hot data)
- Rate limiting counters
- Real-time leaderboards

## Polyglot Persistence

**Modern applications use multiple databases:**

\`\`\`
Application Architecture:

PostgreSQL (Primary)
├─ User accounts, orders, products (ACID transactions)
├─ Core business logic

Redis (Cache)
├─ Session storage
├─ Cache frequently accessed data
└─ Rate limiting counters

Elasticsearch (Search)
├─ Product search
└─ Log aggregation

Prometheus (Metrics)
├─ Application metrics
└─ Infrastructure monitoring

S3 (Object Storage)
└─ User uploads, backups
\`\`\`

**Benefits:**
- Use the right tool for each job
- Optimize performance and cost
- Scale different components independently

**Trade-offs:**
- More complexity
- Data consistency challenges
- More operational overhead

## Interview Tips

**Q: "When would you use a time-series database?"**
- High write throughput (millions of data points/sec)
- Time-based queries and range scans
- Need compression and downsampling
- Examples: monitoring, IoT, financial data

**Q: "What's the difference between InfluxDB and Prometheus?"**
- InfluxDB: Push model, general-purpose TSDB
- Prometheus: Pull model, monitoring-focused, built-in alerting
- Prometheus is standard for monitoring, InfluxDB for broader time-series use cases

**Q: "How do time-series databases achieve better compression?"**
- Delta encoding (store differences)
- Specialized algorithms for timestamps
- Columnar storage
- Run-length encoding for repeated values
- 10-20x better compression than traditional RDBMS

**Q: "What is downsampling and why is it important?"**
- Reducing data resolution over time
- Keep raw data for recent time, aggregates for old data
- Saves storage and improves query performance
- Example: 1-second data → 1-minute → 1-hour aggregates

## Key Takeaways

1. **Time-series databases optimize for append-only, time-stamped data**
2. **10-20x better compression than traditional RDBMS**
3. **Built-in features: downsampling, retention policies, time-based functions**
4. **Choose based on use case:** Prometheus (monitoring), InfluxDB (general TSDB), TimescaleDB (PostgreSQL + time-series)
5. **Not suitable for:** Low write volume, complex relationships, frequent updates
6. **Graph databases excel at relationship queries**
7. **Search databases (Elasticsearch) for full-text search and logs**
8. **Polyglot persistence:** Use multiple databases for different needs
9. **Time-series compression uses delta encoding and pattern detection**
10. **Downsampling reduces storage and improves long-range query performance**

## Summary

Time-series databases are specialized for append-only, time-stamped data with high write throughput. They provide 10-20x better compression through delta encoding and specialized algorithms. Built-in features like downsampling, retention policies, and time-based functions make them ideal for monitoring, IoT, and financial data. Choose Prometheus for monitoring, InfluxDB for general time-series, or TimescaleDB for PostgreSQL-based deployments. Other specialized databases (graph, search, columnar, in-memory) optimize for specific use cases. Modern applications use polyglot persistence, combining multiple database types to leverage their strengths.
`,
};
