/**
 * Quiz questions for Time-Series and Specialized Databases section
 */

export const timeseriesspecializedQuiz = [
  {
    id: 'ts-disc-1',
    question:
      'Design a monitoring and observability system for a microservices platform with 100 services, each instance emitting metrics every 10 seconds. The system should store metrics, provide alerting, enable debugging, and support long-term trend analysis. Choose appropriate databases (time-series, search, etc.), discuss data retention policies, and explain how to handle 10 million data points per minute.',
    sampleAnswer: `Comprehensive monitoring and observability system design:

**System Requirements:**
- 100 microservices
- Each service has 10 instances (1000 instances total)
- Each instance emits 100 metrics every 10 seconds
- Data rate: 1000 instances × 100 metrics × 6/minute = 600,000 metrics/minute = 10,000/second
- Need: Real-time monitoring, alerting, debugging, long-term trends

**Architecture Overview:**

\`\`\`
Application Instances
    ↓
Metrics Collection (Prometheus)
    ↓
Long-term Storage (Thanos / Cortex)
    ↓
Visualization (Grafana)
    ↓
Alerting (Alertmanager)

Logs → Elasticsearch → Kibana
Traces → Jaeger / Tempo
\`\`\`

**1. Metrics (Time-Series Data)**

**Choice: Prometheus + Thanos**

*Why Prometheus:*
- Industry standard for monitoring
- Pull-based (service discovery)
- Powerful query language (PromQL)
- Low operational overhead

*Why Thanos:*
- Long-term storage (Prometheus stores only 15-30 days locally)
- Horizontal scalability
- Global query view across multiple Prometheus instances

**Architecture:**

\`\`\`
┌─────────────────┐
│ Microservices   │
│ (Expose /metrics)│
└────────┬────────┘
         │ scrape every 10s
         ↓
┌────────────────────┐
│ Prometheus (Per DC)│  ← 15-day local storage
│ - US-East          │
│ - US-West          │
│ - EU-West          │
└────────┬───────────┘
         │ upload blocks
         ↓
┌────────────────────┐
│ Thanos             │
│ - Object Storage   │  ← Long-term storage (S3)
│ - Query Frontend   │  ← Unified query interface
│ - Compactor        │  ← Downsampling
└────────────────────┘
\`\`\`

**Data Model:**

\`\`\`python
# Application exposes metrics
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status',]
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint',]
)

# System metrics
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')

# Business metrics
active_users = Gauge('active_users_total', 'Number of active users')
order_value = Histogram('order_value_dollars', 'Order value in dollars')

# Expose metrics endpoint
start_http_server(8000)  # Prometheus scrapes /metrics
\`\`\`

**Prometheus Configuration:**

\`\`\`yaml
# prometheus.yml
global:
  scrape_interval: 10s
  evaluation_interval: 10s
  external_labels:
    cluster: 'us-east-1'
    environment: 'production'

scrape_configs:
  # Service discovery (Kubernetes)
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: ([^:]+)(?::\\d+)?;(\\d+)
        replacement: $1:$2

# Thanos sidecar
thanos:
  sidecar:
    objstore:
      type: S3
      config:
        bucket: "thanos-metrics"
        endpoint: "s3.amazonaws.com"
        region: "us-east-1"
\`\`\`

**Data Retention Policy:**

\`\`\`yaml
# Thanos Compactor config
retention:
  # Raw data (10-second granularity)
  raw: 15d
  
  # 5-minute downsampling
  5m: 90d
  
  # 1-hour downsampling
  1h: 365d
  
  # After 1 year: delete
\`\`\`

**Storage Calculation:**

\`\`\`
Raw Data:
- 600,000 metrics/minute × 8 bytes/metric = 4.8 MB/minute
- Per day: 4.8 MB × 60 × 24 = 6.9 GB/day
- 15 days (Prometheus local): 104 GB
- With compression (10:1): 10.4 GB

Downsampled (5-minute):
- 600,000 metrics/minute / 5 = 120,000 metrics/5min
- Per day: 1.4 GB/day
- 90 days: 126 GB
- With compression: 12.6 GB

Downsampled (1-hour):
- Per day: 115 MB/day
- 365 days: 42 GB
- With compression: 4.2 GB

Total storage (1 year): ~27 GB
Cost (S3): ~$0.65/month
\`\`\`

**2. Logs (Search and Aggregation)**

**Choice: Elasticsearch + Fluentd + Kibana (EFK Stack)**

**Architecture:**

\`\`\`
Application Logs (JSON)
    ↓
Fluentd (Log Aggregator)
    ↓
Elasticsearch (Search and Storage)
    ↓
Kibana (Visualization and Search UI)
\`\`\`

**Log Format:**

\`\`\`python
import logging
import json

# Structured logging
logger = logging.getLogger(__name__)

log_record = {
    "timestamp": "2024-01-01T12:00:00Z",
    "level": "ERROR",
    "service": "order-service",
    "instance": "order-service-7f9d8-abc123",
    "trace_id": "a1b2c3d4e5f6",
    "span_id": "12345",
    "user_id": "user123",
    "message": "Failed to process order",
    "error": {
        "type": "PaymentError",
        "message": "Payment gateway timeout",
        "stack_trace": "..."
    },
    "context": {
        "order_id": "order-789",
        "amount": 99.99,
        "payment_method": "credit_card"
    }
}

logger.error (json.dumps (log_record))
\`\`\`

**Elasticsearch Index Strategy:**

\`\`\`json
// Index template
{
  "index_patterns": ["logs-*",],
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "index.lifecycle.name": "logs-policy",
    "index.codec": "best_compression"
  },
  "mappings": {
    "properties": {
      "timestamp": { "type": "date" },
      "level": { "type": "keyword" },
      "service": { "type": "keyword" },
      "message": { "type": "text" },
      "trace_id": { "type": "keyword" },
      "user_id": { "type": "keyword" }
    }
  }
}

// Index Lifecycle Management (ILM)
{
  "policy": "logs-policy",
  "phases": {
    "hot": {
      "actions": {
        "rollover": {
          "max_size": "50GB",
          "max_age": "1d"
        }
      }
    },
    "warm": {
      "min_age": "7d",
      "actions": {
        "forcemerge": {
          "max_num_segments": 1
        },
        "shrink": {
          "number_of_shards": 1
        }
      }
    },
    "cold": {
      "min_age": "30d",
      "actions": {
        "freeze": {}
      }
    },
    "delete": {
      "min_age": "90d",
      "actions": {
        "delete": {}
      }
    }
  }
}
\`\`\`

**Log Retention:**
- Hot (SSD): 7 days
- Warm (HDD): 30 days
- Cold (compressed): 90 days
- Delete after 90 days

**3. Distributed Tracing**

**Choice: Jaeger (or Tempo)**

\`\`\`python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger import JaegerExporter

# Setup tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger-agent",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor (jaeger_exporter)
)

tracer = trace.get_tracer(__name__)

# Instrument code
@app.route('/api/order', methods=['POST',])
def create_order():
    with tracer.start_as_current_span("create_order") as span:
        span.set_attribute("user_id", user_id)
        
        # Call payment service
        with tracer.start_as_current_span("payment_service.process"):
            payment_result = payment_service.process (amount)
        
        # Call inventory service
        with tracer.start_as_current_span("inventory_service.reserve"):
            inventory_service.reserve (items)
        
        return {"order_id": order_id}
\`\`\`

**4. Alerting**

**Prometheus Alertmanager Configuration:**

\`\`\`yaml
# alerting_rules.yml
groups:
  - name: service_health
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          sum (rate (http_requests_total{status=~"5.."}[5m])) by (service)
          / sum (rate (http_requests_total[5m])) by (service) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in {{ $labels.service }}"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, 
            sum (rate (http_request_duration_seconds_bucket[5m])) by (service, le)
          ) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency in {{ $labels.service }}"
          description: "P95 latency is {{ $value }}s"
      
      # Service down
      - alert: ServiceDown
        expr: up{job="kubernetes-pods"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.instance }} is down"
      
      # High CPU usage
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 80
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU on {{ $labels.instance }}"
          description: "CPU usage is {{ $value }}%"

# alertmanager.yml
route:
  group_by: ['alertname', 'service',]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'team-pager'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
    - match:
        severity: warning
      receiver: 'slack'

receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '<pagerduty-key>'
  
  - name: 'slack'
    slack_configs:
      - api_url: '<slack-webhook>'
        channel: '#alerts'
\`\`\`

**5. Visualization (Grafana)**

\`\`\`json
// Grafana Dashboard (JSON)
{
  "dashboard": {
    "title": "Service Overview",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "sum (rate (http_requests_total[5m])) by (service)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": 'sum (rate (http_requests_total{status=~"5.."}[5m])) / sum (rate (http_requests_total[5m]))'
          }
        ]
      },
      {
        "title": "Latency (P50, P95, P99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum (rate (http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, sum (rate (http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, sum (rate (http_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P99"
          }
        ]
      }
    ]
  }
}
\`\`\`

**6. Cost Optimization**

**Storage Costs (per month):**
- Prometheus (local SSD): 10 GB × 3 regions × $0.10/GB = $3
- Thanos (S3): 27 GB × $0.023/GB = $0.62
- Elasticsearch (hot): 50 GB × $0.10/GB = $5
- Elasticsearch (warm): 100 GB × $0.05/GB = $5
- Jaeger (traces): 10 GB × $0.023/GB = $0.23
**Total: ~$14/month for storage**

**Compute Costs:**
- Prometheus instances: 3 × $50/month = $150
- Elasticsearch cluster: $200/month
- Grafana: $0 (open-source)
**Total: ~$350/month**

**7. Key Design Decisions**

✅ Prometheus for metrics (industry standard)
✅ Thanos for long-term storage (cost-effective)
✅ Elasticsearch for logs (powerful search)
✅ Structured logging (JSON format)
✅ Distributed tracing (Jaeger)
✅ Automated downsampling (10s → 5m → 1h)
✅ Tiered storage (hot → warm → cold → delete)
✅ Multi-level alerting (critical → PagerDuty, warning → Slack)
✅ Correlation (trace_id links metrics, logs, traces)

This architecture handles 10M data points/minute efficiently with <$400/month operational cost.`,
    keyPoints: [
      'Time-series databases provide 10-20x better compression than RDBMS',
      'Choose Prometheus for monitoring, InfluxDB for general time-series, TimescaleDB for PostgreSQL compatibility',
      'Downsampling: keep raw data short-term, aggregate for long-term',
      'Tiered storage: hot (fast SSD), warm (cheaper storage), cold (object storage)',
      'Specialized databases optimize for specific workloads vs general-purpose RDBMS',
    ],
  },
  {
    id: 'ts-disc-2',
    question:
      'You are storing IoT sensor data from 1 million smart home devices. Each device sends temperature, humidity, and motion data every 30 seconds. After 6 months, you notice query performance degrading and storage costs ballooning. Design a comprehensive data retention, downsampling, and archival strategy. Include specific time windows, aggregation levels, and cost-benefit analysis.',
    sampleAnswer: `**IoT Data Management Strategy:**

**Current State Analysis:**

\`\`\`python
# Data volume calculation
devices = 1_000_000
metrics_per_device = 3  # temperature, humidity, motion
interval_seconds = 30
readings_per_day = 86400 / 30  # 2,880 readings/day per metric

# Daily data points
daily_points = devices * metrics_per_device * readings_per_day
# = 1M × 3 × 2,880 = 8.64 billion points/day

# Storage estimate (uncompressed)
# Each point: timestamp (8 bytes) + device_id (8 bytes) + value (8 bytes) = 24 bytes
daily_storage = daily_points * 24 / (1024**3)  # GB
# = 8.64B × 24 / 1024^3 = 193 GB/day uncompressed

# 6 months storage
six_month_storage = daily_storage * 180
# = 193 GB × 180 = 34.7 TB uncompressed

# With time-series compression (10x)
compressed_storage = six_month_storage / 10
# = 3.47 TB
\`\`\`

**Problem Identified:**
- 3.47 TB of data after 6 months
- Most queries only need recent data or historical trends (not raw data)
- Querying billions of points causes performance degradation

---

**Solution: Tiered Data Retention Strategy**

**Retention Tiers:**

| Tier | Duration | Granularity | Compression | Storage | Use Case |
|------|----------|-------------|-------------|---------|----------|
| **Hot** | 7 days | Raw (30s) | Standard (10x) | 1.35 GB | Real-time monitoring, alerts |
| **Warm** | 30 days | 5-minute avg | Downsampled (50x) | 2.3 GB | Recent analysis, debugging |
| **Cold** | 90 days | 1-hour avg | Downsampled (200x) | 3.5 GB | Trend analysis |
| **Archive** | 2 years | Daily avg | Max (1000x) | 14.6 GB | Compliance, long-term trends |
| **Delete** | >2 years | - | - | 0 | GDPR/compliance |

**Total Storage: ~22 GB (vs 3.47 TB = 99.4% reduction!)**

---

**Implementation Design:**

**1. Hot Tier (0-7 days): Raw Data**

\`\`\`python
# InfluxDB configuration for hot data
# influxdb.conf
[retention]
  enabled = true
  check-interval = "30m"

# Create retention policy
CREATE RETENTION POLICY "hot" ON "iot_data" 
  DURATION 7d 
  REPLICATION 1 
  DEFAULT

# Write raw data
from influxdb_client import InfluxDBClient, Point

def write_sensor_data (device_id, temperature, humidity, motion):
    point = Point("sensors") \\
        .tag("device_id", device_id) \\
        .tag("room_type", get_room_type (device_id)) \\
        .field("temperature", temperature) \\
        .field("humidity", humidity) \\
        .field("motion", motion) \\
        .time (datetime.utcnow(), WritePrecision.NS)
    
    write_api.write (bucket="hot", record=point)

# Query hot data (fast)
query = ''
FROM(bucket: "hot")
  |> range (start: -7d)
  |> filter (fn: (r) => r._measurement == "sensors")
  |> filter (fn: (r) => r.device_id == "device_123")
''
\`\`\`

**2. Warm Tier (7-30 days): 5-Minute Aggregates**

\`\`\`python
# Continuous Query for downsampling (InfluxDB)
CREATE CONTINUOUS QUERY "downsample_5m" ON "iot_data"
BEGIN
  SELECT 
    mean (temperature) AS temperature_avg,
    max (temperature) AS temperature_max,
    min (temperature) AS temperature_min,
    mean (humidity) AS humidity_avg,
    sum (motion) AS motion_count
  INTO "warm"."sensors_5m"
  FROM "hot"."sensors"
  GROUP BY time(5m), device_id
END

# Retention policy for warm tier
CREATE RETENTION POLICY "warm" ON "iot_data" 
  DURATION 30d 
  REPLICATION 1

# Result: 
# - Raw: 2,880 points/day → Warm: 288 points/day (10x reduction)
\`\`\`

**3. Cold Tier (30-90 days): 1-Hour Aggregates**

\`\`\`python
# Downsample to 1-hour aggregates
CREATE CONTINUOUS QUERY "downsample_1h" ON "iot_data"
BEGIN
  SELECT 
    mean (temperature_avg) AS temperature_avg,
    max (temperature_max) AS temperature_max,
    min (temperature_min) AS temperature_min,
    mean (humidity_avg) AS humidity_avg,
    sum (motion_count) AS motion_count
  INTO "cold"."sensors_1h"
  FROM "warm"."sensors_5m"
  GROUP BY time(1h), device_id
END

# Move to cheaper storage (S3)
# Use InfluxDB OSS 2.0 with cloud storage tier
# OR TimescaleDB with compressed chunks

# Result:
# - Warm: 288 points/day → Cold: 24 points/day (12x reduction)
\`\`\`

**4. Archive Tier (90 days - 2 years): Daily Aggregates**

\`\`\`python
# Further downsample to daily aggregates
CREATE CONTINUOUS QUERY "downsample_daily" ON "iot_data"
BEGIN
  SELECT 
    mean (temperature_avg) AS temperature_avg,
    max (temperature_max) AS temperature_max,
    min (temperature_min) AS temperature_min,
    mean (humidity_avg) AS humidity_avg,
    sum (motion_count) AS motion_count
  INTO "archive"."sensors_daily"
  FROM "cold"."sensors_1h"
  GROUP BY time(1d), device_id
END

# Store in S3 with compression
# AWS S3 Glacier for long-term storage
# Cost: $0.004/GB/month (vs $0.10/GB/month for hot storage)

# Result:
# - Cold: 24 points/day → Archive: 1 point/day (24x reduction)
\`\`\`

**5. Automated Tier Migration**

\`\`\`python
import schedule
from datetime import datetime, timedelta

def migrate_hot_to_warm():
    """Run daily: migrate 7-day-old data to warm tier"""
    cutoff = datetime.utcnow() - timedelta (days=7)
    
    # Downsample and write to warm tier
    result = client.query (f''
        SELECT mean (temperature), mean (humidity), sum (motion)
        FROM sensors
        WHERE time < '{cutoff.isoformat()}'
        GROUP BY time(5m), device_id
        INTO warm.sensors_5m
    '')
    
    # Delete from hot tier (automatically handled by retention policy)
    logger.info (f"Migrated {result.count} points to warm tier")

def migrate_warm_to_cold():
    """Run weekly: migrate 30-day-old data to cold tier"""
    cutoff = datetime.utcnow() - timedelta (days=30)
    
    result = client.query (f''
        SELECT mean (temperature_avg), mean (humidity_avg), sum (motion_count)
        FROM warm.sensors_5m
        WHERE time < '{cutoff.isoformat()}'
        GROUP BY time(1h), device_id
        INTO cold.sensors_1h
    '')
    
    logger.info (f"Migrated {result.count} points to cold tier")

def migrate_cold_to_archive():
    """Run monthly: migrate 90-day-old data to archive"""
    cutoff = datetime.utcnow() - timedelta (days=90)
    
    # Export to S3
    result = client.query (f''
        SELECT mean (temperature_avg), max (temperature_max), min (temperature_min)
        FROM cold.sensors_1h
        WHERE time < '{cutoff.isoformat()}'
        GROUP BY time(1d), device_id
    '')
    
    # Write to S3 in Parquet format
    df = result_to_dataframe (result)
    df.to_parquet (f"s3://iot-archive/{cutoff.year}/{cutoff.month}/sensors.parquet")
    
    logger.info (f"Archived {len (df)} points to S3")

# Schedule jobs
schedule.every().day.at("02:00").do (migrate_hot_to_warm)
schedule.every().week.at("03:00").do (migrate_warm_to_cold)
schedule.every().month.at("04:00").do (migrate_cold_to_archive)
\`\`\`

---

**6. Query Optimization Strategy**

\`\`\`python
def query_sensor_data (device_id, start_time, end_time):
    """
    Smart query router: automatically choose appropriate tier
    """
    now = datetime.utcnow()
    days_ago = (now - start_time).days
    
    if days_ago <= 7:
        # Query hot tier (raw data)
        bucket = "hot"
        measurement = "sensors"
        granularity = "30s"
    elif days_ago <= 30:
        # Query warm tier (5-minute aggregates)
        bucket = "warm"
        measurement = "sensors_5m"
        granularity = "5m"
    elif days_ago <= 90:
        # Query cold tier (1-hour aggregates)
        bucket = "cold"
        measurement = "sensors_1h"
        granularity = "1h"
    else:
        # Query archive tier (daily aggregates)
        bucket = "archive"
        measurement = "sensors_daily"
        granularity = "1d"
    
    query = f''
        FROM(bucket: "{bucket}")
          |> range (start: {start_time}, stop: {end_time})
          |> filter (fn: (r) => r._measurement == "{measurement}")
          |> filter (fn: (r) => r.device_id == "{device_id}")
          |> aggregateWindow (every: {granularity}, fn: mean)
    ''
    
    return client.query_api().query (query)
\`\`\`

---

**7. Cost-Benefit Analysis**

**Before Optimization (6 months):**

| Component | Storage | Cost/GB/month | Monthly Cost |
|-----------|---------|---------------|--------------|
| InfluxDB (SSD) | 3.47 TB | $0.10 | $355 |
| **Total** | **3.47 TB** | | **$355/month** |

**After Optimization:**

| Tier | Storage | Cost/GB/month | Monthly Cost |
|------|---------|---------------|--------------|
| Hot (7d, SSD) | 1.35 GB | $0.10 | $0.14 |
| Warm (30d, SSD) | 2.3 GB | $0.10 | $0.23 |
| Cold (90d, SSD) | 3.5 GB | $0.10 | $0.35 |
| Archive (2y, S3 Glacier) | 14.6 GB | $0.004 | $0.06 |
| **Total** | **21.76 GB** | | **$0.78/month** |

**Savings: $354/month (99.8% reduction!)**

**Performance Comparison:**

| Query Type | Before | After | Improvement |
|------------|--------|-------|-------------|
| Last 24 hours | 2.5s | 50ms | 50x faster |
| Last 7 days | 15s | 200ms | 75x faster |
| Last 30 days | 60s | 800ms | 75x faster |
| Last 90 days | 300s | 2s | 150x faster |

---

**8. Key Design Principles**

✅ **Tier data by access patterns** (recent = hot, old = cold)
✅ **Downsample aggressively** (raw → 5m → 1h → daily)
✅ **Automate tier migration** (scheduled jobs)
✅ **Smart query routing** (automatically select appropriate tier)
✅ **Preserve statistical value** (mean, min, max, count)
✅ **Use appropriate storage** (SSD for hot, S3 Glacier for archive)
✅ **Set retention policies** (auto-delete after 2 years)
✅ **Monitor migration jobs** (ensure data not lost during migration)

This strategy reduces storage costs by 99.8% while maintaining query performance and data value for analysis.`,
    keyPoints: [
      'Tier data by access patterns: hot (raw, 7d), warm (5m agg, 30d), cold (1h agg, 90d), archive (daily)',
      'Downsample aggressively as data ages to reduce storage and improve query performance',
      'Automate tier migration with scheduled jobs (daily, weekly, monthly)',
      'Smart query routing: automatically select appropriate tier based on time range',
      'Use cheap object storage (S3 Glacier) for long-term archive',
    ],
  },
  {
    id: 'ts-disc-3',
    question:
      'Compare InfluxDB, TimescaleDB, and Prometheus for time-series data. For each database, describe ideal use cases, strengths, weaknesses, and when you would choose one over the others. Provide specific scenarios and technical reasoning.',
    sampleAnswer: `**Comprehensive Comparison of Time-Series Databases:**

---

## **1. InfluxDB**

**What it is:**
General-purpose time-series database designed for high write throughput and flexible data models.

**Architecture:**
- Custom storage engine (TSM - Time-Structured Merge tree)
- Columnar storage with specialized compression
- InfluxQL (SQL-like) and Flux query languages
- Built-in downsampling and retention policies

**Strengths:**

✅ **High write throughput** - 100,000+ points/second per node
✅ **Purpose-built for time-series** - optimized storage and queries
✅ **Flexible tagging system** - multi-dimensional data modeling
✅ **Built-in downsampling** - continuous queries for automatic aggregation
✅ **Easy to operate** - single binary, no dependencies
✅ **Multiple query languages** - InfluxQL (SQL-like) and Flux (powerful)

**Weaknesses:**

❌ **Clustering complexity** - open-source version is single-node only (InfluxDB Cloud/Enterprise for clustering)
❌ **No JOINs** - can't join with relational data
❌ **Memory intensive** - requires significant RAM for indexing
❌ **Limited ecosystem** - smaller community than PostgreSQL
❌ **Flux learning curve** - powerful but complex query language

**Ideal Use Cases:**1. **IoT sensor data** - millions of devices sending metrics
2. **Application metrics** - non-monitoring use cases (analytics, dashboards)
3. **Real-time analytics** - fast aggregations over time windows
4. **Financial data** - stock prices, trading data

**Example Scenario:**

\`\`\`python
# InfluxDB: IoT temperature monitoring for 1M devices
from influxdb_client import InfluxDBClient, Point

# Write data (very fast)
point = Point("temperature") \\
    .tag("device_id", "sensor_123") \\
    .tag("location", "warehouse_A") \\
    .tag("floor", "3") \\
    .field("value", 22.5) \\
    .time (datetime.utcnow())

write_api.write (bucket="sensors", record=point)

# Query: Get average temperature per floor last 24h
query = ''
from (bucket: "sensors")
  |> range (start: -24h)
  |> filter (fn: (r) => r._measurement == "temperature")
  |> group (columns: ["floor",])
  |> aggregateWindow (every: 1h, fn: mean)
''

# Downsampling (automatic)
CREATE CONTINUOUS QUERY "hourly_avg" ON "iot_data"
BEGIN
  SELECT mean (value) AS value_mean
  INTO "downsampled"."temperature_hourly"
  FROM "temperature"
  GROUP BY time(1h), *
END
\`\`\`

---

## **2. TimescaleDB**

**What it is:**
Time-series extension for PostgreSQL - combines RDBMS power with time-series optimizations.

**Architecture:**
- Built on PostgreSQL (extension, not fork)
- Automatic partitioning (hypertables)
- Compressed columnar storage
- Full SQL support with time-series functions

**Strengths:**

✅ **SQL compatibility** - full PostgreSQL SQL support
✅ **JOINs work** - can join time-series with relational data
✅ **ACID transactions** - full transactional guarantees
✅ **Rich ecosystem** - all PostgreSQL tools work (pgAdmin, connectors)
✅ **Hybrid workloads** - mix time-series and relational data
✅ **Automatic compression** - 90%+ compression for time-series
✅ **Mature ecosystem** - PostgreSQL's stability and community

**Weaknesses:**

❌ **PostgreSQL overhead** - not as fast as specialized TSDBs
❌ **More complex to operate** - PostgreSQL tuning required
❌ **Write performance** - lower than InfluxDB for pure time-series
❌ **Memory usage** - PostgreSQL can be memory-hungry
❌ **Horizontal scaling** - more complex than InfluxDB clustering

**Ideal Use Cases:**1. **Hybrid workloads** - need both time-series AND relational data
2. **Financial systems** - need ACID transactions with time-series data
3. **Migrating from PostgreSQL** - already using PostgreSQL
4. **Complex analytics** - need JOINs, window functions, CTEs

**Example Scenario:**

\`\`\`sql
-- TimescaleDB: E-commerce analytics with user data

-- Create hypertable (automatic partitioning)
CREATE TABLE page_views (
    time TIMESTAMPTZ NOT NULL,
    user_id INTEGER NOT NULL,
    page_url TEXT,
    duration_ms INTEGER,
    FOREIGN KEY (user_id) REFERENCES users (id)  -- Can use FK!
);

SELECT create_hypertable('page_views', 'time');

-- Enable compression
ALTER TABLE page_views SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'user_id'
);

-- Query: JOIN time-series with relational data
SELECT 
    u.email,
    u.subscription_tier,
    AVG(pv.duration_ms) as avg_duration,
    COUNT(*) as page_count
FROM page_views pv
JOIN users u ON pv.user_id = u.id
WHERE pv.time > NOW() - INTERVAL '7 days'
  AND u.subscription_tier = 'premium'
GROUP BY u.email, u.subscription_tier
ORDER BY avg_duration DESC;

-- Continuous aggregates (materialized views)
CREATE MATERIALIZED VIEW page_views_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    user_id,
    COUNT(*) as views,
    AVG(duration_ms) as avg_duration
FROM page_views
GROUP BY bucket, user_id;

-- Automatic refresh
SELECT add_continuous_aggregate_policy('page_views_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
\`\`\`

---

## **3. Prometheus**

**What it is:**
Monitoring and alerting system specifically designed for operational metrics.

**Architecture:**
- Pull-based model (scrapes metrics from targets)
- Local time-series database (limited to single node)
- PromQL query language
- Built-in alerting (Alertmanager)
- Service discovery integration

**Strengths:**

✅ **Monitoring-focused** - built for operational monitoring
✅ **Pull model** - discovers and scrapes targets automatically
✅ **Service discovery** - integrates with Kubernetes, Consul, etc.
✅ **Built-in alerting** - Alertmanager included
✅ **Grafana integration** - de facto standard for dashboards
✅ **Low operational overhead** - single binary, no dependencies
✅ **Open-source ecosystem** - huge ecosystem of exporters

**Weaknesses:**

❌ **Local storage only** - limited retention (15-30 days typically)
❌ **Single-node** - no built-in clustering (need Thanos/Cortex for scale)
❌ **Pull model** - not suitable for all use cases
❌ **No long-term storage** - need Thanos, Cortex, or VictoriaMetrics
❌ **Limited query capabilities** - PromQL less powerful than SQL
❌ **Not general-purpose** - specifically for monitoring

**Ideal Use Cases:**1. **Infrastructure monitoring** - servers, containers, services
2. **Kubernetes monitoring** - native integration
3. **Application metrics** - RED (Rate, Errors, Duration) metrics
4. **Alerting** - operational alerts and on-call

**Example Scenario:**

\`\`\`yaml
# Prometheus: Kubernetes monitoring

# prometheus.yml
scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

# Application exposes metrics
from prometheus_client import Counter, Histogram, start_http_server

request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status',])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint',])

@app.route('/api/users')
@request_duration.labels (method='GET', endpoint='/api/users').time()
def get_users():
    request_count.labels (method='GET', endpoint='/api/users', status=200).inc()
    return users

# Start metrics server on :8000
start_http_server(8000)

# PromQL queries
# Query: Error rate by service
sum (rate (http_requests_total{status=~"5.."}[5m])) by (service)
/ 
sum (rate (http_requests_total[5m])) by (service)

# Query: P95 latency
histogram_quantile(0.95, 
  sum (rate (http_request_duration_seconds_bucket[5m])) by (le, service)
)

# Alert rule
groups:
  - name: example
    rules:
      - alert: HighErrorRate
        expr: rate (http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on {{ $labels.service }}"
\`\`\`

---

## **Decision Matrix**

| Scenario | Choose | Why |
|----------|--------|-----|
| **Infrastructure monitoring** | Prometheus | Built for monitoring, service discovery, alerting |
| **IoT sensor data (millions of devices)** | InfluxDB | High write throughput, purpose-built for time-series |
| **E-commerce analytics (need JOINs)** | TimescaleDB | SQL support, can JOIN with user/product tables |
| **Financial trading data** | TimescaleDB or InfluxDB | TimescaleDB if need ACID, InfluxDB for pure speed |
| **Hybrid workload (time-series + relational)** | TimescaleDB | PostgreSQL compatibility, full SQL support |
| **Kubernetes metrics** | Prometheus | Native Kubernetes integration |
| **Long-term trends (years of data)** | InfluxDB or TimescaleDB | Both support downsampling and retention |
| **Need clustering** | InfluxDB Enterprise or TimescaleDB | Both support multi-node setups |

---

## **Real-World Example: Choose Database for Different Components**

**Scenario: E-commerce platform**

\`\`\`
Component                    | Database Choice  | Reasoning
-----------------------------|------------------|--------------------------------
Infrastructure monitoring    | Prometheus       | Service discovery, alerting
Application metrics (custom) | InfluxDB         | High write throughput
User behavior analytics      | TimescaleDB      | Need JOIN with user table
Product performance tracking | TimescaleDB      | Need JOIN with product catalog
Long-term storage           | Thanos (+ Prom)  | Cost-effective S3 storage
\`\`\`

---

## **My Recommendations:**

**Start with:**
- **Prometheus** if you're monitoring infrastructure/applications
- **TimescaleDB** if you already use PostgreSQL or need SQL
- **InfluxDB** if you have pure time-series workload without relational needs

**Scale with:**
- Prometheus → **Thanos** or **Cortex** for long-term storage
- InfluxDB → **InfluxDB Cloud/Enterprise** for clustering
- TimescaleDB → **TimescaleDB clustering** for horizontal scale

**Best practice:**
Use polyglot persistence - Prometheus for monitoring, TimescaleDB for business analytics, InfluxDB for IoT. Each database excels at different use cases.`,
    keyPoints: [
      'InfluxDB: best for high-write IoT and general time-series (purpose-built, fast writes)',
      'TimescaleDB: best for hybrid workloads needing SQL/JOINs (PostgreSQL compatibility)',
      'Prometheus: best for operational monitoring and alerting (pull model, service discovery)',
      'Use polyglot persistence: different databases for different use cases',
      'Consider clustering needs: Prometheus needs Thanos, InfluxDB OSS is single-node',
    ],
  },
];
