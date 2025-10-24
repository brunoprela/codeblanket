/**
 * Metrics & Monitoring Section
 */

export const metricsMonitoringSection = {
  id: 'metrics-monitoring',
  title: 'Metrics & Monitoring',
  content: `Metrics are the heartbeat of your system. They provide real-time visibility into system health, enable proactive alerting, and help you understand trends over time. Unlike logs (discrete events) and traces (request journeys), metrics are aggregated numerical measurements that answer "how much" and "how fast."

## What Are Metrics?

**Metrics** are numerical measurements collected over time that represent system state and behavior.

**Characteristics**:
- **Time-series data**: Values with timestamps
- **Aggregated**: Counted, summed, averaged
- **Low cardinality**: Limited dimensions
- **Efficient**: Much smaller than logs
- **Real-time**: Updated continuously

**Example Metrics**:
- Request rate: 1500 requests/second
- Error rate: 0.5%
- Latency (p99): 250ms
- CPU usage: 65%
- Active connections: 47

---

## Metrics vs Logs vs Traces

| Aspect | Metrics | Logs | Traces |
|--------|---------|------|--------|
| **Type** | Aggregated numbers | Discrete events | Request paths |
| **Volume** | Low | Very high | Medium |
| **Cost** | Cheap | Expensive | Moderate |
| **Use Case** | Dashboards, alerts | Debugging | Performance |
| **Cardinality** | Low | High | Medium |
| **Retention** | Long (years) | Short (days) | Short (days) |

**When to Use What**:
- **Metrics**: "Is the system healthy?" "Are we meeting SLOs?"
- **Logs**: "Why did this specific request fail?"
- **Traces**: "Where is the bottleneck in request flow?"

---

## Metric Types

### **1. Counter**
Monotonically increasing value (only goes up)

**Examples**:
- Total HTTP requests
- Total errors
- Total bytes sent
- Total messages processed

**Properties**:
- Starts at 0
- Only increases
- Resets on restart

**Usage**:
\`\`\`
http_requests_total{method="GET", status="200"} → 15420
http_requests_total{method="POST", status="201"} → 3201
\`\`\`

**Queries** (Prometheus):
\`\`\`
# Rate of requests per second
rate(http_requests_total[5m])

# Total requests in last hour
increase(http_requests_total[1h])
\`\`\`

### **2. Gauge**
Point-in-time value (can go up or down)

**Examples**:
- CPU usage (%)
- Memory usage (bytes)
- Active connections
- Queue length
- Temperature

**Properties**:
- Current value
- Can increase or decrease
- Represents "right now"

**Usage**:
\`\`\`
cpu_usage_percent{host="server1"} → 65.3
memory_used_bytes{host="server1"} → 4294967296
active_connections{service="api"} → 47
\`\`\`

**Queries**:
\`\`\`
# Average CPU over 5 minutes
avg_over_time(cpu_usage_percent[5m])

# Peak memory in last hour
max_over_time(memory_used_bytes[1h])
\`\`\`

### **3. Histogram**
Samples observations and counts them in buckets

**Examples**:
- HTTP request durations
- Response sizes
- Database query times

**Structure**:
\`\`\`
http_request_duration_seconds_bucket{le="0.1"} → 5000   # <= 100ms
http_request_duration_seconds_bucket{le="0.5"} → 8500   # <= 500ms
http_request_duration_seconds_bucket{le="1.0"} → 9800   # <= 1s
http_request_duration_seconds_bucket{le="+Inf"} → 10000 # All requests
http_request_duration_seconds_sum → 1250.5              # Total time
http_request_duration_seconds_count → 10000             # Total requests
\`\`\`

**Benefits**:
- Calculate percentiles (p50, p95, p99)
- See distribution
- Aggregatable across services

**Queries**:
\`\`\`
# 99th percentile latency
histogram_quantile(0.99, 
  rate(http_request_duration_seconds_bucket[5m]))

# Average latency
rate(http_request_duration_seconds_sum[5m]) /
rate(http_request_duration_seconds_count[5m])
\`\`\`

### **4. Summary**
Similar to histogram but calculates quantiles on client side

**Structure**:
\`\`\`
http_request_duration_seconds{quantile="0.5"} → 0.12    # p50
http_request_duration_seconds{quantile="0.9"} → 0.35    # p90
http_request_duration_seconds{quantile="0.99"} → 0.85   # p99
http_request_duration_seconds_sum → 1250.5
http_request_duration_seconds_count → 10000
\`\`\`

**Histogram vs Summary**:
- **Histogram**: Aggregate across instances, flexible percentiles
- **Summary**: Pre-calculated percentiles, less flexible
- **Recommendation**: Use histograms (more flexible)

---

## Key Monitoring Frameworks

### **RED Method** (for services)

**Rate**: Requests per second
\`\`\`
rate(http_requests_total[5m])
\`\`\`

**Errors**: Error rate
\`\`\`
rate(http_requests_total{status=~"5.."}[5m]) /
rate(http_requests_total[5m])
\`\`\`

**Duration**: Latency distribution
\`\`\`
histogram_quantile(0.99, 
  rate(http_request_duration_seconds_bucket[5m]))
\`\`\`

**When to Use**: Monitoring user-facing services

### **USE Method** (for resources)

**Utilization**: % time resource busy
- CPU usage
- Disk usage
- Memory usage

**Saturation**: Degree of queuing
- CPU run queue length
- Disk I/O wait
- Memory swapping

**Errors**: Error count
- Disk errors
- Network errors
- Dropped packets

**When to Use**: Monitoring infrastructure and resources

### **Four Golden Signals** (Google SRE)

1. **Latency**: Time to serve a request
2. **Traffic**: Demand on system (requests/sec)
3. **Errors**: Rate of failed requests
4. **Saturation**: How "full" the system is

**Why These Four**: Provide comprehensive view of system health

---

## Metric Naming Conventions

### **Prometheus Conventions**

**Format**: \`< namespace > _ < metric > _ < unit > _<suffix>\`

**Examples**:
\`\`\`
http_requests_total           # Counter (suffix: _total)
http_request_duration_seconds # Histogram (unit: seconds)
process_cpu_usage_ratio       # Gauge (unit: ratio 0-1)
database_queries_in_flight    # Gauge (current state)
\`\`\`

**Rules**:
- Use underscores, not camelCase
- Include unit in name (seconds, bytes, ratio)
- Counters end with \`_total\`
- Base units (seconds not milliseconds, bytes not megabytes)

### **Labels (Dimensions)**

**Labels** add dimensions to metrics

**Example**:
\`\`\`
http_requests_total{
  method="GET",
  path="/api/users",
  status="200",
  service="api"
} → 1500
\`\`\`

**Best Practices**:
✅ Use labels for **bounded** dimensions:
- HTTP method: GET, POST, PUT, DELETE (4 values)
- Status code: 200, 404, 500, etc. (~20 values)
- Service name: api, worker, db (~10 values)

❌ Avoid labels for **unbounded** dimensions:
- User IDs (millions of values)
- Request IDs (infinite)
- Email addresses (millions)

**Why**: Each unique combination creates a new time series → cardinality explosion → high costs

---

## Cardinality

**Cardinality** = number of unique time series

**Example**:
\`\`\`
Metric: http_requests_total
Labels: {method, path, status}
method: 4 values (GET, POST, PUT, DELETE)
path: 10 values (/users, /posts, etc.)
status: 5 values (200, 201, 400, 404, 500)

Cardinality = 4 × 10 × 5 = 200 time series
\`\`\`

### **Cardinality Explosion**

**Bad Example**:
\`\`\`
http_requests_total{
  method="GET",
  user_id="user123",    # ❌ Millions of unique values
  request_id="abc-123"  # ❌ Infinite unique values
}
\`\`\`

**Impact**:
- 1M users × 4 methods = 4 million time series
- Storage cost explodes
- Query performance degrades
- Metric system crashes

**Solution**:
- Remove high-cardinality labels
- Use logs for high-cardinality data
- Limit cardinality with guards

---

## Prometheus

**Prometheus** is the de facto standard for metrics in cloud-native systems.

### **Architecture**

\`\`\`
┌─────────────┐
│ Application │ Expose /metrics endpoint
└──────┬──────┘
       │
       ↓ (scrape)
┌─────────────┐
│ Prometheus  │ Collect, store, query
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Grafana   │ Visualize
└─────────────┘
\`\`\`

### **Pull Model**

Prometheus **pulls** metrics from applications:
1. Application exposes \`/ metrics\` HTTP endpoint
2. Prometheus scrapes endpoint every 15s (configurable)
3. Metrics stored in time-series database

**Benefits**:
- Centralized control (Prometheus decides when to scrape)
- Service discovery (dynamic targets)
- Failure detection (scrape failures)
- No client buffering needed

**Drawbacks**:
- Requires open port
- Short-lived jobs need Pushgateway
- Firewall complexity

### **PromQL**

**Prometheus Query Language** for querying metrics

**Basic Queries**:
\`\`\`
# Current value
http_requests_total

# Filter by labels
http_requests_total{method="GET", status="200"}

# Rate (per-second average over 5 minutes)
rate(http_requests_total[5m])

# Sum across all services
sum(rate(http_requests_total[5m])) by (status)
\`\`\`

**Advanced Queries**:
\`\`\`
# Error rate percentage
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m])) * 100

# p99 latency
histogram_quantile(0.99,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# CPU usage across all pods
avg(rate(container_cpu_usage_seconds_total[5m])) by (pod)
\`\`\`

### **Service Discovery**

Prometheus can auto-discover targets:
- Kubernetes (pods, services)
- AWS EC2
- Consul
- DNS
- File-based

**Kubernetes Example**:
\`\`\`yaml
scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
\`\`\`

---

## Metric Aggregation

### **Aggregation Functions**

**Sum**: Total across dimensions
\`\`\`
sum(http_requests_total) by (service)
\`\`\`

**Average**: Mean value
\`\`\`
avg(cpu_usage_percent) by (host)
\`\`\`

**Min/Max**: Extremes
\`\`\`
max(memory_used_bytes) by (pod)
\`\`\`

**Count**: Number of time series
\`\`\`
count(up == 1)  # Number of healthy instances
\`\`\`

**Quantile**: Percentiles
\`\`\`
quantile(0.95, http_request_duration_seconds)
\`\`\`

### **Aggregation Best Practices**

1. **Aggregate Early**: At query time, not collection
2. **Preserve Labels**: Use \`by (label)\` to keep useful dimensions
3. **Without vs By**: \`without(label)\` removes specific labels, \`by(label)\` keeps only specified

---

## Dashboards

### **Dashboard Design Principles**

**1. Hierarchy**
- Top: High-level health (error rate, latency)
- Middle: Service-specific metrics
- Bottom: Infrastructure (CPU, memory)

**2. Time Synchronization**
- All graphs show same time range
- Easy to correlate events

**3. Consistency**
- Same colors for same metrics
- Same scales across services

**4. Context**
- Show SLO thresholds
- Include units in labels
- Add descriptions

### **Grafana**

**Industry Standard** for metric visualization

**Features**:
- Multiple data sources (Prometheus, InfluxDB, etc.)
- Templating (variables)
- Alerting
- Annotations (mark deployments)
- Sharing and permissions

**Dashboard Example**:
\`\`\`
┌─────────────────────────────────────┐
│ Service Health                      │
├─────────────────────────────────────┤
│ Request Rate: 1.5K/s ↑ 10%         │
│ Error Rate: 0.2% ✓ (SLO: < 1%)    │
│ p99 Latency: 250ms ✓ (SLO: < 500ms)│
└─────────────────────────────────────┘

[Request Rate Graph - Last 24h]
[Error Rate Graph - Last 24h]
[Latency Percentiles - Last 24h]
\`\`\`

### **Essential Dashboards**

1. **Service Dashboard**
   - RED metrics
   - Request rate
   - Error rate
   - Latency percentiles

2. **Infrastructure Dashboard**
   - USE metrics
   - CPU, memory, disk
   - Network I/O

3. **Business Dashboard**
   - Revenue/transactions
   - User signups
   - Feature usage

---

## Metric Storage

### **Time-Series Databases**

**Prometheus** (most common):
- Built-in to Prometheus
- Local storage
- Retention: 15 days typical
- Not for long-term storage

**Thanos** (Prometheus long-term):
- Extends Prometheus
- Object storage backend (S3)
- Multi-year retention
- Query across clusters

**Cortex** (Prometheus multi-tenant):
- Multi-tenancy
- Horizontal scalability
- Long-term storage

**InfluxDB**:
- Alternative to Prometheus
- Push model
- SQL-like query language

**VictoriaMetrics**:
- Prometheus-compatible
- Higher performance
- Lower resource usage

### **Storage Optimization**

**1. Downsampling**
- Raw data: 15s resolution for 7 days
- 5min aggregates: 30 days
- 1hour aggregates: 1 year

**2. Retention Policies**
- Short-term: High resolution, fast storage
- Long-term: Low resolution, cheap storage

**3. Compression**
- Time-series compress well (up to 10x)
- Prometheus uses snappy compression

---

## Metric Collection Performance

### **Client Libraries**

**Prometheus Client Libraries** for all languages:
- Python: \`prometheus_client\`
- Go: \`prometheus / client_golang\`
- Java: \`io.prometheus\`
- Node.js: \`prom - client\`

**Example** (Node.js):
\`\`\`javascript
const client = require('prom-client');

// Counter
const httpRequestsTotal = new client.Counter({
  name: 'http_requests_total',
  help: 'Total HTTP requests',
  labelNames: ['method', 'status']
});

// Increment
httpRequestsTotal.inc({ method: 'GET', status: '200' });

// Histogram
const httpDuration = new client.Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration',
  buckets: [0.1, 0.5, 1, 2, 5]
});

// Observe
const end = httpDuration.startTimer();
// ... handle request ...
end();
\`\`\`

### **Performance Impact**

**Metric Collection Overhead**:
- Counters: ~10ns per increment
- Histograms: ~100ns per observation
- Negligible for most applications

**Network Overhead**:
- Scrape every 15s
- Payload: ~1-10KB per scrape
- Bandwidth: < 1KB/s per instance

---

## Best Practices

### **Do's**
✅ Use counters for totals, gauges for current values
✅ Include units in metric names
✅ Keep cardinality low
✅ Expose metrics on \`/ metrics\` endpoint
✅ Use RED/USE method
✅ Create actionable dashboards
✅ Document metrics in code

### **Don'ts**
❌ High-cardinality labels (user IDs, request IDs)
❌ Metrics for non-aggregatable data
❌ Too many metrics (metric explosion)
❌ Inconsistent naming
❌ Metrics without documentation
❌ Dashboards without context

---

## Interview Tips

### **Key Points to Mention**

1. **Metrics Types**: Counter, gauge, histogram, summary
2. **RED Method**: Rate, errors, duration
3. **Cardinality**: Avoid high-cardinality labels
4. **Prometheus**: Industry standard, pull model
5. **Dashboards**: Grafana for visualization

### **Common Questions**

**Q: How would you monitor a new service?**
A: Expose metrics endpoint, track RED (rate, errors, duration), create Grafana dashboard, set up alerts on error rate and latency SLOs.

**Q: What's cardinality and why does it matter?**
A: Number of unique time series. High cardinality (e.g., user_id label with millions of values) explodes storage and degrades performance. Keep dimensions bounded.

**Q: Counter vs Gauge?**
A: Counter only increases (total requests), gauge can go up/down (CPU usage, active connections).

---

## Real-World Examples

### **Google**
- **Monarch**: Planet-scale metrics system
- **Volume**: Billions of time series
- **Lesson**: Hierarchical aggregation, distributed queries

### **Netflix**
- **Atlas**: In-house metrics system
- **Volume**: 1.2 billion metrics/minute
- **Lesson**: Streaming aggregation, real-time alerting

### **Uber**
- **M3**: Open-source metrics platform
- **Scale**: Millions of time series per cluster
- **Lesson**: Multi-datacenter aggregation, long-term storage

---

## Summary

Metrics provide real-time visibility into system health. Key takeaways:

1. **Four Types**: Counter, gauge, histogram, summary
2. **Low Cardinality**: Avoid high-cardinality labels
3. **Prometheus**: Industry standard for metrics
4. **RED/USE**: Frameworks for comprehensive monitoring
5. **Dashboards**: Visualize metrics in Grafana
6. **Storage**: Time-series databases with downsampling

Good metrics enable proactive monitoring, fast incident response, and data-driven decisions. Design metric systems with cardinality in mind and focus on actionable signals over vanity metrics.`,
};
