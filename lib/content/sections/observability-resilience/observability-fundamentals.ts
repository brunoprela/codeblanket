/**
 * Observability Fundamentals Section
 */

export const observabilityFundamentalsSection = {
  id: 'observability-fundamentals',
  title: 'Observability Fundamentals',
  content: `Observability is the ability to understand the internal state of a system by examining its external outputs. In modern distributed systems, observability is not optional—it's essential for maintaining reliability, debugging issues, and understanding system behavior.

## What is Observability?

**Observability** is a measure of how well you can infer the internal state of a system from the knowledge of its external outputs. It answers fundamental questions:
- **What is happening?** System behavior in real-time
- **Why is it happening?** Root cause of issues
- **Where is it happening?** Which components are affected
- **When did it happen?** Timeline of events
- **Who is affected?** Impact scope

### Observability vs Monitoring

**Monitoring**: Answers questions you know to ask
- Pre-defined dashboards
- Known metrics and alerts
- "Is the system up?"
- "Is CPU > 80%?"

**Observability**: Answers questions you didn't know you'd need to ask
- Exploratory investigation
- Ad-hoc queries
- "Why is user X experiencing latency?"
- "What changed in the last hour for feature Y?"

**Key Difference**: Monitoring tells you *that* something is wrong. Observability helps you understand *why* it's wrong.

---

## The Three Pillars of Observability

### 1. **Logs**
**Discrete events that happened at a specific point in time**

**Characteristics**:
- Timestamp + message + context
- Immutable records
- High cardinality (unique events)
- Often unstructured or semi-structured

**Examples**:
\`\`\`
2024-01-15 10:23:45 INFO User 12345 logged in from IP 192.168.1.1
2024-01-15 10:23:46 ERROR Database connection failed: timeout after 5s
2024-01-15 10:23:47 WARN Retry attempt 1 of 3
\`\`\`

**When to Use Logs**:
- Debugging specific events
- Audit trails
- Error investigation
- Understanding request flow

**Challenges**:
- Volume (can be massive)
- Cost (storage + processing)
- Finding signal in noise

### 2. **Metrics**
**Numerical measurements aggregated over time**

**Characteristics**:
- Time-series data
- Aggregated (sum, average, percentile)
- Low cardinality (dimensions)
- Efficient storage

**Examples**:
\`\`\`
http_requests_total{method="GET", status="200"} 15420
api_latency_seconds{endpoint="/users", percentile="p99"} 0.245
database_connections_active 47
\`\`\`

**Common Metric Types**:
- **Counter**: Monotonically increasing (total requests)
- **Gauge**: Point-in-time value (CPU usage, active connections)
- **Histogram**: Distribution of values (latencies)
- **Summary**: Similar to histogram, pre-calculated quantiles

**When to Use Metrics**:
- Real-time dashboards
- Alerting
- Trend analysis
- System health monitoring

**Advantages**:
- Compact (much smaller than logs)
- Fast queries
- Easy to visualize

### 3. **Traces**
**The journey of a request through a distributed system**

**Characteristics**:
- Spans connected in parent-child relationships
- Contains timing information
- Shows service dependencies
- Includes metadata (tags)

**Example Structure**:
\`\`\`
Trace ID: abc-123
├── Span: API Gateway (120ms)
│   ├── Span: Auth Service (20ms)
│   ├── Span: User Service (80ms)
│   │   └── Span: Database Query (60ms)
│   └── Span: Payment Service (15ms)
\`\`\`

**When to Use Traces**:
- Understanding request flow
- Finding performance bottlenecks
- Debugging distributed systems
- Analyzing service dependencies

**Benefits**:
- Visualize request path
- Identify slow operations
- Understand system topology

---

## Beyond the Three Pillars

### 4. **Profiles**
**Continuous profiling of code execution**

- CPU usage by function
- Memory allocation
- Flame graphs
- Tools: pprof, Pyroscope, Datadog Continuous Profiler

### 5. **Events**
**Significant occurrences in the system**

- Deployments
- Configuration changes
- Scaling events
- Incidents

---

## Key Observability Concepts

### **Cardinality**
The number of unique combinations of dimension values

**Low Cardinality** (good for metrics):
- Status codes: {200, 404, 500}
- HTTP methods: {GET, POST, PUT, DELETE}
- Regions: {us-east-1, eu-west-1}

**High Cardinality** (bad for metrics):
- User IDs (millions)
- Session IDs (billions)
- IP addresses

**Why it Matters**: High cardinality metrics explode storage costs and query performance

### **Sampling**
Only collecting a subset of data

**When to Sample**:
- **Traces**: Sample 1% of requests (Trace 1 in 100)
- **Logs**: Sample debug logs in production
- **Metrics**: Usually NO sampling (aggregated already)

**Trade-off**: Cost savings vs completeness

### **Correlation**
Connecting related telemetry across pillars

**Example**:
- Log has trace ID → Find related trace
- Trace has host ID → Find related metrics
- Metric spike → Query logs for errors

**Implementation**: Use correlation IDs consistently

---

## Telemetry Data Collection

### **Push vs Pull**

**Push Model** (application sends data):
- Application pushes to collector
- Examples: Logs to Loki, Traces to Jaeger
- Pros: Real-time, works behind firewalls
- Cons: Application overhead, need buffering

**Pull Model** (collector scrapes data):
- Collector pulls from application
- Examples: Prometheus scraping metrics
- Pros: Central control, backpressure handling
- Cons: Requires open ports, discovery needed

### **Instrumentation**

**Automatic Instrumentation**:
- Agent-based (OpenTelemetry agents)
- Framework integration
- Pros: Quick setup, no code changes
- Cons: Less control, generic spans

**Manual Instrumentation**:
- Explicit code additions
- Custom spans and metrics
- Pros: Full control, business-specific
- Cons: More effort, maintenance burden

---

## OpenTelemetry (OTel)

**The industry standard for observability**

### **What is OpenTelemetry?**
- Unified standard for telemetry collection
- Vendor-neutral (works with any backend)
- Covers logs, metrics, traces
- Auto-instrumentation for many languages

### **Components**:
1. **API**: Instrument code
2. **SDK**: Process and export telemetry
3. **Collector**: Receive, process, export data
4. **Instrumentation**: Pre-built for frameworks

### **Architecture**:
\`\`\`
Application (instrumented with OTel)
    ↓
OTel Collector (optional)
    ↓
Backend (Jaeger, Prometheus, Datadog, etc.)
\`\`\`

### **Benefits**:
- Avoid vendor lock-in
- Consistent instrumentation
- Rich ecosystem
- Production-ready

---

## Observability-Driven Development

### **Principles**1. **Instrument from Day 1**
   - Don't wait for production issues
   - Observability is not "nice to have"

2. **Make it Easy**
   - Auto-instrumentation where possible
   - Consistent patterns
   - Shared libraries

3. **Think About the Reader**
   - Logs should tell a story
   - Metrics should have clear names
   - Traces should show business flow

4. **Balance Cost and Value**
   - Not everything needs to be logged
   - Sample intelligently
   - Aggregate when possible

### **What to Instrument**

**Critical Paths**:
- User authentication
- Payment processing
- Data mutations
- External API calls

**Don't Over-Instrument**:
- Avoid logging in tight loops
- Skip obvious success cases
- Use sampling for high-volume

---

## Observability Data Flow

\`\`\`
┌──────────────┐
│ Application  │ Generate telemetry
└──────┬───────┘
       │
       ↓
┌──────────────┐
│   Collect    │ Buffer, batch, enrich
└──────┬───────┘
       │
       ↓
┌──────────────┐
│   Process    │ Filter, aggregate, sample
└──────┬───────┘
       │
       ↓
┌──────────────┐
│    Store     │ Time-series DB, object storage
└──────┬───────┘
       │
       ↓
┌──────────────┐
│  Analyze     │ Query, visualize, alert
└──────────────┘
\`\`\`

---

## Common Observability Patterns

### **Correlation IDs**
Unique identifier propagated through the system

**Implementation**:
\`\`\`
Request arrives → Generate correlation_id → Pass in headers → Log in every service
\`\`\`

**Benefits**:
- Connect logs across services
- Trace user journey
- Debug distributed issues

### **Structured Logging**
Logs in machine-readable format (JSON)

**Instead of**:
\`\`\`
User john logged in from 192.168.1.1 at 2024-01-15 10:23:45
\`\`\`

**Use**:
\`\`\`json
{
  "timestamp": "2024-01-15T10:23:45Z",
  "level": "INFO",
  "event": "user_login",
  "user_id": "john",
  "ip": "192.168.1.1",
  "trace_id": "abc-123"
}
\`\`\`

**Benefits**: Easy to query, filter, aggregate

### **Semantic Conventions**
Standardized naming for common operations

OpenTelemetry defines conventions:
- \`http.method\`, \`http.status_code\`
- \`db.system\`, \`db.operation\`
- \`messaging.system\`, \`messaging.destination\`

**Why**: Cross-service consistency, tooling compatibility

---

## Observability Challenges in Distributed Systems

### **1. Data Volume**
- Millions of logs per second
- Thousands of metrics per service
- Complex traces

**Solutions**:
- Sampling
- Aggregation
- Tiered storage (hot/warm/cold)

### **2. Context Propagation**
- Correlation IDs must flow everywhere
- Async operations break chains

**Solutions**:
- Use OpenTelemetry context propagation
- Explicit parent-child span relationships

### **3. Cardinality Explosion**
- Unique values in dimensions
- Costs skyrocket

**Solutions**:
- Limit dimensions
- Use logs for high-cardinality data
- Cardinality limits in exporters

### **4. Tool Sprawl**
- Different tools for different pillars
- Context switching

**Solutions**:
- Unified platforms (Datadog, New Relic)
- OpenTelemetry for consistent instrumentation

---

## Best Practices

### **Do's**
✅ Use structured logging
✅ Propagate context (trace IDs)
✅ Instrument critical paths first
✅ Set up alerts on SLIs
✅ Use semantic conventions
✅ Sample traces intelligently
✅ Monitor the monitors (alert fatigue)

### **Don'ts**
❌ Log sensitive data (passwords, PII)
❌ Create high-cardinality metrics
❌ Ignore sampling (traces and logs)
❌ Mix metrics and logs use cases
❌ Forget about cost
❌ Over-instrument

---

## Interview Tips

### **When Discussing Observability**1. **Start with the Three Pillars**
   - Show you understand fundamentals
   - Explain when to use each

2. **Talk About Trade-offs**
   - Sampling reduces cost but loses data
   - Structured logs cost more but enable queries
   - Real-time vs batch processing

3. **Mention OpenTelemetry**
   - Industry standard
   - Vendor-neutral
   - Shows you're up-to-date

4. **Connect to Reliability**
   - Observability enables SLOs
   - Faster MTTR (Mean Time To Recovery)
   - Proactive issue detection

5. **Consider Cost**
   - Observability can be 10-20% of infrastructure cost
   - Sampling strategies
   - Data retention policies

### **Common Interview Questions**

- "How would you debug a latency issue in a distributed system?"
  → Start with traces to identify slow span, then logs for details, metrics for patterns

- "What\'s the difference between logs and metrics?"
  → Logs are discrete events (high cardinality), metrics are aggregated (low cardinality)

- "How do you instrument a new service?"
  → OpenTelemetry auto-instrumentation + custom spans for business logic

---

## Real-World Examples

### **Netflix**
- **Observability at Scale**: Processes trillions of events per day
- **Mantis**: Real-time stream processing for observability
- **Atlas**: Custom metrics system
- **Lessons**: Build for scale from day 1, stream processing enables real-time

### **Uber**
- **Jaeger**: Created by Uber for distributed tracing
- **M3**: Time-series metrics platform
- **Lessons**: Open source observability tools, horizontal scalability

### **Google**
- **Dapper**: Pioneered distributed tracing (2010 paper)
- **Monarch**: Planet-scale monitoring
- **Lessons**: Sampling essential at scale, correlation IDs critical

---

## Summary

Observability is the foundation of reliable distributed systems. The three pillars—logs, metrics, and traces—work together to provide comprehensive visibility. Modern observability requires:

1. **Instrumentation**: Using standards like OpenTelemetry
2. **Collection**: Efficient pipelines that don't overwhelm systems
3. **Storage**: Scalable backends for massive data volumes
4. **Analysis**: Tools that enable fast investigation

As systems grow more complex, observability becomes more critical. Start simple, instrument early, and iterate based on real needs. Remember: you can't improve what you can't measure, and you can't fix what you can't see.`,
};
