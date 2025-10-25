/**
 * Distributed Tracing Section
 */

export const distributedTracingSection = {
  id: 'distributed-tracing',
  title: 'Distributed Tracing',
  content: `Distributed tracing is essential for understanding how requests flow through microservices architectures. While logs tell you what happened in a single service and metrics tell you system health, traces show you the complete journey of a request across all services, helping you identify bottlenecks and debug complex issues.

## What is Distributed Tracing?

**Distributed Tracing** tracks a single request as it travels through multiple services, capturing timing information and metadata at each step.

**The Problem It Solves**:
In a monolith:
\`\`\`
User Request → Application → Database → Response
(Simple to debug: one log file, one call stack)
\`\`\`

In microservices:
\`\`\`
User Request → API Gateway → Auth Service → User Service
                  ↓              ↓              ↓
              Rate Limiter   Redis Cache    Database
                  ↓              ↓              ↓
              Logging      Payment Service   Cache
\`\`\`
(Complex: Which service is slow? Where did the error originate?)

**Tracing Answers**:
- Where is the bottleneck?
- Which service failed?
- How long did each operation take?
- What\'s the service dependency graph?

---

## Core Concepts

### **Trace**
The complete journey of a request through the system

**Characteristics**:
- Unique Trace ID (e.g., \`abc- 123 - def - 456\`)
- Multiple spans
- Start and end timestamps
- Overall duration

**Example**: User loads their profile
\`\`\`
Trace ID: abc-123
Duration: 450ms
Services: API Gateway, Auth, User Service, Database
Status: Success
\`\`\`

### **Span**
A single operation within a trace

**Contains**:
- Span ID (unique)
- Parent Span ID (forms tree)
- Operation name
- Start time
- Duration
- Status (success/error)
- Tags (metadata)
- Logs (events within span)

**Example Span**:
\`\`\`json
{
  "span_id": "span-001",
  "parent_span_id": null,
  "trace_id": "abc-123",
  "operation": "GET /api/users/123",
  "start_time": "2024-01-15T10:23:45.000Z",
  "duration_ms": 450,
  "status": "ok",
  "tags": {
    "http.method": "GET",
    "http.url": "/api/users/123",
    "http.status_code": 200,
    "service": "api-gateway"
  }
}
\`\`\`

### **Span Relationships**

**Parent-Child** (most common):
\`\`\`
Span A (parent)
  ├── Span B (child)
  └── Span C (child)
\`\`\`

**Example**:
\`\`\`
GET /api/users/123 (API Gateway) - 450ms
  ├── Authenticate User (Auth Service) - 50ms
  ├── Fetch User Profile (User Service) - 350ms
  │   └── Database Query (DB) - 300ms
  └── Log Request (Async) - 10ms
\`\`\`

**Follows-From** (asynchronous):
For async operations where B happens after A completes

---

## Trace Visualization

### **Waterfall View** (Gantt Chart)

\`\`\`
Time →  0ms      100ms    200ms    300ms    400ms    500ms
────────────────────────────────────────────────────────────
API Gateway   [════════════════════════════════════════]
  └─Auth      [════]
  └─User Svc          [════════════════════════════]
      └─DB              [═══════════════════════]
  └─Cache                                      [═]
\`\`\`

**What It Shows**:
- Parallel vs sequential operations
- Where time is spent
- Which operation is blocking

### **Service Dependency Graph**

\`\`\`
    ┌──────────────┐
    │ API Gateway  │
    └──────┬───────┘
           │
     ┌─────┴─────┬───────────┐
     │           │           │
┌────▼────┐ ┌────▼──────┐ ┌─▼──────┐
│  Auth   │ │User Service│ │ Cache  │
└─────────┘ └─────┬──────┘ └────────┘
                  │
            ┌─────▼────┐
            │ Database │
            └──────────┘
\`\`\`

---

## Trace Context Propagation

### **The Challenge**
How does service B know it's part of the same trace as service A?

**Solution**: Propagate trace context via headers

### **W3C Trace Context** (Standard)

**Headers**:
1. **traceparent**: Contains trace ID, span ID, flags
2. **tracestate**: Vendor-specific data

**Format**:
\`\`\`
traceparent: 00-abc123def456-span001-01
             │  │            │        │
             │  │            │        └─ Flags (sampled: 01)
             │  │            └────────── Parent Span ID
             │  └─────────────────────── Trace ID
             └────────────────────────── Version
\`\`\`

### **Implementation**

**Service A** (creates trace):
\`\`\`javascript
const traceId = generateId();
const spanId = generateId();

// Make request to Service B
fetch('http://service-b/endpoint', {
  headers: {
    'traceparent': \`00-\${traceId}-\${spanId}-01\`
  }
});
\`\`\`

**Service B** (continues trace):
\`\`\`javascript
// Extract from headers
const traceparent = req.headers['traceparent'];
const [version, traceId, parentSpanId, flags] = traceparent.split('-');

// Create new span in same trace
const newSpanId = generateId();
startSpan({
  traceId: traceId,
  spanId: newSpanId,
  parentSpanId: parentSpanId
});
\`\`\`

### **Context Propagation in Different Protocols**

**HTTP**: Headers
\`\`\`
GET /users HTTP/1.1
traceparent: 00-abc123-span001-01
\`\`\`

**gRPC**: Metadata
\`\`\`
metadata: {
  'traceparent': '00-abc123-span001-01'
}
\`\`\`

**Message Queues**: Message headers
\`\`\`
{
  headers: {
    'traceparent': '00-abc123-span001-01'
  },
  body: { ... }
}
\`\`\`

---

## Sampling

### **Why Sample?**

**Problem**: Tracing every request is expensive
- Storage: 100K requests/second = massive data
- Processing: CPU overhead
- Network: Bandwidth to send traces
- Cost: Storage and analysis

**Solution**: Sample a percentage of traces

### **Sampling Strategies**

**1. Head-Based Sampling** (决定 at start)
Decision made when trace starts

**Probabilistic** (most common):
- Sample 1% of all requests
- Random selection
\`\`\`
if (random() < 0.01) {
  startTrace();
}
\`\`\`

**Rate Limiting**:
- Max 100 traces/second
- Prevent overwhelming system

**2. Tail-Based Sampling** (decide at end)
Decision made after trace completes

**Strategy**:
- Buffer complete trace
- Sample based on outcome:
  - ✅ Keep all errors (100%)
  - ✅ Keep slow requests (latency > 1s) (100%)
  - ✅ Sample successful fast requests (1%)

**Benefits**: Keep interesting traces
**Drawback**: Requires buffering, delayed decision

**3. Adaptive Sampling**
Adjust sample rate dynamically

**Example**:
- Normal: 1% sampling
- Error spike detected: 100% for 10 minutes
- Back to normal: 1%

### **Sampling in Practice**

**Production Systems**:
- High-volume endpoints: 0.1-1%
- Critical endpoints: 10-100%
- Errors: Always 100%

**Trade-off**: Completeness vs cost

---

## Tracing Tools

### **Jaeger** (Uber)

**Architecture**:
\`\`\`
Application → Jaeger Agent → Jaeger Collector → Storage → Jaeger UI
\`\`\`

**Features**:
- Distributed context propagation
- Distributed transaction monitoring
- Root cause analysis
- Service dependency analysis
- Performance optimization

**Storage Options**:
- Cassandra (recommended for production)
- Elasticsearch
- In-memory (development)

**Pros**:
- Open source
- Battle-tested at Uber
- Rich UI

**Cons**:
- Self-hosted complexity
- Need separate storage

### **Zipkin** (Twitter)

**Similar to Jaeger**, earlier project

**Features**:
- Span collection
- Storage (Cassandra, MySQL, Elasticsearch)
- Query service
- Web UI

**Pros**:
- Mature
- Wide language support
- Simple to get started

### **AWS X-Ray**

**Managed tracing service**

**Features**:
- Automatic integration with AWS services
- Service map visualization
- Trace analysis

**Pros**:
- No infrastructure to manage
- Deep AWS integration

**Cons**:
- AWS-only
- Cost

### **Google Cloud Trace**

**GCP managed tracing**

**Features**:
- Automatic tracing for App Engine
- Integration with GKE
- Latency analysis

### **Datadog APM**

**Commercial APM with tracing**

**Features**:
- Distributed tracing
- Service map
- Integration with metrics and logs
- AI-powered insights

**Pros**:
- Unified observability platform
- Great UX

**Cons**:
- Expensive

---

## OpenTelemetry Tracing

**OpenTelemetry** is the industry standard for instrumentation

### **Benefits**:
- Vendor-neutral
- Auto-instrumentation
- Consistent API across languages
- Export to any backend

### **Components**:

**1. Tracer**
Creates spans
\`\`\`javascript
const tracer = trace.getTracer('my-service');
\`\`\`

**2. Span**
\`\`\`javascript
const span = tracer.startSpan('database-query');
span.setAttribute('db.system', 'postgresql');
span.setAttribute('db.query', 'SELECT * FROM users');

try {
  const result = await db.query('SELECT * FROM users');
  span.setStatus({ code: SpanStatusCode.OK });
} catch (error) {
  span.recordException (error);
  span.setStatus({ code: SpanStatusCode.ERROR });
} finally {
  span.end();
}
\`\`\`

**3. Context Propagation**
Automatic across HTTP, gRPC, message queues

### **Auto-Instrumentation**

Many frameworks auto-instrumented:
- Express.js, Fastify (Node.js)
- Flask, Django (Python)
- Spring Boot (Java)
- ASP.NET Core (C#)

**Example** (Node.js):
\`\`\`javascript
const { NodeTracerProvider } = require('@opentelemetry/node');
const { registerInstrumentations } = require('@opentelemetry/instrumentation');
const { HttpInstrumentation } = require('@opentelemetry/instrumentation-http');

// Automatic tracing of HTTP requests
registerInstrumentations({
  instrumentations: [
    new HttpInstrumentation(),
  ],
});
\`\`\`

---

## Span Attributes and Events

### **Attributes** (Metadata)

**Semantic Conventions** (standardized):
\`\`\`javascript
span.setAttribute('http.method', 'GET');
span.setAttribute('http.url', '/api/users/123');
span.setAttribute('http.status_code', 200);
span.setAttribute('db.system', 'postgresql');
span.setAttribute('messaging.system', 'kafka');
\`\`\`

**Custom Attributes**:
\`\`\`javascript
span.setAttribute('user_id', '123');
span.setAttribute('order_id', 'order-456');
span.setAttribute('feature_flag', 'new_checkout');
\`\`\`

### **Events** (Logs within span)

\`\`\`javascript
span.addEvent('cache_miss', {
  key: 'user:123'
});

span.addEvent('retry_attempt', {
  attempt: 2,
  max_attempts: 3
});
\`\`\`

**Use Cases**:
- Important milestones
- Errors and exceptions
- Cache hits/misses
- Retry attempts

---

## Trace Analysis

### **Identifying Bottlenecks**

**1. Longest Span**
Find the span with highest duration
\`\`\`
Database Query: 800ms ← bottleneck!
\`\`\`

**2. Sequential vs Parallel**
Look for operations that could be parallelized
\`\`\`
❌ Sequential (500ms total):
  └─ Fetch User (200ms)
  └─ Fetch Orders (300ms)

✅ Parallel (300ms total):
  ├─ Fetch User (200ms)
  └─ Fetch Orders (300ms)
\`\`\`

**3. Excessive Calls**
N+1 query problem
\`\`\`
❌ Bad:
  Fetch Posts (10ms)
  └─ For each post:
      Fetch Author (5ms) × 20 posts = 100ms

✅ Good:
  Fetch Posts (10ms)
  Fetch All Authors (15ms)
\`\`\`

### **Error Investigation**

**Find Error Span**:
\`\`\`
Trace abc-123 (ERROR)
  API Gateway (OK)
    Auth Service (OK)
    User Service (ERROR) ← error originated here!
      Database Query (ERROR: timeout)
\`\`\`

**Error Details in Span**:
\`\`\`json
{
  "span_id": "span-123",
  "status": "error",
  "error": {
    "type": "DatabaseTimeoutError",
    "message": "Query timeout after 5s",
    "stack_trace": "..."
  }
}
\`\`\`

---

## Correlation with Logs and Metrics

### **Trace ID in Logs**

Include trace_id in all log entries:
\`\`\`json
{
  "timestamp": "2024-01-15T10:23:45Z",
  "level": "ERROR",
  "message": "Database query failed",
  "trace_id": "abc-123",
  "span_id": "span-456",
  "service": "user-service"
}
\`\`\`

**Benefit**: Click on trace → see all related logs

### **Exemplars in Metrics**

Link metrics to example traces:
\`\`\`
http_request_duration_seconds_bucket{le="1.0"} 100
  # Exemplar: trace_id=abc-123, timestamp=...
\`\`\`

**Benefit**: Click on metric spike → see example slow trace

---

## Best Practices

### **Do's**
✅ Use OpenTelemetry for vendor neutrality
✅ Propagate context across all services
✅ Add meaningful span attributes
✅ Sample aggressively in production
✅ Always trace errors
✅ Use semantic conventions
✅ Correlate with logs via trace_id

### **Don'ts**
❌ Trace every request without sampling
❌ Add sensitive data to spans
❌ Create too many spans (overhead)
❌ Forget to end spans
❌ Block on trace export
❌ Ignore trace context

---

## Performance Impact

**Overhead**:
- Creating spans: ~1-5μs per span
- Adding attributes: ~100ns per attribute
- Context propagation: ~1μs
- Network export: async, non-blocking

**Total Impact**: < 1% CPU overhead with sampling

**Best Practice**: Export spans asynchronously, batch multiple spans

---

## Interview Tips

### **Key Concepts**

1. **Trace vs Span**: Trace is full request, span is single operation
2. **Context Propagation**: Headers carry trace_id and span_id
3. **Sampling**: Don't trace everything, sample 1-10%
4. **Use Cases**: Bottleneck identification, error debugging
5. **Tools**: Jaeger, Zipkin, OpenTelemetry

### **Common Questions**

**Q: How would you debug a slow API request in microservices?**
A: Look at distributed trace, identify longest span, check if operations are sequential that could be parallel, look for N+1 queries.

**Q: What\'s the difference between logs and traces?**
A: Logs are discrete events in a single service. Traces show request flow across multiple services with timing.

**Q: How does a downstream service know it's part of a trace?**
A: Upstream service passes trace_id and span_id in headers (W3C traceparent).

---

## Real-World Examples

### **Google (Dapper)**
- **2010 Paper**: Pioneered distributed tracing
- **Sampling**: < 0.01% of requests
- **Overhead**: < 0.01% latency impact
- **Lesson**: Sampling essential at massive scale

### **Uber (Jaeger)**
- **Open Sourced**: 2017
- **Scale**: Processes billions of spans/day
- **Sampling**: Adaptive based on traffic
- **Lesson**: Built for scale from day 1

### **Netflix**
- **Scale**: Traces 100+ services
- **Challenge**: Complex service dependencies
- **Solution**: Mandatory tracing for all services
- **Lesson**: Tracing is not optional in microservices

---

## Summary

Distributed tracing is essential for microservices:

1. **Traces & Spans**: Traces track full request, spans track individual operations
2. **Context Propagation**: Headers carry trace context between services
3. **Sampling**: Sample 1-10% to control costs
4. **Tools**: Jaeger, Zipkin, AWS X-Ray, OpenTelemetry
5. **Use Cases**: Bottleneck identification, error debugging, service dependencies
6. **Best Practice**: Auto-instrument with OpenTelemetry, correlate with logs

Without tracing, debugging microservices is nearly impossible. Invest in tracing infrastructure early.`,
};
