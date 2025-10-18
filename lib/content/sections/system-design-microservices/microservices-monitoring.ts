/**
 * Microservices Monitoring Section
 */

export const microservicesmonitoringSection = {
  id: 'microservices-monitoring',
  title: 'Microservices Monitoring',
  content: `Monitoring microservices is fundamentally different from monoliths. With distributed services, you need comprehensive observability across the entire system.

## The Three Pillars of Observability

### 1. Metrics

**What**: Numeric measurements over time (CPU, memory, request rate, latency).

**Tools**: Prometheus, Grafana, Datadog

**RED Metrics** (for every service):
- **Rate**: Requests per second
- **Errors**: Error rate (%)
- **Duration**: Latency (p50, p95, p99)

**Example metrics**:
\`\`\`
http_requests_total{service="order-service", method="POST", status="200"} 1547
http_request_duration_seconds{service="order-service", quantile="0.95"} 0.245
\`\`\`

**Implementation** (Prometheus):
\`\`\`javascript
const promClient = require('prom-client');
const register = new promClient.Registry();

// Counter: total requests
const httpRequestsTotal = new promClient.Counter({
    name: 'http_requests_total',
    help: 'Total HTTP requests',
    labelNames: ['method', 'route', 'status_code'],
    registers: [register]
});

// Histogram: request duration
const httpRequestDuration = new promClient.Histogram({
    name: 'http_request_duration_seconds',
    help: 'HTTP request duration in seconds',
    labelNames: ['method', 'route', 'status_code'],
    buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10],
    registers: [register]
});

// Middleware
app.use((req, res, next) => {
    const end = httpRequestDuration.startTimer();
    
    res.on('finish', () => {
        httpRequestsTotal.inc({
            method: req.method,
            route: req.route?.path || 'unknown',
            status_code: res.statusCode
        });
        
        end({
            method: req.method,
            route: req.route?.path || 'unknown',
            status_code: res.statusCode
        });
    });
    
    next();
});

// Expose metrics endpoint
app.get('/metrics', async (req, res) => {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
});
\`\`\`

**Prometheus scrapes** the /metrics endpoint every 15s.

**Grafana Dashboard**:
\`\`\`
Order Service Dashboard
━━━━━━━━━━━━━━━━━━━━━━━
Request Rate: 245 req/s  📈
Error Rate: 0.5%         ✅
P95 Latency: 245ms       ⚠️
P99 Latency: 1.2s        ❌

[Graph showing request rate over time]
[Graph showing latency percentiles]
\`\`\`

### 2. Logs

**What**: Text records of events (structured or unstructured).

**Tools**: ELK Stack (Elasticsearch, Logstash, Kibana), Loki, Splunk

**Structured logging**:
\`\`\`javascript
const logger = require('pino')();

// Bad: Unstructured
logger.info('User 123 created order 456 for $99.99');

// Good: Structured
logger.info({
    event: 'order_created',
    userId: '123',
    orderId: '456',
    amount: 99.99,
    currency: 'USD',
    timestamp: new Date().toISOString()
});
\`\`\`

**Correlation ID** for tracing across services:
\`\`\`javascript
app.use((req, res, next) => {
    req.correlationId = req.headers['x-correlation-id'] || uuidv4();
    res.setHeader('x-correlation-id', req.correlationId);
    next();
});

// Log with correlation ID
logger.info({
    correlationId: req.correlationId,
    event: 'processing_order',
    orderId: order.id
});

// Pass to downstream services
await axios.post('http://payment-service/charge', payment, {
    headers: {
        'x-correlation-id': req.correlationId
    }
});
\`\`\`

**Centralized logging** (ELK):
\`\`\`
All services → Logstash → Elasticsearch → Kibana

Query: correlationId:"abc-123"
Results:
[Order Service] 10:00:01 - Order created
[Inventory Service] 10:00:02 - Inventory reserved
[Payment Service] 10:00:03 - Payment charged
[Shipping Service] 10:00:04 - Shipment created
\`\`\`

### 3. Distributed Tracing

**What**: Track requests across multiple services.

**Tools**: Jaeger, Zipkin, AWS X-Ray

**Problem**: Request touches 5 services, where's the bottleneck?

**Solution**: Distributed tracing shows entire request flow.

**Implementation** (OpenTelemetry):
\`\`\`javascript
const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');

const provider = new NodeTracerProvider();
provider.addSpanProcessor(new SimpleSpanProcessor(new JaegerExporter({
    endpoint: 'http://jaeger:14268/api/traces'
})));
provider.register();

const tracer = provider.getTracer('order-service');

// Create span
app.post('/orders', async (req, res) => {
    const span = tracer.startSpan('create_order');
    
    try {
        const order = await createOrder(req.body);
        span.setAttributes({
            'order.id': order.id,
            'user.id': req.body.userId
        });
        
        res.json(order);
    } catch (error) {
        span.recordException(error);
        span.setStatus({ code: SpanStatusCode.ERROR });
        throw error;
    } finally {
        span.end();
    }
});
\`\`\`

**Jaeger UI**:
\`\`\`
Trace ID: abc-123-def-456
Duration: 1.2s

Timeline:
├─ API Gateway (50ms)
│  └─ Order Service (1150ms)
│     ├─ Payment Service (800ms) ⬅ SLOW!
│     │  └─ Fraud Check (750ms) ⬅ BOTTLENECK!
│     └─ Inventory Service (150ms)
│        └─ Database Query (120ms)

Root cause: Fraud check is slow
\`\`\`

---

## Alerting

**Don't** alert on everything. Alert on **actionable** symptoms.

### Good Alerts

**Symptom-based** (user-facing):
\`\`\`yaml
# High error rate
- alert: HighErrorRate
  expr: |
    rate(http_requests_total{status_code=~"5.."}[5m]) 
    / 
    rate(http_requests_total[5m]) > 0.05
  for: 5m
  annotations:
    summary: "Error rate > 5% for 5 minutes"
    
# High latency
- alert: HighLatency
  expr: |
    histogram_quantile(0.95, 
      rate(http_request_duration_seconds_bucket[5m])
    ) > 1
  for: 10m
  annotations:
    summary: "P95 latency > 1s for 10 minutes"

# Service down
- alert: ServiceDown
  expr: up{job="order-service"} == 0
  for: 1m
  annotations:
    summary: "Order service is down"
\`\`\`

### Bad Alerts

**Cause-based** (not actionable):
\`\`\`
❌ Alert: CPU > 80%
   Problem: So what? Is it affecting users?
   Better: Alert on high latency or errors
   
❌ Alert: Disk > 90%
   Problem: When will it fill up? 
   Better: Predict when it will reach 100%
   
❌ Alert: A single request failed
   Problem: Too noisy, not actionable
   Better: Alert on error rate > 5% for 5 minutes
\`\`\`

---

## Service-Level Objectives (SLOs)

**Define** acceptable reliability targets.

### SLI (Service Level Indicator)

**Measurement** of service health:
- Availability: % of requests that succeed
- Latency: % of requests below threshold
- Throughput: Requests per second

### SLO (Service Level Objective)

**Target** for SLI:
- 99.9% of requests succeed (availability)
- 95% of requests complete < 500ms (latency)

### SLA (Service Level Agreement)

**Contract** with users (with penalties):
- If availability < 99.9%, refund 10% of monthly fee

**Example SLO**:
\`\`\`yaml
Service: order-service
SLO:
  - availability: 99.9% (43 minutes downtime/month)
  - latency_p95: < 500ms
  - latency_p99: < 1000ms
  
Error Budget:
  - 0.1% of requests can fail (99.9% SLO)
  - At 1M requests/month: 1,000 errors allowed
  - Current: 500 errors used (50% of budget)
  - Remaining: 500 errors
\`\`\`

**Error Budget Policy**:
- Budget remaining: Keep shipping features
- Budget exhausted: Focus on reliability

---

## Dashboards

### Service Dashboard

**For each service**:
\`\`\`
Order Service Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔷 Health: Healthy ✅

📊 RED Metrics (last 1 hour)
├─ Request Rate: 245 req/s
├─ Error Rate: 0.5%
└─ Latency: P50=120ms, P95=250ms, P99=500ms

💾 Resources
├─ CPU: 45% (3 pods)
├─ Memory: 60% (768MB / 1.5GB)
└─ Disk: 30%

🔗 Dependencies
├─ Payment Service: Healthy ✅
├─ Inventory Service: Healthy ✅
└─ Database: Healthy ✅

⚠️ Recent Alerts
└─ None

📈 [Request Rate Graph]
📈 [Latency Distribution Graph]
📈 [Error Rate Graph]
\`\`\`

### System Dashboard

**Overall health**:
\`\`\`
System Overview
━━━━━━━━━━━━━━━━━━━━━━━━━
🌍 Total Request Rate: 2,500 req/s
📊 Overall Error Rate: 0.3%
⚡ P95 Latency: 300ms

Services (10 total):
✅ 9 Healthy
⚠️ 1 Degraded (Recommendation Service)

[Service Map showing interconnections]
[Request flow waterfall]
\`\`\`

---

## Anomaly Detection

**Automatically** detect unusual patterns.

**Example**: Request rate normally 100 req/s ± 10%

Sudden spike to 1000 req/s → Alert!
Could be: Attack, viral content, or legitimate traffic spike

**Tools**: Datadog anomaly detection, Prometheus predict_linear

\`\`\`yaml
- alert: AnomalousRequestRate
  expr: |
    abs(rate(http_requests_total[5m]) - avg_over_time(rate(http_requests_total[5m])[1h:5m]))
    /
    stddev_over_time(rate(http_requests_total[5m])[1h:5m])
    > 3
  annotations:
    summary: "Request rate is 3 standard deviations from normal"
\`\`\`

---

## Health Checks

**Different levels** of health:

### Liveness

**Is service alive?**
\`\`\`javascript
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});
\`\`\`

Kubernetes restarts if fails.

### Readiness

**Is service ready for traffic?**
\`\`\`javascript
app.get('/ready', async (req, res) => {
    try {
        // Check dependencies
        await database.ping();
        await paymentServiceClient.healthCheck();
        
        res.json({ status: 'ready' });
    } catch (error) {
        res.status(503).json({ 
            status: 'not ready',
            error: error.message
        });
    }
});
\`\`\`

Kubernetes removes from load balancer if fails.

### Deep Health

**Detailed health of all components**:
\`\`\`javascript
app.get('/health/deep', async (req, res) => {
    const health = {
        status: 'healthy',
        checks: {
            database: await checkDatabase(),
            redis: await checkRedis(),
            paymentService: await checkPaymentService(),
            inventoryService: await checkInventoryService()
        }
    };
    
    const hasFailure = Object.values(health.checks)
        .some(check => check.status === 'unhealthy');
    
    if (hasFailure) {
        health.status = 'degraded';
    }
    
    res.json(health);
});

async function checkDatabase() {
    try {
        await database.ping();
        return { status: 'healthy', latency: 5 };
    } catch (error) {
        return { status: 'unhealthy', error: error.message };
    }
}
\`\`\`

---

## Cost Monitoring

**Track infrastructure costs** per service.

**Example**:
\`\`\`
Monthly Costs by Service:
├─ Order Service: $500
│  ├─ Compute: $300 (3 pods @ $100/pod)
│  ├─ Database: $150
│  └─ Networking: $50
├─ Payment Service: $800
│  ├─ Compute: $400
│  ├─ Database: $300
│  └─ External API: $100
└─ Inventory Service: $400

Total: $1,700/month

Cost per Request: $0.0005
\`\`\`

**Optimize**: 
- Autoscaling (reduce idle pods)
- Right-sizing (don't over-provision)
- Reserved instances (save 30-50%)

---

## Monitoring Best Practices

1. **Monitor symptoms, not causes** (alert on errors, not CPU)
2. **Set SLOs** and track error budgets
3. **Use correlation IDs** to trace requests across services
4. **Structured logging** for easy querying
5. **Dashboard per service** + system overview
6. **Alert only on actionable issues** (no noise)
7. **Distributed tracing** for debugging
8. **Monitor costs** per service
9. **Health checks** at multiple levels (liveness, readiness)
10. **Anomaly detection** for unusual patterns

---

## Interview Tips

**Red Flags**:
❌ Only monitoring server metrics (CPU, memory)
❌ No distributed tracing
❌ Alerts on everything (alert fatigue)

**Good Responses**:
✅ Explain three pillars: metrics, logs, traces
✅ Mention RED metrics (Rate, Errors, Duration)
✅ Discuss SLOs and error budgets
✅ Talk about correlation IDs
✅ Mention specific tools (Prometheus, Jaeger, ELK)

**Sample Answer**:
*"For microservices monitoring, I'd implement the three pillars of observability: (1) Metrics - Prometheus scraping RED metrics (Rate, Errors, Duration) from each service with Grafana dashboards, (2) Logs - Structured logging with correlation IDs sent to ELK stack for centralized querying, (3) Distributed Tracing - Jaeger/OpenTelemetry to track requests across services and identify bottlenecks. I'd define SLOs (99.9% availability, P95 < 500ms) and alert on SLO violations, not symptoms. Each service would have liveness and readiness probes for Kubernetes auto-healing."*

---

## Key Takeaways

1. **Three pillars**: Metrics, Logs, Distributed Tracing
2. **RED metrics**: Rate, Errors, Duration (for every service)
3. **Structured logging** with correlation IDs
4. **Distributed tracing** tracks requests across services
5. **SLOs** define reliability targets, error budgets guide priorities
6. **Alert on symptoms** (errors, latency) not causes (CPU)
7. **Health checks** enable automatic recovery
8. **Dashboards** per service + system overview
9. **Tools**: Prometheus, Grafana, Jaeger, ELK`,
};
