/**
 * API Monitoring & Analytics Section
 */

export const apimonitoringSection = {
  id: 'api-monitoring',
  title: 'API Monitoring & Analytics',
  content: `API monitoring is essential for maintaining reliability, performance, and understanding usage patterns. Comprehensive monitoring enables proactive issue detection and data-driven decisions.

## Why Monitor APIs?

### **Benefits**

1. **Detect issues early**: Before users complain
2. **Performance tracking**: Identify slow endpoints
3. **Usage analytics**: Understand how APIs are used
4. **Capacity planning**: Predict resource needs
5. **Security**: Detect unusual patterns
6. **SLA compliance**: Meet uptime guarantees

## Key Metrics to Track

### **1. Golden Signals (Google SRE)**

#### **Latency**

Response time distribution:

\`\`\`javascript
const prometheus = require('prom-client');

const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
});

app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    httpRequestDuration
      .labels(req.method, req.route?.path || 'unknown', res.statusCode)
      .observe(duration);
  });
  
  next();
});
\`\`\`

**Track**:
- p50 (median)
- p95 (95th percentile)
- p99 (99th percentile)
- p99.9 (tail latency)

#### **Traffic**

Request rate:

\`\`\`javascript
const httpRequestsTotal = new prometheus.Counter({
  name: 'http_requests_total',
  help: 'Total HTTP requests',
  labelNames: ['method', 'route', 'status_code']
});

app.use((req, res, next) => {
  res.on('finish', () => {
    httpRequestsTotal
      .labels(req.method, req.route?.path || 'unknown', res.statusCode)
      .inc();
  });
  
  next();
});
\`\`\`

#### **Errors**

Error rate:

\`\`\`javascript
const httpErrorsTotal = new prometheus.Counter({
  name: 'http_errors_total',
  help: 'Total HTTP errors',
  labelNames: ['method', 'route', 'status_code', 'error_type']
});

app.use((err, req, res, next) => {
  httpErrorsTotal
    .labels(
      req.method,
      req.route?.path || 'unknown',
      res.statusCode || 500,
      err.name || 'UnknownError'
    )
    .inc();
  
  next(err);
});
\`\`\`

**Track**:
- 4xx rate (client errors)
- 5xx rate (server errors)
- Error types distribution

#### **Saturation**

Resource utilization:

\`\`\`javascript
const resourceUsage = new prometheus.Gauge({
  name: 'resource_usage_percent',
  help: 'Resource usage percentage',
  labelNames: ['resource_type']
});

// CPU usage
setInterval(() => {
  const cpuUsage = process.cpuUsage();
  resourceUsage.labels('cpu').set(cpuUsage.user / 1000000);
}, 5000);

// Memory usage
setInterval(() => {
  const memUsage = process.memoryUsage();
  const memPercent = (memUsage.heapUsed / memUsage.heapTotal) * 100;
  resourceUsage.labels('memory').set(memPercent);
}, 5000);
\`\`\`

### **2. Business Metrics**

Track API usage patterns:

\`\`\`javascript
const apiCallsByUser = new prometheus.Counter({
  name: 'api_calls_by_user_total',
  help: 'API calls per user',
  labelNames: ['user_id', 'user_plan', 'endpoint']
});

const apiCostByEndpoint = new prometheus.Counter({
  name: 'api_cost_by_endpoint_total',
  help: 'API cost by endpoint',
  labelNames: ['endpoint', 'cost']
});

app.use((req, res, next) => {
  res.on('finish', () => {
    apiCallsByUser
      .labels(req.user?.id, req.user?.plan, req.route?.path)
      .inc();
    
    const cost = OPERATION_COSTS[req.route?.path] || 1;
    apiCostByEndpoint
      .labels(req.route?.path, cost)
      .inc(cost);
  });
  
  next();
});
\`\`\`

## Distributed Tracing

Track requests across services:

### **OpenTelemetry**

\`\`\`javascript
const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');
const { registerInstrumentations } = require('@opentelemetry/instrumentation');
const { HttpInstrumentation } = require('@opentelemetry/instrumentation-http');
const { ExpressInstrumentation } = require('@opentelemetry/instrumentation-express');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');

// Setup tracing
const provider = new NodeTracerProvider();

provider.addSpanProcessor(
  new BatchSpanProcessor(
    new JaegerExporter({
      endpoint: 'http://jaeger:14268/api/traces'
    })
  )
);

provider.register();

registerInstrumentations({
  instrumentations: [
    new HttpInstrumentation(),
    new ExpressInstrumentation()
  ]
});

// Usage automatically traces all HTTP requests
app.get('/users/:id', async (req, res) => {
  // This is automatically traced
  const user = await getUserFromDatabase(req.params.id);
  res.json(user);
});
\`\`\`

**Trace visualization** (Jaeger):
\`\`\`
GET /orders/123
  ├─ GET /users/456 (50ms)
  ├─ GET /products/789 (30ms)
  └─ POST /payments (200ms) ← Slow!
\`\`\`

### **Custom Spans**

\`\`\`javascript
const tracer = require('@opentelemetry/api').trace.getTracer('api-service');

app.get('/dashboard', async (req, res) => {
  const span = tracer.startSpan('dashboard.load');
  
  try {
    // Child span for database query
    const dbSpan = tracer.startSpan('database.query', {
      parent: span
    });
    const data = await fetchDashboardData(req.user.id);
    dbSpan.end();
    
    // Child span for cache
    const cacheSpan = tracer.startSpan('cache.set', {
      parent: span
    });
    await cacheData(data);
    cacheSpan.end();
    
    res.json(data);
  } finally {
    span.end();
  }
});
\`\`\`

## Logging

### **Structured Logging**

\`\`\`javascript
const winston = require('winston');

const logger = winston.createLogger({
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    logger.info({
      method: req.method,
      path: req.path,
      status: res.statusCode,
      duration: Date.now() - start,
      userId: req.user?.id,
      userAgent: req.headers['user-agent'],
      ip: req.ip,
      requestId: req.headers['x-request-id']
    });
  });
  
  next();
});
\`\`\`

### **Log Levels**

\`\`\`javascript
// ERROR: Something failed
logger.error('Payment processing failed', {
  userId: '123',
  orderId: '456',
  error: err.message,
  stack: err.stack
});

// WARN: Potential issue
logger.warn('Rate limit approaching', {
  userId: '123',
  usage: 95,
  limit: 100
});

// INFO: Important events
logger.info('Order created', {
  orderId: '456',
  userId: '123',
  total: 99.99
});

// DEBUG: Detailed debugging
logger.debug('Cache hit', {
  key: 'user:123',
  ttl: 300
});
\`\`\`

### **Request ID Tracking**

\`\`\`javascript
const { v4: uuidv4 } = require('uuid');

app.use((req, res, next) => {
  req.id = req.headers['x-request-id'] || uuidv4();
  res.setHeader('X-Request-ID', req.id);
  next();
});

// Use in logs
logger.info('Processing request', {
  requestId: req.id,
  userId: req.user?.id
});

// Propagate to downstream services
const response = await fetch('http://user-service/users/123', {
  headers: {
    'X-Request-ID': req.id
  }
});
\`\`\`

## Real User Monitoring (RUM)

### **Client-Side Performance**

\`\`\`javascript
// Client sends timing data
fetch('/api/analytics/timing', {
  method: 'POST',
  body: JSON.stringify({
    endpoint: '/api/users',
    method: 'GET',
    duration: performance.now() - startTime,
    status: response.status,
    timestamp: Date.now()
  })
});

// Server aggregates
app.post('/api/analytics/timing', (req, res) => {
  const { endpoint, method, duration, status } = req.body;
  
  rumDuration
    .labels(endpoint, method, status)
    .observe(duration / 1000);
  
  res.status(204).send();
});
\`\`\`

## Alerting

### **Prometheus Alertmanager**

\`\`\`yaml
groups:
  - name: api_alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{status_code=~"5.."}[5m]))
            /
            sum(rate(http_requests_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate (> 5%)"
          description: "{{ $value | humanizePercentage }} of requests are failing"
      
      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            rate(http_request_duration_seconds_bucket[5m])
          ) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High latency (p95 > 1s)"
      
      # API availability
      - alert: APIDown
        expr: up{job="api-server"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "API is down"
\`\`\`

### **PagerDuty Integration**

\`\`\`javascript
const PagerDuty = require('node-pagerduty');
const pd = new PagerDuty(process.env.PAGERDUTY_API_KEY);

async function triggerAlert(severity, message, details) {
  await pd.incidents.createIncident({
    type: 'incident',
    title: message,
    service: {
      id: process.env.PAGERDUTY_SERVICE_ID,
      type: 'service_reference'
    },
    urgency: severity === 'critical' ? 'high' : 'low',
    body: {
      type: 'incident_body',
      details: JSON.stringify(details)
    }
  });
}

// Trigger on high error rate
if (errorRate > 0.05) {
  await triggerAlert('critical', 'High error rate detected', {
    errorRate,
    endpoint: '/api/checkout',
    timestamp: Date.now()
  });
}
\`\`\`

## Dashboard (Grafana)

\`\`\`json
{
  "dashboard": {
    "title": "API Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (route)"
          }
        ]
      },
      {
        "title": "Latency (p50, p95, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status_code=~\\"5..\\"}[5m])) / sum(rate(http_requests_total[5m]))"
          }
        ]
      },
      {
        "title": "Top Slowest Endpoints",
        "targets": [
          {
            "expr": "topk(10, avg by (route) (rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])))"
          }
        ]
      }
    ]
  }
}
\`\`\`

## Health Checks

\`\`\`javascript
app.get('/health', async (req, res) => {
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    checks: {}
  };
  
  // Database check
  try {
    await db.raw('SELECT 1');
    health.checks.database = { status: 'healthy' };
  } catch (err) {
    health.status = 'unhealthy';
    health.checks.database = {
      status: 'unhealthy',
      error: err.message
    };
  }
  
  // Redis check
  try {
    await redis.ping();
    health.checks.redis = { status: 'healthy' };
  } catch (err) {
    health.status = 'degraded';
    health.checks.redis = {
      status: 'unhealthy',
      error: err.message
    };
  }
  
  // Downstream service check
  try {
    const response = await fetch('http://user-service/health', {
      timeout: 2000
    });
    health.checks.userService = {
      status: response.ok ? 'healthy' : 'unhealthy'
    };
  } catch (err) {
    health.checks.userService = {
      status: 'unhealthy',
      error: err.message
    };
  }
  
  const statusCode = health.status === 'healthy' ? 200 :
                     health.status === 'degraded' ? 200 : 503;
  
  res.status(statusCode).json(health);
});
\`\`\`

## Best Practices

1. **Track Golden Signals**: Latency, traffic, errors, saturation
2. **Use distributed tracing**: Understand cross-service requests
3. **Structured logging**: JSON logs for easy parsing
4. **Request IDs**: Track requests across services
5. **Health checks**: Automated uptime monitoring
6. **Alerting thresholds**: p95 latency, error rate, uptime
7. **Dashboard per service**: Grafana dashboards for each API
8. **Log aggregation**: Centralize logs (ELK, Splunk)
9. **Real user monitoring**: Track client-side performance
10. **Synthetic monitoring**: Automated API tests from multiple locations`,
};
