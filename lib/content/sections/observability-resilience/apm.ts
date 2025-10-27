/**
 * Application Performance Monitoring (APM) Section
 */

export const apmSection = {
  id: 'apm',
  title: 'Application Performance Monitoring (APM)',
  content: `Application Performance Monitoring (APM) tools provide comprehensive visibility into application behavior, combining metrics, traces, logs, and code-level insights. While observability's three pillars (logs, metrics, traces) give you raw data, APM platforms synthesize this data into actionable insights for developers and operators.

## What is APM?

**APM** (Application Performance Monitoring) is the practice of monitoring software applications to ensure they meet performance expectations and provide good user experience.

**Core Functions**:
1. **Performance Monitoring**: Track response times, throughput
2. **Error Tracking**: Capture and analyze exceptions
3. **Distributed Tracing**: Follow requests across services
4. **Code-Level Insights**: Identify slow functions
5. **User Experience Monitoring**: Real user metrics
6. **Infrastructure Monitoring**: Server health, resources

**Difference from Basic Observability**:
- **Observability**: Raw telemetry data (logs, metrics, traces)
- **APM**: Synthesized insights with context and intelligence

---

## Why APM Matters

### **Business Impact**

**Slow Application**:
- Lost revenue (\$1M/hour for Amazon)
- Poor user experience
- Customer churn
- Damaged reputation

**Downtime Costs**:
- E-commerce: $10K-$100K per minute
- Financial services: $1M per hour
- Healthcare: Patient safety

**APM Benefits**:
- 10x faster incident resolution
- Proactive issue detection
- Better resource utilization
- Data-driven optimization

### **Developer Impact**

**Without APM**:
- "It\'s slow, figure out why"
- Hours of log searching
- Guessing at root cause
- Can't reproduce in dev

**With APM**:
- "Database query on line 42 is slow"
- Exact slow transactions
- Root cause in minutes
- Production insights

---

## APM Components

### **1. Transaction Monitoring**

**What**: Track end-to-end transactions

**Capabilities**:
- Request flow across services
- Response time breakdown
- Error rate per endpoint
- Throughput (requests/second)

**Example**:
\`\`\`
Transaction: Checkout
Average Duration: 1.2s
├─ API Gateway: 50ms
├─ Inventory Check: 200ms
├─ Payment Processing: 800ms ← bottleneck!
└─ Order Creation: 150ms
\`\`\`

### **2. Error Tracking**

**What**: Capture and analyze application errors

**Capabilities**:
- Stack traces with context
- Error grouping (deduplication)
- Error rate trends
- Affected users
- Resolution tracking

**Example Error**:
\`\`\`json
{
  "error": "NullPointerException",
  "message": "Cannot read property 'id' of null",
  "stack_trace": "...",
  "context": {
    "user_id": "user-123",
    "endpoint": "/api/checkout",
    "browser": "Chrome 120",
    "trace_id": "abc-123"
  },
  "first_seen": "2024-01-15T10:00:00Z",
  "occurrences": 47,
  "affected_users": 23
}
\`\`\`

### **3. Code-Level Profiling**

**What**: Identify slow functions and methods

**Capabilities**:
- CPU profiling
- Memory profiling
- Database query analysis
- External API call timing

**Example**:
\`\`\`
Function: processOrder()
Total Time: 850ms
├─ validateInventory(): 50ms
├─ calculateTax(): 20ms
├─ chargePayment(): 750ms ← bottleneck!
│   └─ stripe.charge(): 720ms
└─ saveOrder(): 30ms
\`\`\`

### **4. Real User Monitoring (RUM)**

**What**: Track actual user experience

**Metrics**:
- Page load time
- Time to First Byte (TTFB)
- First Contentful Paint (FCP)
- Largest Contentful Paint (LCP)
- Cumulative Layout Shift (CLS)
- First Input Delay (FID)

**Browser Performance API**:
\`\`\`javascript
// Automatically captured by APM
performance.timing = {
  navigationStart: 1234567890,
  domContentLoadedEventEnd: 1234568200, // 310ms
  loadEventEnd: 1234569500 // 1.6s total
}
\`\`\`

**Benefits**:
- Understand real user impact
- Geographic performance differences
- Device/browser variations
- User journey tracking

### **5. Synthetic Monitoring**

**What**: Simulated user interactions from bots

**Use Cases**:
- Proactive monitoring (catch issues before users)
- Baseline performance
- Uptime monitoring
- Multi-region checks

**Example**:
\`\`\`
Synthetic Check: Checkout Flow
Frequency: Every 5 minutes
Locations: US-East, EU-West, Asia-Pacific

Results:
  US-East: 1.2s (✓)
  EU-West: 1.5s (✓)
  Asia-Pacific: 3.2s (⚠ Slow)
\`\`\`

### **6. Database Monitoring**

**What**: Track database performance

**Capabilities**:
- Slow query identification
- Query execution plans
- Connection pool usage
- Lock contention
- Transaction volume

**Example Slow Query**:
\`\`\`sql
SELECT * FROM orders
WHERE user_id = 123
ORDER BY created_at DESC
-- Duration: 2.5s
-- Executed: 1000 times/minute
-- Issue: Missing index on user_id
\`\`\`

---

## Popular APM Tools

### **Datadog APM**

**Strengths**:
- Unified platform (metrics, logs, traces)
- Automatic instrumentation
- Service map visualization
- AI-powered insights
- Great UI/UX

**Pricing**: $31-$40/host/month

**Best For**: Teams wanting all-in-one observability

### **New Relic**

**Strengths**:
- Mature product
- Strong error tracking
- Code-level visibility
- Mobile APM
- Extensive integrations

**Pricing**: $25-$99/user/month or usage-based

**Best For**: Established enterprises

### **Dynatrace**

**Strengths**:
- AI/ML-powered root cause analysis (Davis AI)
- Auto-discovery
- Best for complex environments
- Strong infrastructure monitoring

**Pricing**: Custom (expensive)

**Best For**: Large enterprises with complex stacks

### **Elastic APM**

**Strengths**:
- Part of Elastic Stack
- Self-hosted option
- Real-time streaming
- Open source

**Pricing**: Free (self-hosted) or managed Elastic Cloud

**Best For**: Teams already using ELK stack

### **Sentry**

**Strengths**:
- Excellent error tracking
- Great developer experience
- Source map support
- Release tracking
- Affordable

**Pricing**: Free tier, $26/month for teams

**Best For**: Error tracking focus

### **AWS X-Ray**

**Strengths**:
- Native AWS integration
- Serverless tracing
- Low overhead
- Pay-per-use

**Pricing**: $5 per 1M traces

**Best For**: AWS-native applications

### **AppDynamics** (Cisco)

**Strengths**:
- Business transaction monitoring
- Application flow maps
- Enterprise features

**Pricing**: Custom (expensive)

**Best For**: Large enterprises

---

## APM Implementation

### **Auto-Instrumentation**

Most APM tools provide automatic instrumentation:

**Node.js** (Datadog):
\`\`\`javascript
// Step 1: Install
npm install dd-trace

// Step 2: Initialize (first line of app)
require('dd-trace').init();

// That\'s it! Auto-instrumentation of:
// - HTTP requests
// - Database queries
// - Redis operations
// - External API calls
\`\`\`

**Python** (New Relic):
\`\`\`bash
# Install
pip install newrelic

# Run with agent
NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python app.py
\`\`\`

**Java** (Dynatrace):
\`\`\`bash
# Attach OneAgent
java -javaagent:dynatrace-agent.jar -jar myapp.jar
\`\`\`

### **Manual Instrumentation**

For custom business logic:

\`\`\`javascript
const tracer = require('dd-trace').init();

async function processOrder (orderId) {
  const span = tracer.startSpan('process.order');
  span.setTag('order.id', orderId);
  
  try {
    const order = await fetchOrder (orderId);
    span.setTag('order.amount', order.amount);
    
    await chargePayment (order);
    await fulfillOrder (order);
    
    span.setTag('status', 'success');
  } catch (error) {
    span.setTag('error', true);
    span.setTag('error.msg', error.message);
    throw error;
  } finally {
    span.finish();
  }
}
\`\`\`

### **Custom Metrics**

Track business-specific metrics:

\`\`\`javascript
const { metrics } = require('dd-trace');

// Custom metric
metrics.increment('checkout.completed', 1, {
  tags: ['payment_method:credit_card', 'region:us-east']
});

metrics.gauge('cart.value', cartValue);
metrics.histogram('order.processing_time', duration);
\`\`\`

---

## Key APM Metrics

### **Application Metrics**

**Apdex Score** (Application Performance Index):
- Satisfied: Response < 0.5s → Score 1
- Tolerating: 0.5s < Response < 2s → Score 0.5
- Frustrated: Response > 2s → Score 0
- Apdex = (Satisfied + 0.5 × Tolerating) / Total

**Throughput**:
- Requests per minute
- Transactions per second

**Response Time**:
- Average
- p50, p95, p99 percentiles
- Min/max

**Error Rate**:
- Errors per minute
- Error percentage
- HTTP 5xx rate

### **Infrastructure Metrics**

- CPU usage
- Memory usage
- Disk I/O
- Network I/O
- Connection pool usage

### **Business Metrics**

- Revenue per transaction
- Conversion rate
- Cart abandonment
- User signups

---

## Service Maps

### **What Are Service Maps?**

Visual representation of service dependencies

**Example**:
\`\`\`
       ┌──────────────┐
       │ Load Balancer│
       └──────┬───────┘
              │
     ┌────────▼─────────┐
     │   API Gateway    │
     └────────┬─────────┘
              │
      ┌───────┼────────┐
      │       │        │
 ┌────▼───┐ ┌▼──────┐ ┌▼────────┐
 │  Auth  │ │ Users │ │ Orders  │
 └────┬───┘ └┬──────┘ └┬────────┘
      │      │         │
      └──────┼─────────┤
             │         │
        ┌────▼─────┐ ┌▼──────────┐
        │ Database │ │  Payment  │
        └──────────┘ └───────────┘
\`\`\`

**Information Shown**:
- Request rate between services
- Error rate per connection
- Latency per connection
- Service health (green/yellow/red)

**Use Cases**:
- Understand dependencies
- Identify critical path
- Spot cascading failures
- Plan migrations

---

## Error Tracking Deep Dive

### **Error Grouping**

Similar errors grouped together:
\`\`\`
Error Group: NullPointerException in UserController.java:42
Occurrences: 247
First seen: 3 days ago
Last seen: 2 minutes ago
Affected users: 89
Status: Unresolved
\`\`\`

### **Error Context**

**Request Context**:
- URL, method, headers
- Request body
- Query parameters
- User agent

**User Context**:
- User ID
- Email
- Account type
- Browser/device

**Environment Context**:
- Server hostname
- Deployment version
- Feature flags
- Environment (prod/staging)

### **Breadcrumbs**

Events leading up to error:
\`\`\`
1. User logged in
2. Navigated to checkout
3. Added item to cart
4. Applied discount code
5. Clicked "Place Order"
6. ERROR: Payment failed
\`\`\`

### **Release Tracking**

Track errors per release:
\`\`\`
Release v2.5.0 (deployed 2 hours ago)
New errors: 3
Error spike: NullPointerException +250%
Status: Investigating
\`\`\`

---

## Performance Optimization with APM

### **1. Identify Slow Transactions**

**APM Shows**:
- Top 10 slowest endpoints
- p99 latency per endpoint
- Slow transaction examples

**Example Finding**:
\`\`\`
Endpoint: POST /api/orders
p99 Latency: 3.5s ← Needs optimization!
Volume: 1000 req/min
\`\`\`

### **2. Drill Into Slow Request**

**APM Provides**:
- Full trace with timing
- Database queries executed
- External API calls
- Code-level breakdown

**Example**:
\`\`\`
POST /api/orders (3.2s total)
├─ Authentication: 50ms
├─ Validate inventory: 100ms
├─ Calculate taxes: 2.8s ← Problem!
│   └─ Database query: 2.7s
│       Query: SELECT * FROM tax_rates WHERE...
│       Issue: Missing index
└─ Save order: 250ms
\`\`\`

### **3. Fix and Verify**

- Add database index
- Deploy fix
- Monitor in APM
- Verify p99 latency drops to < 500ms

---

## Alerting with APM

### **Smart Alerts**

**Error Rate Spike**:
\`\`\`
ALERT: Error rate > 5% for service "checkout"
Current: 12%
Threshold: 5%
Duration: 5 minutes
Affected transactions: 150
\`\`\`

**Latency Degradation**:
\`\`\`
ALERT: p99 latency > 1s for endpoint "/api/search"
Current: 2.3s
Baseline: 450ms
Change: +400%
\`\`\`

**Anomaly Detection**:
\`\`\`
ALERT: Unusual traffic pattern detected
Service: user-api
Metric: Request rate
Current: 500 req/s (normal: 100 req/s)
Possible: DDoS attack or viral content
\`\`\`

### **Alert Fatigue Prevention**

**Intelligent Grouping**:
- Group related alerts
- Suppress duplicates
- Escalate based on severity

**Anomaly-Based Alerts**:
- Learn normal patterns
- Alert on deviations
- Reduce false positives

---

## APM Best Practices

### **Do's**
✅ Start with auto-instrumentation
✅ Add custom instrumentation for business logic
✅ Track both technical and business metrics
✅ Set up meaningful alerts (not too many!)
✅ Use error tracking for all exceptions
✅ Monitor real user experience (RUM)
✅ Create service dependency maps
✅ Track performance per release

### **Don'ts**
❌ Over-instrument (performance impact)
❌ Alert on everything (fatigue)
❌ Ignore APM data (waste of money)
❌ Only monitor in production (use in dev/staging too)
❌ Forget to track business metrics
❌ Send sensitive data to APM

---

## Cost Considerations

### **Pricing Models**

**Per-Host**: $30-$100/host/month
- Datadog, New Relic

**Per-User**: $25-$99/user/month
- New Relic, AppDynamics

**Usage-Based**: Pay for data ingested
- Elastic Cloud, AWS X-Ray

**Hybrid**: Base + usage
- Datadog (host + custom metrics)

### **Cost Optimization**

**Sampling**:
- Trace 1-10% of requests
- Sample more on errors

**Data Retention**:
- Keep high-resolution data for 7 days
- Downsample for longer retention

**Filter Noise**:
- Exclude health checks
- Filter success responses (keep errors)

**Use Open Source**:
- Elastic APM (self-hosted)
- Jaeger + Grafana

---

## Interview Tips

### **Key Points**1. **APM = Observability + Intelligence**2. **Components**: Transaction monitoring, error tracking, RUM, profiling
3. **Tools**: Datadog, New Relic, Dynatrace, Elastic APM
4. **Use Cases**: Performance optimization, error debugging, user experience
5. **Best Practice**: Auto-instrument + custom business metrics

### **Common Questions**

**Q: How would you identify why an API is slow?**
A: Use APM to trace slow requests, identify bottleneck span (e.g., database query), drill into code-level profiling, optimize the slow operation.

**Q: What\'s the difference between APM and logging?**
A: Logs are raw events. APM synthesizes logs, metrics, and traces into actionable insights with code-level context.

**Q: How do you balance APM costs with observability needs?**
A: Sample traces (1-10%), keep errors at 100%, downsample metrics, filter noise, use appropriate retention.

---

## Real-World Examples

### **Shopify**
- **Challenge**: Black Friday traffic spikes
- **Solution**: APM for proactive monitoring
- **Result**: Zero downtime during 10x traffic

### **Airbnb**
- **Challenge**: Slow search performance
- **Solution**: APM identified N+1 queries
- **Result**: 60% latency reduction

### **Netflix**
- **Challenge**: 1000+ microservices
- **APM**: Custom tools + vendor solutions
- **Result**: 99.99% availability

---

## Summary

APM tools provide comprehensive application visibility:

1. **Transaction Monitoring**: Track request flow and performance
2. **Error Tracking**: Capture and analyze exceptions with context
3. **Code-Level Profiling**: Identify slow functions
4. **Real User Monitoring**: Understand actual user experience
5. **Service Maps**: Visualize dependencies

Modern APM combines observability's three pillars with intelligence, making it essential for production systems. Start with auto-instrumentation, add custom metrics for business logic, and use insights to proactively optimize performance.`,
};
