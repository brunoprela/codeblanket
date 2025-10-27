/**
 * Logging Best Practices Section
 */

export const loggingBestPracticesSection = {
  id: 'logging-best-practices',
  title: 'Logging Best Practices',
  content: `Logging is the most fundamental form of observability, yet it's often done poorly. Well-designed logs can accelerate debugging by 10x, while bad logs create noise and waste resources. This section covers production-grade logging practices used at top tech companies.

## What is Logging?

**Logging** is the practice of recording discrete events that happen in your application. Each log entry captures:
- **What happened**: The event or operation
- **When it happened**: Timestamp
- **Where it happened**: Service, function, line
- **Context**: Related data (user ID, request ID, etc.)

**Purpose**:
- Debugging production issues
- Audit trails
- Security analysis
- Understanding system behavior

---

## Log Levels

### **Standard Log Levels** (Most to Least Verbose)

#### **TRACE**
**When**: Extremely detailed debugging
**Example**: "Entering function calculateDiscount with params: {userId: 123, items: [...]}"
**Production**: Usually OFF (too verbose)

#### **DEBUG**
**When**: Diagnostic information
**Example**: "Cache miss for key: user:123, querying database"
**Production**: OFF or sampled (10% of requests)

#### **INFO**
**When**: Normal operations, significant events
**Example**: "User 123 logged in successfully from IP 192.168.1.1"
**Production**: ON (but be selective)

#### **WARN**
**When**: Unexpected but handled situations
**Example**: "Payment gateway latency high (2.5s), retrying"
**Production**: ON (always investigate)

#### **ERROR**
**When**: Errors that prevent operation
**Example**: "Failed to process payment for order 456: timeout"
**Production**: ON (alerts should fire)

#### **FATAL/CRITICAL**
**When**: System-breaking errors
**Example**: "Database connection pool exhausted, service shutting down"
**Production**: ON (immediate page-out)

---

## Structured Logging

### **Why Structured Logs?**

**Unstructured** (bad):
\`\`\`
User john made purchase of $49.99 for product laptop
\`\`\`

**Problems**:
- Hard to parse
- Inconsistent format
- Can't aggregate
- Difficult to query

**Structured** (good):
\`\`\`json
{
  "timestamp": "2024-01-15T10:23:45.123Z",
  "level": "INFO",
  "event": "purchase_completed",
  "user_id": "john",
  "amount": 49.99,
  "currency": "USD",
  "product": "laptop",
  "trace_id": "abc-123",
  "service": "checkout-api",
  "environment": "production"
}
\`\`\`

**Benefits**:
- Machine-readable
- Easy to query: \`amount > 100\`
- Consistent structure
- Enables aggregation

### **Format: JSON vs Others**

**JSON** (recommended):
- Universal support
- Easy to parse
- Flexible schema
\`\`\`json
{"level":"INFO","message":"User login","user_id":123}
\`\`\`

**Logfmt** (alternative):
- Human-readable
- Easier to read raw
- Still structured
\`\`\`
level=INFO message="User login" user_id=123
\`\`\`

**Choice**: Use JSON for production systems (better tooling support)

---

## What to Log

### **Always Log**

✅ **Critical Business Events**
- User registration, login, logout
- Purchases, payments, refunds
- Data mutations (create, update, delete)

✅ **Errors and Exceptions**
- Stack traces
- Error codes
- Context leading to error

✅ **External Service Calls**
- API calls to third parties
- Database queries (slow ones)
- Message queue operations

✅ **Security Events**
- Authentication attempts (success/failure)
- Authorization failures
- Suspicious activity

### **Avoid Logging**

❌ **Sensitive Data**
- Passwords (even encrypted!)
- Credit card numbers
- Personal Identifiable Information (PII)
- API keys, tokens, secrets

❌ **High-Frequency Operations**
- Every database query (too noisy)
- Successful health checks
- Internal function calls

❌ **Non-Events**
- Variables during computation
- Intermediate states
- Debug artifacts left in code

---

## Contextual Logging

### **Correlation IDs**

Every request should have a unique ID that flows through all services.

**Example Flow**:
\`\`\`
API Gateway: trace_id=abc-123 → generates ID
Auth Service: trace_id=abc-123 → propagates ID
User Service: trace_id=abc-123 → propagates ID
Database:     trace_id=abc-123 → propagates ID
\`\`\`

**Implementation** (Express.js):
\`\`\`javascript
app.use((req, res, next) => {
  req.id = req.headers['x-correlation-id'] || generateId();
  res.setHeader('x-correlation-id', req.id);
  next();
});
\`\`\`

**Benefits**:
- Track request across services
- Debug distributed issues
- Measure end-to-end latency

### **Standard Fields**

Include in every log:
- **timestamp**: ISO 8601 format
- **level**: INFO, ERROR, etc.
- **service**: Service name
- **trace_id**: Correlation ID
- **user_id**: When available
- **environment**: prod, staging, dev

**Example Structure**:
\`\`\`json
{
  "timestamp": "2024-01-15T10:23:45.123Z",
  "level": "ERROR",
  "service": "payment-service",
  "trace_id": "abc-123",
  "user_id": "user-456",
  "environment": "production",
  "event": "payment_failed",
  "error": "Gateway timeout",
  "details": {...}
}
\`\`\`

---

## Centralized Logging

### **Why Centralize?**

**Without Centralization**:
- SSH into each server
- grep through files
- Correlate manually
- Lose data when servers die

**With Centralization**:
- Single query across all servers
- Persistent storage
- Full-text search
- Historical analysis

### **Architecture**

\`\`\`
Application → Log Forwarder → Aggregator → Storage → Query UI
  (app.log)     (Filebeat)     (Logstash)   (Elasticsearch) (Kibana)
\`\`\`

**Popular Stacks**:

1. **ELK Stack** (Elasticsearch, Logstash, Kibana)
   - Most popular
   - Self-hosted
   - Powerful search

2. **Loki + Grafana** (Grafana Labs)
   - Label-based indexing
   - Lower cost than ELK
   - Integrates with metrics

3. **Splunk** (Commercial)
   - Enterprise features
   - Expensive
   - Advanced analytics

4. **Cloud Services**
   - CloudWatch (AWS)
   - Stackdriver (GCP)
   - Azure Monitor

---

## Log Forwarders

### **Filebeat** (Elastic)
- Lightweight shipper
- Tails log files
- Buffers during outages
- Minimal CPU/memory

**Configuration**:
\`\`\`yaml
filebeat.inputs:
  - type: log
    paths:
      - /var/log/app/*.log
    fields:
      service: my-api
output.logstash:
  hosts: ["logstash:5044"]
\`\`\`

### **Fluentd**
- Unified logging layer
- Rich plugin ecosystem
- Transforms logs
- Supports multiple outputs

**Configuration**:
\`\`\`xml
<source>
  @type tail
  path /var/log/app/*.log
  format json
  tag app.logs
</source>

<match app.logs>
  @type elasticsearch
  host elasticsearch
  port 9200
</match>
\`\`\`

### **Vector** (Datadog)
- Modern alternative
- Written in Rust (fast)
- Built-in transforms
- Lower resource usage

---

## Log Retention and Lifecycle

### **Hot-Warm-Cold Architecture**

**Hot Storage** (0-7 days):
- Fast SSD storage
- Full indexing
- Real-time search
- Most expensive

**Warm Storage** (7-30 days):
- Slower HDD storage
- Partial indexing
- Acceptable latency
- Medium cost

**Cold Storage** (30+ days):
- Object storage (S3)
- No indexing
- Compliance/audit
- Cheapest

**Archive** (1+ years):
- Compressed archives
- Rarely accessed
- Legal requirements
- Very cheap

### **Retention Policies**

**By Log Level**:
- DEBUG: 1 day
- INFO: 7 days
- WARN: 30 days
- ERROR: 90 days
- FATAL: 1 year

**By Service**:
- Critical services: 90 days
- Non-critical: 30 days
- Test environments: 7 days

**Compliance**: Often requires longer (GDPR, HIPAA)

---

## Log Sampling

### **Why Sample?**

At scale, logging everything is:
- Too expensive (storage)
- Too slow (I/O)
- Too noisy (signal to noise)

### **Sampling Strategies**

**1. Level-Based**
\`\`\`
ERROR, FATAL: 100% (log everything)
WARN: 100%
INFO: 10% (sample 1 in 10)
DEBUG: 1% (sample 1 in 100)
\`\`\`

**2. Adaptive Sampling**
- Normal: 1% sampling
- Error spike detected: 100% for 5 minutes
- Return to normal: 1%

**3. Tail-Based Sampling** (traces)
- Buffer logs for request
- If request errors → keep all logs
- If request succeeds → sample

**4. Rate Limiting**
- Max 1000 logs/second per service
- Prevents log storms
- Emit "logs dropped" counter

---

## Log Aggregation Patterns

### **1. Application-Side**
Application sends directly to logging service

**Pros**: No agent needed
**Cons**: Increases app latency, coupling

### **2. Sidecar Pattern**
Agent runs alongside application (Kubernetes)

**Pros**: Decoupled, offloads work
**Cons**: More complex

### **3. File + Forwarder**
App writes to file, forwarder ships logs

**Pros**: Reliable, buffering
**Cons**: Disk I/O, delay

**Best Practice**: Sidecar for Kubernetes, File + Forwarder for VMs

---

## Log Security

### **What NOT to Log**

❌ **Credentials**
- Passwords, API keys, tokens
- Even hashed passwords
- SSH keys

❌ **PII** (Personal Identifiable Information)
- Full names, SSN, addresses
- Email addresses (sometimes)
- Phone numbers

❌ **Financial Data**
- Credit card numbers
- Bank account details
- Crypto private keys

### **Redaction Strategies**

**1. At Source** (best)
\`\`\`javascript
logger.info('User login', {
  user_id: user.id,
  email: redact (user.email),  // j***@example.com
  ip: req.ip
});
\`\`\`

**2. In Forwarder**
Configure Filebeat/Fluentd to redact patterns

**3. In Aggregator**
Logstash filters to remove sensitive data

### **Access Control**

- Logs contain sensitive business data
- Role-based access (RBAC)
- Audit log access itself
- Encrypt logs at rest and in transit

---

## Common Logging Anti-Patterns

### **1. Logging Exceptions Without Context**
❌ Bad:
\`\`\`
logger.error (error.message)
\`\`\`

✅ Good:
\`\`\`
logger.error('Failed to process payment', {
  error: error.message,
  stack: error.stack,
  order_id: orderId,
  user_id: userId,
  amount: amount,
  trace_id: traceId
});
\`\`\`

### **2. Log Spam**
❌ Bad:
\`\`\`
for (user of users) {
  logger.debug('Processing user', user);
}
\`\`\`

✅ Good:
\`\`\`
logger.info('Processing batch', {
  user_count: users.length,
  batch_id: batchId
});
\`\`\`

### **3. Inconsistent Formats**
❌ Bad:
\`\`\`
logger.info('User login: john')
logger.info('User: jane logged in')
logger.info('Login successful for user: bob')
\`\`\`

✅ Good:
\`\`\`
logger.info('User login', { event: 'user_login', user_id: 'john' })
logger.info('User login', { event: 'user_login', user_id: 'jane' })
logger.info('User login', { event: 'user_login', user_id: 'bob' })
\`\`\`

### **4. Logging in Hot Paths**
❌ Bad:
\`\`\`
for (let i = 0; i < 1000000; i++) {
  logger.debug('Loop iteration', i);
  process (i);
}
\`\`\`

### **5. Not Logging Errors**
❌ Bad:
\`\`\`
try {
  processPayment();
} catch (e) {
  // Silent failure
}
\`\`\`

✅ Good:
\`\`\`
try {
  processPayment();
} catch (e) {
  logger.error('Payment processing failed', {
    error: e.message,
    stack: e.stack,
    order_id: orderId
  });
  throw e;
}
\`\`\`

---

## Performance Considerations

### **Async Logging**

**Synchronous** (blocks):
\`\`\`javascript
logger.info('Message');  // Waits for I/O
// Next line executes after log written
\`\`\`

**Asynchronous** (non-blocking):
\`\`\`javascript
logger.info('Message');  // Returns immediately
// Log written in background
\`\`\`

**Trade-off**: Async is faster but logs might be lost on crash

**Best Practice**: Use async with buffering and flush on shutdown

### **Batching**

Instead of sending each log immediately:
1. Buffer logs in memory (100 entries or 5 seconds)
2. Send batch to logging service
3. Reduces network calls 100x

### **Cost of Logging**

**CPU**: JSON serialization, formatting
**Memory**: Buffering
**Network**: Sending to aggregator
**Storage**: Disk space

**Benchmark**: Logging can add 5-10% overhead at high volumes

---

## Practical Implementation

### **Node.js (Pino)**
\`\`\`javascript
const pino = require('pino');
const logger = pino({
  level: 'info',
  base: { service: 'api' }
});

logger.info({ user_id: 123, trace_id: 'abc' }, 'User logged in');
\`\`\`

### **Python (structlog)**
\`\`\`python
import structlog
logger = structlog.get_logger()

logger.info("user_login",
  user_id=123,
  trace_id="abc")
\`\`\`

### **Go (zap)**
\`\`\`go
logger, _ := zap.NewProduction()
logger.Info("User logged in",
  zap.String("user_id", "123"),
  zap.String("trace_id", "abc"))
\`\`\`

### **Java (Log4j2 with JSON)**
\`\`\`java
logger.info(LogMarkers.USER_LOGIN,
  "user_id={} trace_id={}",
  userId, traceId);
\`\`\`

---

## Interview Tips

### **When Discussing Logging**1. **Mention Structured Logging**
   - Shows modern practices
   - JSON format

2. **Talk About Scale**
   - Sampling at high volume
   - Cost considerations
   - Hot-warm-cold storage

3. **Security Awareness**
   - Never log secrets
   - PII handling
   - Access control

4. **Centralization**
   - ELK stack or Loki
   - Why centralized logs matter
   - Retention policies

5. **Correlation IDs**
   - Tracking requests across services
   - Essential for microservices

---

## Real-World Examples

### **Google**
- **Log Volume**: Petabytes per day
- **Approach**: Extensive sampling, structured logs
- **Tools**: Internal systems (Dremel for querying)

### **Netflix**
- **Log Volume**: Terabytes per day
- **Approach**: Real-time log processing (Apache Kafka)
- **Tools**: Custom stream processing (Mantis)

### **Uber**
- **Challenge**: Millions of rides, need audit trail
- **Solution**: Structured logs with ride_id in every log
- **Retention**: Hot data for 7 days, warm for 30, archive for 1 year

---

## Summary

Effective logging is critical for production systems:

1. **Structure**: Use JSON for machine readability
2. **Context**: Include correlation IDs and relevant fields
3. **Levels**: Use appropriate levels (ERROR for errors, not DEBUG)
4. **Security**: Never log secrets or PII
5. **Centralize**: Aggregate logs from all services
6. **Sample**: Reduce volume at scale
7. **Retain**: Hot-warm-cold based on access patterns

Good logging accelerates debugging, enables auditing, and provides system insights. Bad logging creates noise and wastes money. Design logging systems with the same rigor as application code.`,
};
