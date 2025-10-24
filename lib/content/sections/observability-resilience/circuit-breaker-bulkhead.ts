/**
 * Circuit Breaker & Bulkhead Patterns Section
 */

export const circuitBreakerBulkheadSection = {
  id: 'circuit-breaker-bulkhead',
  title: 'Circuit Breaker & Bulkhead Patterns',
  content: `In distributed systems, failures are inevitable. Services go down, networks partition, and dependencies fail. Without proper protection, a single failing service can cascade and bring down your entire system. Circuit Breaker and Bulkhead patterns are essential resilience patterns that prevent cascading failures and protect system stability.

## The Cascading Failure Problem

### **Scenario Without Protection**

\`\`\`
User Request → API → Database (down)
                      ↓
                   Hangs for 30s timeout
                      ↓
                   Thread blocked
                      ↓
                   100 requests = 100 threads blocked
                      ↓
                   Thread pool exhausted
                      ↓
                   API crashes
                      ↓
                   All users affected!
\`\`\`

**The Problem**: One failing dependency crashes the entire service

**Real-World Impact**:
- 2016: AWS outage cascaded across regions
- 2017: GitLab.com 18-hour outage from database issue
- 2020: Cloudflare global outage from single bad configuration

### **Solution Patterns**

1. **Circuit Breaker**: Stop calling failing services
2. **Bulkhead**: Isolate failures to prevent spread
3. **Timeout**: Don't wait forever
4. **Fallback**: Gracefully degrade

---

## Circuit Breaker Pattern

### **What is a Circuit Breaker?**

**Circuit Breaker** prevents an application from repeatedly trying to execute an operation that's likely to fail.

**Electrical Analogy**:
- Electrical circuit breaker trips when current too high
- Protects circuit from damage
- Must be manually reset

**Software Circuit Breaker**:
- Trips when failure rate too high
- Protects service from cascade
- Auto-resets after timeout

### **Circuit States**

**1. Closed** (Normal Operation)
\`\`\`
Requests → Pass through → Dependency
Success → Stay closed
Failure → Increment counter
If failures > threshold → Open circuit
\`\`\`

**2. Open** (Failing Fast)
\`\`\`
Requests → BLOCKED → Return error immediately
No calls to dependency
After timeout → Half-Open
\`\`\`

**3. Half-Open** (Testing Recovery)
\`\`\`
Requests → Limited pass through → Dependency
If success → Close circuit
If failure → Re-open circuit
\`\`\`

### **State Diagram**

\`\`\`
       ┌──────────┐
       │  Closed  │
       │ (Normal) │
       └────┬─────┘
            │
      Failures > threshold
            │
            ↓
       ┌──────────┐
  ┌───│   Open   │
  │   │ (Failing)│
  │   └────┬─────┘
  │        │
  │   Timeout expires
  │        │
  │        ↓
  │   ┌──────────────┐
  │   │  Half-Open   │
  │   │  (Testing)   │
  │   └──────┬───────┘
  │          │
  │     Success │ Failure
  └──────────┬───────┘
\`\`\`

### **Configuration**

**Key Parameters**:

**1. Failure Threshold**
- Number or percentage of failures before opening
- Example: Open after 5 consecutive failures

**2. Timeout**
- How long to wait in open state before testing
- Example: 30 seconds

**3. Success Threshold** (Half-Open)
- Number of successes needed to close
- Example: 2 consecutive successes

**4. Sliding Window**
- Time window for counting failures
- Example: 10 failures in 1 minute

**Example Configuration**:
\`\`\`javascript
{
  failureThreshold: 5,        // Open after 5 failures
  failureRate: 50,            // Or 50% failure rate
  timeout: 30000,             // 30 seconds in open state
  successThreshold: 2,        // 2 successes to close
  slidingWindowSize: 100,     // Last 100 requests
  slidingWindowTime: 60000    // Or last 60 seconds
}
\`\`\`

### **Implementation Example**

**Node.js (Opossum)**:
\`\`\`javascript
const CircuitBreaker = require('opossum');

// Wrap your async function
function callExternalAPI() {
  return fetch('https://api.example.com/data');
}

const breaker = new CircuitBreaker(callExternalAPI, {
  timeout: 3000,              // 3s timeout
  errorThresholdPercentage: 50, // Open at 50% failure
  resetTimeout: 30000         // Try again after 30s
});

// Use it
breaker.fire()
  .then(result => console.log(result))
  .catch(err => console.log('Circuit breaker caught:', err));

// Events
breaker.on('open', () => console.log('Circuit opened'));
breaker.on('halfOpen', () => console.log('Circuit half-open'));
breaker.on('close', () => console.log('Circuit closed'));
\`\`\`

**Java (Resilience4j)**:
\`\`\`java
CircuitBreakerConfig config = CircuitBreakerConfig.custom()
    .failureRateThreshold(50)
    .waitDurationInOpenState(Duration.ofSeconds(30))
    .slidingWindowSize(100)
    .build();

CircuitBreaker breaker = CircuitBreaker.of("myService", config);

// Decorate your function
Supplier<String> decorated = CircuitBreaker.decorateSupplier(
    breaker,
    () -> callExternalAPI()
);

// Execute
try {
    String result = decorated.get();
} catch (Exception e) {
    // Fallback logic
}
\`\`\`

**Python (pybreaker)**:
\`\`\`python
from pybreaker import CircuitBreaker

breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=30
)

@breaker
def call_external_api():
    return requests.get('https://api.example.com/data')

# Use it
try:
    result = call_external_api()
except CircuitBreakerError:
    # Fallback logic
    result = get_cached_data()
\`\`\`

### **Fallback Strategies**

When circuit is open, what to return?

**1. Cached Data**
\`\`\`javascript
try {
  return await breaker.fire();
} catch (err) {
  return cache.get('last_good_value');
}
\`\`\`

**2. Default Value**
\`\`\`javascript
try {
  return await breaker.fire();
} catch (err) {
  return { status: 'unavailable', data: [] };
}
\`\`\`

**3. Degraded Functionality**
\`\`\`javascript
try {
  return await getPersonalizedRecommendations();
} catch (err) {
  return getPopularItems(); // Fallback to non-personalized
}
\`\`\`

**4. Error Response**
\`\`\`javascript
try {
  return await breaker.fire();
} catch (err) {
  return { error: 'Service temporarily unavailable' };
}
\`\`\`

---

## Bulkhead Pattern

### **What is a Bulkhead?**

**Bulkhead** isolates elements of an application into pools so that if one fails, others continue to function.

**Ship Analogy**:
- Ships have watertight compartments (bulkheads)
- If one compartment floods, others stay dry
- Ship stays afloat even with damage

**Software Bulkhead**:
- Partition resources (threads, connections)
- If one partition exhausted, others unaffected
- System stays partially functional

### **The Problem**

**Without Bulkheads**:
\`\`\`
Thread Pool (50 threads)
├─ Payment API (slow): Uses 40 threads
├─ User API: Needs 10 threads, only 10 available
└─ Search API: Needs 5 threads, NONE AVAILABLE ← Starved!

Result: One slow service blocks everything
\`\`\`

**With Bulkheads**:
\`\`\`
Thread Pool Partition 1 (20 threads): Payment API
  └─ Can use max 20 threads, no more

Thread Pool Partition 2 (20 threads): User API
  └─ Always has 20 threads available

Thread Pool Partition 3 (10 threads): Search API
  └─ Always has 10 threads available

Result: Payment slowness doesn't affect Search
\`\`\`

### **Types of Bulkheads**

**1. Thread Pool Bulkhead**

Separate thread pools for different operations

\`\`\`javascript
// Payment service thread pool
const paymentPool = new ThreadPool({ size: 20 });

// User service thread pool
const userPool = new ThreadPool({ size: 20 });

// Search service thread pool
const searchPool = new ThreadPool({ size: 10 });
\`\`\`

**2. Semaphore Bulkhead**

Limit concurrent executions using semaphores

\`\`\`javascript
// Max 10 concurrent calls to payment service
const paymentSemaphore = new Semaphore(10);

async function callPaymentService() {
  await paymentSemaphore.acquire();
  try {
    return await paymentAPI.charge();
  } finally {
    paymentSemaphore.release();
  }
}
\`\`\`

**3. Connection Pool Bulkhead**

Separate connection pools for different databases/services

\`\`\`javascript
// Primary database connection pool
const primaryDB = new ConnectionPool({
  host: 'primary-db',
  maxConnections: 50
});

// Analytics database connection pool
const analyticsDB = new ConnectionPool({
  host: 'analytics-db',
  maxConnections: 20
});
\`\`\`

### **Implementation Example**

**Resilience4j (Java)**:
\`\`\`java
// Thread Pool Bulkhead
ThreadPoolBulkheadConfig config = ThreadPoolBulkheadConfig.custom()
    .maxThreadPoolSize(10)
    .coreThreadPoolSize(5)
    .queueCapacity(20)
    .build();

ThreadPoolBulkhead bulkhead = ThreadPoolBulkhead.of(
    "paymentService",
    config
);

// Use it
CompletableFuture<String> future = bulkhead.executeSupplier(
    () -> callPaymentService()
);
\`\`\`

**Node.js (Custom)**:
\`\`\`javascript
class Bulkhead {
  constructor(maxConcurrent) {
    this.maxConcurrent = maxConcurrent;
    this.current = 0;
    this.queue = [];
  }

  async execute(fn) {
    if (this.current >= this.maxConcurrent) {
      // Wait in queue
      await new Promise(resolve => this.queue.push(resolve));
    }

    this.current++;
    try {
      return await fn();
    } finally {
      this.current--;
      if (this.queue.length > 0) {
        const next = this.queue.shift();
        next();
      }
    }
  }
}

// Usage
const paymentBulkhead = new Bulkhead(10);
const userBulkhead = new Bulkhead(20);

paymentBulkhead.execute(() => callPaymentAPI());
userBulkhead.execute(() => callUserAPI());
\`\`\`

---

## Combining Patterns

### **Circuit Breaker + Bulkhead**

Best used together for maximum resilience

\`\`\`
Request
  ↓
[Bulkhead] ← Limit concurrency
  ↓
[Circuit Breaker] ← Fail fast if service down
  ↓
External Service
\`\`\`

**Example**:
\`\`\`javascript
// Thread pool bulkhead
const bulkhead = new Bulkhead(10);

// Circuit breaker
const breaker = new CircuitBreaker(callAPI, {
  failureThreshold: 5,
  timeout: 30000
});

// Combined
async function callWithResilience() {
  return bulkhead.execute(async () => {
    try {
      return await breaker.fire();
    } catch (err) {
      return fallback();
    }
  });
}
\`\`\`

### **Circuit Breaker + Retry + Timeout**

Full resilience stack

\`\`\`
Request
  ↓
[Timeout] ← Don't wait forever (3s)
  ↓
[Retry] ← Try again (3 attempts)
  ↓
[Circuit Breaker] ← Fail fast if broken
  ↓
External Service
\`\`\`

---

## Monitoring Circuit Breakers

### **Key Metrics**

**State Metrics**:
\`\`\`
circuit_breaker_state{name="payment_api"} = 0 (closed)
circuit_breaker_state{name="payment_api"} = 1 (open)
circuit_breaker_state{name="payment_api"} = 2 (half-open)
\`\`\`

**Call Metrics**:
\`\`\`
circuit_breaker_calls_total{name="payment_api", result="success"} = 1000
circuit_breaker_calls_total{name="payment_api", result="failure"} = 50
circuit_breaker_calls_total{name="payment_api", result="rejected"} = 200
\`\`\`

**Timing Metrics**:
\`\`\`
circuit_breaker_call_duration_seconds{name="payment_api", quantile="0.99"} = 0.5
\`\`\`

### **Dashboards**

**Circuit Breaker Status**:
\`\`\`
[payment_api: CLOSED ✓]
[user_api: CLOSED ✓]
[search_api: OPEN ⚠]
[notification_api: HALF-OPEN ⚠]
\`\`\`

**Success Rate**:
\`\`\`
payment_api: 98% (last 5 min)
user_api: 99.5%
search_api: 45% ← Problem!
\`\`\`

### **Alerts**

**Circuit Opened**:
\`\`\`
ALERT: Circuit breaker opened for "search_api"
Failure rate: 60%
Last 100 requests: 40 failures
Action: Investigate search_api
\`\`\`

**Circuit Stays Open**:
\`\`\`
WARNING: Circuit breaker open for 10 minutes
Service: "payment_api"
Action: Check if service is actually down
\`\`\`

---

## Best Practices

### **Circuit Breaker**

✅ **Do's**:
- Use for external dependencies
- Configure appropriate timeouts
- Implement fallback logic
- Monitor circuit state
- Log state transitions
- Test circuit opening/closing

❌ **Don'ts**:
- Use for local operations
- Set threshold too low (flapping)
- Ignore open circuits
- Forget to implement fallback
- Use globally (per-service is better)

### **Bulkhead**

✅ **Do's**:
- Partition by criticality
- Size pools based on load
- Monitor pool utilization
- Reserve resources for critical paths
- Test under load

❌ **Don'ts**:
- Over-partition (too many pools)
- Under-provision critical services
- Ignore queue growth
- Forget to handle rejection

---

## Tuning

### **Circuit Breaker Tuning**

**Failure Threshold**:
- Too low (3): Opens too easily, false positives
- Too high (100): Opens too late, damage done
- **Sweet spot**: 5-10 failures or 50% rate

**Timeout**:
- Too short (5s): Doesn't give time to recover
- Too long (5min): Prolongs outage
- **Sweet spot**: 30-60 seconds

**Success Threshold** (Half-Open):
- Typically 1-2 successes
- More = safer but slower recovery

### **Bulkhead Tuning**

**Pool Size**:
- Too small: Under-utilized, poor performance
- Too large: No isolation, defeats purpose
- **Start with**: Expected peak load × 1.5

**Example**:
\`\`\`
Payment API:
  - Peak load: 100 req/s
  - Avg latency: 200ms
  - Concurrent: 100 × 0.2 = 20 requests
  - Pool size: 20 × 1.5 = 30 threads
\`\`\`

---

## Interview Tips

### **Key Concepts**

1. **Circuit Breaker**: Stop calling failing services
2. **Three States**: Closed, open, half-open
3. **Bulkhead**: Isolate failures with resource partitioning
4. **Fallback**: What to do when breaker is open
5. **Combine**: Use together for resilience

### **Common Questions**

**Q: How does circuit breaker prevent cascading failures?**
A: Fails fast when service is down, preventing thread exhaustion. Instead of waiting 30s for timeout, immediately returns error, freeing threads for other requests.

**Q: Difference between circuit breaker and retry?**
A: Retry attempts operation multiple times (good for transient failures). Circuit breaker stops attempting after threshold (good for sustained failures). Use both together.

**Q: How do you size bulkhead partitions?**
A: Based on expected load and latency. concurrent_requests = requests_per_sec × latency_seconds. Add 50% buffer.

**Q: When would you NOT use circuit breaker?**
A: For critical operations where failure is not acceptable (e.g., financial transactions). For local operations (no network). When fallback is not possible.

---

## Real-World Examples

### **Netflix (Hystrix)**
- **Created**: Hystrix library (now in maintenance)
- **Scale**: Protects 1000s of microservices
- **Result**: Survived massive failures gracefully

### **AWS**
- **Circuit Breakers**: Between all services
- **Bulkheads**: Separate resource pools per service
- **Result**: Partial failures don't cascade

### **Uber**
- **Ringpop**: Circuit breakers for service mesh
- **Approach**: Automatic failure detection and isolation
- **Result**: High availability despite complex dependencies

---

## Summary

Circuit Breaker and Bulkhead patterns are essential for resilient distributed systems:

**Circuit Breaker**:
1. **Three states**: Closed, open, half-open
2. **Purpose**: Fail fast, prevent cascading failures
3. **Configuration**: Failure threshold, timeout, success threshold
4. **Fallback**: Cache, defaults, degraded functionality

**Bulkhead**:
1. **Purpose**: Isolate failures
2. **Types**: Thread pools, semaphores, connection pools
3. **Configuration**: Pool sizes based on load
4. **Benefit**: One failure doesn't exhaust all resources

**Combined**: Maximum resilience

These patterns are not optional for production systems. Implement them for all external dependencies. The investment in resilience pays off during the inevitable failures.`,
};
