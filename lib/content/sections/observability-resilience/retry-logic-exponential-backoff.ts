/**
 * Retry Logic & Exponential Backoff Section
 */

export const retryLogicSection = {
  id: 'retry-logic-exponential-backoff',
  title: 'Retry Logic & Exponential Backoff',
  content: `Network requests fail. Databases timeout. External APIs return errors. In distributed systems, transient failures are normal. Retry logic with exponential backoff is a fundamental pattern for handling these failures gracefully. However, naive retry logic can make problems worse. This section covers how to implement retries correctly.

## Why Retry?

### **Transient Failures**

Failures that resolve themselves quickly:

**Network Glitches**:
\`\`\`
Request → Network blip → Failed
Retry → Success
\`\`\`

**Temporary Overload**:
\`\`\`
Request → Server busy → 503 Service Unavailable
Wait 1 second
Retry → Success
\`\`\`

**Database Deadlock**:
\`\`\`
Transaction → Deadlock detected → Rollback
Retry → Success (deadlock resolved)
\`\`\`

**Brief Outage**:
\`\`\`
Request → Service restarting → Failed
Wait 5 seconds
Retry → Success
\`\`\`

### **Non-Transient Failures**

Failures that won't resolve with retry:

**Invalid Input**:
\`\`\`
Request with bad data → 400 Bad Request
Retry → Still 400 (don't retry!)
\`\`\`

**Authentication Failure**:
\`\`\`
Request with invalid token → 401 Unauthorized
Retry → Still 401 (don't retry!)
\`\`\`

**Not Found**:
\`\`\`
Request for deleted resource → 404 Not Found
Retry → Still 404 (don't retry!)
\`\`\`

**Business Logic Error**:
\`\`\`
Insufficient funds → 400 Payment Failed
Retry → Still insufficient (don't retry!)
\`\`\`

---

## When to Retry

### **Retry on These**

✅ **5xx Server Errors**
- 500 Internal Server Error
- 502 Bad Gateway
- 503 Service Unavailable
- 504 Gateway Timeout

✅ **Network Errors**
- Connection timeout
- Connection refused (if service is restarting)
- DNS resolution failure (temporary)

✅ **Database Errors**
- Deadlock detected
- Connection pool exhausted (temporary)
- Replica lag (eventual consistency)

✅ **Rate Limit** (with backoff)
- 429 Too Many Requests

### **Don't Retry on These**

❌ **4xx Client Errors** (except 429)
- 400 Bad Request
- 401 Unauthorized
- 403 Forbidden
- 404 Not Found
- 409 Conflict

❌ **Application Logic Errors**
- Invalid input
- Business rule violations
- Insufficient permissions

❌ **Non-Idempotent Operations** (without safeguards)
- POST /create-user (might create duplicate)
- POST /charge-payment (might double-charge)

---

## Idempotency

### **What is Idempotency?**

**Idempotent**: Operation that produces the same result no matter how many times it's executed

**Examples**:

**Idempotent** ✅:
\`\`\`
PUT /users/123 { name: "Alice" }
→ Run once: User 123 name = "Alice"
→ Run 10 times: User 123 name = "Alice" (same result)
\`\`\`

**Not Idempotent** ❌:
\`\`\`
POST /users { name: "Alice" }
→ Run once: Creates user with ID 456
→ Run 10 times: Creates 10 users (different results!)
\`\`\`

**Idempotent** ✅:
\`\`\`
DELETE /users/123
→ Run once: User deleted
→ Run 10 times: User still deleted (same result)
\`\`\`

### **Making Operations Idempotent**

**Idempotency Keys**:

\`\`\`javascript
// Client generates unique key per request
const idempotencyKey = generateUUID(); // "abc-123-def-456"

await fetch('/api/payments', {
  method: 'POST',
  headers: {
    'Idempotency-Key': idempotencyKey
  },
  body: JSON.stringify({
    amount: 100,
    userId: 'user-123'
  })
});
\`\`\`

**Server Side**:
\`\`\`javascript
app.post('/api/payments', async (req, res) => {
  const idempotencyKey = req.headers['idempotency-key'];
  
  // Check if request with this key already processed
  const cached = await cache.get(idempotencyKey);
  if (cached) {
    // Return cached result (idempotent!)
    return res.json(cached);
  }
  
  // Process payment
  const result = await processPayment(req.body);
  
  // Cache result for 24 hours
  await cache.set(idempotencyKey, result, { ttl: 86400 });
  
  return res.json(result);
});
\`\`\`

**Real-World Examples**:
- **Stripe**: Requires idempotency keys for payments
- **AWS**: Uses idempotency tokens for EC2 operations
- **Shopify**: Idempotency keys for order creation

---

## Retry Strategies

### **1. Fixed Retry**

Wait same amount of time between retries

\`\`\`javascript
async function retryFixed(fn, maxRetries = 3, delay = 1000) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await sleep(delay); // Always wait 1 second
    }
  }
}

// Usage
await retryFixed(() => callAPI(), 3, 1000);
// Attempt 1 → fail → wait 1s
// Attempt 2 → fail → wait 1s
// Attempt 3 → fail → throw
\`\`\`

**Problems**:
- Doesn't adapt to server load
- Can overwhelm recovering service
- Retry storm risk

### **2. Exponential Backoff**

Exponentially increase wait time

\`\`\`javascript
async function retryExponential(fn, maxRetries = 5, baseDelay = 1000) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      
      const delay = baseDelay * Math.pow(2, i);
      await sleep(delay);
    }
  }
}

// Usage
await retryExponential(() => callAPI(), 5, 1000);
// Attempt 1 → fail → wait 1s (2^0)
// Attempt 2 → fail → wait 2s (2^1)
// Attempt 3 → fail → wait 4s (2^2)
// Attempt 4 → fail → wait 8s (2^3)
// Attempt 5 → fail → throw
\`\`\`

**Benefits**:
- Gives service time to recover
- Reduces load on struggling service
- Prevents retry storm

**Formula**: \`delay = baseDelay × 2^attempt\`

### **3. Exponential Backoff with Jitter**

Add randomness to prevent thundering herd

**The Problem**:
\`\`\`
1000 clients all failed at same time
→ All retry after exactly 1 second
→ All retry after exactly 2 seconds
→ Synchronized retry storm!
\`\`\`

**The Solution**: Add random jitter
\`\`\`javascript
async function retryExponentialJitter(fn, maxRetries = 5, baseDelay = 1000) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      
      const exponentialDelay = baseDelay * Math.pow(2, i);
      const jitter = Math.random() * exponentialDelay;
      const delay = exponentialDelay + jitter;
      
      await sleep(delay);
    }
  }
}

// Usage
await retryExponentialJitter(() => callAPI(), 5, 1000);
// Attempt 1 → fail → wait 1-2s (randomized)
// Attempt 2 → fail → wait 2-4s (randomized)
// Attempt 3 → fail → wait 4-8s (randomized)
\`\`\`

**Jitter Types**:

**Full Jitter** (recommended):
\`\`\`javascript
const delay = Math.random() * (baseDelay * Math.pow(2, attempt));
\`\`\`

**Equal Jitter**:
\`\`\`javascript
const temp = baseDelay * Math.pow(2, attempt);
const delay = temp / 2 + Math.random() * (temp / 2);
\`\`\`

**Decorrelated Jitter**:
\`\`\`javascript
const delay = Math.random() * (prevDelay * 3);
\`\`\`

**AWS Recommendation**: Use full jitter

### **4. Capped Exponential Backoff**

Limit maximum wait time

\`\`\`javascript
async function retryCapped(fn, maxRetries = 10, baseDelay = 100, maxDelay = 30000) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      
      const exponentialDelay = baseDelay * Math.pow(2, i);
      const jitter = Math.random() * exponentialDelay;
      const delay = Math.min(exponentialDelay + jitter, maxDelay);
      
      await sleep(delay);
    }
  }
}

// Usage
await retryCapped(() => callAPI());
// Delay capped at 30 seconds max
\`\`\`

**Why Cap?**
- Prevent waiting too long (hours)
- User experience
- Resource utilization

---

## Retry Budget

### **What is a Retry Budget?**

**Retry Budget**: Limit on total retries to prevent overwhelming system

**Problem Without Budget**:
\`\`\`
Service experiencing issues
→ 10,000 requests/sec
→ All retry 3 times
→ 30,000 requests/sec total
→ Service completely overwhelmed!
\`\`\`

**Solution: Retry Budget**:
\`\`\`javascript
class RetryBudget {
  constructor(maxRetryRate = 0.1) { // 10% retry rate
    this.maxRetryRate = maxRetryRate;
    this.attempts = 0;
    this.retries = 0;
  }
  
  canRetry() {
    this.attempts++;
    const currentRetryRate = this.retries / this.attempts;
    return currentRetryRate < this.maxRetryRate;
  }
  
  recordRetry() {
    this.retries++;
  }
}

// Usage
const budget = new RetryBudget(0.1);

async function callWithBudget(fn) {
  try {
    return await fn();
  } catch (error) {
    if (budget.canRetry()) {
      budget.recordRetry();
      return await fn(); // Retry
    } else {
      throw error; // Budget exhausted
    }
  }
}
\`\`\`

**Benefits**:
- Prevents retry storms
- Adapts to system health
- Protects struggling services

---

## Advanced Retry Patterns

### **1. Retry with Circuit Breaker**

Combine retry and circuit breaker

\`\`\`javascript
async function retryWithCircuitBreaker(fn, breaker, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await breaker.fire(fn);
    } catch (error) {
      if (error.name === 'CircuitBreakerOpen') {
        throw error; // Don't retry if circuit open
      }
      if (i === maxRetries - 1) throw error;
      await sleep(1000 * Math.pow(2, i));
    }
  }
}
\`\`\`

**Logic**:
- Retry on transient failures
- Stop retrying if circuit breaker opens
- Best of both worlds

### **2. Selective Retry**

Retry only specific errors

\`\`\`javascript
const RETRYABLE_ERRORS = [
  'ETIMEDOUT',
  'ECONNRESET',
  'ECONNREFUSED'
];

const RETRYABLE_STATUS = [
  429, // Too Many Requests
  500, // Internal Server Error
  502, // Bad Gateway
  503, // Service Unavailable
  504  // Gateway Timeout
];

async function retrySelective(fn, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      const shouldRetry = 
        RETRYABLE_ERRORS.includes(error.code) ||
        RETRYABLE_STATUS.includes(error.response?.status);
      
      if (!shouldRetry || i === maxRetries - 1) {
        throw error;
      }
      
      await sleep(1000 * Math.pow(2, i));
    }
  }
}
\`\`\`

### **3. Hedged Requests**

Send multiple requests, use first success

\`\`\`javascript
async function hedgedRequest(fn, hedgeDelay = 1000) {
  return Promise.race([
    fn(),
    sleep(hedgeDelay).then(() => fn()), // Hedged request
  ]);
}

// Usage
const result = await hedgedRequest(() => callSlowAPI(), 1000);
// Send initial request
// If no response after 1s, send second request
// Use whichever responds first
\`\`\`

**Use Case**: Tail latency reduction

**Trade-off**: 2x load

### **4. Adaptive Retry**

Adjust retry strategy based on success rate

\`\`\`javascript
class AdaptiveRetry {
  constructor() {
    this.successRate = 1.0;
    this.attempts = 0;
    this.successes = 0;
  }
  
  getMaxRetries() {
    if (this.successRate > 0.99) return 5; // High success, retry more
    if (this.successRate > 0.95) return 3;
    if (this.successRate > 0.80) return 2;
    return 1; // Low success, retry less
  }
  
  recordOutcome(success) {
    this.attempts++;
    if (success) this.successes++;
    this.successRate = this.successes / this.attempts;
  }
}
\`\`\`

---

## Implementation Libraries

### **Node.js**

**async-retry**:
\`\`\`javascript
const retry = require('async-retry');

await retry(
  async () => {
    return await callAPI();
  },
  {
    retries: 5,
    factor: 2,
    minTimeout: 1000,
    maxTimeout: 30000,
    randomize: true, // Jitter
  }
);
\`\`\`

**axios-retry**:
\`\`\`javascript
const axios = require('axios');
const axiosRetry = require('axios-retry');

axiosRetry(axios, {
  retries: 3,
  retryDelay: axiosRetry.exponentialDelay,
  retryCondition: (error) => {
    return axiosRetry.isNetworkOrIdempotentRequestError(error) ||
           error.response.status === 429;
  }
});
\`\`\`

### **Python**

**tenacity**:
\`\`\`python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True
)
def call_api():
    return requests.get('https://api.example.com')
\`\`\`

### **Java**

**Resilience4j**:
\`\`\`java
RetryConfig config = RetryConfig.custom()
    .maxAttempts(3)
    .waitDuration(Duration.ofSeconds(1))
    .retryOnException(e -> e instanceof TimeoutException)
    .build();

Retry retry = Retry.of("api", config);

Supplier<String> decorated = Retry.decorateSupplier(
    retry,
    () -> callAPI()
);
\`\`\`

---

## Best Practices

### **Do's**
✅ Use exponential backoff with jitter
✅ Set maximum retry limit (3-5 attempts)
✅ Cap maximum delay (30-60 seconds)
✅ Only retry idempotent operations (or use idempotency keys)
✅ Only retry transient failures (5xx, timeouts)
✅ Use retry budgets to prevent storms
✅ Log retry attempts
✅ Monitor retry rate metrics

### **Don'ts**
❌ Retry 4xx errors (except 429)
❌ Retry immediately without delay
❌ Unlimited retries
❌ Retry non-idempotent operations without safeguards
❌ Fixed delay without jitter
❌ Forget timeout on individual attempts
❌ Ignore retry amplification

---

## Monitoring Retries

### **Key Metrics**

**Retry Rate**:
\`\`\`
retries_total / requests_total
Target: < 5%
\`\`\`

**Success After Retry**:
\`\`\`
successful_retries / total_retries
Higher is better
\`\`\`

**Retry Amplification**:
\`\`\`
total_attempts / original_requests
If > 2x, you have a problem
\`\`\`

### **Alerts**

**High Retry Rate**:
\`\`\`
ALERT: Retry rate > 10%
Current: 15%
Service: payment-api
Action: Investigate service health
\`\`\`

**Retry Storm**:
\`\`\`
ALERT: Request volume 3x normal due to retries
Original: 1000 req/s
Actual: 3000 req/s
Action: Check retry configuration
\`\`\`

---

## Interview Tips

### **Key Concepts**

1. **Exponential Backoff**: Double delay each retry
2. **Jitter**: Add randomness to prevent thundering herd
3. **Idempotency**: Safe to retry
4. **Retry Budget**: Limit total retries
5. **Selective Retry**: Only retry 5xx and timeouts

### **Common Questions**

**Q: How would you implement retry logic for an API call?**
A: Use exponential backoff with jitter (1s, 2s, 4s), maximum 5 attempts, only retry on 5xx/timeouts, use idempotency keys for non-idempotent operations.

**Q: What's the thundering herd problem?**
A: When many clients retry at exact same time, overwhelming service. Solution: Add jitter (randomness) to retry delays.

**Q: Should you retry a 404 Not Found error?**
A: No, 404 is not transient. Retrying won't change the result.

**Q: How do you prevent retry storms?**
A: Exponential backoff with jitter, retry budgets, circuit breakers, maximum retry limits.

---

## Real-World Examples

### **AWS SDK**
- **Default**: Exponential backoff with jitter
- **Max Attempts**: 3
- **Jitter**: Full jitter
- **Lesson**: Battle-tested defaults

### **Google Cloud**
- **Retry**: Automatic for 5xx
- **Backoff**: Exponential
- **Cap**: 32 seconds max delay
- **Lesson**: Cap delays for user experience

### **Stripe**
- **Idempotency**: Required for payments
- **Retry**: Client responsibility
- **Recommendation**: Exponential backoff
- **Lesson**: Protect critical operations

---

## Summary

Retry logic with exponential backoff is essential for distributed systems:

1. **When to Retry**: 5xx errors, timeouts, network failures
2. **When NOT to Retry**: 4xx errors (except 429), non-transient failures
3. **Exponential Backoff**: Double delay each retry (1s, 2s, 4s, 8s)
4. **Jitter**: Add randomness to prevent thundering herd
5. **Idempotency**: Use keys for non-idempotent operations
6. **Retry Budget**: Limit total retries to prevent storms
7. **Circuit Breaker**: Combine with retries for maximum resilience

Done right, retries make systems more resilient. Done wrong, they make outages worse. Always use exponential backoff with jitter, cap maximum delays, and limit retry attempts.`,
};
