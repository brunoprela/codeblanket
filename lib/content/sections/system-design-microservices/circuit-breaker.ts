/**
 * Circuit Breaker Pattern Section
 */

export const circuitbreakerSection = {
  id: 'circuit-breaker',
  title: 'Circuit Breaker Pattern',
  content: `The Circuit Breaker pattern prevents cascading failures in distributed systems by failing fast when a service is down, rather than waiting for timeouts.

## The Problem: Cascading Failures

**Scenario**: Payment Service is down

**Without Circuit Breaker**:
\`\`\`
1. User clicks "Checkout"
2. Order Service calls Payment Service
3. Payment Service is down
4. Order Service waits 30 seconds (timeout)
5. Returns error to user
6. 100 users click checkout simultaneously
7. 100 threads in Order Service blocked for 30 seconds
8. Order Service runs out of threads → crashes
9. Now both Order and Payment Service are down! 💥
\`\`\`

This is a **cascading failure** - one service failure brings down others.

**With Circuit Breaker**:
\`\`\`
1. First few requests to Payment Service fail
2. Circuit breaker "opens" - stops calling Payment Service
3. Subsequent requests fail immediately (no waiting)
4. Order Service stays healthy
5. After timeout, circuit breaker "half-opens" - allows test request
6. If test succeeds, circuit "closes" - normal operation resumes
\`\`\`

---

## Circuit Breaker States

Circuit breaker acts like an electrical circuit breaker.

### State Machine:

\`\`\`
        ┌──────────┐
        │  CLOSED  │  (Normal operation)
        │  ✅      │
        └────┬─────┘
             │ Failures exceed threshold
             ▼
        ┌──────────┐
        │   OPEN   │  (Fail fast)
        │  ❌      │
        └────┬─────┘
             │ After timeout period
             ▼
        ┌──────────┐
        │HALF-OPEN │  (Testing)
        │  ⚠️      │
        └────┬─────┘
          │      │
    Success│      │More failures
          │      │
          ▼      ▼
      CLOSED   OPEN
\`\`\`

### 1. CLOSED (Normal Operation)

**Behavior**: All requests pass through to service

**Tracking**: Count failures

**Transition**: If failures exceed threshold → OPEN

\`\`\`javascript
// Example: 5 failures in 10 seconds
if (failureCount >= 5 && timeWindow <= 10s) {
    state = OPEN;
}
\`\`\`

### 2. OPEN (Failing Fast)

**Behavior**: All requests fail immediately (don't call service)

**Response**: Return error or fallback value

**Transition**: After timeout (e.g., 30 seconds) → HALF_OPEN

\`\`\`javascript
// Fail immediately
if (state === OPEN) {
    throw new Error('Circuit breaker open');
}
\`\`\`

### 3. HALF_OPEN (Testing)

**Behavior**: Allow limited requests through (test if service recovered)

**Transition**: 
- If test succeeds → CLOSED (resume normal operation)
- If test fails → OPEN (back to failing fast)

\`\`\`javascript
if (state === HALF_OPEN) {
    if (testRequest succeeds) {
        state = CLOSED;
        resetFailureCount();
    } else {
        state = OPEN;
        startTimeout();
    }
}
\`\`\`

---

## Implementation

### Basic Circuit Breaker

\`\`\`javascript
class CircuitBreaker {
    constructor (options = {}) {
        this.failureThreshold = options.failureThreshold || 5;
        this.timeout = options.timeout || 60000; // 60 seconds
        this.resetTimeout = options.resetTimeout || 30000; // 30 seconds
        
        this.state = 'CLOSED';
        this.failureCount = 0;
        this.nextAttempt = Date.now();
        this.successCount = 0;
    }
    
    async call (fn) {
        // Check state
        if (this.state === 'OPEN') {
            if (Date.now() < this.nextAttempt) {
                throw new Error('Circuit breaker is OPEN');
            }
            // Time to try again
            this.state = 'HALF_OPEN';
        }
        
        try {
            // Execute function
            const result = await this.executeWithTimeout (fn, this.timeout);
            
            // Success!
            this.onSuccess();
            return result;
            
        } catch (error) {
            // Failure
            this.onFailure();
            throw error;
        }
    }
    
    async executeWithTimeout (fn, timeout) {
        return Promise.race([
            fn(),
            new Promise((_, reject) => 
                setTimeout(() => reject (new Error('Timeout')), timeout)
            )
        ]);
    }
    
    onSuccess() {
        this.failureCount = 0;
        
        if (this.state === 'HALF_OPEN') {
            this.successCount++;
            // Require N successes to fully close
            if (this.successCount >= 3) {
                this.state = 'CLOSED';
                this.successCount = 0;
            }
        }
    }
    
    onFailure() {
        this.failureCount++;
        this.successCount = 0;
        
        if (this.failureCount >= this.failureThreshold) {
            this.state = 'OPEN';
            this.nextAttempt = Date.now() + this.resetTimeout;
        }
    }
    
    getState() {
        return this.state;
    }
}
\`\`\`

### Usage

\`\`\`javascript
const paymentServiceBreaker = new CircuitBreaker({
    failureThreshold: 5,     // Open after 5 failures
    timeout: 3000,           // Request timeout: 3s
    resetTimeout: 30000      // Try again after 30s
});

async function chargePayment (paymentData) {
    try {
        return await paymentServiceBreaker.call (async () => {
            return await paymentService.charge (paymentData);
        });
    } catch (error) {
        if (error.message === 'Circuit breaker is OPEN') {
            // Handle gracefully
            return { status: 'PENDING', message: 'Payment service temporarily unavailable' };
        }
        throw error;
    }
}
\`\`\`

---

## Fallback Strategies

When circuit breaker is OPEN, what do you return?

### 1. Default Value

Return sensible default.

\`\`\`javascript
async function getRecommendations (userId) {
    try {
        return await circuitBreaker.call(() => 
            recommendationService.get (userId)
        );
    } catch (error) {
        // Fallback: return popular products
        return await productService.getPopular();
    }
}
\`\`\`

### 2. Cached Value

Return last known good value.

\`\`\`javascript
const cache = new Map();

async function getProductPrice (productId) {
    try {
        const price = await circuitBreaker.call(() => 
            pricingService.getPrice (productId)
        );
        cache.set (productId, price);  // Update cache
        return price;
    } catch (error) {
        // Fallback: return cached price
        const cachedPrice = cache.get (productId);
        if (cachedPrice) {
            return { ...cachedPrice, stale: true };
        }
        throw error;
    }
}
\`\`\`

### 3. Degraded Functionality

Reduce features rather than complete failure.

\`\`\`javascript
async function getProductDetails (productId) {
    const product = await productService.get (productId);
    
    // Try to get recommendations (non-critical)
    try {
        product.recommendations = await circuitBreaker.call(() =>
            recommendationService.getRelated (productId)
        );
    } catch (error) {
        // Degrade gracefully - product still works without recommendations
        product.recommendations = [];
    }
    
    return product;
}
\`\`\`

### 4. Queue for Later

Queue request for processing when service recovers.

\`\`\`javascript
async function sendNotification (notification) {
    try {
        return await circuitBreaker.call(() => 
            notificationService.send (notification)
        );
    } catch (error) {
        // Fallback: queue for later
        await notificationQueue.enqueue (notification);
        return { status: 'QUEUED' };
    }
}
\`\`\`

---

## Real-World Example: Netflix Hystrix

**Hystrix** is Netflix\'s circuit breaker library (now in maintenance mode, but concepts remain).

**Features**:
- Circuit breaker
- Timeouts
- Fallbacks
- Request collapsing
- Real-time monitoring dashboard

**Example (Java)**:
\`\`\`java
@HystrixCommand(
    fallbackMethod = "getDefaultRecommendations",
    commandProperties = {
        @HystrixProperty (name = "execution.isolation.thread.timeoutInMilliseconds", value = "3000"),
        @HystrixProperty (name = "circuitBreaker.requestVolumeThreshold", value = "20"),
        @HystrixProperty (name = "circuitBreaker.errorThresholdPercentage", value = "50"),
        @HystrixProperty (name = "circuitBreaker.sleepWindowInMilliseconds", value = "60000")
    }
)
public List<Movie> getRecommendations(String userId) {
    return recommendationService.getForUser (userId);
}

public List<Movie> getDefaultRecommendations(String userId) {
    // Fallback: return popular movies
    return movieService.getPopular();
}
\`\`\`

**Configuration Explained**:
- timeout: 3 seconds
- requestVolumeThreshold: Minimum 20 requests before circuit can open
- errorThresholdPercentage: Open circuit if 50% of requests fail
- sleepWindowInMilliseconds: Wait 60s before trying again (HALF_OPEN)

---

## Modern Alternative: Resilience4j

**Resilience4j** is a lightweight alternative to Hystrix.

**Features**:
- Circuit breaker
- Rate limiter
- Bulkhead (limit concurrent calls)
- Retry
- Time limiter

**Example (Java)**:
\`\`\`java
CircuitBreakerConfig config = CircuitBreakerConfig.custom()
    .failureRateThreshold(50)                    // 50% failure rate
    .waitDurationInOpenState(Duration.ofSeconds(30))  // Wait 30s
    .slidingWindowSize(20)                       // Last 20 requests
    .build();

CircuitBreaker circuitBreaker = CircuitBreaker.of("paymentService", config);

// Decorate function
Supplier<PaymentResult> decoratedSupplier = CircuitBreaker
    .decorateSupplier (circuitBreaker, () -> paymentService.charge (payment));

// Execute
try {
    PaymentResult result = decoratedSupplier.get();
} catch (CallNotPermittedException e) {
    // Circuit breaker is open
    return handleFallback();
}
\`\`\`

**JavaScript equivalent** (using opossum library):
\`\`\`javascript
const CircuitBreaker = require('opossum');

const options = {
    timeout: 3000,              // 3 second timeout
    errorThresholdPercentage: 50,  // Open at 50% failure
    resetTimeout: 30000         // Try again after 30s
};

const breaker = new CircuitBreaker (paymentService.charge, options);

// Add fallback
breaker.fallback(() => ({status: 'PENDING', message: 'Service unavailable'}));

// Add listeners
breaker.on('open', () => console.log('Circuit opened!'));
breaker.on('halfOpen', () => console.log('Circuit half-open, testing...'));
breaker.on('close', () => console.log('Circuit closed!'));

// Use it
const result = await breaker.fire (paymentData);
\`\`\`

---

## Monitoring Circuit Breakers

**Metrics to track**:
1. **Circuit state**: CLOSED, OPEN, HALF_OPEN
2. **Success rate**: % of successful requests
3. **Failure rate**: % of failed requests
4. **Latency**: Response time distribution (p50, p95, p99)
5. **Timeout rate**: % of requests that timeout
6. **Fallback rate**: % of requests using fallback

**Dashboard Example**:
\`\`\`
Service: payment-service
State: OPEN ❌
Failures: 47 / 50 (94%)
Uptime: 12min since last transition
Next attempt: in 18 seconds

Last 100 requests:
✅✅✅❌❌❌❌❌❌❌ ... (mostly failures)

Fallback responses: 235
\`\`\`

**Alerting**:
\`\`\`
ALERT: Circuit breaker for payment-service has been OPEN for > 5 minutes
Action: Investigate payment-service health
\`\`\`

---

## Circuit Breaker vs Retry

**Circuit Breaker**: Stop calling failing service
**Retry**: Try again on failure

**They complement each other!**

\`\`\`javascript
// Retry THEN circuit breaker
const retryPolicy = {
    maxRetries: 3,
    backoff: 'exponential'
};

async function callServiceWithResilience (data) {
    return await circuitBreaker.call (async () => {
        return await retryWithBackoff (async () => {
            return await service.call (data);
        }, retryPolicy);
    });
}
\`\`\`

**When to use each**:

| Pattern | Use Case |
|---------|----------|
| **Retry** | Transient failures (network blip) |
| **Circuit Breaker** | Service down or degraded |
| **Both** | Retry for transient issues, circuit breaker to stop retry storm |

---

## Best Practices

### 1. Tune Thresholds

Don't use default values blindly.

**Consider**:
- **Request volume**: Don't open circuit on 2 failures if you only get 10 req/min
- **Error rate**: 50% might be too aggressive for some services
- **Recovery time**: How long does service need to recover?

**Example**:
\`\`\`javascript
// Critical service - more lenient
const paymentBreaker = new CircuitBreaker({
    failureThreshold: 10,      // Higher threshold
    resetTimeout: 10000        // Retry sooner
});

// Non-critical service - stricter
const recommendationBreaker = new CircuitBreaker({
    failureThreshold: 3,       // Lower threshold
    resetTimeout: 60000        // Wait longer
});
\`\`\`

### 2. Bulkheading

Isolate circuit breakers per dependency.

❌ **Bad**: Single circuit breaker for entire service
\`\`\`javascript
// All external calls share one breaker
const breaker = new CircuitBreaker();
await breaker.call(() => paymentService.charge());
await breaker.call(() => emailService.send());
\`\`\`

**Problem**: Email service failure opens circuit for payment too!

✅ **Good**: Separate breakers per dependency
\`\`\`javascript
const paymentBreaker = new CircuitBreaker();
const emailBreaker = new CircuitBreaker();

await paymentBreaker.call(() => paymentService.charge());
await emailBreaker.call(() => emailService.send());
\`\`\`

### 3. Fail Fast

Don't retry when circuit is open.

❌ **Bad**:
\`\`\`javascript
for (let i = 0; i < 3; i++) {
    try {
        return await breaker.call (fn);
    } catch (error) {
        // Circuit is open, but we keep retrying! 🤦
    }
}
\`\`\`

✅ **Good**:
\`\`\`javascript
try {
    return await breaker.call (fn);
} catch (error) {
    if (error.message === 'Circuit breaker is OPEN') {
        return fallback();  // Don't retry
    }
    throw error;
}
\`\`\`

### 4. Monitor and Alert

Circuit breaker state changes are important signals.

\`\`\`javascript
breaker.on('open', () => {
    logger.error('Circuit breaker opened for payment-service');
    metrics.increment('circuit_breaker.opened');
    alerting.notify('Payment service is down!');
});

breaker.on('halfOpen', () => {
    logger.info('Circuit breaker half-open, testing payment-service');
});

breaker.on('close', () => {
    logger.info('Circuit breaker closed, payment-service recovered');
    metrics.increment('circuit_breaker.closed');
});
\`\`\`

---

## Interview Tips

**Red Flags**:
❌ Not knowing about cascading failures
❌ Thinking circuit breaker is same as retry
❌ Not mentioning fallback strategies

**Good Responses**:
✅ Explain the three states (CLOSED, OPEN, HALF_OPEN)
✅ Discuss cascading failure prevention
✅ Mention fallback strategies
✅ Talk about monitoring

**Sample Answer**:
*"I'd implement circuit breakers to prevent cascading failures. When Payment Service fails repeatedly (e.g., 5 failures in 10 seconds), the circuit breaker opens and fails fast instead of waiting for timeouts. This prevents Order Service from running out of threads. The circuit stays open for 30 seconds, then enters half-open state to test if the service recovered. For fallbacks, I'd queue payment requests or show 'Payment pending' to users. We'd monitor circuit breaker state changes and alert on-call engineers when circuits open. I'd use a library like Resilience4j (Java) or Opossum (Node.js) rather than building from scratch."*

---

## Key Takeaways

1. **Circuit breaker** prevents cascading failures by failing fast
2. **Three states**: CLOSED (normal), OPEN (failing fast), HALF_OPEN (testing)
3. **Fail fast**: Don't wait for timeout when service is known to be down
4. **Fallbacks**: Default values, cached data, degraded functionality, or queuing
5. **Monitor**: Track state changes, failure rates, fallback usage
6. **Tune thresholds**: Based on traffic patterns and service criticality
7. **Bulkhead**: Separate circuit breakers per dependency
8. **Use with retry**: Retry for transient failures, circuit breaker for sustained outages`,
};
