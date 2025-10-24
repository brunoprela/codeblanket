/**
 * Quiz questions for Retry Logic & Exponential Backoff section
 */

export const retryLogicQuiz = [
  {
    id: 'q1',
    question:
      'Explain exponential backoff with jitter. Why is jitter critical, and what problem does it solve? Provide examples with and without jitter.',
    sampleAnswer:
      'Exponential backoff increases wait time exponentially between retries (1s, 2s, 4s, 8s), while jitter adds randomness to prevent thundering herd problem. **Exponential Backoff Without Jitter**: Attempt 1 → fail → wait 1s, Attempt 2 → fail → wait 2s, Attempt 3 → fail → wait 4s, Attempt 4 → fail → wait 8s. **Problem - Thundering Herd**: 1000 clients all fail at exact same time (e.g., service restart) → All retry after exactly 1 second (synchronized) → Service gets 1000 requests simultaneously → Overwhelmed → All fail again → All retry after exactly 2 seconds → Still synchronized! This creates retry storm that prevents recovery. **Exponential Backoff With Jitter**: Add randomness to delay. **Full Jitter (Recommended)**: delay = random(0, baseDelay × 2^attempt). Attempt 1 → wait 0-2s (random), Attempt 2 → wait 0-4s (random), Attempt 3 → wait 0-8s (random). **With 1000 Clients**: Time 0: All 1000 fail. Time 0-2s: Clients retry randomly spread across 2 seconds (not synchronized!). Time 0-4s: Second attempts spread across 4 seconds. Result: Load distributed evenly, service can recover. **Real-World Example Without Jitter**: Database goes down for 1 minute → 10,000 requests queued → Database restarts → All 10,000 retry simultaneously after 1s → Database crashes from load → Repeat cycle. **With Jitter**: Same scenario → 10,000 retries spread across 2 seconds → Database gets 5,000 req/s (manageable) instead of 10,000 simultaneously → Recovers successfully. **Jitter Types**: Full Jitter: random(0, exponentialDelay) - Most spread, AWS recommendation. Equal Jitter: exponentialDelay/2 + random(0, exponentialDelay/2) - Balanced. Decorrelated Jitter: random(0, previousDelay × 3) - Adapts to patterns. **Implementation**: Without jitter: delay = 1000 × Math.pow(2, attempt); With jitter: delay = Math.random() × (1000 × Math.pow(2, attempt)). **Best Practice**: Always use jitter in production systems. Cost is negligible (few lines of code). Benefit is enormous (prevents retry storms).',
    keyPoints: [
      'Exponential backoff: Double delay each retry (1s, 2s, 4s, 8s)',
      'Jitter: Add randomness to prevent synchronized retries',
      'Without jitter: Thundering herd (all retry at same time)',
      'With jitter: Retries spread over time, enables recovery',
      'Full jitter (random(0, delay)) is AWS recommendation',
    ],
  },
  {
    id: 'q2',
    question:
      'What is idempotency, why is it important for retries, and how do you make non-idempotent operations safe to retry?',
    sampleAnswer:
      'Idempotency means an operation produces the same result no matter how many times it\'s executed. Critical for retries because network failures can cause duplicate requests. **Idempotent Operations (Safe to Retry)**: GET /users/123 → Always returns same user, no side effects. PUT /users/123 {name: "Alice"} → Set name to Alice (running 10 times still results in name = "Alice"). DELETE /users/123 → Delete user (running 10 times still results in user deleted). **Non-Idempotent Operations (NOT Safe to Retry)**: POST /users {name: "Alice"} → Creates new user each time (10 retries = 10 users!). POST /payments {amount: 100} → Charges $100 each time (10 retries = $1000 charge!). POST /increment-counter → Increments each time (10 retries = counter += 10). **The Problem**: Client sends: POST /payments {amount: 100}. Request succeeds on server, but response lost in network. Client sees timeout, retries → User charged $200! **Solution 1: Idempotency Keys**: Client generates unique key per logical operation: Key: "purchase-order-abc-123" (same key for all retries of same purchase). Request: POST /payments {amount: 100} Headers: {"Idempotency-Key": "purchase-order-abc-123"}. Server logic: if (cache.has(idempotencyKey)) return cache.get(idempotencyKey); // Return cached result, result = processPayment(request); cache.set(idempotencyKey, result, ttl=24hours); return result; First request: Processes payment, caches result. Retry requests: Returns cached result (idempotent!). **Solution 2: Natural Idempotency**: Use PUT instead of POST with client-generated ID: POST /payments → PUT /payments/order-abc-123 {amount: 100}. Server uses order ID as unique constraint. Duplicate requests update same record (idempotent). **Real-World Examples**: Stripe: Requires idempotency keys for all POST requests. AWS: Uses idempotency tokens for EC2 operations. Shopify: Idempotency keys for order creation. **Key Management**: Client generates UUID for each logical operation. Keeps same key for all retries of that operation. Server stores keys for 24 hours. After 24 hours, key expires (client must generate new key). **Best Practice**: Always use idempotency keys for financial operations. For other non-idempotent operations, evaluate risk vs complexity.',
    keyPoints: [
      'Idempotent: Same result no matter how many times executed',
      'Safe to retry: GET, PUT, DELETE (idempotent)',
      'NOT safe to retry: POST (non-idempotent, creates duplicate)',
      'Solution: Idempotency keys - cache results, return cached for duplicates',
      'Used by: Stripe, AWS, Shopify for financial/critical operations',
    ],
  },
  {
    id: 'q3',
    question:
      'When should you NOT retry a request? Provide examples of errors that should not be retried and explain why retry budgets are important.',
    sampleAnswer:
      "Not all failures should be retried. Retrying non-transient failures wastes resources and can make problems worse. **NEVER Retry These**: **4xx Client Errors (except 429)**: 400 Bad Request: Invalid input, retry won't fix it. 401 Unauthorized: Invalid credentials, retry won't help. 403 Forbidden: Lack permission, retry won't grant it. 404 Not Found: Resource doesn't exist, retry won't create it. 409 Conflict: Business logic error (e.g., duplicate email), retry won't resolve. Why: These are permanent failures. Fix requires client to change request. **Non-Transient Errors**: Application Logic Errors: \"Insufficient funds\" for payment, \"Account locked\", \"Invalid discount code\". Why: Retry won't fix business rule violations. **SHOULD Retry These**: **5xx Server Errors**: 500 Internal Server Error (might be transient crash). 502 Bad Gateway (upstream temporarily down). 503 Service Unavailable (overloaded, restarting). 504 Gateway Timeout (temporary slowness). **Network Errors**: Connection timeout (temporary network issue). Connection refused (service restarting). DNS resolution failure (temporary). **Special Case - 429 Too Many Requests**: Should retry with backoff. Respect Retry-After header if provided. Use exponential backoff to avoid overwhelming service. **Retry Budgets - Why Important**: **Problem Without Budget**: Service experiencing issues → All 10,000 req/s fail → All retry 3 times → 30,000 req/s total load → Service completely overwhelmed, can't recover → Retry storm. **Solution - Retry Budget**: Limit: Max 10% of requests can be retries. Example: 10,000 req/s, max 1,000 retries/s allowed. If retry rate exceeds budget, stop retrying new requests. Implementation: track retries/second, if > budget, fail fast without retry. **Benefits**: (1) Prevents retry amplification (retries making problem worse). (2) Allows service to recover (not overwhelmed by retries). (3) Protects overall system stability. **Real Example**: Netflix: Strict retry budgets to prevent cascading failures. AWS SDK: Limited retry attempts with exponential backoff. Google: Retry budget as percentage of error budget. **Best Practice**: Only retry 5xx, timeouts, network errors. Never retry 4xx (except 429). Implement retry budget (10-20% of traffic). Use exponential backoff with jitter. Set max retry attempts (3-5).",
    keyPoints: [
      'NEVER retry: 4xx client errors (permanent failures)',
      'SHOULD retry: 5xx server errors, timeouts, network errors',
      'Exception: Retry 429 with backoff',
      'Retry budget: Limit retries to 10-20% of traffic',
      'Prevents retry amplification (retries overwhelming failing service)',
    ],
  },
];
