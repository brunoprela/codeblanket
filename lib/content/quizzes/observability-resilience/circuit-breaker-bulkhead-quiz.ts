/**
 * Quiz questions for Circuit Breaker & Bulkhead Patterns section
 */

export const circuitBreakerBulkheadQuiz = [
  {
    id: 'q1',
    question:
      'Explain how the Circuit Breaker pattern prevents cascading failures. Walk through what happens in each state (Closed, Open, Half-Open) with a concrete example.',
    sampleAnswer:
      'Circuit Breaker prevents cascading failures by failing fast when a dependency is unhealthy, rather than waiting for timeouts that exhaust resources. **The Cascading Failure Problem Without Circuit Breaker**: API → calls Database (down) → Each request waits 30s timeout → 100 requests = 100 threads blocked → Thread pool exhausted → API crashes → All users affected. **With Circuit Breaker - Three States**: **CLOSED (Normal)**: Requests pass through to database. Success → Stay closed. Failures → Increment counter. If failures > threshold (e.g., 5 consecutive) → Trip to OPEN. **OPEN (Failing Fast)**: All requests immediately return error (don\'t call database). No threads blocked waiting for timeout. After configured timeout (e.g., 30 seconds) → Transition to HALF-OPEN. **HALF-OPEN (Testing Recovery)**: Allow limited requests through (e.g., 1 request) to test if database recovered. If success → Close circuit (back to normal). If failure → Re-open circuit (back to OPEN). **Concrete Example - Payment Service**: Time 10:00: Database goes down. 10:00:05: Circuit Breaker CLOSED, first 5 requests fail (30s timeout each). 10:00:35: 5 failures threshold met, circuit OPENS. 10:00:36: New requests fail immediately with "Circuit Open" (no database calls, instant response). 10:00:40: User gets error in 50ms instead of waiting 30s. Threads freed immediately. 10:01:05: 30 seconds elapsed, circuit goes HALF-OPEN. 10:01:06: Single test request attempts database call. Scenario A (Database Still Down): Test request fails → Circuit re-opens → Wait another 30s. Scenario B (Database Back Up): Test request succeeds → Circuit closes → Normal operation resumes. **Benefits**: (1) Fail Fast: 50ms error instead of 30s timeout. (2) Resource Protection: Threads not blocked. (3) Auto-Recovery: Automatically tests and recovers. (4) User Experience: Immediate error instead of hang. **Fallback**: While open, return cached data or degraded functionality instead of error.',
    keyPoints: [
      'Prevents cascading failures by failing fast instead of waiting for timeouts',
      'CLOSED: Normal operation, counts failures',
      "OPEN: Fail immediately, don't call dependency, test after timeout",
      'HALF-OPEN: Test if dependency recovered, close if success',
      'Protects thread pools from exhaustion, enables auto-recovery',
    ],
  },
  {
    id: 'q2',
    question:
      'What is the Bulkhead pattern, and how does it complement the Circuit Breaker pattern? Provide an example where both are needed.',
    sampleAnswer:
      "Bulkhead pattern isolates resources into separate pools so failure in one area doesn't exhaust resources for other areas. Named after ship bulkheads (watertight compartments) that prevent entire ship from flooding. **The Problem Without Bulkheads**: Shared thread pool (50 threads) servicing three APIs: Payment API (slow): Uses 45 threads, all waiting on slow database. User API: Needs 5 threads, gets 5. Search API: Needs 10 threads, gets 0 (starved!) → FAILS. Result: One slow API blocks unrelated APIs. **With Bulkheads**: Partition resources into isolated pools: Payment API Pool: 20 threads dedicated. User API Pool: 20 threads dedicated. Search API Pool: 10 threads dedicated. Result: Payment slowness uses only its 20 threads. Search still has 10 threads available, works fine. **Circuit Breaker vs Bulkhead**: Circuit Breaker: Protects from calling failing dependencies. Bulkhead: Protects from resource exhaustion. **Why You Need Both - Example**: E-commerce service calling: Payment Service (external), User Service (internal database), Search Service (Elasticsearch). **Without Protection**: Payment service goes down → Circuit breaker opens (good!) But before opening, exhausted 45 of 50 threads → User/Search APIs starved (bad!). **With Circuit Breaker Only**: Payment service down → Circuit breaker opens after 5 failures. But 5 failures × 30s timeout = 150 thread-seconds wasted. Other APIs degraded during those 150 seconds. **With Bulkhead Only**: Payment service gets 20 threads max. But all 20 threads blocked on slow Payment service. Payment API still timing out for users. **With Both**: Bulkhead: Payment service can only use 20 threads (isolated). Circuit Breaker: After 5 failures, circuit opens, payment requests fail fast. Result: Payment service failures isolated (bulkhead) and quick to detect (circuit breaker). User/Search APIs completely unaffected. **Implementation**: Code: paymentBreaker = new Circuit Breaker(config); paymentBulkhead = new Bulkhead(20); Request flow: req → bulkhead.execute(() → breaker.fire(() → callPaymentAPI()))",
    keyPoints: [
      'Bulkhead: Isolates resources (thread pools) to prevent starvation',
      'Circuit Breaker: Fails fast when dependency unhealthy',
      'Bulkhead prevents: One slow service exhausting all resources',
      'Both together: Isolation (bulkhead) + fail fast (circuit breaker)',
      'Use bulkhead for all external dependencies to prevent cross-contamination',
    ],
  },
  {
    id: 'q3',
    question:
      'How would you configure circuit breaker parameters (failure threshold, timeout, success threshold) for a production service? Walk through your reasoning.',
    sampleAnswer:
      "Circuit breaker configuration balances fast failure detection vs avoiding false positives. **Key Parameters**: (1) **Failure Threshold**: Failures before opening circuit. (2) **Timeout**: How long to stay open before testing. (3) **Success Threshold**: Successes needed in half-open to close. **Configuration Process**: **Step 1: Analyze Current Performance**: Measure service: Normal success rate: 99.5%, Normal latency: 100ms p99, Occasional transient failures: 1-2 per minute. **Step 2: Set Failure Threshold**: Too Low (2 failures): Opens on transient blips, false positives. Too High (100 failures): Opens too late, many users affected. **Sweet Spot (5-10 failures)**: Filters single transient errors. Detects sustained problems quickly. For 100 req/s service, 5 failures = 50ms of bad traffic. **Recommendation**: failureThreshold: 5 consecutive failures OR 50% failure rate in sliding window of 100 requests. **Step 3: Set Timeout (Open Duration)**: Too Short (5 seconds): Doesn't give dependency time to recover, constant testing. Too Long (5 minutes): Prolongs outage unnecessarily. **Sweet Spot (30-60 seconds)**: Most services recover in < 1 minute (restart, scale, rollback). Balances recovery time vs testing overhead. **Recommendation**: timeout: 30 seconds for internal services, 60 seconds for external services (less control over recovery). **Step 4: Set Success Threshold (Half-Open)**: Too Low (1 success): Might close on lucky success, still unstable. Too High (10 successes): Takes too long to recover, conservative. **Sweet Spot (1-2 successes)**: Fast recovery when actually healthy. Multiple successes confirm stability. **Recommendation**: successThreshold: 1 success for fast recovery, 2 successes for cautious approach. **Example Production Config**: Internal Database Call: failureThreshold: 5 consecutive OR 50% in window of 100, timeout: 30 seconds, successThreshold: 1, requestTimeout: 5 seconds (don't wait forever per request). External Payment API: failureThreshold: 3 consecutive (less tolerance for payment failures), timeout: 60 seconds (external, less control), successThreshold: 2 (be cautious with payments), requestTimeout: 10 seconds. **Step 5: Monitor and Tune**: Track: Circuit open frequency, Time in open state, False positives (opens when service actually healthy). Adjust: If opening too often on transients → increase failure threshold. If taking too long to detect failures → decrease threshold.",
    keyPoints: [
      'Failure threshold: 5-10 failures (balances transient vs sustained)',
      'Timeout: 30-60 seconds (enough to recover, not too long)',
      'Success threshold: 1-2 successes (fast recovery, confirm stability)',
      'Tune based on service: Internal (faster) vs external (slower)',
      'Monitor and adjust: Track open frequency, false positives',
    ],
  },
];
