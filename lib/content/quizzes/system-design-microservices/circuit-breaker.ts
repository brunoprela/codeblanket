/**
 * Quiz questions for Circuit Breaker Pattern section
 */

export const circuitbreakerQuiz = [
  {
    id: 'q1-circuit',
    question:
      'Explain how circuit breakers prevent cascading failures. What would happen WITHOUT a circuit breaker when a downstream service fails?',
    sampleAnswer:
      "Without circuit breaker: When Payment Service goes down, Order Service keeps trying to call it. Each request waits for timeout (e.g., 30 seconds). With 100 concurrent users, 100 threads get blocked waiting. Order Service runs out of threads → can't handle new requests → crashes. Now both services are down (cascading failure). WITH circuit breaker: After N failures, circuit opens → subsequent requests fail immediately (no waiting) → Order Service stays healthy → circuit half-opens after timeout → tests if Payment Service recovered → closes if test succeeds. Circuit breaker isolates failure to one service, prevents thread pool exhaustion, and allows graceful degradation.",
    keyPoints: [
      'Without CB: threads block waiting for timeout → thread pool exhaustion',
      'Cascading failure: one service failure brings down others',
      'Circuit breaker fails fast → no thread blocking',
      'Allows graceful degradation with fallbacks',
      'Auto-recovery via half-open state testing',
    ],
  },
  {
    id: 'q2-circuit',
    question:
      'Describe the three states of a circuit breaker and the transitions between them.',
    sampleAnswer:
      'CLOSED (normal): All requests pass through. Track failures. Transition: If failures >= threshold → OPEN. OPEN (failing fast): All requests fail immediately without calling service. Saves resources. Transition: After timeout period → HALF_OPEN. HALF_OPEN (testing): Allow limited test requests through. Transition: If test succeeds → CLOSED (resume normal), if fails → OPEN (back to failing fast). Example: threshold=5 failures, timeout=30s. After 5 failures, circuit opens. For 30 seconds, all requests fail fast. Then circuit half-opens, sends test request. If test succeeds, circuit closes; otherwise back to open for another 30s.',
    keyPoints: [
      'CLOSED: normal operation, tracking failures',
      'OPEN: fail fast, no calls to service',
      'HALF_OPEN: testing recovery with limited requests',
      'Transitions based on failure threshold and timeout',
      'Auto-recovery mechanism prevents manual intervention',
    ],
  },
  {
    id: 'q3-circuit',
    question:
      'What are fallback strategies for when a circuit breaker is OPEN? Give examples for different scenarios.',
    sampleAnswer:
      'Fallback strategies: (1) Default/Popular values - Recommendation service down → return popular products instead of personalized. (2) Cached data - Pricing service down → return last known prices (mark as potentially stale). (3) Degraded functionality - Product page works but without recommendations. (4) Queue for later - Notification service down → queue emails for retry when service recovers. (5) Error with retry - Critical operation → return error, ask user to try again. Choose based on: Is operation critical? (payment = yes, recommendations = no), Can we show stale data? (prices = maybe with disclaimer, inventory = risky), Can we defer? (notifications = yes, checkout = no).',
    keyPoints: [
      'Default/popular values for non-critical features',
      'Cached data with staleness indicators',
      'Degraded functionality (reduce features)',
      'Queue for asynchronous operations',
      'Choose based on criticality and data freshness requirements',
    ],
  },
];
