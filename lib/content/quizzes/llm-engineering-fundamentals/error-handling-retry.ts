/**
 * Quiz questions for Error Handling & Retry Logic section
 */

export const errorhandlingretryQuiz = [
  {
    id: 'q1',
    question:
      'Explain why exponential backoff with jitter is superior to simple fixed-delay retries. Provide a concrete example showing how jitter prevents the "thundering herd" problem.',
    sampleAnswer:
      'Exponential backoff with jitter solves two critical problems: (1) Exponential backoff (1s, 2s, 4s, 8s) gives the API time to recover from overload - if rate limited, waiting 1 second and immediately retrying likely hits the same limit. Doubling delay gives progressively more recovery time. (2) Jitter (random variance) prevents thundering herd. Concrete example: Imagine 1000 clients hit rate limit simultaneously at t=0. With fixed 5s retry, all 1000 retry at t=5, causing another thundering herd. With exponential backoff but no jitter, all retry at t=1, t=2, t=4 - still synchronized spikes. With jitter: Client A waits 1s + random(0-1s) = 1.3s, Client B waits 1s + random(0-1s) = 1.7s, etc. This spreads the 1000 retries across 1-2s window instead of hitting all at once. On second retry, they are already desynchronized and spread further (2-4s window). Mathematical impact: Without jitter, 1000 requests become 1000-request spike. With jitter, 1000 requests spread to ~100 requests/second over 10s - 10x reduction in peak load. This prevents re-triggering rate limits. Implementation: wait_time = base_delay * (2 ** attempt) + random.uniform(0, base_delay). The random component (0 to base_delay) provides enough variance to desynchronize while keeping waits reasonable. Production evidence: AWS and Google both recommend exponential backoff with jitter - it is the proven pattern for distributed systems. Systems without jitter often see cascading failures where retries cause more rate limits in an endless loop.',
    keyPoints: [
      'Exponential backoff gives API time to recover',
      'Jitter prevents synchronized retry spikes',
      'Without jitter, all clients retry simultaneously',
      'Spreads load over time instead of peaks',
      'Industry standard for reliable retry logic',
    ],
  },
  {
    id: 'q2',
    question:
      'You are implementing retry logic for LLM API calls. Which errors should be retried and which should fail immediately? Explain your reasoning for each category.',
    sampleAnswer:
      "RETRY IMMEDIATELY: (1) 429 Rate Limit - temporary, will succeed after backoff; use exponential backoff (1s, 2s, 4s, 8s, 16s). (2) 503 Service Unavailable - API temporarily overloaded; retry with backoff, API will recover. (3) 500 Internal Server Error - transient server issues; might work on retry, use backoff. (4) Timeout errors - network or processing timeout; might succeed with retry, but reduce timeout gradually. (5) Connection errors - network interruption; retry immediately once, then with backoff. FAIL IMMEDIATELY (NO RETRY): (1) 401 Unauthorized - invalid API key; retrying will never succeed, requires fixing the key. User action needed. (2) 400 Bad Request - malformed input; same request will always fail, need to fix request parameters. Retrying wastes time and money. (3) 403 Forbidden - account suspended or permission issue; requires account-level fix, not transient. (4) 413 Payload Too Large - request exceeds limits; need to reduce size, retry will always fail. (5) Content Policy Violations - content filtered; need to modify content, retry unchanged content will fail. CONTEXT-DEPENDENT: (1) 404 Not Found - might be model name typo (don't retry) or temporary unavailability(retry once).Check error message. (2) 408 Request Timeout - if client timeout, increase timeout and retry; if server timeout, might indicate too complex request(don't retry endlessly). Implementation: Classify errors into transient vs permanent, only retry transient with max 3-5 attempts, for permanent errors, return clear message to user about what to fix, log all non-retried errors for debugging. This prevents wasting API calls on requests that will never succeed while ensuring transient issues are handled gracefully.",
    keyPoints: [
      'Retry transient errors (rate limits, timeouts, server errors)',
      'Never retry auth, bad requests, or policy violations',
      'Permanent errors need user intervention, not retries',
      'Classify errors to avoid wasting API calls',
      'Clear error messages for non-retried failures',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe how you would implement a circuit breaker pattern for LLM API calls. What metrics would you track, and how would you determine when to open and close the circuit?',
    sampleAnswer:
      'Circuit breaker prevents cascading failures by stopping requests after repeated failures. Implementation: States: (1) CLOSED - normal operation, requests flow through. (2) OPEN - too many failures, block all requests for timeout period (e.g., 60s). (3) HALF_OPEN - testing recovery, allow one request; if succeeds, close circuit; if fails, reopen. Metrics to track: (1) Failure count in sliding window - track last N requests or last M seconds. (2) Failure rate % - e.g., 50%+ failures triggers circuit. (3) Error types - distinguish between rate limits (maybe keep trying) vs service down (stop immediately). (4) Recovery time - how long circuit has been open. (5) Success count in half-open - need consecutive successes before fully closing. Opening conditions: (1) Threshold: 5+ failures in last 10 requests, OR (2) Rate: >50% failure rate in last 60 seconds, OR (3) Consecutive: 3 consecutive failures on critical endpoints. Closing conditions: (1) Timeout expires (60s open period). (2) Enter HALF_OPEN state. (3) Test request succeeds. (4) 2-3 consecutive successes in half-open. (5) Return to CLOSED. Benefits: Prevents hammering failing API (saves costs), faster failure response (fail fast vs waiting for timeout), gives API time to recover, improves system resilience. Implementation: Use per-endpoint circuit breakers (API might be partially down), expose circuit state in metrics/dashboard, alert when circuit opens (indicates systemic issue), manual override to reset circuit if needed. Real-world example: If OpenAI has outage, circuit opens after 5 failures, blocks requests for 60s instead of 1000s of failed requests, tests recovery with single request, gracefully resumes once API recovers.',
    keyPoints: [
      'Circuit breaker has CLOSED, OPEN, HALF_OPEN states',
      'Open circuit after threshold failures (e.g., 5 in 10 requests)',
      'Block requests during open period to allow recovery',
      'Test with single request before fully reopening',
      'Prevents cascading failures and wasted API calls',
    ],
  },
];
