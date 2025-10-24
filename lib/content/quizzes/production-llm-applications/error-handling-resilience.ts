export const errorHandlingResilienceQuiz = [
  {
    id: 'pllm-q-7-1',
    question:
      'Design a comprehensive error handling strategy for an LLM application that handles rate limits, timeouts, context length errors, and provider outages. Include specific retry logic, fallback strategies, and user communication.',
    sampleAnswer:
      'Error categorization: 1) Retriable (rate limits, timeouts, 5xx errors) - retry with exponential backoff, 2) Non-retriable (400 bad request, token limit exceeded) - fail immediately with helpful message, 3) Auth errors (401) - alert admins, dont retry. Retry logic: Use tenacity library with @retry decorator, max 3 attempts, exponential backoff (2^n seconds), jitter to prevent thundering herd, retry only on openai.error.RateLimitError, APIError, Timeout. Rate limit (429): Wait for retry-after header or 60s, switch to alternative provider if available, queue request for later. Timeout: Start with 30s timeout, retry with 60s, then 90s, use streaming to detect early failures. Context length: Catch InvalidRequestError about max context, truncate conversation history (keep system prompt + last 10 messages), retry. Provider outage (503): Circuit breaker pattern - after 5 failures, open circuit for 60s, use cached responses or alternative provider. Fallback chain: GPT-4 → GPT-3.5-turbo → Claude → cached similar response → helpful error message. User communication: Real-time status updates (Retrying...), clear error messages (Token limit exceeded, please shorten your prompt), estimated wait time, option to cancel.',
    keyPoints: [
      'Categorize errors into retriable vs non-retriable with appropriate handling',
      'Circuit breaker and fallback chain for resilience',
      'Clear user communication with actionable error messages',
    ],
  },
  {
    id: 'pllm-q-7-2',
    question:
      'Explain the circuit breaker pattern for LLM applications. When would you open/close the circuit, and how do you test if the service has recovered? Provide implementation details.',
    sampleAnswer:
      'Circuit breaker prevents repeatedly calling failing services, giving them time to recover. Three states: Closed (normal), Open (blocking calls), Half-Open (testing recovery). State transitions: Closed → Open after failure_threshold (5) consecutive failures. Open → Half-Open after recovery_timeout (60s). Half-Open → Closed after successful test call. Half-Open → Open if test fails. Implementation: Track failure count, last failure time, current state in Redis (shared across instances). On each call: if Closed, execute normally, increment failure count on error, reset on success; if Open, check if timeout passed → try Half-Open test, else return cached response or error; if Half-Open, allow one test request, Closed on success, Open on failure. Testing recovery: Make lightweight test request (simple prompt), 10s timeout, success criteria: valid response in <10s. If successful, mark as recovered. Monitoring: Track circuit state changes, time in Open state, recovery success rate. Advanced: Per-model circuits (GPT-4 might fail while GPT-3.5 works), gradual recovery (allow 10% traffic in Half-Open, then 25%, then 50%), health score based on recent success rate. Use cases: API provider outage, rate limit exceeded, consistent timeouts. Example: After 5 rate limit errors, open circuit for 5min, preventing wasteful retry attempts.',
    keyPoints: [
      'Three states with clear transition criteria based on failure thresholds',
      'Lightweight test requests to verify recovery before opening',
      'Shared state in Redis for distributed systems',
    ],
  },
  {
    id: 'pllm-q-7-3',
    question:
      'Describe graceful degradation strategies for an LLM application when primary services fail. How do you maintain partial functionality while communicating limitations to users?',
    sampleAnswer:
      'Service modes: Normal (full features, best models), Degraded (limited features, cheaper models), Maintenance (read-only, cached responses). Automatic degradation triggers: Error rate >10% for 5min → Degraded, >25% → Maintenance. In Degraded mode: Use GPT-3.5 instead of GPT-4, reduce max_tokens to 500 (faster, cheaper), increase cache TTL to 7 days, disable non-essential features (voice, image generation), return cached responses when available. In Maintenance mode: Only serve cached responses, disable new generations, show status message, allow viewing history. User communication: Banner showing current mode (We are experiencing high load, using faster models), set expectations (Responses may be shorter), provide ETA for full restoration, offer manual refresh option. Maintain critical functions: Authentication still works, conversation history accessible, cached responses available, admin functions operational. Recovery: Monitor health metrics (error rate, latency, success rate), automatic return to Normal when error rate <1% for 10min, gradual rollback (Maintenance → Degraded → Normal), alert team on mode changes. Implementation: global service_mode variable, check before operations, ServiceDegraded exception with helpful message, metrics tracking time in each mode, dashboard showing current mode and health.',
    keyPoints: [
      'Three service modes with automatic transitions based on health metrics',
      'Maintain core functionality with reduced capacity',
      'Clear communication of limitations and expected restoration time',
    ],
  },
];
