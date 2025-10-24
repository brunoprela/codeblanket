export const rateLimitingThrottlingQuiz = [
  {
    id: 'pllm-q-6-1',
    question:
      'Compare token bucket, sliding window, and fixed window rate limiting algorithms for LLM applications. Which would you choose for a multi-tier SaaS product and why? Include implementation details.',
    sampleAnswer:
      'Token bucket: Allows bursts up to capacity, refills at steady rate. Best for LLM apps because users occasionally need bursts. Sliding window: More accurate than fixed window, counts requests in moving time window. Fixed window: Simple but allows double capacity at window boundaries. For multi-tier SaaS: Use token bucket with Redis. Free tier: 10 req/min capacity, 0.17 req/sec refill. Pro: 100 req/min capacity, 1.67 req/sec refill. Enterprise: 1000 req/min, 16.7 req/sec. Implementation: Lua script in Redis for atomic operations, track tokens and last_update per user, calculate elapsed time and refill tokens = min(capacity, current + elapsed * rate), consume if enough tokens. Advantages: smooth rate control, allows legitimate bursts, fair across time, scales with Redis. Return headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset. On limit: 429 status with Retry-After header. Also implement cost-based limiting: track spend per hour, alert at 80% of budget.',
    keyPoints: [
      'Token bucket best for LLM apps due to burst allowance',
      'Redis Lua scripts for atomic distributed rate limiting',
      'Multi-tier limits with comprehensive response headers',
    ],
  },
  {
    id: 'pllm-q-6-2',
    question:
      'Design a rate limiting system that prevents abuse while maintaining excellent user experience. How do you handle edge cases like rapid retries, legitimate traffic spikes, and distinguish between malicious and normal usage?',
    sampleAnswer:
      'Implement multiple layers: 1) Per-IP rate limit (100/min) to prevent DDoS, 2) Per-API-key limit based on tier, 3) Per-endpoint limits (expensive endpoints lower limits), 4) Cost-based limits ($/hour), 5) Concurrent request limits. Edge cases: Rapid retries - exponential backoff hints in response, block after 10 retries in 1 min. Legitimate spikes - allow burst capacity (2x normal rate for 10s), queue requests instead of rejecting. Distinguish abuse: Track patterns (requests at regular intervals = bot, varied timing = human), analyze request distribution (all to same endpoint = suspicious), monitor success rates (high error rate = probing), check user-agent and headers. Graceful handling: Return helpful errors explaining limit and upgrade path, offer queuing for non-urgent requests, provide usage dashboard showing consumption, send email warnings at 80% of limit, allow temporary limit increases for verified users. For malicious: CAPTCHA after repeated limit hits, temporary IP blocks (increasing duration), require re-authentication, notify security team. Maintain UX: cache aggressively to avoid hits, batch operations, provide cost estimates before expensive operations, show rate limit status in dashboard.',
    keyPoints: [
      'Multi-layered rate limiting with different scopes',
      'Pattern analysis to distinguish malicious from legitimate usage',
      'Graceful degradation and helpful error messages',
    ],
  },
  {
    id: 'pllm-q-6-3',
    question:
      'Explain how you would implement cost-based rate limiting where users have a daily budget rather than request count limit. How do you handle estimation errors and prevent budget overruns?',
    sampleAnswer:
      'Track spending per user per day in Redis with atomic increment. Daily budget by tier: Free $1, Pro $50, Enterprise $1000. Before each request: 1) Estimate cost (input_tokens/1000 * input_price + estimated_output_tokens/1000 * output_price), 2) Check if current_spend + estimated_cost > budget, 3) If yes, reject with budget error, 4) If no, reserve estimated cost, 5) After completion, calculate actual cost, 6) Adjust reservation (refund if over-estimated, charge difference if under). Handle estimation errors: Conservative estimates (assume max_tokens for output), track estimation accuracy per model, adjust multiplier based on historical data (if consistently 50% over, reduce estimates). Prevent overruns: Set hard limits at 100% of budget, soft alerts at 80% and 90%, reject requests that would exceed even with minimal output, provide real-time budget dashboard. Edge cases: Streaming responses - reserve max cost upfront, refund unused after completion. Concurrent requests - use Redis transactions to prevent race conditions. Budget rollovers - allow 10% rollover to next day to smooth usage. Emergency overrides - admin can increase budget temporarily. Implementation: Redis key per user-day, INCRBYFLOAT for atomic updates, expire keys after 48 hours, track in database for billing and analytics.',
    keyPoints: [
      'Estimate-then-adjust pattern with conservative estimates',
      'Atomic Redis operations to prevent race conditions',
      'Soft and hard limits with real-time budget tracking',
    ],
  },
];
