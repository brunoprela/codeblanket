/**
 * Quiz questions for Rate Limiting Algorithms section
 */

export const ratelimitingalgorithmsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the boundary burst problem in Fixed Window Counter algorithm and how Sliding Window Counter solves it. Use a concrete example.',
    sampleAnswer:
      'FIXED WINDOW PROBLEM: Counts requests in fixed time windows (0-60s, 60-120s). Window resets at boundary. Limit: 100 req/min. BOUNDARY BURST SCENARIO: 00:30-00:59: User makes 100 requests (allowed, within limit). Window 1 counter = 100. 01:00: Window 2 starts, counter resets to 0. 01:00-01:29: User makes 100 requests (allowed, new window). Result: 200 requests in 1 minute (00:30-01:30), DOUBLE the limit! SLIDING WINDOW COUNTER SOLUTION: Uses weighted average of current and previous windows. At 01:15 (15 seconds into new window): Current window (01:00-02:00): 100 requests. Previous window (00:00-01:00): 100 requests. Elapsed in current: 15s = 25% of window. Weighted count = (100 * 75%) + (100 * 100%) = 75 + 100 = 175. Next request: 176 > 100 limit → DENIED. RESULT: Smoothly enforces limit across sliding 60-second window, prevents boundary bursts. This is why production systems (Cloudflare, Kong) use sliding window counter instead of fixed window. Trade-off: Slightly more complex (2 counters instead of 1) but prevents 2x limit violations.',
    keyPoints: [
      'Fixed window: resets at boundaries, allows 2x limit in edge case',
      'Example: 100 at end of window 1 + 100 at start of window 2 = 200 in 60s',
      'Sliding window: weighted average of current + previous windows',
      'Smoothly enforces limit across any 60-second period',
      'Production standard (Cloudflare, Kong) for this reason',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare Token Bucket vs Leaky Bucket algorithms. When would you choose each? What are the key behavioral differences?',
    sampleAnswer:
      'TOKEN BUCKET: Bucket holds tokens, refills at constant rate. Request consumes token. If tokens available → allow (even in burst). Capacity: 10, refill: 2/sec. Behavior: Can burst up to 10 requests instantly (if bucket full). Then limited to 2 req/sec sustained. Good for: API rate limiting where occasional bursts acceptable (AWS API Gateway, Stripe). User experience: Fast responses during burst, then throttle. LEAKY BUCKET: Requests enter queue, processed at constant rate. Capacity: 10 requests in queue. Behavior: Processes exactly 2 req/sec regardless of input. Burst of 20 requests: 10 queued, 10 dropped, process 2/sec. Good for: Traffic shaping, protecting downstream services. Network: Smooth output traffic to prevent overwhelming backend. KEY DIFFERENCES: Token bucket ALLOWS bursts (consume accumulated tokens). Leaky bucket SMOOTHS bursts (constant output rate). Token bucket: Output rate varies (burst then steady). Leaky bucket: Output rate constant (drops excess). DECISION: Use token bucket for user-facing APIs (better UX, handle burst). Use leaky bucket for backend protection (consistent load). Example: Stripe uses token bucket (allows burst API calls). Network routers use leaky bucket (smooth traffic to prevent congestion).',
    keyPoints: [
      'Token bucket: allows bursts up to capacity, then steady rate',
      'Leaky bucket: constant output rate, queues or drops bursts',
      'Token bucket: better UX, accumulated credits enable bursts',
      'Leaky bucket: protects downstream, smooths traffic spikes',
      'Use case: Token (API limits), Leaky (network/backend protection)',
    ],
  },
  {
    id: 'q3',
    question:
      'Design a distributed rate limiting system using Redis for an API gateway serving 10,000 users with 1000 requests/minute per user limit.',
    sampleAnswer:
      'DISTRIBUTED RATE LIMITING ARCHITECTURE: (1) ALGORITHM: Sliding window counter (accurate, memory-efficient). (2) STORAGE: Redis cluster (centralized state, atomic operations). (3) KEY DESIGN: "rate_limit:{user_id}:{minute_timestamp}". Example: "rate_limit:user123:1633024800". (4) IMPLEMENTATION: On each request: Check Redis: count = GET rate_limit:user123:1633024800. If count < 1000: INCR rate_limit:user123:1633024800, EXPIRE 120 (keep 2 windows). Return 200 + headers (X-RateLimit-Remaining). Else: Return 429 Too Many Requests + Retry-After: 37. (5) LUA SCRIPT (atomic): local current = redis.call("GET", current_key) or 0. local prev = redis.call("GET", prev_key) or 0. local weight = (60 - seconds_elapsed) / 60. local estimated = (prev * weight) + current. if estimated < limit then redis.call("INCR", current_key); return 1 else return 0 end. (6) SCALE: 10K users × 2 keys (current + prev) × 8 bytes = 160 KB (tiny!). Redis: 100K ops/sec, easily handles 10K users × 1000 req/min = 166K req/sec distributed. (7) HIGH AVAILABILITY: Redis Cluster with replication. Fallback: If Redis down, local rate limiting (eventual consistency). This is production-ready: Cloudflare, Kong, AWS API Gateway use similar architecture. Memory efficient, accurate, scales horizontally.',
    keyPoints: [
      'Sliding window counter for accuracy + efficiency',
      'Redis for distributed state with atomic operations',
      'Lua script for atomic read-modify-write',
      'Key per user per minute: 160 KB for 10K users',
      'Redis cluster handles 166K req/sec easily',
    ],
  },
];
