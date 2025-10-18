/**
 * Quiz questions for Rate Limiting & Counters section
 */

export const ratelimitingQuiz = [
  {
    id: 'q1',
    question:
      'Explain the boundary spike problem in Fixed Window rate limiting and how Sliding Window fixes it.',
    sampleAnswer:
      'Fixed Window resets counter at fixed intervals (0-60s, 60-120s, etc.). The problem: a user can make 100 requests at t=59s (end of window 1) and another 100 at t=60s (start of window 2), getting 200 requests in 1 second while limit is 100/minute. Sliding Window fixes this by looking at any 60-second period ending at current time, not fixed boundaries. At t=60s with sliding window, we count requests from t=0 to t=60, including the 100 at t=59, so we\'d only allow 0 more. Sliding window gives true "100 per rolling minute" while fixed window gives "100 per calendar minute" which can be exploited at boundaries.',
    keyPoints: [
      'Fixed window: resets at boundaries',
      'Exploit: 100 at t=59s + 100 at t=60s',
      'Sliding window: any 60s period ending now',
      'Sliding counts from t-60 to t',
      'Prevents boundary exploitation',
    ],
  },
  {
    id: 'q2',
    question:
      'Why is Token Bucket the industry standard for rate limiting? What advantages does it have?',
    sampleAnswer:
      'Token Bucket is industry standard because it elegantly handles bursts while maintaining average rate. Key advantages: (1) Allows controlled bursts - if user hasn\'t used API for a while, they can burst up to capacity, which feels natural. (2) Smooth refill - tokens accumulate steadily, not all at once. (3) Simple to reason about - "you have N tokens, requests cost 1 token". (4) Easily distributed - can store in Redis. (5) Flexible - different costs for different operations. Real APIs (AWS, GCP, GitHub) use this because users expect occasional bursts without penalty, but sustained abuse is still blocked. Fixed window feels harsh (sudden reset), Token Bucket feels fair.',
    keyPoints: [
      'Allows controlled bursts (up to capacity)',
      'Smooth token refill over time',
      'Simple mental model (tokens = credits)',
      'Easily distributed via Redis',
      'Used by AWS, GCP, GitHub - proven',
    ],
  },
  {
    id: 'q3',
    question:
      'How would you implement rate limiting in a distributed system with multiple servers?',
    sampleAnswer:
      'In distributed systems, use centralized state in Redis or similar. Each server checks/updates Redis before allowing request. Implementation: (1) Use Redis INCR for atomic counter increment. (2) Set TTL with EXPIRE for automatic cleanup. (3) For sliding window, use sorted sets with scores=timestamps, ZREMRANGEBYSCORE to remove old entries. (4) For token bucket, store tokens and last_refill in Redis hash. (5) Handle Redis failures gracefully - either fail open (allow request) or fail closed (deny request) based on requirements. Alternative for extreme scale: use consistent hashing to shard users across Redis instances. Trade-off: slight inconsistency (could exceed limit by milliseconds) for massive scale.',
    keyPoints: [
      'Centralized state in Redis',
      'Atomic operations (INCR, EXPIRE)',
      'Sorted sets for sliding window',
      'Hash for token bucket state',
      'Handle Redis failures (fail open/closed)',
    ],
  },
];
