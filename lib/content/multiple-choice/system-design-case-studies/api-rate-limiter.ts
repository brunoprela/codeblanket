/**
 * Design API Rate Limiter Multiple Choice Questions
 */

export const rateLimiterMultipleChoice = [
    {
        question: "A user makes 80 requests in the previous window (12:00:00-12:00:59) and 40 requests in the current window (12:01:00-12:01:59). At 12:01:45, should the user's 121st total request be allowed if the limit is 100 requests/minute using sliding window counter algorithm?",
        options: [
            "Allow: The current window only has 40 requests, which is under the 100 limit",
            "Allow: 45 seconds into the window gives weighted estimate of (80 × 0.25) + 40 = 60 requests",
            "Deny: 45 seconds into the window gives weighted estimate of (80 × 0.25) + 41 = 61, under limit, so request 121 would make it 61, but we're counting total requests wrong",
            "Deny: The weighted estimate is (80 × (1 - 45/60)) + 40 = (80 × 0.25) + 40 = 60, and this is the 41st request in the current window, making estimated total 61 which is under 100, so this explanation is wrong. Let me recalculate: at 45 seconds in, weight = 45/60 = 0.75, so estimate = (80 × (1 - 0.75)) + 40 = 20 + 40 = 60. The new request would be the 41st in current window, making it (80 × 0.25) + 41 = 61, still under 100, so ALLOW"
        ],
        correctAnswer: 1,
        explanation: "The correct answer is Allow, with a weighted estimate of 60 requests. Let's carefully work through the sliding window counter calculation. At time 12:01:45, we are 45 seconds into the current 60-second window. The weight (fraction of current window elapsed) is 45/60 = 0.75. The sliding window formula is: estimated_count = (prev_window_count × (1 - weight)) + curr_window_count = (80 × (1 - 0.75)) + 40 = (80 × 0.25) + 40 = 20 + 40 = 60 requests in the last 60 seconds. Since 60 < 100, the request is allowed. After allowing it, the current window count becomes 41, so the next request would estimate (80 × 0.25) + 41 = 61. The sliding window counter gives more weight to the current window as time progresses: at 0 seconds (start of window), weight = 0, so we fully count previous window (80 × 1 = 80); at 30 seconds (halfway), weight = 0.5, giving (80 × 0.5) + 40 = 40 + 40 = 80; at 60 seconds (end of window), weight = 1, so we fully count current window (40 × 1 = 40). This smoothly transitions from previous to current window, avoiding the fixed window boundary problem. The key insight: the algorithm estimates requests in a rolling 60-second window, not just the current fixed window, providing more accurate rate limiting."
    },
    {
        question: "You implement rate limiting in Redis with these commands: `GET rate_limit:user:123`, check if under limit, then `INCR rate_limit:user:123`. Under high concurrency (10,000 requests/second for user 123 with a 100 req/sec limit), what is the most likely outcome?",
        options: [
            "Exactly 100 requests/second allowed due to Redis single-threaded execution model",
            "Approximately 100-150 requests/second allowed due to race conditions between GET and INCR",
            "All requests blocked after the first 100 due to Redis optimistic locking",
            "Unpredictable behavior ranging from 100-1000 requests allowed depending on network latency"
        ],
        correctAnswer: 1,
        explanation: "The most likely outcome is approximately 100-150 requests/second allowed (potentially more) due to race conditions. Here's why: Even though Redis itself is single-threaded and processes commands atomically, the GET and INCR commands are separate operations. The race condition occurs in the following scenario: 10,000 API servers simultaneously process requests for user 123. They all execute GET rate_limit:user:123 and receive count=99 (under the 100 limit). All 10,000 servers see that 99 < 100, so they all decide to allow the request. Then all 10,000 servers execute INCR, incrementing the counter to 10,099. The user just made 10,000 requests that were all 'allowed' despite a 100 req/sec limit. In practice, network latency and timing variations mean not all requests arrive at exactly the same millisecond, so the over-limit amount is smaller (perhaps 100-150 rather than 10,000), but the race condition still occurs. This is why production systems use Lua scripts for atomicity—the script performs GET, check, and INCR in a single atomic operation, ensuring no other commands execute in between. Option A is wrong because single-threaded execution doesn't prevent race conditions across multiple separate commands. Option C mentions optimistic locking, which Redis doesn't provide automatically (you'd need WATCH/MULTI/EXEC, which adds complexity). Option D overstates the unpredictability—it's not 1000, more like 10-50% over-limit. The fix: Lua script that atomically reads, checks, and increments."
    },
    {
        question: "Your rate limiter uses Redis with a 60-second window. A key is set as `SETEX rate_limit:user:123:1672531200 120 50`. Why is the TTL set to 120 seconds (2× window size) instead of 60 seconds?",
        options: [
            "Safety margin: If Redis cleanup is delayed, the key won't expire prematurely",
            "Sliding window counters need both current and previous windows, so we keep 2 windows (120 seconds)",
            "Redis SETEX has a bug where TTL shorter than 60 seconds can cause data loss",
            "Performance optimization: Longer TTL means fewer EXPIRE commands, reducing Redis load"
        ],
        correctAnswer: 1,
        explanation: "The correct answer is that sliding window counters require access to both the current and previous windows. Here's the detailed reasoning: At any given time, the sliding window calculation references two windows: the current window and the previous window. For example, at 12:01:30, we need data from both the 12:01:00-12:01:59 window (current) and the 12:00:00-12:00:59 window (previous) to calculate the weighted average. If we set TTL to only 60 seconds, the previous window would be deleted exactly when the current window begins, making the sliding window calculation impossible—we'd have no previous window data. By setting TTL to 120 seconds, we ensure the previous window's data is retained throughout the entire current window's duration. The data flow: at 12:00:00, create window 12:00:00 with TTL=120s (expires at 12:02:00). At 12:01:00, create window 12:01:00 with TTL=120s (expires at 12:03:00). At 12:01:30, we access both windows (12:00:00 and 12:01:00) for the sliding calculation. At 12:02:00, window 12:00:00 expires (no longer needed—it's 2 windows old). This pattern ensures we always have exactly 2 windows available: current and previous. Option A (safety margin) is a minor benefit but not the primary reason. Option C is false—Redis has no such bug. Option D is misleading—the TTL is set once during SETEX, not repeatedly. The core principle: sliding window counters depend on historical data (previous window), so TTL must cover 2 windows."
    },
    {
        question: "Your API rate limiter returns HTTP 429 with header `Retry-After: 45`. A client implements exponential backoff: retry after 1s, 2s, 4s, 8s, 16s... What is the problem with this approach?",
        options: [
            "Exponential backoff is correct; the Retry-After header is advisory and can be ignored for more sophisticated retry strategies",
            "The client should respect Retry-After: 45 and wait exactly 45 seconds, not use exponential backoff",
            "Exponential backoff starts too short (1s) and should start at 5s minimum for rate limiting",
            "The client should combine both: wait 45 seconds (Retry-After) + exponential backoff (1s, 2s, 4s) on subsequent retries"
        ],
        correctAnswer: 1,
        explanation: "The correct answer is that the client should respect the Retry-After: 45 header and wait exactly 45 seconds. The Retry-After header is not advisory for rate limiting—it's a directive that tells the client precisely when their rate limit window resets. Here's why this matters: The server calculated that the user's rate limit will reset in 45 seconds (e.g., current time is 12:01:15, limit resets at 12:02:00, so 45 seconds remain). If the client ignores this and retries after 1 second, 2 seconds, 4 seconds, etc., all those retries will also hit the rate limit (receiving more 429s), wasting both client and server resources. The client will make 6-7 failed requests (1s + 2s + 4s + 8s + 16s = 31s, still before 45s reset) before finally succeeding after 45 seconds anyway. This is inefficient and can appear as an attack (aggressive retrying) to the server, potentially leading to IP blocking. The proper approach: When receiving 429 with Retry-After, sleep for that exact duration, then retry once. If that retry also fails (unlikely unless the client miscalculated), then exponential backoff can be applied. Option A is wrong—Retry-After is not advisory for rate limiting; it's precise. Option C's suggestion to start at 5s doesn't address the fundamental issue of ignoring the directive. Option D is redundant—if you wait 45 seconds as instructed, the retry should succeed; exponential backoff on subsequent failures protects against other errors (500, network issues) but shouldn't be needed for 429. Respecting Retry-After is both efficient and polite, demonstrating good API citizenship."
    },
    {
        question: "You're implementing a token bucket rate limiter with capacity=100 tokens and refill_rate=1.67 tokens/second (100 tokens/minute). At time T=0, the bucket is full (100 tokens). A client makes 100 requests instantly at T=0, consuming all tokens. When will the bucket be full again?",
        options: [
            "T=60 seconds (100 tokens ÷ 1.67 tokens/sec = 60 seconds)",
            "T=59.88 seconds (100 ÷ 1.67 = 59.88)",
            "Never fully refills; refill_rate of 1.67 is an approximation, so the bucket asymptotically approaches 100",
            "T=0 seconds; the bucket refills instantly upon requests being processed"
        ],
        correctAnswer: 1,
        explanation: "The correct answer is T=59.88 seconds, calculated as 100 tokens ÷ 1.67 tokens/second = 59.88 seconds. Here's the precise token bucket behavior: At T=0, bucket has 100 tokens (full). Client makes 100 requests instantly, each consuming 1 token, leaving bucket with 0 tokens at T=0. Tokens refill continuously at 1.67 tokens/second. At T=1, bucket has 1.67 tokens. At T=10, bucket has 16.7 tokens. At T=30, bucket has 50.1 tokens. At T=59.88, bucket has ~100 tokens (capacity reached). After reaching capacity, refilling stops (bucket can't exceed 100 tokens). The refill_rate of 1.67 tokens/sec corresponds to 100 tokens/minute (1.67 × 60 = 100.2, close enough due to rounding). Option A (60 seconds) is close but slightly imprecise—the exact calculation is 100 / 1.67 = 59.88 seconds. Option C is incorrect; the bucket fully refills to capacity (100)—there's no asymptotic behavior. Option D misunderstands token bucket mechanics; refilling happens continuously over time based on elapsed seconds, not instantly. Token bucket advantages: (1) Allows bursts—client can use all 100 tokens instantly if bucket is full. (2) Smooths traffic—refills gradually, preventing sustained high rates. (3) Forgives quiet periods—if client doesn't use tokens for 60 seconds, bucket refills to 100, allowing another burst. This is why token bucket is widely used (AWS API Gateway, Google Cloud) for rate limiting APIs with bursty traffic patterns."
    }
];

