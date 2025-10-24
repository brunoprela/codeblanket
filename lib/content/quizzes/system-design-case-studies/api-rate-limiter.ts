/**
 * Design API Rate Limiter Quiz
 */

export const rateLimiterQuiz = [
    {
        question: "Compare fixed window counter and sliding window counter algorithms for rate limiting. Explain the 'boundary problem' in fixed window counters with a concrete example, then describe how sliding window counters solve this problem using weighted calculations. When would fixed window counters still be acceptable despite their limitations?",
        sampleAnswer: "Fixed window counters divide time into discrete windows (e.g., 12:00:00-12:00:59, 12:01:00-12:01:59) and count requests within each window. The boundary problem occurs at window transitions: a user can make 100 requests at 12:00:59, then immediately make 100 more at 12:01:00, resulting in 200 requests in 2 seconds—double the intended 100 requests/minute limit. This violates the spirit of rate limiting and enables abuse. Sliding window counters solve this by calculating a weighted average of the previous and current windows. Example: At 12:01:30, we're 30 seconds (50%) into the current window. If the previous window (12:00:00-12:00:59) had 80 requests and the current window (12:01:00-12:01:59) has 40 requests, the estimate is: (80 × (1 - 0.5)) + 40 = 40 + 40 = 80 requests in the last 60 seconds. This smooths the transition between windows and provides more accurate enforcement. The trade-off is slight complexity and minor inaccuracy (it's an approximation, not exact like sliding window log). Fixed window counters are acceptable when: (1) simplicity is paramount, (2) boundary abuse is unlikely (internal APIs with trusted users), (3) the window is very short (1 second window has minimal boundary impact), or (4) slight over-limit is acceptable (non-critical systems). Most production systems (Twitter, Stripe) use sliding window counters as the best balance.",
        keyPoints: [
            "Fixed window boundary problem: 200 requests in 2 seconds at window transition violates 100 req/min limit",
            "Sliding window counters use weighted average: (prev_count × (1 - weight)) + curr_count for smooth enforcement",
            "Sliding window provides more accurate rate limiting without the memory overhead of storing all timestamps",
            "Fixed window is acceptable for internal APIs, very short windows, or when simplicity outweighs accuracy",
            "Production systems favor sliding window counter for distributed rate limiting (Redis-based)"
        ]
    },
    {
        question: "You're implementing distributed rate limiting across 10,000 API servers using Redis. Explain why you need Lua scripts for atomicity, what race conditions could occur without them, and how Lua scripts solve this problem. Additionally, discuss the fallback strategy if Redis becomes unavailable.",
        sampleAnswer: "Without Lua scripts, checking and incrementing the rate limit counter involves multiple Redis commands: GET (read current count), conditional logic (check if under limit), INCR (increment counter), EXPIRE (set TTL). Between these commands, other API servers can execute their own requests, causing race conditions. Example: Two API servers simultaneously check the count (99 requests), both see it's under the 100 limit, both increment (now 101), and the user exceeds the limit without detection. With millions of concurrent requests, this happens frequently, undermining rate limiting effectiveness. Lua scripts solve this by executing all operations atomically—Redis processes the entire script without interruption, guaranteeing no other commands execute in between. The script reads both windows, calculates the estimated count, checks the limit, and increments in a single atomic operation. This eliminates race conditions and ensures accuracy even with 1 million concurrent requests. Regarding Redis failure: implement a circuit breaker pattern. If Redis is unavailable (connection timeout, error rate >5%), fail open (allow all requests) or fail to local rate limiting (each server enforces limits independently using in-memory counters). Failing open maintains API availability but loses rate limiting protection—acceptable for short outages. Failing to local limits is safer: each server limits users to 100 req/min, but users might get 100 × 10,000 servers = 1M total requests (no global coordination). Choose based on priorities: availability (fail open) vs security (fail to local limits). Monitor Redis health and alert on failures.",
        keyPoints: [
            "Race conditions occur when multiple servers read-check-increment concurrently, allowing users to exceed limits",
            "Lua scripts execute atomically in Redis, preventing interleaving of commands from different servers",
            "Atomic execution ensures rate limit accuracy even with millions of concurrent requests across distributed systems",
            "Circuit breaker pattern: fail open (allow all) or fail to local limits (per-server enforcement) if Redis unavailable",
            "Trade-off: availability (fail open) vs security (local limits)—choose based on system priorities"
        ]
    },
    {
        question: "Design a multi-tier rate limiting system where free users get 100 req/min, premium users get 1000 req/min, and enterprise users get 10,000 req/min. Additionally, implement per-endpoint limits where POST /api/upload is limited to 10 req/min regardless of user tier. Explain how you'd structure Redis keys, handle tier lookup efficiently, and prevent abuse through multiple free accounts.",
        sampleAnswer: "I would implement a hierarchical rate limiting system with multiple checks. First, structure Redis keys to include both user_id and endpoint: 'rate_limit:user:{user_id}:endpoint:{endpoint}:{window}'. For global user limits, use 'rate_limit:user:{user_id}:{window}'. When a request arrives: (1) Look up user tier from cache (Redis hash: 'user:tier:{user_id}' → 'premium', with 1-hour TTL). Cache hit avoids database queries for every request. On cache miss, query database and populate cache. (2) Check endpoint-specific limit first (POST /api/upload = 10 req/min for all tiers). If this fails, reject immediately—no need to check user tier. (3) If endpoint limit passes, check user tier limit (100/1000/10000 based on tier). (4) Optionally check global system limit (e.g., 100K req/sec total) to prevent system overload even if all user limits are respected. To prevent abuse through multiple free accounts: (1) Rate limit by IP address in addition to user_id for anonymous/free users: 'rate_limit:ip:{ip_address}:{window}'. (2) Device fingerprinting: track browser fingerprint (user agent, screen res, timezone, canvas) and limit unique fingerprints per IP (e.g., max 5 accounts per fingerprint). (3) Credit card fingerprinting: if free trial requires payment info, limit trials per card. (4) Email domain verification: block disposable email domains (mailinator.com, guerrillamail.com). (5) Behavioral analysis: detect rapid account creation from same IP (CAPTCHA challenge). This layered approach makes it expensive for attackers to abuse free tiers while maintaining good UX for legitimate users.",
        keyPoints: [
            "Hierarchical key structure: user tier limits + per-endpoint limits, checked in order (most restrictive first)",
            "Cache user tier in Redis (1-hour TTL) to avoid database queries on every API request",
            "Per-endpoint limits apply to all tiers (POST /api/upload = 10 req/min overrides user tier limits)",
            "Prevent multi-account abuse: IP-based limiting, device fingerprinting, credit card verification, email domain blocking",
            "Layered defense increases cost of abuse while maintaining low friction for legitimate users"
        ]
    }
];

