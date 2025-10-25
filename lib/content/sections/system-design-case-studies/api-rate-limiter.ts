/**
 * Design API Rate Limiter Section
 */

export const rateLimiterSection = {
  id: 'api-rate-limiter',
  title: 'Design API Rate Limiter',
  content: `An API rate limiter controls how many requests a client can make to an API within a time window. It prevents abuse, ensures fair resource allocation, protects against DDoS attacks, and maintains service quality for all users. Rate limiting is critical for public APIs (Twitter, GitHub, Stripe) and internal microservices.

## Problem Statement

Design a rate limiter that:
- **Enforces Limits**: 100 requests/minute per user
- **Multiple Rules**: Different limits for different API endpoints
- **Distributed**: Works across multiple API servers
- **Low Latency**: Decision in < 5ms (don't slow down API)
- **Flexible Policies**: Tiered limits (free: 100/min, premium: 1000/min)
- **Graceful Degradation**: Inform users of limit, remaining quota
- **Bypass for Critical Users**: Whitelist, emergency access

**Scale**: 10,000 API servers, 100 million users, 1 million requests/second

---

## Step 1: Rate Limiting Algorithms

### 1. Token Bucket

**Concept**: Bucket holds tokens. Each request consumes 1 token. Tokens refill at fixed rate.

\`\`\`python
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity  # Max tokens (burst size)
        self.tokens = capacity
        self.refill_rate = refill_rate  # Tokens per second
        self.last_refill = time.time()
    
    def allow_request (self):
        now = time.time()
        
        # Refill tokens based on elapsed time
        elapsed = now - self.last_refill
        self.tokens = min (self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        # Try to consume 1 token
        if self.tokens >= 1:
            self.tokens -= 1
            return True  # Allow
        else:
            return False  # Reject (rate limited)

# Example: 100 requests/minute = 1.67 tokens/second
bucket = TokenBucket (capacity=100, refill_rate=1.67)
\`\`\`

**Pros**:
- ✅ Allows bursts (if bucket full, can consume all tokens at once)
- ✅ Smooth traffic over time
- ✅ Simple implementation

**Cons**:
- ❌ Bucket state per user (memory intensive)
- ❌ Distributed systems challenge (shared state)

**Use Case**: General API rate limiting (AWS API Gateway uses this)

---

### 2. Leaky Bucket

**Concept**: Requests enter bucket (queue). Leak out at fixed rate. If bucket full, reject.

\`\`\`python
import queue

class LeakyBucket:
    def __init__(self, capacity, leak_rate):
        self.queue = queue.Queue (maxsize=capacity)
        self.leak_rate = leak_rate  # Requests per second
        self.last_leak = time.time()
    
    def allow_request (self):
        now = time.time()
        
        # Leak tokens
        elapsed = now - self.last_leak
        num_leaks = int (elapsed * self.leak_rate)
        for _ in range (num_leaks):
            if not self.queue.empty():
                self.queue.get()
        self.last_leak = now
        
        # Try to add request to queue
        try:
            self.queue.put_nowait(None)  # Non-blocking put
            return True  # Allowed
        except queue.Full:
            return False  # Rejected

# Example: 100 requests/minute
bucket = LeakyBucket (capacity=100, leak_rate=1.67)
\`\`\`

**Pros**:
- ✅ Smooth output rate (guaranteed)
- ✅ Good for rate-sensitive systems (video streaming)

**Cons**:
- ❌ No burst handling (strict rate)
- ❌ Queue overhead

**Use Case**: Traffic shaping, network packet scheduling

---

### 3. Fixed Window Counter

**Concept**: Count requests in fixed time windows (12:00-12:01, 12:01-12:02).

\`\`\`python
class FixedWindow:
    def __init__(self, max_requests, window_size):
        self.max_requests = max_requests
        self.window_size = window_size  # Seconds
        self.counter = {}  # window_start → count
    
    def allow_request (self, user_id):
        now = time.time()
        window_start = int (now // self.window_size) * self.window_size
        
        key = f"{user_id}:{window_start}"
        count = self.counter.get (key, 0)
        
        if count < self.max_requests:
            self.counter[key] = count + 1
            return True  # Allowed
        else:
            return False  # Rate limited

# Example: 100 requests/minute (60-second window)
limiter = FixedWindow (max_requests=100, window_size=60)
\`\`\`

**Pros**:
- ✅ Simple to implement
- ✅ Memory efficient (only current window)

**Cons**:
- ❌ **Boundary Issue**: User can make 100 requests at 12:00:59, then 100 more at 12:01:00 (200 in 2 seconds!)
- ❌ Unfair at window edges

**Use Case**: Simple rate limiting with acceptable edge cases

---

### 4. Sliding Window Log

**Concept**: Store timestamp of each request. Count requests in last N seconds.

\`\`\`python
from collections import deque

class SlidingWindowLog:
    def __init__(self, max_requests, window_size):
        self.max_requests = max_requests
        self.window_size = window_size  # Seconds
        self.logs = {}  # user_id → deque of timestamps
    
    def allow_request (self, user_id):
        now = time.time()
        
        if user_id not in self.logs:
            self.logs[user_id] = deque()
        
        # Remove old requests (outside window)
        while self.logs[user_id] and self.logs[user_id][0] < now - self.window_size:
            self.logs[user_id].popleft()
        
        # Check limit
        if len (self.logs[user_id]) < self.max_requests:
            self.logs[user_id].append (now)
            return True  # Allowed
        else:
            return False  # Rate limited

# Example: 100 requests/minute
limiter = SlidingWindowLog (max_requests=100, window_size=60)
\`\`\`

**Pros**:
- ✅ Accurate (no boundary issues)
- ✅ Precise rate enforcement

**Cons**:
- ❌ Memory intensive (store every request timestamp)
- ❌ Expensive for high-traffic users (1M requests = 1M timestamps)

**Use Case**: When precision matters, moderate traffic

---

### 5. Sliding Window Counter (Hybrid)

**Concept**: Combine fixed window efficiency with sliding window accuracy.

\`\`\`python
class SlidingWindowCounter:
    def __init__(self, max_requests, window_size):
        self.max_requests = max_requests
        self.window_size = window_size
        self.counters = {}  # user_id → {window_start → count}
    
    def allow_request (self, user_id):
        now = time.time()
        current_window = int (now // self.window_size) * self.window_size
        previous_window = current_window - self.window_size
        
        # Weight: how far into current window are we?
        elapsed_in_current = now - current_window
        weight = elapsed_in_current / self.window_size
        
        # Estimate: (prev_window_count * (1 - weight)) + current_window_count
        prev_count = self.counters.get (f"{user_id}:{previous_window}", 0)
        curr_count = self.counters.get (f"{user_id}:{current_window}", 0)
        estimated_count = (prev_count * (1 - weight)) + curr_count
        
        if estimated_count < self.max_requests:
            self.counters[f"{user_id}:{current_window}"] = curr_count + 1
            return True  # Allowed
        else:
            return False  # Rate limited

# Example: 100 requests/minute
limiter = SlidingWindowCounter (max_requests=100, window_size=60)
\`\`\`

**Calculation Example**:
- Window size: 60 seconds
- Current time: 12:01:30 (30 seconds into current window)
- Previous window (12:00-12:01): 80 requests
- Current window (12:01-12:02): 40 requests
- Weight: 30/60 = 0.5
- Estimated: (80 × 0.5) + 40 = 40 + 40 = 80 requests in last 60 seconds

**Pros**:
- ✅ Memory efficient (only 2 counters per user)
- ✅ Smooth (no sharp boundary issues)
- ✅ Accurate approximation

**Cons**:
- ❌ Slight inaccuracy (estimates, not exact)

**Use Case**: **Best for distributed systems** (Twitter, Stripe use this)

---

## Step 2: Distributed Rate Limiting (Redis-Based)

**Challenge**: 10,000 API servers, need shared rate limit state.

**Architecture**:

\`\`\`
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│API Server 1  │       │API Server 2  │       │API Server 3  │
└──────┬───────┘       └──────┬───────┘       └──────┬───────┘
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              │
                     ┌────────▼────────┐
                     │  Redis Cluster  │
                     │ (Rate Limit State)│
                     └─────────────────┘

# Key: "rate_limit:user:123:1672531200"  (user_id + window)
# Value: 45  (request count)
\`\`\`

### Implementation: Sliding Window Counter in Redis

\`\`\`python
import redis
import time

redis_client = redis.Redis (host='localhost', port=6379, decode_responses=True)

def rate_limit (user_id, max_requests=100, window_size=60):
    now = time.time()
    current_window = int (now // window_size) * window_size
    previous_window = current_window - window_size
    
    current_key = f"rate_limit:user:{user_id}:{current_window}"
    previous_key = f"rate_limit:user:{user_id}:{previous_window}"
    
    # Get counts
    prev_count = int (redis_client.get (previous_key) or 0)
    curr_count = int (redis_client.get (current_key) or 0)
    
    # Calculate estimated count (sliding window)
    elapsed_in_current = now - current_window
    weight = elapsed_in_current / window_size
    estimated_count = (prev_count * (1 - weight)) + curr_count
    
    if estimated_count >= max_requests:
        return {
            "allowed": False,
            "limit": max_requests,
            "remaining": 0,
            "reset": current_window + window_size
        }
    
    # Increment current window counter
    pipe = redis_client.pipeline()
    pipe.incr (current_key)
    pipe.expire (current_key, window_size * 2)  # Keep 2 windows
    pipe.execute()
    
    return {
        "allowed": True,
        "limit": max_requests,
        "remaining": int (max_requests - estimated_count - 1),
        "reset": current_window + window_size
    }

# Usage
result = rate_limit (user_id=123, max_requests=100, window_size=60)
if not result["allowed"]:
    return Response("Rate limit exceeded", status=429, headers={
        "X-RateLimit-Limit": result["limit"],
        "X-RateLimit-Remaining": result["remaining"],
        "X-RateLimit-Reset": result["reset"]
    })
\`\`\`

**Benefits**:
- ✅ Shared state (all API servers see same counts)
- ✅ Fast (Redis: < 1ms latency)
- ✅ Atomic operations (INCR is thread-safe)
- ✅ Auto-expiry (old windows deleted automatically)

---

## Step 3: Redis Lua Script (Atomic Operations)

**Problem**: Multiple Redis commands (GET, INCR, EXPIRE) can race if requests concurrent.

**Solution**: Lua script (atomic execution).

\`\`\`lua
-- rate_limit.lua
local current_key = KEYS[1]
local previous_key = KEYS[2]
local max_requests = tonumber(ARGV[1])
local window_size = tonumber(ARGV[2])
local weight = tonumber(ARGV[3])

local prev_count = tonumber (redis.call('GET', previous_key) or 0)
local curr_count = tonumber (redis.call('GET', current_key) or 0)

local estimated_count = (prev_count * (1 - weight)) + curr_count

if estimated_count >= max_requests then
    return {0, max_requests, 0}  -- Denied
end

redis.call('INCR', current_key)
redis.call('EXPIRE', current_key, window_size * 2)

local remaining = max_requests - estimated_count - 1
return {1, max_requests, remaining}  -- Allowed
\`\`\`

**Python Usage**:

\`\`\`python
with open('rate_limit.lua') as f:
    lua_script = redis_client.register_script (f.read())

def rate_limit_atomic (user_id, max_requests=100, window_size=60):
    now = time.time()
    current_window = int (now // window_size) * window_size
    previous_window = current_window - window_size
    
    current_key = f"rate_limit:user:{user_id}:{current_window}"
    previous_key = f"rate_limit:user:{user_id}:{previous_window}"
    
    elapsed_in_current = now - current_window
    weight = elapsed_in_current / window_size
    
    # Execute Lua script atomically
    allowed, limit, remaining = lua_script(
        keys=[current_key, previous_key],
        args=[max_requests, window_size, weight]
    )
    
    return {
        "allowed": bool (allowed),
        "limit": limit,
        "remaining": remaining
    }
\`\`\`

**Why Lua?**
- ✅ Atomic: All commands in script execute without interruption
- ✅ No race conditions (even with 1M concurrent requests)
- ✅ Lower latency (single round-trip to Redis)

---

## Step 4: Multi-Tier Rate Limiting

**Scenario**: Different limits for different user tiers.

\`\`\`
Free Tier:      100 requests/minute
Premium Tier:  1000 requests/minute
Enterprise:   10000 requests/minute
\`\`\`

**Implementation**:

\`\`\`python
def get_user_tier (user_id):
    # Look up in database
    user = db.query("SELECT tier FROM users WHERE user_id = ?", user_id)
    return user.tier  # "free", "premium", "enterprise"

def rate_limit_with_tier (user_id):
    tier = get_user_tier (user_id)
    
    limits = {
        "free": 100,
        "premium": 1000,
        "enterprise": 10000
    }
    
    max_requests = limits.get (tier, 100)  # Default: free
    return rate_limit (user_id, max_requests=max_requests, window_size=60)
\`\`\`

---

## Step 5: Per-Endpoint Rate Limiting

**Scenario**: Different limits for different API endpoints.

\`\`\`
POST /api/upload:     10 requests/minute  (expensive operation)
GET  /api/users:     100 requests/minute
GET  /api/health:  10000 requests/minute  (lightweight)
\`\`\`

**Implementation**:

\`\`\`python
def rate_limit_endpoint (user_id, endpoint):
    endpoint_limits = {
        "/api/upload": 10,
        "/api/users": 100,
        "/api/health": 10000
    }
    
    max_requests = endpoint_limits.get (endpoint, 100)
    
    # Use endpoint-specific key
    key = f"rate_limit:user:{user_id}:endpoint:{endpoint}"
    return rate_limit (user_id, max_requests=max_requests, window_size=60)
\`\`\`

---

## Step 6: Global Rate Limiting (Prevent System Overload)

**Problem**: Even if per-user limits respected, 100K users × 100 req/min = 10M req/min (167K req/sec) could overwhelm system.

**Solution**: Global rate limit (total requests across all users).

\`\`\`python
def global_rate_limit (max_requests_per_second=100000):
    now = int (time.time())
    key = f"rate_limit:global:{now}"
    
    count = redis_client.incr (key)
    redis_client.expire (key, 2)  # Keep for 2 seconds
    
    if count > max_requests_per_second:
        return False  # System overloaded
    return True

# Check global limit first
if not global_rate_limit():
    return Response("System overloaded", status=503)

# Then check user limit
if not rate_limit (user_id):
    return Response("Rate limit exceeded", status=429)
\`\`\`

---

## Step 7: Rate Limiter Middleware (API Integration)

\`\`\`python
from flask import Flask, request, jsonify

app = Flask(__name__)

def rate_limiter_middleware():
    # Extract user ID (from JWT, API key, etc.)
    user_id = request.headers.get("X-User-ID")
    endpoint = request.path
    
    # Check rate limit
    result = rate_limit_endpoint (user_id, endpoint)
    
    if not result["allowed"]:
        return jsonify({
            "error": "Rate limit exceeded",
            "limit": result["limit"],
            "remaining": result["remaining"],
            "reset": result["reset"]
        }), 429
    
    # Add headers to response
    response = make_response()
    response.headers["X-RateLimit-Limit"] = result["limit"]
    response.headers["X-RateLimit-Remaining"] = result["remaining"]
    response.headers["X-RateLimit-Reset"] = result["reset"]
    return response

# Apply to all routes
@app.before_request
def before_request():
    return rate_limiter_middleware()

@app.route("/api/users")
def get_users():
    return jsonify({"users": [...]})
\`\`\`

---

## Step 8: Handling Rate Limit Responses

**HTTP 429 Response**:

\`\`\`http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1672531260
Retry-After: 45

{
  "error": "Rate limit exceeded",
  "message": "You have exceeded 100 requests per minute. Try again in 45 seconds."
}
\`\`\`

**Client-Side Handling**:

\`\`\`python
import requests
import time

def api_call_with_retry (url):
    response = requests.get (url)
    
    if response.status_code == 429:
        # Respect Retry-After header
        retry_after = int (response.headers.get("Retry-After", 60))
        print(f"Rate limited. Waiting {retry_after} seconds...")
        time.sleep (retry_after)
        return api_call_with_retry (url)  # Retry
    
    return response
\`\`\`

---

## Step 9: Distributed Rate Limiting Challenges

**Challenge 1: Redis Latency**

**Problem**: Redis call adds latency to every API request.

**Solution**:
- Use Redis Cluster (multiple nodes, low latency)
- Local cache (approximate limits): Check local counter first, sync with Redis every 10 seconds
- Accept slight inaccuracy (user gets 105 requests instead of 100)

**Challenge 2: Redis Failure**

**Problem**: Redis down = all APIs fail?

**Solution**:
- Circuit breaker: If Redis unavailable, allow all requests (fail open)
- Or: Use local rate limiting (per-server limits) as fallback

\`\`\`python
def rate_limit_with_fallback (user_id):
    try:
        return rate_limit (user_id)  # Redis-based
    except redis.ConnectionError:
        # Fallback: local rate limiting (less accurate, but API stays up)
        return local_rate_limit (user_id)
\`\`\`

**Challenge 3: Clock Skew**

**Problem**: Different API servers have slightly different clocks (1-2 seconds apart).

**Solution**:
- Use Redis timestamp (single source of truth)
- Or: NTP sync (Network Time Protocol) for all servers

---

## Step 10: Advanced Patterns

### Dynamic Rate Limiting

**Scenario**: Adjust limits based on system load.

\`\`\`python
def dynamic_rate_limit (user_id):
    # Check system load
    cpu_usage = get_cpu_usage()
    
    if cpu_usage > 80:
        max_requests = 50  # Reduce limits when system stressed
    else:
        max_requests = 100
    
    return rate_limit (user_id, max_requests=max_requests)
\`\`\`

### Whitelisting/Blacklisting

\`\`\`python
WHITELIST = [999, 1000, 1001]  # VIP users, no limits
BLACKLIST = [666, 667]  # Banned users

def rate_limit_with_rules (user_id):
    if user_id in WHITELIST:
        return {"allowed": True}  # Bypass
    
    if user_id in BLACKLIST:
        return {"allowed": False}  # Always block
    
    return rate_limit (user_id)
\`\`\`

### Burst Handling (Token Bucket)

\`\`\`python
# Allow short bursts (10 requests/sec) but limit average (100 requests/minute)
def rate_limit_with_burst (user_id):
    # Short-term limit (10 req/sec)
    short_term = rate_limit (user_id, max_requests=10, window_size=1)
    if not short_term["allowed"]:
        return short_term
    
    # Long-term limit (100 req/min)
    long_term = rate_limit (user_id, max_requests=100, window_size=60)
    return long_term
\`\`\`

---

## Step 11: Monitoring & Alerting

**Metrics**:

1. **Rate Limit Hit Rate**: % of requests rejected
2. **Top Rate-Limited Users**: Who's hitting limits most?
3. **Endpoint-Specific Limits**: Which endpoints rate-limited most?
4. **Redis Latency**: p95, p99 latency for rate limit checks

**Alerts**:
- Rate limit hit rate > 10% → Investigate (DDoS? Legitimate spike?)
- Redis latency > 10ms → Scale Redis cluster
- Redis errors > 1% → Failover to replica

---

## Trade-offs

**Accuracy vs Performance**:
- Sliding window log: Most accurate, memory intensive
- Fixed window: Fast, boundary issues
- Sliding window counter: Balanced (recommended)

**Strict vs Lenient**:
- Strict: Reject at exactly 100 requests (poor UX if user retries)
- Lenient: Allow 105 requests (5% buffer), smoother experience

**Centralized vs Distributed**:
- Redis (centralized): Accurate, single point of failure
- Local counters (distributed): Fault-tolerant, less accurate

---

## Interview Tips

**Clarify**:
1. Limits: Per-user? Per-IP? Per-API-key?
2. Window: Fixed or sliding? Duration?
3. Distribution: Single server or distributed?
4. Scale: QPS? Number of users?

**Emphasize**:
1. **Algorithm Choice**: Sliding window counter (best for distributed systems)
2. **Redis Implementation**: Lua scripts for atomicity
3. **Response Headers**: X-RateLimit-Limit, Remaining, Reset
4. **Graceful Degradation**: Return 429 with Retry-After
5. **Monitoring**: Track hit rates, latencies

**Common Mistakes**:
- Using sliding window log for high traffic (memory explosion)
- No atomicity (race conditions with INCR/GET)
- Not handling Redis failures (fail closed = API down)
- Fixed window without considering boundary issues

**Follow-up Questions**:
- "How to handle distributed clock skew? (Use Redis timestamp)"
- "What if Redis goes down? (Fail open or local fallback)"
- "How to rate limit by IP for anonymous users? (IP-based keys)"
- "How to prevent abuse of free trials? (IP + email + credit card fingerprinting)"

---

## Summary

**Core Algorithm**: **Sliding Window Counter** (best balance of accuracy, memory, distributed systems)

**Implementation**:
- Redis for shared state (10,000 API servers)
- Lua scripts for atomic operations (no race conditions)
- Keys: \`rate_limit:user:{user_id}:{window_start}\`
- TTL: 2 × window size (keep current + previous window)

**Response Format** (HTTP 429):
\`\`\`http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1672531260
Retry-After: 45
\`\`\`

**Key Decisions**:
- ✅ Sliding window counter (not fixed window or log)
- ✅ Redis with Lua scripts (atomic, distributed)
- ✅ Per-user + per-endpoint limits
- ✅ Graceful degradation (fail open if Redis down)
- ✅ Monitoring rate limit hit rates

**Capacity**:
- 1 million requests/second
- 100 million users
- < 5ms latency overhead
- 99.99% accuracy (sliding window approximation)

A production rate limiter ensures **fair resource allocation**, **prevents abuse**, and **maintains service quality** for all users while handling massive scale across distributed systems.`,
};
