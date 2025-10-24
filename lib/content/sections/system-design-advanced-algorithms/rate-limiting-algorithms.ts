/**
 * Rate Limiting Algorithms Section
 */

export const ratelimitingalgorithmsSection = {
  id: 'rate-limiting-algorithms',
  title: 'Rate Limiting Algorithms',
  content: `Rate limiting is a critical technique for protecting systems from abuse and overload. It answers: **"How do we limit the number of requests a user can make in a time period?"**

## Why Rate Limiting?

**Without rate limiting**:
- ❌ DDoS attacks overwhelm servers
- ❌ Abusive scrapers exhaust resources
- ❌ Bugs cause infinite request loops
- ❌ Expensive API costs spiral out of control

**With rate limiting**:
- ✅ Protect against abuse (100 req/min per user)
- ✅ Ensure fair resource allocation
- ✅ Prevent cost overruns (API quotas)
- ✅ Maintain SLA for paying customers

---

## Core Rate Limiting Algorithms

### 1. Token Bucket

**Concept**: Bucket holds tokens. Each request consumes 1 token. Tokens refill at constant rate.

\`\`\`
Capacity: 10 tokens
Refill rate: 2 tokens/second

Initial: [🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙] (10 tokens)
Request 1: [🪙🪙🪙🪙🪙🪙🪙🪙🪙] (9 tokens, allowed)
After 1 sec: [🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙] (11 tokens, capped at 10)
Burst of 10: [] (0 tokens)
Immediate 11th request: DENIED (no tokens)
After 5 sec: [🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙] (refilled to 10)
\`\`\`

**Properties**:
- ✅ Allows bursts (up to bucket capacity)
- ✅ Smooth average rate
- ✅ Simple to implement

\`\`\`python
import time

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity  # Max tokens
        self.tokens = capacity  # Current tokens
        self.refill_rate = refill_rate  # Tokens per second
        self.last_refill = time.time()
    
    def allow_request(self):
        # Refill tokens based on time passed
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
        
        # Check if request allowed
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

# Usage: 10 requests/second, burst of 20
bucket = TokenBucket(capacity=20, refill_rate=10)
\`\`\`

**Use cases**: API rate limiting (AWS, Stripe), network traffic shaping

---

### 2. Leaky Bucket

**Concept**: Requests enter queue (bucket). Process at constant rate (leak).

\`\`\`
Queue capacity: 10 requests
Process rate: 2 requests/second

Requests arrive: [RRRRRR] (6 requests queued)
After 1 sec: [RRRR] (processed 2, 4 remain)
Burst of 8: [RRRRRRRRRRRR] (12 requests)
→ 2 requests DROPPED (queue full at 10)
\`\`\`

**Properties**:
- ✅ Constant output rate (smooth traffic)
- ✅ Protects downstream services
- ❌ Queuing delay for bursts
- ❌ Drops requests when queue full

\`\`\`python
from collections import deque
import time

class LeakyBucket:
    def __init__(self, capacity, leak_rate):
        self.capacity = capacity
        self.queue = deque()
        self.leak_rate = leak_rate  # Requests per second
        self.last_leak = time.time()
    
    def add_request(self, request):
        self._leak()
        
        if len(self.queue) < self.capacity:
            self.queue.append(request)
            return True
        return False  # Queue full, reject
    
    def _leak(self):
        now = time.time()
        elapsed = now - self.last_leak
        leaks = int(elapsed * self.leak_rate)
        
        for _ in range(min(leaks, len(self.queue))):
            self.queue.popleft()  # Process request
        
        self.last_refill = now
\`\`\`

**Use cases**: Traffic shaping, protecting downstream services

---

### 3. Fixed Window Counter

**Concept**: Count requests in fixed time windows (e.g., 0-60s, 60-120s).

\`\`\`
Limit: 100 requests per minute

Window 1 (0-60s):
  00:00 - 00:59: 100 requests → All allowed
  00:59: Counter = 100

Window 2 (60-120s):  
  01:00: Counter resets to 0
  01:00 - 01:59: 100 requests → All allowed
\`\`\`

**Problem: Burst at boundary**

\`\`\`
00:30 - 00:59: 100 requests (allowed)
01:00 - 01:29: 100 requests (allowed)  

→ 200 requests in 1 minute (00:30 - 01:30)!
   Double the limit!
\`\`\`

**Implementation**:

\`\`\`python
import time

class FixedWindowCounter:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size  # seconds
        self.counter = 0
        self.window_start = time.time()
    
    def allow_request(self):
        now = time.time()
        
        # New window?
        if now - self.window_start >= self.window_size:
            self.counter = 0
            self.window_start = now
        
        # Check limit
        if self.counter < self.limit:
            self.counter += 1
            return True
        return False

# Usage: 100 requests per 60 seconds
limiter = FixedWindowCounter(limit=100, window_size=60)
\`\`\`

**Properties**:
- ✅ Simple, memory efficient
- ❌ Burst at window boundaries (2x limit possible)

**Use cases**: Simple rate limiting when burst is acceptable

---

### 4. Sliding Window Log

**Concept**: Store timestamp of each request. Count requests in last N seconds.

\`\`\`
Limit: 10 requests per minute

Timestamps: [10:00:05, 10:00:15, 10:00:30, ...]

Request at 10:01:00:
  - Remove timestamps < 10:00:00 (older than 60s)
  - Count remaining timestamps
  - If count < 10: Allow
\`\`\`

**Implementation**:

\`\`\`python
import time
from collections import deque

class SlidingWindowLog:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.requests = deque()  # Timestamps
    
    def allow_request(self):
        now = time.time()
        
        # Remove old timestamps
        while self.requests and self.requests[0] < now - self.window_size:
            self.requests.popleft()
        
        # Check limit
        if len(self.requests) < self.limit:
            self.requests.append(now)
            return True
        return False
\`\`\`

**Properties**:
- ✅ Accurate (no boundary issues)
- ❌ Memory intensive (stores every timestamp)
- ❌ O(N) lookup time (scan old timestamps)

**Use cases**: When accuracy critical, traffic moderate

---

### 5. Sliding Window Counter

**Concept**: Combine fixed windows with weighted average.

\`\`\`
Limit: 100 requests per minute

11:00:40 request:
  - Current window (11:00-11:01): 70 requests
  - Previous window (10:59-11:00): 80 requests
  - Time in current window: 40s (66.7%)
  
Estimate = (80 * 33.3%) + (70 * 100%)
         = 26.7 + 70
         = 96.7 requests
         
< 100 → Allow
\`\`\`

**Implementation**:

\`\`\`python
import time

class SlidingWindowCounter:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.current_window_start = time.time()
        self.current_count = 0
        self.previous_count = 0
    
    def allow_request(self):
        now = time.time()
        
        # Roll window if needed
        elapsed = now - self.current_window_start
        if elapsed >= self.window_size:
            self.previous_count = self.current_count
            self.current_count = 0
            self.current_window_start = now
        
        # Calculate weighted count
        elapsed_percent = elapsed / self.window_size
        previous_weight = 1 - elapsed_percent
        estimated_count = (self.previous_count * previous_weight + 
                          self.current_count)
        
        # Check limit
        if estimated_count < self.limit:
            self.current_count += 1
            return True
        return False
\`\`\`

**Properties**:
- ✅ Smooth (no boundary bursts)
- ✅ Memory efficient (only 2 counters)
- ✅ Good approximation

**Use cases**: Production systems (Cloudflare, Kong)

---

## Distributed Rate Limiting

**Challenge**: Multiple servers must coordinate limits.

### Redis-Based Rate Limiting

\`\`\`python
import redis
import time

class DistributedRateLimiter:
    def __init__(self, redis_client, limit, window):
        self.redis = redis_client
        self.limit = limit
        self.window = window
    
    def allow_request(self, user_id):
        key = f"rate_limit:{user_id}"
        now = int(time.time())
        
        # Fixed window with Redis
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, self.window)
        results = pipe.execute()
        
        count = results[0]
        return count <= self.limit

# Usage
redis_client = redis.Redis()
limiter = DistributedRateLimiter(redis_client, limit=100, window=60)

if limiter.allow_request("user123"):
    # Process request
    pass
else:
    # Return 429 Too Many Requests
    pass
\`\`\`

### Sliding Window in Redis

\`\`\`lua
-- Lua script (atomic in Redis)
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

-- Remove old entries
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- Count current requests  
local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window)
    return 1
else
    return 0
end
\`\`\`

---

## Algorithm Comparison

| Algorithm | Memory | Accuracy | Bursts | Use Case |
|-----------|--------|----------|--------|----------|
| **Token Bucket** | O(1) | Good | Allows | API rate limiting |
| **Leaky Bucket** | O(N) | Perfect | Smooths | Traffic shaping |
| **Fixed Window** | O(1) | Poor | Boundary issue | Simple limits |
| **Sliding Log** | O(N) | Perfect | Prevents | Critical systems |
| **Sliding Counter** | O(1) | Good | Prevents | Production (best) |

**Recommendation**: Sliding Window Counter for production

---

## Real-World Examples

### 1. Stripe API (Token Bucket)

\`\`\`
Rate limits:
- 100 requests/second (burst)
- Sustained: 1000 requests/hour

Returns:
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 73
X-RateLimit-Reset: 1625097600
\`\`\`

### 2. GitHub API (Fixed Window)

\`\`\`
5000 requests per hour (authenticated)
60 requests per hour (unauthenticated)

Reset at start of each hour
\`\`\`

### 3. Twitter API (Sliding Window)

\`\`\`
15-minute windows
300 requests per 15 minutes

Smooth enforcement (no boundary bursts)
\`\`\`

---

## Interview Tips

### Key Points

1. **Problem**: Protect against abuse, ensure fairness
2. **Algorithms**: Token bucket (bursts), Sliding window (smooth), Fixed window (simple)
3. **Distributed**: Redis for coordination
4. **Trade-offs**: Accuracy vs memory vs complexity
5. **Production**: Sliding window counter (Cloudflare, Kong)

### Common Questions

**"Which algorithm would you use?"**
- Sliding window counter: Best balance (accurate, memory-efficient)
- Token bucket: Allow reasonable bursts (API limits)
- Fixed window: Simple but boundary issues

**"How do you implement distributed rate limiting?"**
- Redis with atomic operations (INCR + EXPIRE)
- Lua scripts for sliding window (atomic)
- Alternative: Local limits + eventual consistency

**"How do you handle 429 responses?"**
- Return Retry-After header
- Exponential backoff on client
- Queue non-critical requests

### Design Exercise

Design rate limiting for API gateway:

\`\`\`
Requirement: 1000 req/min per user, 10K users

Solution:
1. Algorithm: Sliding window counter (smooth, efficient)
2. Storage: Redis cluster (distributed)
3. Key: "rate_limit:{user_id}:{minute}"
4. On request:
   - Check Redis counter
   - If < 1000: Increment, allow
   - If >= 1000: Return 429
5. Headers:
   - X-RateLimit-Limit: 1000
   - X-RateLimit-Remaining: 247
   - Retry-After: 42 (seconds)
6. Scale: 10K users × 1 key × 8 bytes = 80 KB (tiny!)
\`\`\`

---

## Summary

**Rate limiting algorithms** protect systems from abuse and overload.

**Key algorithms**:
- ✅ **Token Bucket**: Allows bursts (AWS, Stripe)
- ✅ **Sliding Window Counter**: Production standard (Cloudflare)
- ✅ **Leaky Bucket**: Smooth traffic (network)
- ❌ **Fixed Window**: Simple but boundary bursts

**Distributed**: Redis with atomic operations

Understanding rate limiting is **essential** for API design and system protection.`,
};
