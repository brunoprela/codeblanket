/**
 * Rate Limiting & Counters Section
 */

export const ratelimitingSection = {
  id: 'rate-limiting',
  title: 'Rate Limiting & Counters',
  content: `Rate limiting controls how frequently users can perform actions. It's critical for:
- **API protection**: Prevent abuse and overload
- **Fair usage**: Ensure all users get access
- **Cost control**: Limit expensive operations
- **Security**: Prevent brute force attacks

**Real-world examples**: Twitter limits tweets/hour, APIs limit requests/second, login attempts limited per minute.

---

## Design Hit Counter

**Problem**: Count hits in the last N seconds (typically 300 = 5 minutes).

**Operations**:
- \`hit(timestamp)\`: Record a hit at given time
- \`getHits(timestamp)\`: Return hits in last N seconds

### Approach 1: Queue/Deque (Simple)

**Idea**: Store all timestamps, remove old ones.

\`\`\`python
from collections import deque

class HitCounter:
    def __init__(self):
        self.hits = deque()  # Store timestamps
        self.window = 300  # 5 minutes
    
    def hit(self, timestamp):
        self.hits.append(timestamp)  # O(1)
    
    def getHits(self, timestamp):
        # Remove hits older than timestamp - window
        while self.hits and self.hits[0] <= timestamp - self.window:
            self.hits.popleft()  # O(1) per old hit
        return len(self.hits)  # O(1)
\`\`\`

**Time Complexity**:
- hit(): O(1)
- getHits(): O(N) worst case if many old hits to remove, but amortized O(1) per hit

**Space Complexity**: O(N) where N is total hits in window

**Pros**: Simple, exact count  
**Cons**: Memory grows with hit count

### Approach 2: Time Buckets (Optimized)

**Idea**: Divide time into buckets, store count per bucket.

\`\`\`python
class HitCounter:
    def __init__(self):
        self.buckets = [0] * 300  # 300 seconds
        self.timestamps = [0] * 300  # Last update time per bucket
    
    def hit(self, timestamp):
        idx = timestamp % 300
        # If bucket from old window, reset it
        if self.timestamps[idx] != timestamp:
            self.timestamps[idx] = timestamp
            self.buckets[idx] = 1
        else:
            self.buckets[idx] += 1
    
    def getHits(self, timestamp):
        total = 0
        for i in range(300):
            # Only count if timestamp is within window
            if timestamp - self.timestamps[i] < 300:
                total += self.buckets[i]
        return total
\`\`\`

**Time Complexity**:
- hit(): O(1)
- getHits(): O(300) = O(1) for fixed window

**Space Complexity**: O(300) = O(1)

**Pros**: Fixed memory, good for high traffic  
**Cons**: Less accurate (bucket granularity), getHits() scans all buckets

### Approach 3: Hybrid (Best of Both)

**Idea**: Use deque but with bucketing for extreme scale.

\`\`\`python
class HitCounter:
    def __init__(self):
        self.hits = deque()  # Store (timestamp, count) pairs
        self.window = 300
    
    def hit(self, timestamp):
        # If last hit is same second, increment count
        if self.hits and self.hits[-1][0] == timestamp:
            self.hits[-1] = (timestamp, self.hits[-1][1] + 1)
        else:
            self.hits.append((timestamp, 1))
    
    def getHits(self, timestamp):
        # Remove old entries
        while self.hits and self.hits[0][0] <= timestamp - self.window:
            self.hits.popleft()
        
        return sum(count for ts, count in self.hits)
\`\`\`

**Optimization**: Stores (timestamp, count) instead of individual hits. If 1000 hits in same second, stores once instead of 1000 times.

---

## Rate Limiter Algorithms

### 1. Fixed Window Counter

**Idea**: Count requests in fixed time windows (e.g., 0-60s, 60-120s).

\`\`\`python
class FixedWindowRateLimiter:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.window_start = 0
        self.count = 0
    
    def allow_request(self, timestamp):
        # Check if we're in a new window
        window_num = timestamp // self.window_size
        if window_num > self.window_start:
            # Reset for new window
            self.window_start = window_num
            self.count = 0
        
        # Check if under limit
        if self.count < self.limit:
            self.count += 1
            return True
        return False
\`\`\`

**Problem**: Boundary spike issue!
\`\`\`
Window 1: [0-60s] - 100 requests at t=59s (allowed)
Window 2: [60-120s] - 100 requests at t=60s (allowed)
→ 200 requests in 2 seconds! (burst at boundary)
\`\`\`

### 2. Sliding Window Log

**Idea**: Store timestamp of each request, count in sliding window.

\`\`\`python
class SlidingWindowLog:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.requests = deque()  # Store timestamps
    
    def allow_request(self, timestamp):
        # Remove old requests outside window
        cutoff = timestamp - self.window_size
        while self.requests and self.requests[0] <= cutoff:
            self.requests.popleft()
        
        # Check if under limit
        if len(self.requests) < self.limit:
            self.requests.append(timestamp)
            return True
        return False
\`\`\`

**Pros**: Accurate, no boundary issues  
**Cons**: O(N) memory for N requests

### 3. Sliding Window Counter (Best)

**Idea**: Weighted combination of previous and current window.

\`\`\`python
class SlidingWindowCounter:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.prev_count = 0
        self.curr_count = 0
        self.curr_window_start = 0
    
    def allow_request(self, timestamp):
        window_num = timestamp // self.window_size
        
        if window_num > self.curr_window_start:
            # Move to new window
            self.prev_count = self.curr_count
            self.curr_count = 0
            self.curr_window_start = window_num
        
        # Calculate weighted count
        elapsed = timestamp % self.window_size
        weight = 1 - (elapsed / self.window_size)
        estimated_count = self.prev_count * weight + self.curr_count
        
        if estimated_count < self.limit:
            self.curr_count += 1
            return True
        return False
\`\`\`

**Pros**: O(1) memory, accurate approximation  
**Cons**: Slightly complex math

### 4. Token Bucket (Industry Standard)

**Idea**: Bucket fills with tokens at fixed rate. Requests consume tokens.

\`\`\`python
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity  # Max tokens
        self.tokens = capacity  # Current tokens
        self.refill_rate = refill_rate  # Tokens per second
        self.last_refill = time.time()
    
    def allow_request(self):
        now = time.time()
        # Refill tokens based on time elapsed
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, 
                         self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
\`\`\`

**Pros**: 
- Allows bursts (up to capacity)
- Smooth rate limiting
- Used by AWS, GCP, most APIs

**Example**: Capacity=10, rate=1/sec
- Can burst 10 requests immediately
- Then limited to 1 per second
- Unused capacity accumulates (up to 10)

---

## Comparison Table

| Algorithm | Memory | Accuracy | Burst Handling | Use Case |
|-----------|--------|----------|----------------|----------|
| **Fixed Window** | O(1) | ❌ Boundary spikes | ❌ Double at boundary | Simple systems |
| **Sliding Log** | O(N) | ✅ Perfect | ✅ Perfect | Low traffic |
| **Sliding Counter** | O(1) | ✅ Good | ✅ Good | High traffic |
| **Token Bucket** | O(1) | ✅ Good | ✅ Controlled | Production APIs |

---

## Design Considerations

### Distributed Systems

**Problem**: Rate limiting across multiple servers

**Solutions**:
1. **Redis with atomic counters**: INCR and EXPIRE
2. **Sliding window in Redis**: Sorted sets with timestamps
3. **Token bucket in Redis**: Store tokens and last_refill

\`\`\`python
# Redis-based rate limiter
def allow_request(user_id, redis_client):
    key = f"rate_limit:{user_id}"
    current = redis_client.incr(key)
    
    if current == 1:
        redis_client.expire(key, 60)  # 60 second window
    
    return current <= 100  # 100 requests per minute
\`\`\`

### Per-User vs Global

- **Per-user**: Each user has own limit (fair)
- **Global**: Total limit across all users (protect system)
- **Hybrid**: Per-user + global cap

### 429 Status Code

Return "429 Too Many Requests" with headers:
\`\`\`
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1617891234
Retry-After: 60
\`\`\`

---

## Interview Tips

1. **Clarify requirements**:
   - What's the rate? (100/min, 1000/hour?)
   - Per user or global?
   - Fixed or sliding window?
   - Distributed or single server?

2. **Start simple**: Fixed window, then discuss improvements

3. **Mention production**: "In production, I'd use Token Bucket with Redis"

4. **Discuss trade-offs**: Memory vs accuracy, simplicity vs perfect fairness

5. **Test edge cases**: Boundary times, burst traffic, user with 0 requests

**Common Mistake**: Not removing old timestamps in sliding window implementations!`,
};
