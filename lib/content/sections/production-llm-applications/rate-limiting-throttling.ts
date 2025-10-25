export const rateLimitingThrottlingContent = `
# Rate Limiting & Throttling

## Introduction

Rate limiting is critical for production LLM applications to prevent abuse, manage costs, stay within provider limits, and ensure fair resource distribution. Without proper rate limiting, a single user could consume your entire API quota, rack up thousands in costs, or cause service degradation for others.

In this section, we'll explore various rate limiting algorithms, implementation strategies, and best practices for LLM applications. We'll cover token bucket, sliding window, per-user limits, and how to handle rate limit errors gracefully.

## Why Rate Limiting Matters for LLM Apps

**Cost Control**: A runaway script could make thousands of expensive API calls in minutes, costing thousands of dollars.

**Provider Limits**: LLM providers enforce rate limits. You need to stay within them and distribute capacity fairly among users.

**Fair Usage**: Prevent a single user from monopolizing resources and degrading service for others.

**Abuse Prevention**: Detect and block malicious actors trying to abuse your service.

**Quality of Service**: Maintain consistent performance by preventing system overload.

### Real-World Example

Without rate limiting:
- User submits 10,000 requests in 1 minute
- Each request costs $0.01
- Total cost: $100 in one minute = $144,000/day
- Your LLM provider blocks you for exceeding limits
- Other users experience timeouts and failures

With rate limiting:
- User limited to 100 requests/hour
- Maximum cost per user: $1/hour = $24/day
- Service remains stable for all users
- You stay within provider limits

## Token Bucket Algorithm

The token bucket algorithm is the most popular for LLM rate limiting because it allows burst traffic while maintaining an average rate.

### How It Works

1. A "bucket" holds tokens, with a maximum capacity
2. Tokens are added to the bucket at a constant rate
3. Each request consumes one or more tokens
4. Requests are rejected if not enough tokens available
5. Allows bursts when bucket is full, then throttles to steady rate

\`\`\`python
import time
from typing import Optional
from dataclasses import dataclass
from threading import Lock

@dataclass
class TokenBucket:
    """
    Token bucket rate limiter.
    
    Allows burst traffic up to capacity, then steady-state at rate.
    """
    
    rate: float  # Tokens per second
    capacity: float  # Maximum tokens in bucket
    tokens: float = None  # Current tokens
    last_update: float = None  # Last update timestamp
    lock: Lock = None
    
    def __post_init__(self):
        if self.tokens is None:
            self.tokens = self.capacity
        if self.last_update is None:
            self.last_update = time.monotonic()
        if self.lock is None:
            self.lock = Lock()
    
    def consume (self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            
            # Refill tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            # Check if enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def wait_time (self, tokens: int = 1) -> float:
        """
        Calculate how long to wait until tokens available.
        """
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.rate


# Usage example
limiter = TokenBucket (rate=10.0, capacity=100.0)  # 10 requests/sec, burst of 100

# Fast requests work (burst)
for i in range(100):
    if limiter.consume():
        print(f"Request {i} allowed")
    else:
        print(f"Request {i} rejected")

# After burst, steady-state at 10 req/sec
time.sleep(1)
if limiter.consume():
    print("Request after 1 second allowed")
\`\`\`

### Production Implementation

\`\`\`python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import redis
import time
import json

app = FastAPI()
redis_client = redis.Redis (host='localhost', port=6379, decode_responses=True)

class DistributedTokenBucket:
    """
    Token bucket using Redis for distributed rate limiting.
    Works across multiple server instances.
    """
    
    def __init__(self, redis_client, rate: float, capacity: float):
        self.redis = redis_client
        self.rate = rate
        self.capacity = capacity
    
    def consume (self, key: str, tokens: int = 1) -> tuple[bool, dict]:
        """
        Try to consume tokens for a given key.
        
        Returns:
            (success, info_dict) where info_dict contains remaining tokens
        """
        now = time.time()
        
        # Lua script for atomic token bucket operation
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local rate = tonumber(ARGV[2])
        local tokens_to_consume = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        -- Get current bucket state
        local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
        local tokens = tonumber (bucket[1]) or capacity
        local last_update = tonumber (bucket[2]) or now
        
        -- Calculate token refill
        local elapsed = now - last_update
        tokens = math.min (capacity, tokens + elapsed * rate)
        
        -- Try to consume
        if tokens >= tokens_to_consume then
            tokens = tokens - tokens_to_consume
            redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
            redis.call('EXPIRE', key, 3600)  -- 1 hour expiry
            return {1, tokens}  -- Success
        else
            redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
            redis.call('EXPIRE', key, 3600)
            return {0, tokens}  -- Failure
        end
        """
        
        # Execute atomic operation
        result = self.redis.eval(
            lua_script,
            1,  # Number of keys
            key,
            self.capacity,
            self.rate,
            tokens,
            now
        )
        
        success = bool (result[0])
        remaining_tokens = float (result[1])
        
        return success, {
            'remaining': int (remaining_tokens),
            'limit': int (self.capacity),
            'reset_in': int((self.capacity - remaining_tokens) / self.rate) if not success else 0
        }


# Rate limiter instance
limiter = DistributedTokenBucket(
    redis_client,
    rate=10.0,  # 10 requests per second
    capacity=100.0  # Burst of 100
)


@app.middleware("http")
async def rate_limit_middleware (request: Request, call_next):
    """Apply rate limiting to all requests."""
    
    # Skip docs
    if request.url.path in ['/docs', '/openapi.json']:
        return await call_next (request)
    
    # Get API key
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"error": "API key required"}
        )
    
    # Check rate limit
    success, info = limiter.consume (f"ratelimit:{api_key}", tokens=1)
    
    if not success:
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": "Too many requests",
                "retry_after": info['reset_in']
            },
            headers={
                "X-RateLimit-Limit": str (info['limit']),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str (int (time.time()) + info['reset_in']),
                "Retry-After": str (info['reset_in'])
            }
        )
    
    # Process request
    response = await call_next (request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str (info['limit'])
    response.headers["X-RateLimit-Remaining"] = str (info['remaining'])
    
    return response
\`\`\`

## Sliding Window Algorithm

Sliding window provides more accurate rate limiting than fixed windows.

\`\`\`python
from collections import deque
from datetime import datetime, timedelta

class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.
    
    Tracks exact timestamps of requests within window.
    More accurate than fixed windows but uses more memory.
    """
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = timedelta (seconds=window_seconds)
        self.requests = {}  # key -> deque of timestamps
    
    def is_allowed (self, key: str) -> tuple[bool, dict]:
        """
        Check if request is allowed.
        
        Returns:
            (allowed, info_dict)
        """
        now = datetime.utcnow()
        window_start = now - self.window
        
        # Get or create request queue for this key
        if key not in self.requests:
            self.requests[key] = deque()
        
        request_queue = self.requests[key]
        
        # Remove old requests outside window
        while request_queue and request_queue[0] < window_start:
            request_queue.popleft()
        
        # Check if under limit
        current_count = len (request_queue)
        
        if current_count < self.max_requests:
            # Allow request
            request_queue.append (now)
            return True, {
                'remaining': self.max_requests - current_count - 1,
                'limit': self.max_requests,
                'reset_at': (request_queue[0] + self.window).isoformat() if request_queue else None
            }
        else:
            # Reject request
            oldest_request = request_queue[0]
            reset_at = oldest_request + self.window
            retry_after = int((reset_at - now).total_seconds())
            
            return False, {
                'remaining': 0,
                'limit': self.max_requests,
                'reset_at': reset_at.isoformat(),
                'retry_after': max(1, retry_after)
            }


# Usage
limiter = SlidingWindowRateLimiter (max_requests=100, window_seconds=3600)  # 100/hour

allowed, info = limiter.is_allowed("user:123")
if allowed:
    print(f"Request allowed. {info['remaining']} remaining.")
else:
    print(f"Rate limit exceeded. Retry in {info['retry_after']}s")
\`\`\`

## Tiered Rate Limiting

Different limits for different user tiers:

\`\`\`python
from enum import Enum

class UserTier(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class TieredRateLimiter:
    """
    Rate limiter with different limits per user tier.
    """
    
    TIER_LIMITS = {
        UserTier.FREE: {
            'requests_per_minute': 10,
            'requests_per_day': 1000,
            'burst_capacity': 20
        },
        UserTier.PRO: {
            'requests_per_minute': 100,
            'requests_per_day': 50000,
            'burst_capacity': 200
        },
        UserTier.ENTERPRISE: {
            'requests_per_minute': 1000,
            'requests_per_day': 1000000,
            'burst_capacity': 2000
        }
    }
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def get_user_tier (self, api_key: str) -> UserTier:
        """Get user tier from database."""
        user = get_user_by_api_key (api_key)
        return UserTier (user.get('tier', 'free'))
    
    def check_limit (self, api_key: str) -> tuple[bool, dict]:
        """
        Check all rate limits for user.
        """
        tier = self.get_user_tier (api_key)
        limits = self.TIER_LIMITS[tier]
        
        # Check per-minute limit
        minute_key = f"rl:minute:{api_key}:{int (time.time() // 60)}"
        minute_count = self.redis.incr (minute_key)
        self.redis.expire (minute_key, 60)
        
        if minute_count > limits['requests_per_minute']:
            return False, {
                'error': 'minute_limit_exceeded',
                'limit': limits['requests_per_minute'],
                'period': 'minute'
            }
        
        # Check per-day limit
        day_key = f"rl:day:{api_key}:{datetime.utcnow().date()}"
        day_count = self.redis.incr (day_key)
        self.redis.expire (day_key, 86400)
        
        if day_count > limits['requests_per_day']:
            return False, {
                'error': 'daily_limit_exceeded',
                'limit': limits['requests_per_day'],
                'period': 'day'
            }
        
        # Check burst limit using token bucket
        bucket = DistributedTokenBucket(
            self.redis,
            rate=limits['requests_per_minute'] / 60,
            capacity=limits['burst_capacity']
        )
        
        success, info = bucket.consume (f"burst:{api_key}")
        
        if not success:
            return False, {
                'error': 'burst_limit_exceeded',
                **info
            }
        
        # All limits passed
        return True, {
            'tier': tier.value,
            'minute_remaining': limits['requests_per_minute'] - minute_count,
            'day_remaining': limits['requests_per_day'] - day_count,
            'burst_remaining': info['remaining']
        }


# FastAPI integration
limiter = TieredRateLimiter (redis_client)

@app.middleware("http")
async def tiered_rate_limit (request: Request, call_next):
    """Apply tiered rate limiting."""
    api_key = request.headers.get('X-API-Key')
    
    if not api_key:
        return JSONResponse (status_code=401, content={"error": "API key required"})
    
    allowed, info = limiter.check_limit (api_key)
    
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                **info
            }
        )
    
    response = await call_next (request)
    
    # Add tier info to response headers
    response.headers["X-RateLimit-Tier"] = info['tier']
    response.headers["X-RateLimit-Remaining-Minute"] = str (info['minute_remaining'])
    response.headers["X-RateLimit-Remaining-Day"] = str (info['day_remaining'])
    
    return response
\`\`\`

## Cost-Based Rate Limiting

Limit based on cost rather than just request count:

\`\`\`python
class CostBasedRateLimiter:
    """
    Rate limit based on API cost rather than request count.
    
    Useful for LLM APIs where costs vary by request.
    """
    
    def __init__(self, redis_client, max_cost_per_hour: float = 10.0):
        self.redis = redis_client
        self.max_cost_per_hour = max_cost_per_hour
    
    def estimate_cost (self, prompt: str, model: str) -> float:
        """
        Estimate cost of a request.
        """
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        input_tokens = len (prompt) // 4
        output_tokens = 500  # Estimate average output
        
        # Cost per 1K tokens
        costs = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-3.5-turbo': {'input': 0.001, 'output': 0.002}
        }
        
        model_cost = costs.get (model, costs['gpt-3.5-turbo'])
        
        cost = (
            (input_tokens / 1000) * model_cost['input'] +
            (output_tokens / 1000) * model_cost['output']
        )
        
        return cost
    
    def check_limit (self, api_key: str, estimated_cost: float) -> tuple[bool, dict]:
        """
        Check if request is within cost budget.
        """
        hour_key = f"cost:hour:{api_key}:{int (time.time() // 3600)}"
        
        # Get current spend this hour
        current_spend = float (self.redis.get (hour_key) or 0.0)
        
        # Check if this request would exceed budget
        if current_spend + estimated_cost > self.max_cost_per_hour:
            return False, {
                'current_spend': current_spend,
                'estimated_cost': estimated_cost,
                'budget': self.max_cost_per_hour,
                'remaining': max(0, self.max_cost_per_hour - current_spend)
            }
        
        # Track estimated spend
        pipe = self.redis.pipeline()
        pipe.incrbyfloat (hour_key, estimated_cost)
        pipe.expire (hour_key, 3600)
        pipe.execute()
        
        return True, {
            'estimated_cost': estimated_cost,
            'total_spend': current_spend + estimated_cost,
            'budget': self.max_cost_per_hour,
            'remaining': self.max_cost_per_hour - (current_spend + estimated_cost)
        }
    
    def record_actual_cost (self, api_key: str, actual_cost: float):
        """
        Record actual cost after request completes.
        Adjusts the hourly tracking.
        """
        hour_key = f"cost:hour:{api_key}:{int (time.time() // 3600)}"
        self.redis.incrbyfloat (hour_key, actual_cost)
        self.redis.expire (hour_key, 3600)


# Usage in endpoint
cost_limiter = CostBasedRateLimiter (redis_client, max_cost_per_hour=10.0)

@app.post("/generate")
async def generate (prompt: str, model: str, api_key: str):
    """Generate with cost-based rate limiting."""
    
    # Estimate cost
    estimated_cost = cost_limiter.estimate_cost (prompt, model)
    
    # Check limit
    allowed, info = cost_limiter.check_limit (api_key, estimated_cost)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "cost_limit_exceeded",
                "message": f"Would exceed hourly budget of \${info['budget']:.2f}",
    ** info
            }
        )
    
    # Make LLM call
response = openai.ChatCompletion.create(
    model = model,
    messages = [{ "role": "user", "content": prompt }]
)
    
    # Calculate actual cost
actual_cost = calculate_actual_cost (response.usage, model)
    
    # Record actual cost
cost_limiter.record_actual_cost (api_key, actual_cost)

return {
    "result": response.choices[0].message.content,
    "cost": actual_cost,
    "remaining_budget": info['remaining']
}
\`\`\`

## Graceful Degradation

Handle rate limits gracefully rather than failing hard:

\`\`\`python
from enum import Enum

class FallbackStrategy(Enum):
    QUEUE = "queue"  # Queue for later processing
    CACHE = "cache"  # Return cached/stale response
    CHEAPER_MODEL = "cheaper_model"  # Use cheaper model
    REJECT = "reject"  # Reject request

class GracefulRateLimiter:
    """
    Rate limiter with graceful degradation strategies.
    """
    
    def __init__(self, redis_client, cache, queue):
        self.redis = redis_client
        self.cache = cache
        self.queue = queue
        self.primary_limiter = DistributedTokenBucket(
            redis_client,
            rate=10.0,
            capacity=100.0
        )
    
    async def handle_request(
        self,
        api_key: str,
        prompt: str,
        model: str = "gpt-4",
        fallback_strategy: FallbackStrategy = FallbackStrategy.CHEAPER_MODEL
    ):
        """
        Handle request with graceful degradation.
        """
        # Try primary rate limit
        success, info = self.primary_limiter.consume (f"primary:{api_key}")
        
        if success:
            # Within limits, proceed normally
            return await self._call_llm (prompt, model)
        
        # Rate limited, apply fallback strategy
        if fallback_strategy == FallbackStrategy.QUEUE:
            # Queue for later processing
            task_id = self.queue.enqueue(
                process_prompt,
                prompt=prompt,
                model=model,
                api_key=api_key
            )
            return {
                "status": "queued",
                "task_id": task_id,
                "message": "Request queued due to rate limit"
            }
        
        elif fallback_strategy == FallbackStrategy.CACHE:
            # Try to return cached response
            cached = self.cache.get_semantic_match (prompt, threshold=0.85)
            if cached:
                return {
                    "result": cached,
                    "cached": True,
                    "message": "Returning similar cached response due to rate limit"
                }
            else:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded and no cached response available"
                )
        
        elif fallback_strategy == FallbackStrategy.CHEAPER_MODEL:
            # Try with cheaper model
            if model == "gpt-4":
                cheaper_model = "gpt-3.5-turbo"
                
                # Check secondary rate limit for cheaper model
                secondary_limiter = DistributedTokenBucket(
                    self.redis,
                    rate=50.0,  # Higher rate for cheaper model
                    capacity=500.0
                )
                
                success, _ = secondary_limiter.consume (f"secondary:{api_key}")
                
                if success:
                    result = await self._call_llm (prompt, cheaper_model)
                    result['model_downgraded'] = True
                    result['message'] = "Used cheaper model due to rate limit"
                    return result
            
            # Fall through to reject
            raise HTTPException (status_code=429, detail="Rate limit exceeded")
        
        else:  # REJECT
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "retry_after": info['reset_in']
                }
            )
    
    async def _call_llm (self, prompt: str, model: str):
        """Make LLM API call."""
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "result": response.choices[0].message.content,
            "model": model
        }


# Usage
limiter = GracefulRateLimiter (redis_client, cache, queue)

@app.post("/generate")
async def generate(
    prompt: str,
    model: str = "gpt-4",
    api_key: str = Depends (get_api_key)
):
    """Generate with graceful degradation."""
    return await limiter.handle_request(
        api_key=api_key,
        prompt=prompt,
        model=model,
        fallback_strategy=FallbackStrategy.CHEAPER_MODEL
    )
\`\`\`

## Monitoring Rate Limits

Track rate limit metrics for optimization:

\`\`\`python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
rate_limit_hits = Counter(
    'rate_limit_hits_total',
    'Total rate limit violations',
    ['tier', 'limit_type']
)

rate_limit_remaining = Gauge(
    'rate_limit_remaining',
    'Remaining rate limit capacity',
    ['api_key', 'tier']
)

request_latency = Histogram(
    'request_latency_seconds',
    'Request latency',
    ['rate_limited']
)

class MonitoredRateLimiter:
    """Rate limiter with monitoring."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.limiter = TieredRateLimiter (redis_client)
    
    def check_with_monitoring (self, api_key: str) -> tuple[bool, dict]:
        """Check rate limit and record metrics."""
        tier = self.limiter.get_user_tier (api_key)
        allowed, info = self.limiter.check_limit (api_key)
        
        if not allowed:
            # Record rate limit hit
            rate_limit_hits.labels(
                tier=tier.value,
                limit_type=info.get('error', 'unknown')
            ).inc()
        
        # Record remaining capacity
        if 'minute_remaining' in info:
            rate_limit_remaining.labels(
                api_key=api_key,
                tier=tier.value
            ).set (info['minute_remaining'])
        
        return allowed, info


# Dashboard queries:
# rate_limit_hits_total - Total violations
# rate (rate_limit_hits_total[5m]) - Violations per second
# rate_limit_remaining - Current capacity per user
\`\`\`

## Best Practices

1. **Use token bucket** for most LLM applications (allows bursts)

2. **Implement tiered limits** based on user subscription level

3. **Consider cost-based limiting** in addition to request-based

4. **Provide clear error messages** with retry-after information

5. **Add rate limit headers** to all responses for transparency

6. **Use Redis** for distributed rate limiting across instances

7. **Monitor rate limit hits** to optimize limits

8. **Implement graceful degradation** rather than hard failures

9. **Set per-endpoint limits** for expensive operations

10. **Test rate limiting** under load before production

Rate limiting is essential for controlling costs, ensuring fair usage, and maintaining service quality in production LLM applications.
`;
