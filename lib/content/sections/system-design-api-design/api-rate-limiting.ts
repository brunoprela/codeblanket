/**
 * API Rate Limiting Strategies Section
 */

export const apiratelimitingSection = {
  id: 'api-rate-limiting',
  title: 'API Rate Limiting Strategies',
  content: `Rate limiting prevents API abuse, ensures fair resource allocation, and protects backend services from overload. Implementing effective rate limiting is critical for production APIs.

## Why Rate Limiting?

### **Benefits**1. **Prevent abuse**: Block malicious actors and scrapers
2. **Fair usage**: Ensure no single client monopolizes resources
3. **Cost control**: Limit expensive operations
4. **Service protection**: Prevent cascade failures
5. **Monetization**: Tiered pricing based on usage

### **Without Rate Limiting**

- Attackers can DDoS your API
- Single client can overwhelm system
- Expensive operations (analytics, search) drain resources
- No way to monetize API usage tiers

## Rate Limiting Algorithms

### **1. Fixed Window**

Count requests in fixed time windows:

\`\`\`javascript
const redis = require('redis');
const client = redis.createClient();

async function fixedWindowRateLimit (userId, limit, windowSeconds) {
  const key = \`ratelimit:\${userId}:\${Math.floor(Date.now() / 1000 / windowSeconds)}\`;
  
  const current = await client.incr (key);
  
  if (current === 1) {
    // First request in window, set expiry
    await client.expire (key, windowSeconds);
  }
  
  if (current > limit) {
    return {
      allowed: false,
      remaining: 0,
      resetAt: (Math.floor(Date.now() / 1000 / windowSeconds) + 1) * windowSeconds
    };
  }
  
  return {
    allowed: true,
    remaining: limit - current,
    resetAt: (Math.floor(Date.now() / 1000 / windowSeconds) + 1) * windowSeconds
  };
}

// Example: 100 requests per 60 seconds
const result = await fixedWindowRateLimit('user123', 100, 60);
\`\`\`

**Problem**: **Burst at window boundaries**

\`\`\`
Window 1 (0-60s): 100 requests at t=59s
Window 2 (60-120s): 100 requests at t=60s
Result: 200 requests in 1 second! 💥
\`\`\`

### **2. Sliding Window Log**

Track timestamp of each request:

\`\`\`javascript
async function slidingWindowLogRateLimit (userId, limit, windowMs) {
  const key = \`ratelimit:log:\${userId}\`;
  const now = Date.now();
  const windowStart = now - windowMs;
  
  // Remove old entries
  await client.zremrangebyscore (key, 0, windowStart);
  
  // Count requests in window
  const count = await client.zcard (key);
  
  if (count >= limit) {
    return {
      allowed: false,
      remaining: 0,
      resetAt: await client.zrange (key, 0, 0, 'WITHSCORES')
        .then(([_, timestamp]) => parseInt (timestamp) + windowMs)
    };
  }
  
  // Add current request
  await client.zadd (key, now, \`\${now}-\${Math.random()}\`);
  await client.expire (key, Math.ceil (windowMs / 1000));
  
  return {
    allowed: true,
    remaining: limit - count - 1,
    resetAt: now + windowMs
  };
}
\`\`\`

**Pros**: Accurate, no burst issues

**Cons**: Memory intensive (stores every request timestamp)

### **3. Sliding Window Counter (Hybrid)**

Best of both worlds:

\`\`\`javascript
async function slidingWindowCounter (userId, limit, windowSeconds) {
  const now = Date.now() / 1000;
  const currentWindow = Math.floor (now / windowSeconds);
  const previousWindow = currentWindow - 1;
  
  const currentKey = \`ratelimit:\${userId}:\${currentWindow}\`;
  const previousKey = \`ratelimit:\${userId}:\${previousWindow}\`;
  
  // Get counts
  const currentCount = parseInt (await client.get (currentKey) || '0');
  const previousCount = parseInt (await client.get (previousKey) || '0');
  
  // Calculate weighted count based on time elapsed in current window
  const percentageInCurrent = (now % windowSeconds) / windowSeconds;
  const weightedCount = 
    previousCount * (1 - percentageInCurrent) + currentCount;
  
  if (weightedCount >= limit) {
    return {
      allowed: false,
      remaining: 0,
      resetAt: (currentWindow + 1) * windowSeconds
    };
  }
  
  // Increment current window
  await client.incr (currentKey);
  await client.expire (currentKey, windowSeconds * 2);
  
  return {
    allowed: true,
    remaining: Math.floor (limit - weightedCount - 1),
    resetAt: (currentWindow + 1) * windowSeconds
  };
}
\`\`\`

**Pros**: Memory efficient, smooth rate limiting

**Cons**: Slightly more complex logic

**This is the recommended approach for most use cases.**

### **4. Token Bucket**

Tokens replenish at fixed rate:

\`\`\`javascript
async function tokenBucketRateLimit (userId, capacity, refillRate) {
  const key = \`ratelimit:bucket:\${userId}\`;
  
  const data = await client.get (key);
  let tokens, lastRefill;
  
  if (data) {
    ({ tokens, lastRefill } = JSON.parse (data));
  } else {
    tokens = capacity;
    lastRefill = Date.now();
  }
  
  // Refill tokens based on time elapsed
  const now = Date.now();
  const elapsed = (now - lastRefill) / 1000;
  const tokensToAdd = elapsed * refillRate;
  tokens = Math.min (capacity, tokens + tokensToAdd);
  
  if (tokens < 1) {
    return {
      allowed: false,
      remaining: 0,
      retryAfter: (1 - tokens) / refillRate
    };
  }
  
  // Consume 1 token
  tokens -= 1;
  
  await client.set (key, JSON.stringify({
    tokens,
    lastRefill: now
  }), 'EX', 3600);
  
  return {
    allowed: true,
    remaining: Math.floor (tokens),
    retryAfter: null
  };
}

// Example: 100 token capacity, refill 10 tokens/second
await tokenBucketRateLimit('user123', 100, 10);
\`\`\`

**Pros**: Allows bursts (up to capacity), smooth refill

**Use case**: APIs where bursts are acceptable

### **5. Leaky Bucket**

Process requests at fixed rate, queue overflow:

\`\`\`javascript
async function leakyBucketRateLimit (userId, capacity, leakRate) {
  const queueKey = \`ratelimit:queue:\${userId}\`;
  const lastLeakKey = \`ratelimit:lastleak:\${userId}\`;
  
  // Get current queue size
  let queueSize = await client.llen (queueKey);
  const lastLeak = parseInt (await client.get (lastLeakKey) || Date.now());
  
  // Leak tokens based on time elapsed
  const now = Date.now();
  const elapsed = (now - lastLeak) / 1000;
  const tokensToLeak = Math.floor (elapsed * leakRate);
  
  if (tokensToLeak > 0) {
    const leaked = Math.min (tokensToLeak, queueSize);
    for (let i = 0; i < leaked; i++) {
      await client.rpop (queueKey);
    }
    queueSize -= leaked;
    await client.set (lastLeakKey, now);
  }
  
  if (queueSize >= capacity) {
    return {
      allowed: false,
      retryAfter: (queueSize - capacity + 1) / leakRate
    };
  }
  
  // Add request to queue
  await client.lpush (queueKey, now);
  await client.expire (queueKey, 3600);
  
  return {
    allowed: true,
    remaining: capacity - queueSize - 1
  };
}
\`\`\`

**Pros**: Smooth traffic, prevents bursts

**Use case**: APIs with consistent processing capacity

## Rate Limiting Strategies

### **1. Per-User Rate Limiting**

\`\`\`javascript
app.use (async (req, res, next) => {
  const userId = req.user?.id || req.ip;
  
  const result = await slidingWindowCounter (userId, 1000, 3600); // 1000/hour
  
  res.set({
    'X-RateLimit-Limit': 1000,
    'X-RateLimit-Remaining': result.remaining,
    'X-RateLimit-Reset': result.resetAt
  });
  
  if (!result.allowed) {
    return res.status(429).json({
      error: 'Rate limit exceeded',
      retryAfter: result.resetAt
    });
  }
  
  next();
});
\`\`\`

### **2. Tiered Rate Limiting**

Different limits per plan:

\`\`\`javascript
const RATE_LIMITS = {
  free: { requests: 100, window: 3600 },
  basic: { requests: 1000, window: 3600 },
  premium: { requests: 10000, window: 3600 }
};

app.use (async (req, res, next) => {
  const plan = req.user?.plan || 'free';
  const { requests, window } = RATE_LIMITS[plan];
  
  const result = await slidingWindowCounter (req.user.id, requests, window);
  
  if (!result.allowed) {
    return res.status(429).json({
      error: 'Rate limit exceeded',
      plan,
      upgradeUrl: '/pricing'
    });
  }
  
  next();
});
\`\`\`

### **3. Endpoint-Specific Rate Limiting**

Different limits per endpoint:

\`\`\`javascript
// Expensive search endpoint
app.get('/search', 
  rateLimiter({ requests: 10, window: 60 }),  // 10/min
  async (req, res) => {
    // ...
  }
);

// Lightweight read
app.get('/users/:id',
  rateLimiter({ requests: 1000, window: 60 }),  // 1000/min
  async (req, res) => {
    // ...
  }
);

// Write operations
app.post('/posts',
  rateLimiter({ requests: 100, window: 60 }),  // 100/min
  async (req, res) => {
    // ...
  }
);
\`\`\`

### **4. Cost-Based Rate Limiting**

Different costs per operation:

\`\`\`javascript
const OPERATION_COSTS = {
  'GET /users/:id': 1,
  'GET /search': 10,
  'POST /analytics': 50,
  'GET /reports': 100
};

async function costBasedRateLimit (userId, operation, budget) {
  const cost = OPERATION_COSTS[operation] || 1;
  const key = \`ratelimit:cost:\${userId}\`;
  
  const spent = parseInt (await client.get (key) || '0');
  
  if (spent + cost > budget) {
    return {
      allowed: false,
      cost,
      spent,
      budget
    };
  }
  
  await client.incrby (key, cost);
  await client.expire (key, 3600);  // Reset hourly
  
  return {
    allowed: true,
    cost,
    spent: spent + cost,
    budget
  };
}

app.use (async (req, res, next) => {
  const operation = \`\${req.method} \${req.route.path}\`;
  const result = await costBasedRateLimit (req.user.id, operation, 10000);
  
  if (!result.allowed) {
    return res.status(429).json({
      error: 'Cost budget exceeded',
      cost: result.cost,
      spent: result.spent,
      budget: result.budget
    });
  }
  
  next();
});
\`\`\`

### **5. Concurrent Request Limiting**

Limit simultaneous requests:

\`\`\`javascript
const activeSemaphores = new Map();

async function concurrentLimiter (userId, maxConcurrent) {
  const key = \`ratelimit:concurrent:\${userId}\`;
  
  const current = await client.incr (key);
  
  if (current > maxConcurrent) {
    await client.decr (key);
    return {
      allowed: false,
      current: current - 1,
      max: maxConcurrent
    };
  }
  
  await client.expire (key, 300);  // Safety: expire after 5 min
  
  return {
    allowed: true,
    current,
    max: maxConcurrent,
    release: async () => {
      await client.decr (key);
    }
  };
}

app.use (async (req, res, next) => {
  const limiter = await concurrentLimiter (req.user.id, 10);
  
  if (!limiter.allowed) {
    return res.status(429).json({
      error: 'Too many concurrent requests',
      current: limiter.current,
      max: limiter.max
    });
  }
  
  // Release on response finish
  res.on('finish', limiter.release);
  res.on('close', limiter.release);
  
  next();
});
\`\`\`

## Response Headers

Standard rate limit headers:

\`\`\`javascript
app.use (async (req, res, next) => {
  const result = await rateLimit (req.user.id);
  
  // Standard headers (GitHub, Stripe, Twitter use these)
  res.set({
    'X-RateLimit-Limit': result.limit,
    'X-RateLimit-Remaining': result.remaining,
    'X-RateLimit-Reset': result.resetAt,  // Unix timestamp
    'X-RateLimit-Used': result.used
  });
  
  if (!result.allowed) {
    res.set({
      'Retry-After': result.retryAfter  // Seconds until retry
    });
    
    return res.status(429).json({
      error: 'Too many requests',
      message: 'Rate limit exceeded. Try again later.',
      limit: result.limit,
      remaining: 0,
      resetAt: result.resetAt,
      retryAfter: result.retryAfter
    });
  }
  
  next();
});
\`\`\`

## Distributed Rate Limiting

### **Redis-Based (Most Common)**

\`\`\`javascript
const Redis = require('ioredis');

// Redis Cluster for high availability
const redis = new Redis.Cluster([
  { host: 'redis-1', port: 6379 },
  { host: 'redis-2', port: 6379 },
  { host: 'redis-3', port: 6379 }
]);

// Lua script for atomic rate limiting
const RATE_LIMIT_SCRIPT = \`
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local current = redis.call('INCR', key)
if current == 1 then
  redis.call('EXPIRE', key, window)
end
if current > limit then
  return {0, limit, 0}
else
  local ttl = redis.call('TTL', key)
  return {1, limit, limit - current, ttl}
end
\`;

async function distributedRateLimit (userId, limit, window) {
  const key = \`ratelimit:\${userId}:\${Math.floor(Date.now() / 1000 / window)}\`;
  
  const [allowed, max, remaining, ttl] = await redis.eval(
    RATE_LIMIT_SCRIPT,
    1,
    key,
    limit,
    window
  );
  
  return {
    allowed: allowed === 1,
    limit: max,
    remaining,
    resetAt: Date.now() / 1000 + ttl
  };
}
\`\`\`

### **Rate Limiting at Multiple Layers**

\`\`\`
┌─────────────────┐
│  CDN / WAF      │  ← Layer 1: DDoS protection (IP-based)
└────────┬────────┘
         │
┌────────▼────────┐
│  Load Balancer  │  ← Layer 2: Connection limits
└────────┬────────┘
         │
┌────────▼────────┐
│  API Gateway    │  ← Layer 3: Per-user rate limits
└────────┬────────┘
         │
┌────────▼────────┐
│  Service        │  ← Layer 4: Cost-based limits
└─────────────────┘
\`\`\`

## Handling Rate Limit Errors (Client-Side)

### **Exponential Backoff with Jitter**

\`\`\`javascript
async function fetchWithRetry (url, options = {}, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch (url, options);
      
      if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After');
        const resetAt = response.headers.get('X-RateLimit-Reset');
        
        if (attempt < maxRetries - 1) {
          let delay;
          
          if (retryAfter) {
            // Use server-provided retry-after
            delay = parseInt (retryAfter) * 1000;
          } else if (resetAt) {
            // Calculate delay until reset
            delay = (parseInt (resetAt) * 1000) - Date.now();
          } else {
            // Exponential backoff with jitter
            const baseDelay = Math.pow(2, attempt) * 1000;
            const jitter = Math.random() * 1000;
            delay = baseDelay + jitter;
          }
          
          console.log(\`Rate limited. Retrying in \${delay}ms...\`);
          await new Promise (resolve => setTimeout (resolve, delay));
          continue;
        }
      }
      
      return response;
    } catch (error) {
      if (attempt === maxRetries - 1) throw error;
    }
  }
}
\`\`\`

## Bypass Rate Limits (Whitelist)

\`\`\`javascript
const WHITELISTED_IPS = new Set([
  '10.0.0.0/8',    // Internal services
  '203.0.113.0'    // Trusted partner
]);

const WHITELISTED_USERS = new Set([
  'admin-user-id',
  'monitoring-service'
]);

app.use (async (req, res, next) => {
  // Check whitelist
  if (WHITELISTED_IPS.has (req.ip) || 
      WHITELISTED_USERS.has (req.user?.id)) {
    return next();  // Skip rate limiting
  }
  
  // Apply rate limiting
  const result = await rateLimit (req.user.id);
  if (!result.allowed) {
    return res.status(429).json({ error: 'Rate limit exceeded' });
  }
  
  next();
});
\`\`\`

## Monitoring Rate Limits

\`\`\`javascript
const prometheus = require('prom-client');

const rateLimitCounter = new prometheus.Counter({
  name: 'api_rate_limit_exceeded_total',
  help: 'Number of rate limit exceeded errors',
  labelNames: ['user_plan', 'endpoint']
});

const rateLimitUsage = new prometheus.Histogram({
  name: 'api_rate_limit_usage_percent',
  help: 'Rate limit usage percentage',
  labelNames: ['user_plan'],
  buckets: [10, 25, 50, 75, 90, 95, 100]
});

app.use (async (req, res, next) => {
  const result = await rateLimit (req.user.id);
  
  const usagePercent = ((result.used / result.limit) * 100);
  rateLimitUsage.labels (req.user.plan).observe (usagePercent);
  
  if (!result.allowed) {
    rateLimitCounter.labels (req.user.plan, req.route.path).inc();
    return res.status(429).json({ error: 'Rate limit exceeded' });
  }
  
  next();
});
\`\`\`

## Best Practices

1. **Use sliding window counter** for most cases (balance of accuracy and memory)
2. **Set headers** (\`X-RateLimit-*\`) so clients know their usage
3. **Return 429 status code** for rate limit errors
4. **Provide \`Retry-After\`** header with seconds until retry
5. **Tier limits** by user plan (free, basic, premium)
6. **Different limits** for different endpoints (expensive vs cheap)
7. **Whitelist** internal services and admins
8. **Monitor** rate limit usage and exceeded counts
9. **Use Redis** for distributed systems
10. **Document** rate limits clearly in API docs

## Real-World Examples

**GitHub API**:
- 5,000 requests/hour for authenticated users
- 60 requests/hour for unauthenticated
- Cost-based: search queries cost more

**Stripe API**:
- Rolling rate limits (not fixed windows)
- Different limits per endpoint
- Test vs production limits

**Twitter API**:
- Tiered plans (free, basic, pro, enterprise)
- Per-user and per-app limits
- 15-minute windows

## Rate Limiting Pitfalls

### **❌ Fixed Window Burst**

Problem: 200 requests at window boundary

Solution: Use sliding window counter

### **❌ No Retry-After Header**

Clients spam retries, making it worse

Solution: Always return \`Retry-After\`

### **❌ Same Limit for All Endpoints**

Expensive operations (search, analytics) should have lower limits

Solution: Cost-based or endpoint-specific limits

### **❌ In-Memory Rate Limiting (Single Instance)**

Doesn't work with multiple servers

Solution: Use Redis for distributed rate limiting`,
};
