/**
 * Rate Limiting & Throttling Section
 */

export const ratelimitingSection = {
  id: 'rate-limiting',
  title: 'Rate Limiting & Throttling',
  content: `Rate limiting and throttling are critical for protecting APIs and services from abuse, ensuring fair resource usage, and maintaining system stability. This section covers algorithms, implementation patterns, and distributed rate limiting strategies.
    
    ## What is Rate Limiting?
    
    **Rate Limiting** restricts the number of requests a client can make to an API within a time window.
    
    **Goals**:
    - Prevent abuse and DDoS attacks
    - Ensure fair resource allocation
    - Protect backend services from overload
    - Enforce pricing tiers (free vs paid)
    
    **Example**:
    \`\`\`
    User can make:
    - 100 requests per minute (free tier)
    - 1000 requests per minute (pro tier)
    - Unlimited requests (enterprise tier)
    \`\`\`
    
    ---
    
    ## Rate Limiting Algorithms
    
    ### **1. Token Bucket**
    
    **Most Popular Algorithm** - Used by AWS, Stripe, Shopify
    
    **How it Works**:
    - Bucket has capacity of N tokens
    - Tokens added at rate R per second
    - Each request consumes 1 token
    - Request rejected if bucket empty
    
    **Visualization**:
    \`\`\`
    Bucket capacity: 10 tokens
    Refill rate: 2 tokens/second
    
    Time 0s: [**********] (10 tokens) → Request (9 tokens)
    Time 1s: [**********] (10 tokens, refilled)
    Time 2s: [**********] (10 tokens)
    
    Burst: Can use all 10 immediately
    Sustained: Limited to 2/second long-term
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    class TokenBucket {
      constructor (capacity, refillRate) {
        this.capacity = capacity;
        this.refillRate = refillRate; // tokens per second
        this.tokens = capacity;
        this.lastRefill = Date.now();
      }
      
      tryConsume (tokens = 1) {
        this.refill();
        
        if (this.tokens >= tokens) {
          this.tokens -= tokens;
          return true;
        }
        
        return false;
      }
      
      refill() {
        const now = Date.now();
        const timePassed = (now - this.lastRefill) / 1000; // seconds
        const tokensToAdd = timePassed * this.refillRate;
        
        this.tokens = Math.min (this.capacity, this.tokens + tokensToAdd);
        this.lastRefill = now;
      }
      
      getWaitTime() {
        if (this.tokens >= 1) return 0;
        return (1 - this.tokens) / this.refillRate * 1000; // ms
      }
    }
    
    // Usage
    const bucket = new TokenBucket(100, 2); // 100 capacity, 2/sec
    
    app.use((req, res, next) => {
      const userId = req.user.id;
      const userBucket = buckets.get (userId) || new TokenBucket(100, 2);
      
      if (userBucket.tryConsume()) {
        buckets.set (userId, userBucket);
        next();
      } else {
        const waitTime = Math.ceil (userBucket.getWaitTime() / 1000);
        res.status(429).json({
          error: 'Rate limit exceeded',
          retryAfter: waitTime
        });
      }
    });
    \`\`\`
    
    **Pros**:
    - Allows bursts (good for variable traffic)
    - Memory efficient
    - Smooth rate limiting
    
    **Cons**:
    - Slightly complex to implement
    - Need to track last refill time
    
    ---
    
    ### **2. Leaky Bucket**
    
    **Smooth, Constant Rate** - Used for traffic shaping
    
    **How it Works**:
    - Requests added to queue (bucket)
    - Processed at fixed rate
    - Overflow requests rejected
    
    **Visualization**:
    \`\`\`
    Requests → [Queue: 5/10] → Process at 2/sec → Backend
                  ↓ (full)
               Reject
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    class LeakyBucket {
      constructor (capacity, leakRate) {
        this.capacity = capacity;
        this.leakRate = leakRate; // requests per second
        this.queue = [];
        this.processing = false;
      }
      
      async tryAdd (request) {
        if (this.queue.length >= this.capacity) {
          return false; // Bucket full
        }
        
        this.queue.push (request);
        
        if (!this.processing) {
          this.startLeaking();
        }
        
        return true;
      }
      
      startLeaking() {
        this.processing = true;
        
        const interval = 1000 / this.leakRate; // ms between requests
        
        const leak = setInterval(() => {
          if (this.queue.length === 0) {
            clearInterval (leak);
            this.processing = false;
            return;
          }
          
          const request = this.queue.shift();
          this.processRequest (request);
        }, interval);
      }
      
      async processRequest (request) {
        // Process request at constant rate
        await handleRequest (request);
      }
    }
    \`\`\`
    
    **Pros**:
    - Smooth, constant output rate
    - Good for protecting downstream services
    
    **Cons**:
    - No bursts allowed
    - Requests delayed (queued)
    - More memory (stores queue)
    
    ---
    
    ### **3. Fixed Window**
    
    **Simple but Flawed**
    
    **How it Works**:
    - Count requests in fixed time windows
    - Reset counter at window boundary
    
    **Example**:
    \`\`\`
    Window: 1 minute
    Limit: 100 requests
    
    12:00:00 - 12:00:59 → 100 requests (allowed)
    12:01:00 - 12:01:59 → Counter resets, 100 more allowed
    \`\`\`
    
    **Problem: Burst at Window Boundaries**:
    \`\`\`
    12:00:50 → 100 requests (allowed)
    12:01:01 → 100 requests (allowed)
    Total: 200 requests in 11 seconds! (Burst at boundary)
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    class FixedWindow {
      constructor (limit, windowSizeMs) {
        this.limit = limit;
        this.windowSizeMs = windowSizeMs;
        this.counters = new Map(); // userId -> {count, windowStart}
      }
      
      tryRequest (userId) {
        const now = Date.now();
        const userData = this.counters.get (userId) || { count: 0, windowStart: now };
        
        // Check if we're in a new window
        if (now - userData.windowStart >= this.windowSizeMs) {
          userData.count = 0;
          userData.windowStart = now;
        }
        
        if (userData.count < this.limit) {
          userData.count++;
          this.counters.set (userId, userData);
          return true;
        }
        
        return false;
      }
    }
    \`\`\`
    
    **Pros**:
    - Very simple
    - Memory efficient
    
    **Cons**:
    - Burst problem at boundaries
    - Not accurate
    
    ---
    
    ### **4. Sliding Window Log**
    
    **Accurate but Memory Intensive**
    
    **How it Works**:
    - Store timestamp of each request
    - Count requests in rolling window
    - Remove old timestamps
    
    **Implementation**:
    \`\`\`javascript
    class SlidingWindowLog {
      constructor (limit, windowSizeMs) {
        this.limit = limit;
        this.windowSizeMs = windowSizeMs;
        this.logs = new Map(); // userId -> [timestamps]
      }
      
      tryRequest (userId) {
        const now = Date.now();
        const userLog = this.logs.get (userId) || [];
        
        // Remove timestamps outside window
        const windowStart = now - this.windowSizeMs;
        const validLog = userLog.filter (timestamp => timestamp > windowStart);
        
        if (validLog.length < this.limit) {
          validLog.push (now);
          this.logs.set (userId, validLog);
          return true;
        }
        
        return false;
      }
    }
    \`\`\`
    
    **Pros**:
    - Very accurate
    - No burst problem
    
    **Cons**:
    - Memory intensive (stores all timestamps)
    - Expensive (filter on every request)
    
    ---
    
    ### **5. Sliding Window Counter** ⭐
    
    **Best of Both Worlds** - Combines fixed window + sliding window
    
    **How it Works**:
    - Use two fixed windows
    - Estimate count in sliding window using weighted average
    
    **Formula**:
    \`\`\`
    Current window: 70% complete
    Previous window count: 100
    Current window count: 40
    
    Estimated count = (100 × 0.3) + (40 × 1.0) = 30 + 40 = 70
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    class SlidingWindowCounter {
      constructor (limit, windowSizeMs) {
        this.limit = limit;
        this.windowSizeMs = windowSizeMs;
        this.counters = new Map(); // userId -> {current, previous, windowStart}
      }
      
      tryRequest (userId) {
        const now = Date.now();
        const userData = this.counters.get (userId) || {
          current: 0,
          previous: 0,
          windowStart: now
        };
        
        const elapsed = now - userData.windowStart;
        
        // New window?
        if (elapsed >= this.windowSizeMs) {
          userData.previous = userData.current;
          userData.current = 0;
          userData.windowStart = now;
        }
        
        // Calculate weighted count
        const windowProgress = elapsed / this.windowSizeMs;
        const previousWeight = 1 - windowProgress;
        const estimatedCount = 
          (userData.previous * previousWeight) + userData.current;
        
        if (estimatedCount < this.limit) {
          userData.current++;
          this.counters.set (userId, userData);
          return true;
        }
        
        return false;
      }
    }
    \`\`\`
    
    **Pros**:
    - Accurate (no burst at boundaries)
    - Memory efficient (only 2 counters)
    - Fast (no filtering)
    
    **Cons**:
    - Slightly more complex
    
    **Recommended for most use cases!**
    
    ---
    
    ## Distributed Rate Limiting
    
    **Challenge**: Multiple API servers need to share rate limit state
    
    ### **Solution 1: Redis with Sliding Window**
    
    \`\`\`javascript
    const Redis = require('ioredis');
    const redis = new Redis();
    
    async function rateLimitRedis (userId, limit, windowSec) {
      const key = \`rate_limit:\${userId}\`;
      const now = Date.now();
      const windowStart = now - (windowSec * 1000);
      
      // Use Redis sorted set (score = timestamp)
      const pipeline = redis.pipeline();
      
      // Remove old entries
      pipeline.zremrangebyscore (key, '-inf', windowStart);
      
      // Count requests in window
      pipeline.zcard (key);
      
      // Add current request
      pipeline.zadd (key, now, \`\${now}-\${Math.random()}\`);
      
      // Set expiry
      pipeline.expire (key, windowSec * 2);
      
      const results = await pipeline.exec();
      const count = results[1][1]; // Count from zcard
      
      if (count < limit) {
        return { allowed: true, remaining: limit - count - 1 };
      } else {
        // Remove the request we just added
        await redis.zrem (key, \`\${now}-\${Math.random()}\`);
        return { allowed: false, remaining: 0 };
      }
    }
    
    // Usage
    app.use (async (req, res, next) => {
      const result = await rateLimitRedis (req.user.id, 100, 60);
      
      if (result.allowed) {
        res.setHeader('X-RateLimit-Remaining', result.remaining);
        next();
      } else {
        res.status(429).json({ error: 'Rate limit exceeded' });
      }
    });
    \`\`\`
    
    ### **Solution 2: Redis with Token Bucket (More Efficient)**
    
    \`\`\`javascript
    async function tokenBucketRedis (userId, capacity, refillRate) {
      const key = \`token_bucket:\${userId}\`;
      
      // Lua script for atomic token bucket operation
      const script = \`
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refillRate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'lastRefill')
        local tokens = tonumber (bucket[1]) or capacity
        local lastRefill = tonumber (bucket[2]) or now
        
        -- Refill tokens
        local timePassed = (now - lastRefill) / 1000
        local tokensToAdd = timePassed * refillRate
        tokens = math.min (capacity, tokens + tokensToAdd)
        
        -- Try to consume
        if tokens >= 1 then
          tokens = tokens - 1
          redis.call('HMSET', key, 'tokens', tokens, 'lastRefill', now)
          redis.call('EXPIRE', key, 3600)
          return {1, math.floor (tokens)}
        else
          return {0, 0}
        end
      \`;
      
      const result = await redis.eval(
        script,
        1,
        key,
        capacity,
        refillRate,
        Date.now()
      );
      
      return {
        allowed: result[0] === 1,
        remaining: result[1]
      };
    }
    \`\`\`
    
    **Why Lua Script?**
    - Atomic operation (no race conditions)
    - Single round-trip to Redis
    - Consistent across all servers
    
    ---
    
    ## Rate Limiting Patterns
    
    ### **1. Per-User Rate Limiting**
    
    \`\`\`javascript
    app.use (async (req, res, next) => {
      const userId = req.user.id;
      const result = await rateLimit (userId, 100, 60);
      
      if (result.allowed) {
        next();
      } else {
        res.status(429).json({ error: 'Too many requests' });
      }
    });
    \`\`\`
    
    ### **2. Per-IP Rate Limiting** (for anonymous users)
    
    \`\`\`javascript
    app.use (async (req, res, next) => {
      const ip = req.ip || req.connection.remoteAddress;
      const result = await rateLimit (ip, 20, 60);
      
      if (result.allowed) {
        next();
      } else {
        res.status(429).json({ error: 'Too many requests from this IP' });
      }
    });
    \`\`\`
    
    ### **3. Per-Endpoint Rate Limiting**
    
    \`\`\`javascript
    // Different limits for different endpoints
    const limits = {
      'POST /api/login': { limit: 5, window: 60 }, // 5 per minute
      'GET /api/users': { limit: 100, window: 60 }, // 100 per minute
      'POST /api/upload': { limit: 10, window: 3600 } // 10 per hour
    };
    
    app.use (async (req, res, next) => {
      const endpoint = \`\${req.method} \${req.path}\`;
      const config = limits[endpoint] || { limit: 60, window: 60 };
      
      const key = \`\${req.user.id}:\${endpoint}\`;
      const result = await rateLimit (key, config.limit, config.window);
      
      if (result.allowed) {
        next();
      } else {
        res.status(429).json({ error: 'Rate limit exceeded for this endpoint' });
      }
    });
    \`\`\`
    
    ### **4. Tiered Rate Limiting**
    
    \`\`\`javascript
    const tiers = {
      free: { limit: 100, window: 3600 },
      pro: { limit: 1000, window: 3600 },
      enterprise: { limit: 10000, window: 3600 }
    };
    
    app.use (async (req, res, next) => {
      const userTier = req.user.tier || 'free';
      const config = tiers[userTier];
      
      const result = await rateLimit (req.user.id, config.limit, config.window);
      
      res.setHeader('X-RateLimit-Limit', config.limit);
      res.setHeader('X-RateLimit-Remaining', result.remaining);
      
      if (result.allowed) {
        next();
      } else {
        res.status(429).json({
          error: 'Rate limit exceeded',
          upgrade: 'Upgrade to Pro for higher limits'
        });
      }
    });
    \`\`\`
    
    ---
    
    ## Response Headers
    
    **Standard Headers** (from [RFC 6585](https://tools.ietf.org/html/rfc6585)):
    
    \`\`\`
    X-RateLimit-Limit: 100          # Max requests per window
    X-RateLimit-Remaining: 75       # Requests left in current window
    X-RateLimit-Reset: 1699564800   # Unix timestamp when limit resets
    Retry-After: 30                 # Seconds until retry (when rate limited)
    \`\`\`
    
    **Example**:
    \`\`\`javascript
    app.use (async (req, res, next) => {
      const result = await tokenBucket.tryConsume (req.user.id);
      
      res.setHeader('X-RateLimit-Limit', 100);
      res.setHeader('X-RateLimit-Remaining', result.remaining);
      res.setHeader('X-RateLimit-Reset', result.reset);
      
      if (result.allowed) {
        next();
      } else {
        res.setHeader('Retry-After', result.retryAfter);
        res.status(429).json({
          error: 'Too Many Requests',
          retryAfter: result.retryAfter
        });
      }
    });
    \`\`\`
    
    ---
    
    ## Common Mistakes
    
    ### **❌ Mistake 1: Rate Limiting After Authentication**
    
    \`\`\`javascript
    // Bad: Attacker can DDoS by sending invalid credentials
    app.post('/api/login', authenticate, rateLimit, (req, res) => {
      // Login logic
    });
    
    // Good: Rate limit BEFORE authentication
    app.post('/api/login', rateLimit, authenticate, (req, res) => {
      // Login logic
    });
    \`\`\`
    
    ### **❌ Mistake 2: Not Handling Clock Drift**
    
    \`\`\`javascript
    // Bad: Uses server timestamp (clock drift issues)
    const now = Date.now();
    
    // Good: Use Redis TIME command for distributed consistency
    const [seconds, microseconds] = await redis.time();
    const now = seconds * 1000 + Math.floor (microseconds / 1000);
    \`\`\`
    
    ### **❌ Mistake 3: No Exponential Backoff for Retries**
    
    \`\`\`javascript
    // Bad: Client retries immediately
    if (response.status === 429) {
      retry();
    }
    
    // Good: Exponential backoff
    if (response.status === 429) {
      const retryAfter = response.headers.get('Retry-After');
      const delay = retryAfter ? parseInt (retryAfter) * 1000 : 1000;
      setTimeout(() => retry(), delay * Math.pow(2, attempts));
    }
    \`\`\`
    
    ---
    
    ## Key Takeaways
    
    1. **Token Bucket recommended** for most use cases (allows bursts, memory efficient)
    2. **Sliding Window Counter** for accurate rate limiting without burst problems
    3. **Redis + Lua scripts** for distributed rate limiting (atomic operations)
    4. **Rate limit BEFORE authentication** to prevent DDoS
    5. **Different limits for different endpoints** (login: 5/min, read: 1000/min)
    6. **Tiered limits** enforce pricing (free: 100/hr, pro: 1000/hr)
    7. **Return standard headers** (X-RateLimit-*, Retry-After)
    8. **Per-user AND per-IP** rate limiting for security
    9. **Monitor rate limit hits** to detect abuse or legitimate high usage
    10. **Graceful degradation**: Return 429 with clear error message and retry guidance`,
};
