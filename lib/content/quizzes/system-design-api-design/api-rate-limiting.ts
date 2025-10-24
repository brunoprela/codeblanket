/**
 * Quiz questions for API Rate Limiting Strategies section
 */

export const apiratelimitingQuiz = [
  {
    id: 'ratelimit-d1',
    question:
      'Design a comprehensive rate limiting system for a SaaS API with free, basic, and premium tiers. Include per-endpoint limits, cost-based limits, and handling of burst traffic.',
    sampleAnswer: `Comprehensive rate limiting system for SaaS API:

**1. Tiered Rate Limits**

\`\`\`javascript
const RATE_LIMITS = {
  free: {
    hourly: 100,
    daily: 1000,
    concurrent: 2,
    costBudget: 1000
  },
  basic: {
    hourly: 1000,
    daily: 20000,
    concurrent: 10,
    costBudget: 10000
  },
  premium: {
    hourly: 10000,
    daily: 500000,
    concurrent: 50,
    costBudget: 100000
  }
};

// Endpoint-specific costs
const OPERATION_COSTS = {
  // Reads (cheap)
  'GET /users/:id': 1,
  'GET /posts/:id': 1,
  
  // List operations (moderate)
  'GET /users': 5,
  'GET /posts': 5,
  
  // Search (expensive)
  'GET /search': 10,
  'POST /search/advanced': 20,
  
  // Analytics (very expensive)
  'POST /analytics/reports': 50,
  'POST /analytics/export': 100,
  
  // Writes (moderate)
  'POST /users': 3,
  'PUT /users/:id': 3,
  'DELETE /users/:id': 3
};
\`\`\`

**2. Multi-Layer Rate Limiter**

\`\`\`javascript
const Redis = require('ioredis');
const redis = new Redis.Cluster([
  { host: 'redis-1', port: 6379 },
  { host: 'redis-2', port: 6379 }
]);

class ComprehensiveRateLimiter {
  // Sliding window counter for hourly/daily limits
  async slidingWindowLimit(userId, limit, windowSeconds) {
    const now = Date.now() / 1000;
    const currentWindow = Math.floor(now / windowSeconds);
    const previousWindow = currentWindow - 1;
    
    const currentKey = \`ratelimit:\${userId}:\${windowSeconds}:\${currentWindow}\`;
    const previousKey = \`ratelimit:\${userId}:\${windowSeconds}:\${previousWindow}\`;
    
    const [currentCount, previousCount] = await Promise.all([
      redis.get(currentKey).then(v => parseInt(v || '0')),
      redis.get(previousKey).then(v => parseInt(v || '0'))
    ]);
    
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
    
    await redis.incr(currentKey);
    await redis.expire(currentKey, windowSeconds * 2);
    
    return {
      allowed: true,
      remaining: Math.floor(limit - weightedCount - 1),
      resetAt: (currentWindow + 1) * windowSeconds
    };
  }
  
  // Cost-based limiting
  async costBasedLimit(userId, operation, budget) {
    const cost = OPERATION_COSTS[operation] || 1;
    const key = \`ratelimit:cost:\${userId}\`;
    const now = Date.now();
    const hourStart = Math.floor(now / 1000 / 3600);
    const hourKey = \`\${key}:\${hourStart}\`;
    
    const spent = parseInt(await redis.get(hourKey) || '0');
    
    if (spent + cost > budget) {
      return {
        allowed: false,
        cost,
        spent,
        budget,
        remaining: 0
      };
    }
    
    await redis.incrby(hourKey, cost);
    await redis.expire(hourKey, 7200); // 2 hours
    
    return {
      allowed: true,
      cost,
      spent: spent + cost,
      budget,
      remaining: budget - spent - cost
    };
  }
  
  // Concurrent request limiting
  async concurrentLimit(userId, maxConcurrent) {
    const key = \`ratelimit:concurrent:\${userId}\`;
    const current = await redis.incr(key);
    
    if (current > maxConcurrent) {
      await redis.decr(key);
      return {
        allowed: false,
        current: current - 1,
        max: maxConcurrent
      };
    }
    
    await redis.expire(key, 300);
    
    return {
      allowed: true,
      current,
      max: maxConcurrent,
      release: async () => await redis.decr(key)
    };
  }
  
  // Token bucket for burst handling
  async tokenBucketLimit(userId, capacity, refillRate) {
    const key = \`ratelimit:burst:\${userId}\`;
    
    const data = await redis.get(key);
    let tokens, lastRefill;
    
    if (data) {
      const parsed = JSON.parse(data);
      tokens = parsed.tokens;
      lastRefill = parsed.lastRefill;
    } else {
      tokens = capacity;
      lastRefill = Date.now();
    }
    
    // Refill tokens
    const now = Date.now();
    const elapsed = (now - lastRefill) / 1000;
    const tokensToAdd = elapsed * refillRate;
    tokens = Math.min(capacity, tokens + tokensToAdd);
    
    if (tokens < 1) {
      return {
        allowed: false,
        remaining: 0,
        retryAfter: (1 - tokens) / refillRate
      };
    }
    
    tokens -= 1;
    
    await redis.set(key, JSON.stringify({
      tokens,
      lastRefill: now
    }), 'EX', 3600);
    
    return {
      allowed: true,
      remaining: Math.floor(tokens)
    };
  }
}

const limiter = new ComprehensiveRateLimiter();
\`\`\`

**3. Middleware Implementation**

\`\`\`javascript
async function rateLimitMiddleware(req, res, next) {
  const userId = req.user.id;
  const plan = req.user.plan || 'free';
  const limits = RATE_LIMITS[plan];
  const operation = \`\${req.method} \${req.route.path}\`;
  
  // Check 1: Hourly limit (sliding window)
  const hourly = await limiter.slidingWindowLimit(
    \`\${userId}:hourly\`,
    limits.hourly,
    3600
  );
  
  if (!hourly.allowed) {
    return rateLimitError(res, {
      type: 'hourly_limit',
      limit: limits.hourly,
      resetAt: hourly.resetAt
    });
  }
  
  // Check 2: Daily limit
  const daily = await limiter.slidingWindowLimit(
    \`\${userId}:daily\`,
    limits.daily,
    86400
  );
  
  if (!daily.allowed) {
    return rateLimitError(res, {
      type: 'daily_limit',
      limit: limits.daily,
      resetAt: daily.resetAt
    });
  }
  
  // Check 3: Cost-based limit
  const cost = await limiter.costBasedLimit(
    userId,
    operation,
    limits.costBudget
  );
  
  if (!cost.allowed) {
    return rateLimitError(res, {
      type: 'cost_budget',
      cost: cost.cost,
      spent: cost.spent,
      budget: cost.budget
    });
  }
  
  // Check 4: Concurrent requests
  const concurrent = await limiter.concurrentLimit(
    userId,
    limits.concurrent
  );
  
  if (!concurrent.allowed) {
    return rateLimitError(res, {
      type: 'concurrent_limit',
      current: concurrent.current,
      max: concurrent.max
    });
  }
  
  // Release concurrent slot on finish
  res.on('finish', concurrent.release);
  res.on('close', concurrent.release);
  
  // Check 5: Burst protection (token bucket)
  const burst = await limiter.tokenBucketLimit(
    userId,
    50,    // 50 token capacity
    10     // Refill 10 tokens/second
  );
  
  if (!burst.allowed) {
    return rateLimitError(res, {
      type: 'burst_limit',
      retryAfter: burst.retryAfter
    });
  }
  
  // Set rate limit headers
  res.set({
    'X-RateLimit-Limit-Hourly': limits.hourly,
    'X-RateLimit-Remaining-Hourly': hourly.remaining,
    'X-RateLimit-Reset-Hourly': hourly.resetAt,
    
    'X-RateLimit-Limit-Daily': limits.daily,
    'X-RateLimit-Remaining-Daily': daily.remaining,
    'X-RateLimit-Reset-Daily': daily.resetAt,
    
    'X-RateLimit-Cost-Budget': limits.costBudget,
    'X-RateLimit-Cost-Remaining': cost.remaining,
    'X-RateLimit-Cost-Used': cost.spent,
    
    'X-RateLimit-Concurrent-Max': limits.concurrent,
    'X-RateLimit-Concurrent-Current': concurrent.current
  });
  
  next();
}

function rateLimitError(res, details) {
  res.status(429).json({
    error: 'Rate limit exceeded',
    type: details.type,
    details,
    retryAfter: details.retryAfter || 
      (details.resetAt - Date.now() / 1000)
  });
}

app.use(rateLimitMiddleware);
\`\`\`

**4. Admin Dashboard**

\`\`\`javascript
// Real-time rate limit monitoring
app.get('/admin/rate-limits/:userId', async (req, res) => {
  const { userId } = req.params;
  const plan = await getUserPlan(userId);
  const limits = RATE_LIMITS[plan];
  
  const [hourly, daily, costSpent, concurrent] = await Promise.all([
    redis.get(\`ratelimit:\${userId}:hourly:*\`),
    redis.get(\`ratelimit:\${userId}:daily:*\`),
    redis.get(\`ratelimit:cost:\${userId}:*\`),
    redis.get(\`ratelimit:concurrent:\${userId}\`)
  ]);
  
  res.json({
    userId,
    plan,
    hourly: {
      limit: limits.hourly,
      used: parseInt(hourly || '0'),
      remaining: limits.hourly - parseInt(hourly || '0')
    },
    daily: {
      limit: limits.daily,
      used: parseInt(daily || '0'),
      remaining: limits.daily - parseInt(daily || '0')
    },
    cost: {
      budget: limits.costBudget,
      spent: parseInt(costSpent || '0'),
      remaining: limits.costBudget - parseInt(costSpent || '0')
    },
    concurrent: {
      max: limits.concurrent,
      current: parseInt(concurrent || '0')
    }
  });
});
\`\`\`

**5. Whitelist**

\`\`\`javascript
const WHITELISTED_USERS = new Set([
  'admin-user',
  'monitoring-service',
  'internal-api'
]);

// Skip rate limiting for whitelisted users
async function rateLimitMiddleware(req, res, next) {
  if (WHITELISTED_USERS.has(req.user.id)) {
    return next();
  }
  
  // Apply rate limiting...
}
\`\`\`

**Key Design Decisions**:

1. **Multi-layer checks**: Hourly, daily, cost-based, concurrent, burst
2. **Sliding window counter**: Smooth rate limiting without burst issues
3. **Cost-based**: Expensive operations cost more points
4. **Token bucket**: Allows short bursts while maintaining average rate
5. **Comprehensive headers**: Clients know their usage across all dimensions
6. **Redis cluster**: Distributed, high-availability rate limiting

This system prevents abuse while allowing legitimate usage patterns.`,
    keyPoints: [
      'Multi-layer rate limiting: hourly, daily, cost-based, concurrent, burst',
      'Sliding window counter prevents burst issues at window boundaries',
      'Cost-based limiting assigns higher costs to expensive operations',
      'Token bucket allows short bursts while enforcing average rate',
      'Comprehensive headers inform clients of usage across all dimensions',
    ],
  },
  {
    id: 'ratelimit-d2',
    question:
      'Your API is experiencing abuse from sophisticated attackers who rotate IP addresses and create multiple free accounts. How would you implement advanced rate limiting to prevent this?',
    sampleAnswer: `Advanced rate limiting strategies against sophisticated abuse:

**1. Fingerprinting-Based Rate Limiting**

\`\`\`javascript
const FingerprintJS = require('@fingerprintjs/fingerprintjs');

// Generate device fingerprint
async function getDeviceFingerprint(req) {
  const factors = [
    req.headers['user-agent',],
    req.headers['accept-language',],
    req.headers['accept-encoding',],
    req.connection.remoteAddress,
    // Additional factors from client fingerprinting
    req.body.screenResolution,
    req.body.timezone,
    req.body.browserPlugins
  ];
  
  const hash = crypto.createHash('sha256')
    .update(factors.join('|'))
    .digest('hex');
  
  return hash;
}

// Rate limit by fingerprint (catches IP rotation)
async function fingerprintRateLimit(req, res, next) {
  const fingerprint = await getDeviceFingerprint(req);
  const key = \`ratelimit:fingerprint:\${fingerprint}\`;
  
  const count = await redis.incr(key);
  await redis.expire(key, 3600);
  
  if (count > 100) {  // 100 requests/hour per device
    return res.status(429).json({
      error: 'Rate limit exceeded',
      message: 'Too many requests from this device'
    });
  }
  
  next();
}
\`\`\`

**2. Behavioral Analysis**

\`\`\`javascript
// Track suspicious patterns
class BehavioralAnalyzer {
  async analyzeRequest(req) {
    const userId = req.user?.id || req.ip;
    const patterns = await this.getPatterns(userId);
    
    const signals = {
      // Signal 1: Rapid account creation
      newAccountRate: await this.checkAccountCreationRate(req.ip),
      
      // Signal 2: Unusual access patterns
      accessPattern: await this.checkAccessPattern(userId),
      
      // Signal 3: Low-value interactions
      interactionQuality: await this.checkInteractionQuality(userId),
      
      // Signal 4: Automation detection
      isAutomated: await this.detectAutomation(req)
    };
    
    const suspicionScore = this.calculateSuspicionScore(signals);
    
    return {
      suspicionScore,
      signals,
      action: this.determineAction(suspicionScore)
    };
  }
  
  async checkAccountCreationRate(ip) {
    const key = \`behavioral:accounts:\${ip}\`;
    const accounts = await redis.zcount(
      key,
      Date.now() - 3600000,  // Last hour
      Date.now()
    );
    
    // More than 5 accounts/hour from same IP = suspicious
    return accounts > 5 ? 1.0 : accounts / 5;
  }
  
  async checkAccessPattern(userId) {
    const key = \`behavioral:pattern:\${userId}\`;
    const requests = await redis.lrange(key, 0, -1);
    
    // Check for bot-like patterns:
    // - Too consistent timing (exactly every N seconds)
    // - Sequential resource access (user1, user2, user3...)
    // - No variation in user-agent
    
    const timestamps = requests.map(r => JSON.parse(r).timestamp);
    const intervals = [];
    for (let i = 1; i < timestamps.length; i++) {
      intervals.push(timestamps[i] - timestamps[i-1]);
    }
    
    // Calculate variance (bots have low variance)
    const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((sum, interval) => 
      sum + Math.pow(interval - mean, 2), 0) / intervals.length;
    
    // Low variance = bot-like
    return variance < 100 ? 1.0 : 0.0;
  }
  
  async checkInteractionQuality(userId) {
    const key = \`behavioral:quality:\${userId}\`;
    const actions = await redis.hgetall(key);
    
    const reads = parseInt(actions.reads || '0');
    const writes = parseInt(actions.writes || '0');
    
    // Attackers mostly read (scraping), legitimate users write
    const readWriteRatio = writes === 0 ? reads : reads / writes;
    
    // High read-to-write ratio = suspicious
    return readWriteRatio > 100 ? 1.0 : readWriteRatio / 100;
  }
  
  async detectAutomation(req) {
    // Check for automation signals
    const signals = [
      // No cookies/session
      !req.headers.cookie,
      
      // Suspicious user-agent
      /bot|crawler|spider|scraper/i.test(req.headers['user-agent',]),
      
      // Missing common headers
      !req.headers['accept-language',],
      
      // Headless browser detection
      req.headers['chrome-lighthouse',] !== undefined,
      
      // Too fast (< 100ms between requests)
      await this.checkRequestTiming(req.user?.id || req.ip)
    ];
    
    const automationScore = signals.filter(Boolean).length / signals.length;
    return automationScore;
  }
  
  calculateSuspicionScore(signals) {
    // Weighted scoring
    return (
      signals.newAccountRate * 0.3 +
      signals.accessPattern * 0.3 +
      signals.interactionQuality * 0.2 +
      signals.isAutomated * 0.2
    );
  }
  
  determineAction(score) {
    if (score > 0.8) return 'block';
    if (score > 0.6) return 'challenge';  // CAPTCHA
    if (score > 0.4) return 'throttle';   // Stricter rate limit
    return 'allow';
  }
}

const analyzer = new BehavioralAnalyzer();

app.use(async (req, res, next) => {
  const analysis = await analyzer.analyzeRequest(req);
  
  if (analysis.action === 'block') {
    return res.status(403).json({
      error: 'Access denied',
      message: 'Suspicious activity detected'
    });
  }
  
  if (analysis.action === 'challenge') {
    // Require CAPTCHA
    if (!req.body.captchaToken) {
      return res.status(403).json({
        error: 'CAPTCHA required',
        challengeUrl: '/captcha'
      });
    }
  }
  
  if (analysis.action === 'throttle') {
    // Apply stricter rate limit
    req.rateLimit = { requests: 10, window: 3600 };  // 10/hour instead of normal
  }
  
  next();
});
\`\`\`

**3. Progressive Rate Limiting**

\`\`\`javascript
// Gradually tighten limits for suspicious users
async function progressiveRateLimit(userId) {
  const violationsKey = \`ratelimit:violations:\${userId}\`;
  const violations = parseInt(await redis.get(violationsKey) || '0');
  
  // Base limit
  let limit = 1000;
  let window = 3600;
  
  // Reduce limit with each violation
  if (violations > 0) {
    limit = Math.max(10, limit / Math.pow(2, violations));
  }
  
  const result = await slidingWindowCounter(userId, limit, window);
  
  if (!result.allowed) {
    // Increment violations
    await redis.incr(violationsKey);
    await redis.expire(violationsKey, 86400);  // Reset daily
    
    // Exponentially increase penalty
    const penaltySeconds = Math.min(
      3600,  // Max 1 hour
      Math.pow(2, violations) * 60  // Double each time
    );
    
    return {
      allowed: false,
      remaining: 0,
      penaltySeconds,
      violations: violations + 1
    };
  }
  
  return result;
}
\`\`\`

**4. Distributed Deduplication**

\`\`\`javascript
// Detect multiple accounts from same entity
async function detectSiblingAccounts(userId, fingerprint, email) {
  const emailDomain = email.split('@')[1];
  
  // Find accounts with similar signals
  const siblingKeys = [
    \`siblings:fingerprint:\${fingerprint}\`,
    \`siblings:email:\${emailDomain}\`
  ];
  
  const siblings = new Set();
  for (const key of siblingKeys) {
    const accounts = await redis.smembers(key);
    accounts.forEach(id => siblings.add(id));
  }
  
  // Add current user to sibling groups
  for (const key of siblingKeys) {
    await redis.sadd(key, userId);
    await redis.expire(key, 86400);
  }
  
  // Share rate limit across sibling accounts
  if (siblings.size > 1) {
    const combinedKey = \`ratelimit:siblings:\${Array.from(siblings).sort().join(':')}\`;
    return combinedKey;  // Use this key for rate limiting
  }
  
  return \`ratelimit:user:\${userId}\`;
}
\`\`\`

**5. CAPTCHA Integration**

\`\`\`javascript
const axios = require('axios');

async function verifyCaptcha(token) {
  const response = await axios.post(
    'https://www.google.com/recaptcha/api/siteverify',
    null,
    {
      params: {
        secret: process.env.RECAPTCHA_SECRET,
        response: token
      }
    }
  );
  
  return response.data.success && response.data.score > 0.5;
}

app.use(async (req, res, next) => {
  const suspicionScore = await analyzer.analyzeRequest(req);
  
  if (suspicionScore.action === 'challenge') {
    if (!req.body.captchaToken) {
      return res.status(403).json({
        error: 'CAPTCHA required',
        message: 'Please complete CAPTCHA to continue'
      });
    }
    
    const captchaValid = await verifyCaptcha(req.body.captchaToken);
    if (!captchaValid) {
      return res.status(403).json({
        error: 'Invalid CAPTCHA',
        message: 'CAPTCHA verification failed'
      });
    }
  }
  
  next();
});
\`\`\`

**6. IP Reputation Service**

\`\`\`javascript
// Integrate with IP reputation services
async function checkIPReputation(ip) {
  // Check against known VPN/proxy/datacenter IPs
  const response = await axios.get(
    \`https://ipqualityscore.com/api/json/ip/\${process.env.IPQS_KEY}/\${ip}\`
  );
  
  const { proxy, vpn, tor, recent_abuse, fraud_score } = response.data;
  
  if (proxy || vpn || tor || recent_abuse || fraud_score > 75) {
    return {
      suspicious: true,
      reason: proxy ? 'proxy' : vpn ? 'vpn' : 'high_fraud_score',
      score: fraud_score
    };
  }
  
  return { suspicious: false };
}

app.use(async (req, res, next) => {
  const reputation = await checkIPReputation(req.ip);
  
  if (reputation.suspicious) {
    // Apply stricter rate limit or require CAPTCHA
    req.rateLimit = { requests: 10, window: 3600 };
  }
  
  next();
});
\`\`\`

**Key Strategies**:

1. **Device fingerprinting**: Catch IP rotation
2. **Behavioral analysis**: Detect bot patterns
3. **Progressive penalties**: Increase restrictions with violations
4. **Sibling account detection**: Share limits across related accounts
5. **CAPTCHA challenges**: Human verification for suspicious activity
6. **IP reputation**: Block known malicious IPs

This multi-layered approach makes it extremely difficult for attackers to abuse the API.`,
    keyPoints: [
      'Device fingerprinting catches attackers rotating IP addresses',
      'Behavioral analysis detects bot-like patterns (timing, access patterns)',
      'Progressive rate limiting increases penalties with repeated violations',
      'Sibling account detection shares rate limits across related accounts',
      'CAPTCHA challenges provide human verification for suspicious activity',
    ],
  },
  {
    id: 'ratelimit-d3',
    question:
      'Compare different rate limiting algorithms (Fixed Window, Sliding Window, Token Bucket, Leaky Bucket) for a real-time chat API. Which would you choose and why?',
    sampleAnswer: `Comparison of rate limiting algorithms for real-time chat API:

**Scenario: Real-Time Chat API**

Requirements:
- Users send messages in bursts (normal conversation)
- Need to prevent spam (rapid message flooding)
- Must feel responsive (low latency)
- Should handle reconnections gracefully

**Algorithm Comparison**:

**1. Fixed Window**

\`\`\`javascript
// 10 messages per 60 seconds
async function fixedWindowChat(userId, limit = 10, window = 60) {
  const key = \`chat:ratelimit:\${userId}:\${Math.floor(Date.now() / 1000 / window)}\`;
  const count = await redis.incr(key);
  await redis.expire(key, window);
  
  return {
    allowed: count <= limit,
    remaining: Math.max(0, limit - count)
  };
}
\`\`\`

**Pros**:
- Simple, fast
- Low memory usage

**Cons**:
- Burst at boundaries: user sends 10 messages at t=59s, then 10 more at t=60s = 20 messages in 1 second
- Poor UX: sudden cut-off at window boundary

**Verdict for Chat**: ❌ Not suitable (burst issues)

**2. Sliding Window Counter**

\`\`\`javascript
// 10 messages per 60 seconds (smooth)
async function slidingWindowChat(userId, limit = 10, window = 60) {
  const now = Date.now() / 1000;
  const currentWindow = Math.floor(now / window);
  const previousWindow = currentWindow - 1;
  
  const currentKey = \`chat:\${userId}:\${currentWindow}\`;
  const previousKey = \`chat:\${userId}:\${previousWindow}\`;
  
  const currentCount = parseInt(await redis.get(currentKey) || '0');
  const previousCount = parseInt(await redis.get(previousKey) || '0');
  
  const percentageInCurrent = (now % window) / window;
  const weightedCount = 
    previousCount * (1 - percentageInCurrent) + currentCount;
  
  if (weightedCount >= limit) {
    return { allowed: false, remaining: 0 };
  }
  
  await redis.incr(currentKey);
  await redis.expire(currentKey, window * 2);
  
  return {
    allowed: true,
    remaining: Math.floor(limit - weightedCount - 1)
  };
}
\`\`\`

**Pros**:
- Smooth rate limiting (no burst at boundaries)
- Memory efficient (2 counters)
- Accurate

**Cons**:
- Doesn't allow bursts (chat is naturally bursty)

**Verdict for Chat**: ✅ Good, but might be too strict for conversation bursts

**3. Token Bucket** (RECOMMENDED)

\`\`\`javascript
// 10 token capacity, refill 1 token every 6 seconds
async function tokenBucketChat(userId, capacity = 10, refillRate = 1/6) {
  const key = \`chat:bucket:\${userId}\`;
  
  const data = await redis.get(key);
  let tokens, lastRefill;
  
  if (data) {
    ({ tokens, lastRefill } = JSON.parse(data));
  } else {
    tokens = capacity;
    lastRefill = Date.now();
  }
  
  // Refill tokens based on time elapsed
  const now = Date.now();
  const elapsed = (now - lastRefill) / 1000;
  const tokensToAdd = elapsed * refillRate;
  tokens = Math.min(capacity, tokens + tokensToAdd);
  
  if (tokens < 1) {
    return {
      allowed: false,
      remaining: 0,
      retryAfter: (1 - tokens) / refillRate
    };
  }
  
  // Consume 1 token
  tokens -= 1;
  
  await redis.set(key, JSON.stringify({
    tokens,
    lastRefill: now
  }), 'EX', 3600);
  
  return {
    allowed: true,
    remaining: Math.floor(tokens)
  };
}
\`\`\`

**Pros**:
- **Allows bursts** (up to capacity): Perfect for conversations!
- Smooth refill: tokens accumulate over time
- User can send 10 messages quickly, then must wait
- Feels natural for chat (burst then pause)

**Cons**:
- Slightly more complex than fixed window
- Requires storing float (tokens can be fractional)

**Verdict for Chat**: ✅✅ **BEST CHOICE** (allows natural conversation bursts)

**4. Leaky Bucket**

\`\`\`javascript
// Process messages at fixed rate (1 message every 6 seconds)
async function leakyBucketChat(userId, capacity = 10, leakRate = 1/6) {
  const queueKey = \`chat:queue:\${userId}\`;
  const lastLeakKey = \`chat:lastleak:\${userId}\`;
  
  let queueSize = await redis.llen(queueKey);
  const lastLeak = parseInt(await redis.get(lastLeakKey) || Date.now());
  
  // Leak messages
  const now = Date.now();
  const elapsed = (now - lastLeak) / 1000;
  const messagesToLeak = Math.floor(elapsed * leakRate);
  
  if (messagesToLeak > 0) {
    const leaked = Math.min(messagesToLeak, queueSize);
    for (let i = 0; i < leaked; i++) {
      await redis.rpop(queueKey);
    }
    queueSize -= leaked;
    await redis.set(lastLeakKey, now);
  }
  
  if (queueSize >= capacity) {
    return {
      allowed: false,
      queueFull: true,
      retryAfter: (queueSize - capacity + 1) / leakRate
    };
  }
  
  await redis.lpush(queueKey, now);
  await redis.expire(queueKey, 3600);
  
  return {
    allowed: true,
    queueSize: queueSize + 1
  };
}
\`\`\`

**Pros**:
- Smooth processing rate
- Good for backend with fixed capacity

**Cons**:
- **Delays messages**: User sends message but it's queued
- Poor UX for chat (feels laggy)
- Doesn't fit real-time requirement

**Verdict for Chat**: ❌ Not suitable (introduces latency)

**Comparison Table for Chat API**:

| Algorithm | Burst Handling | Smoothness | Complexity | UX | Verdict |
|-----------|----------------|------------|------------|----|---------| 
| Fixed Window | Poor (boundary bursts) | Poor | Low | Bad | ❌ |
| Sliding Window | None (strict) | Excellent | Medium | Good | ✅ |
| Token Bucket | Excellent (capacity) | Good | Medium | Excellent | ✅✅ |
| Leaky Bucket | None (queues) | Excellent | High | Poor (lag) | ❌ |

**Recommendation: Token Bucket**

**Implementation for Chat**:

\`\`\`javascript
const express = require('express');
const WebSocket = require('ws');

const app = express();
const wss = new WebSocket.Server({ port: 8080 });

// Token bucket rate limiter
class ChatRateLimiter {
  constructor() {
    this.capacity = 10;      // Allow burst of 10 messages
    this.refillRate = 1/6;   // Refill 1 token every 6 seconds (10/minute)
  }
  
  async checkLimit(userId) {
    return await tokenBucketChat(
      userId,
      this.capacity,
      this.refillRate
    );
  }
}

const rateLimiter = new ChatRateLimiter();

wss.on('connection', (ws, req) => {
  const userId = req.user.id;
  
  ws.on('message', async (message) => {
    // Rate limit check
    const limit = await rateLimiter.checkLimit(userId);
    
    if (!limit.allowed) {
      ws.send(JSON.stringify({
        type: 'error',
        error: 'Rate limit exceeded',
        message: 'Please slow down. You can send another message in ' + 
                 Math.ceil(limit.retryAfter) + ' seconds.',
        retryAfter: limit.retryAfter
      }));
      return;
    }
    
    // Send rate limit info
    ws.send(JSON.stringify({
      type: 'ratelimit',
      remaining: limit.remaining,
      capacity: rateLimiter.capacity
    }));
    
    // Broadcast message
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify({
          type: 'message',
          userId,
          text: message,
          timestamp: Date.now()
        }));
      }
    });
  });
});

app.listen(3000);
\`\`\`

**Why Token Bucket for Chat**:

1. **Burst-friendly**: Users can send 10 messages quickly (natural conversation)
2. **Prevents spam**: After burst, must wait for tokens to refill
3. **Smooth refill**: Tokens accumulate gradually (1 every 6 seconds)
4. **Good UX**: Feels responsive, not restrictive
5. **Flexible**: Can adjust capacity (burst size) and refill rate independently

**Example Usage Patterns**:

\`\`\`
User sends: "Hey" "How are you?" "What's up?"
Result: 3 tokens consumed, 7 remaining (burst allowed)

User tries to send 15 messages rapidly:
Messages 1-10: ✅ Allowed (burst capacity)
Messages 11-15: ❌ Blocked (wait ~30 seconds for refill)

After 60 seconds:
Tokens refilled to 10 (ready for next conversation)
\`\`\`

**Alternative: Hybrid Approach**

For production chat, consider combining Token Bucket + Sliding Window:

\`\`\`javascript
async function hybridChatRateLimit(userId) {
  // Token bucket: Allow bursts
  const burst = await tokenBucketChat(userId, 10, 1/6);
  if (!burst.allowed) {
    return burst;
  }
  
  // Sliding window: Hard limit (100/hour)
  const hourly = await slidingWindowChat(userId, 100, 3600);
  if (!hourly.allowed) {
    return hourly;
  }
  
  return {
    allowed: true,
    burstRemaining: burst.remaining,
    hourlyRemaining: hourly.remaining
  };
}
\`\`\`

This prevents:
- Short-term spam (token bucket)
- Long-term abuse (sliding window)

**Final Verdict**: Token Bucket is the best algorithm for real-time chat APIs due to its burst-friendliness and natural feel.`,
    keyPoints: [
      'Token Bucket is best for chat: allows natural conversation bursts',
      'Fixed Window has burst issues at window boundaries',
      'Sliding Window is too strict (no bursts), poor for bursty chat',
      'Leaky Bucket introduces latency (queues messages), bad UX',
      'Hybrid approach combines Token Bucket (short-term) + Sliding Window (long-term)',
    ],
  },
];
