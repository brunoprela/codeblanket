/**
 * Quiz questions for Rate Limiting & Throttling section
 */

export const ratelimitingQuiz = [
  {
    id: 'rate-limit-distributed-system',
    question:
      'Design a distributed rate limiting system for an API with 100 servers handling 1 million requests per second. The system must support per-user limits (10,000 req/hr), per-IP limits (1,000 req/hr), and per-endpoint limits. Explain your architecture, choice of algorithm, Redis configuration, handling of edge cases (clock drift, Redis failures), and monitoring strategy.',
    sampleAnswer: `**Distributed Rate Limiting System Design**
    
    **1. Architecture**
    
    \`\`\`
    Client Request
        ↓
    Load Balancer
        ↓
    API Server (1 of 100)
        ↓
    Rate Limit Middleware
        ↓
    Redis Cluster (rate limit state)
        ↓
    Backend Services
    \`\`\`
    
    **2. Algorithm Choice: Token Bucket with Redis**
    
    **Why Token Bucket**:
    - Allows bursts (better UX)
    - Memory efficient (2 values per key)
    - Natural fit for time-based limits
    - Fast (constant time operations)
    
    **3. Redis Cluster Configuration**
    
    \`\`\`yaml
    # redis.conf
    cluster-enabled yes
    cluster-node-timeout 5000
    
    # Persistence (for rate limit state)
    save 900 1        # Save after 900 sec if 1 key changed
    appendonly yes    # AOF for durability
    appendfsync everysec
    
    # Memory
    maxmemory 16gb
    maxmemory-policy allkeys-lru  # Evict old rate limit keys
    
    # Replication
    min-replicas-to-write 1  # Require 1 replica acknowledgment
    \`\`\`
    
    **Cluster Setup**:
    - 6 Redis nodes (3 masters + 3 replicas)
    - Hash slot distribution: 16384 slots / 3 masters
    - Each master handles ~333K requests/sec
    - Replication for high availability
    
    **4. Implementation**
    
    **Lua Script** (atomic token bucket):
    \`\`\`lua
    -- rate_limit.lua
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local cost = tonumber(ARGV[4]) or 1
    
    -- Get current state
    local tokens = tonumber(redis.call('HGET', key, 'tokens'))
    local last_refill = tonumber(redis.call('HGET', key, 'last_refill'))
    
    -- Initialize if not exists
    if not tokens then
      tokens = capacity
      last_refill = now
    end
    
    -- Refill tokens
    local time_passed = (now - last_refill) / 1000  -- seconds
    local tokens_to_add = time_passed * refill_rate
    tokens = math.min(capacity, tokens + tokens_to_add)
    
    -- Try to consume
    if tokens >= cost then
      tokens = tokens - cost
      redis.call('HSET', key, 'tokens', tokens, 'last_refill', now)
      redis.call('EXPIRE', key, 7200)  -- 2 hour TTL
      
      return {1, math.floor(tokens), 0}  -- allowed, remaining, retry_after
    else
      -- Calculate retry after (seconds)
      local tokens_needed = cost - tokens
      local retry_after = math.ceil(tokens_needed / refill_rate)
      
      return {0, 0, retry_after}  -- denied, remaining=0, retry_after
    end
    \`\`\`
    
    **Node.js Middleware**:
    \`\`\`javascript
    const Redis = require('ioredis');
    const fs = require('fs');
    
    // Redis cluster client
    const redis = new Redis.Cluster([
      { host: 'redis-1', port: 6379 },
      { host: 'redis-2', port: 6379 },
      { host: 'redis-3', port: 6379 }
    ], {
      redisOptions: {
        password: process.env.REDIS_PASSWORD
      }
    });
    
    // Load Lua script
    const rateLimitScript = fs.readFileSync('rate_limit.lua', 'utf8');
    const scriptSha = await redis.script('LOAD', rateLimitScript);
    
    // Rate limit configuration
    const limits = {
      perUser: { capacity: 10000, refillRate: 2.78 }, // 10K per hour = 2.78/sec
      perIP: { capacity: 1000, refillRate: 0.278 },   // 1K per hour = 0.278/sec
      perEndpoint: {
        'POST /api/login': { capacity: 5, refillRate: 0.083 }, // 5 per minute
        'POST /api/upload': { capacity: 10, refillRate: 0.0028 } // 10 per hour
      }
    };
    
    async function rateLimit(identifier, config) {
      const key = \`rate_limit:\${identifier}\`;
      
      try {
        // Use Redis TIME for consistency across servers (handles clock drift)
        const [seconds, microseconds] = await redis.time();
        const now = seconds * 1000 + Math.floor(microseconds / 1000);
        
        const result = await redis.evalsha(
          scriptSha,
          1,
          key,
          config.capacity,
          config.refillRate,
          now,
          1 // cost
        );
        
        return {
          allowed: result[0] === 1,
          remaining: result[1],
          retryAfter: result[2]
        };
      } catch (error) {
        // Fail open on Redis errors (allow request but log)
        logger.error('Rate limit check failed', { identifier, error });
        return { allowed: true, remaining: -1, retryAfter: 0 };
      }
    }
    
    // Middleware
    app.use(async (req, res, next) => {
      const userId = req.user?.id;
      const ip = req.ip || req.connection.remoteAddress;
      const endpoint = \`\${req.method} \${req.path}\`;
      
      // Check multiple limits
      const checks = [];
      
      // Per-user limit (if authenticated)
      if (userId) {
        checks.push({
          name: 'per-user',
          identifier: \`user:\${userId}\`,
          config: limits.perUser
        });
      }
      
      // Per-IP limit
      checks.push({
        name: 'per-ip',
        identifier: \`ip:\${ip}\`,
        config: limits.perIP
      });
      
      // Per-endpoint limit
      const endpointConfig = limits.perEndpoint[endpoint];
      if (endpointConfig) {
        checks.push({
          name: 'per-endpoint',
          identifier: \`\${userId || ip}:\${endpoint}\`,
          config: endpointConfig
        });
      }
      
      // Run all checks in parallel
      const results = await Promise.all(
        checks.map(check => 
          rateLimit(check.identifier, check.config)
            .then(result => ({ ...result, check }))
        )
      );
      
      // Find first limit exceeded
      const blocked = results.find(r => !r.allowed);
      
      if (blocked) {
        const config = blocked.check.config;
        
        res.setHeader('X-RateLimit-Limit', config.capacity);
        res.setHeader('X-RateLimit-Remaining', 0);
        res.setHeader('Retry-After', blocked.retryAfter);
        
        return res.status(429).json({
          error: 'Too Many Requests',
          limit: blocked.check.name,
          retryAfter: blocked.retryAfter
        });
      }
      
      // Set headers for successful request
      const userLimit = results.find(r => r.check.name === 'per-user');
      if (userLimit) {
        res.setHeader('X-RateLimit-Limit', limits.perUser.capacity);
        res.setHeader('X-RateLimit-Remaining', userLimit.remaining);
      }
      
      next();
    });
    \`\`\`
    
    **5. Handling Edge Cases**
    
    **Clock Drift**:
    \`\`\`javascript
    // Use Redis TIME instead of server time
    const [seconds, microseconds] = await redis.time();
    const now = seconds * 1000 + Math.floor(microseconds / 1000);
    
    // Redis provides consistent time across all servers
    \`\`\`
    
    **Redis Failure**:
    \`\`\`javascript
    async function rateLimit(identifier, config) {
      try {
        // Normal rate limiting
        return await rateLimitRedis(identifier, config);
      } catch (error) {
        logger.error('Redis error, failing open', { error });
        
        // Strategy 1: Fail open (allow request)
        return { allowed: true, remaining: -1 };
        
        // Strategy 2: Fail closed (deny request)
        // return { allowed: false, remaining: 0, retryAfter: 60 };
        
        // Strategy 3: Local rate limiting (fallback)
        return await rateLimitLocal(identifier, config);
      }
    }
    \`\`\`
    
    **Redis Cluster Split-Brain**:
    \`\`\`yaml
    # redis.conf
    cluster-require-full-coverage yes  # Stop serving if cluster unhealthy
    min-replicas-to-write 1           # Require replica acknowledgment
    \`\`\`
    
    **6. Monitoring & Alerting**
    
    **Metrics to Track**:
    \`\`\`javascript
    const metrics = {
      // Rate limit hits
      rateLimitHitsTotal: new Counter({
        name: 'rate_limit_hits_total',
        help: 'Total rate limit hits',
        labelNames: ['limit_type', 'endpoint']
      }),
      
      // Allowed requests
      rateLimitAllowed: new Counter({
        name: 'rate_limit_allowed_total',
        help: 'Allowed requests',
        labelNames: ['limit_type']
      }),
      
      // Redis latency
      redisLatency: new Histogram({
        name: 'redis_latency_seconds',
        help: 'Redis operation latency'
      }),
      
      // Tokens remaining (gauge)
      tokensRemaining: new Gauge({
        name: 'rate_limit_tokens_remaining',
        help: 'Tokens remaining per user'
      })
    };
    
    // Track in middleware
    if (blocked) {
      metrics.rateLimitHitsTotal.inc({
        limit_type: blocked.check.name,
        endpoint: req.path
      });
    } else {
      metrics.rateLimitAllowed.inc({
        limit_type: 'per-user'
      });
    }
    \`\`\`
    
    **Grafana Dashboard**:
    \`\`\`yaml
    # Key metrics to visualize
    - Rate limit hit rate (by type, endpoint)
    - Top rate-limited users/IPs
    - Redis latency (p50, p95, p99)
    - Redis cluster health (nodes up, replication lag)
    - Request distribution across limits
    \`\`\`
    
    **Alerts**:
    \`\`\`yaml
    # Prometheus alerts
    - alert: HighRateLimitHitRate
      expr: rate(rate_limit_hits_total[5m]) > 100
      annotations:
        summary: "High rate limit hit rate: {{ $value }}/sec"
    
    - alert: RedisClusterDown
      expr: redis_cluster_state != 1
      annotations:
        summary: "Redis cluster unhealthy"
    
    - alert: RedisHighLatency
      expr: histogram_quantile(0.95, redis_latency_seconds) > 0.1
      annotations:
        summary: "Redis p95 latency > 100ms"
    \`\`\`
    
    **7. Performance Optimizations**
    
    **Connection Pooling**:
    \`\`\`javascript
    // Reuse Redis connection
    const redis = new Redis.Cluster([...], {
      poolSize: 10,      // Connection pool per node
      enableReadyCheck: true,
      maxRetriesPerRequest: 3
    });
    \`\`\`
    
    **Pipelining for Multiple Checks**:
    \`\`\`javascript
    // Instead of sequential checks
    const result1 = await rateLimit('user:123', limits.perUser);
    const result2 = await rateLimit('ip:1.2.3.4', limits.perIP);
    
    // Use pipeline (parallel)
    const pipeline = redis.pipeline();
    pipeline.evalsha(scriptSha, 1, 'user:123', ...);
    pipeline.evalsha(scriptSha, 1, 'ip:1.2.3.4', ...);
    const results = await pipeline.exec();
    \`\`\`
    
    **8. Testing Strategy**
    
    **Load Test**:
    \`\`\`bash
    # Simulate 1M req/sec
    artillery run --target https://api.example.com \\
      --count 10000 \\
      --rate 100 \\
      rate-limit-test.yml
    \`\`\`
    
    **Test Cases**:
    \`\`\`javascript
    describe('Rate Limiting', () => {
      it('should allow requests under limit', async () => {
        for (let i = 0; i < 100; i++) {
          const res = await request(app).get('/api/data');
          expect(res.status).toBe(200);
        }
      });
      
      it('should block requests over limit', async () => {
        // Make 101 requests
        for (let i = 0; i < 101; i++) {
          const res = await request(app).get('/api/data');
          if (i < 100) {
            expect(res.status).toBe(200);
          } else {
            expect(res.status).toBe(429);
            expect(res.headers['retry-after']).toBeDefined();
          }
        }
      });
      
      it('should reset after window expires', async () => {
        // Hit limit
        for (let i = 0; i < 100; i++) {
          await request(app).get('/api/data');
        }
        
        // Wait for refill
        await sleep(60000); // 1 minute
        
        // Should allow again
        const res = await request(app).get('/api/data');
        expect(res.status).toBe(200);
      });
    });
    \`\`\`
    
    **Key Takeaways**:
    
    1. **Token Bucket + Redis** for distributed rate limiting
    2. **Lua scripts** ensure atomic operations (prevent race conditions)
    3. **Redis TIME** handles clock drift across servers
    4. **Fail open** on Redis errors (better UX) but log for monitoring
    5. **Multiple limits** (per-user, per-IP, per-endpoint) for flexibility
    6. **Redis Cluster** (3 masters + 3 replicas) for 1M req/sec
    7. **Connection pooling** and pipelining for performance
    8. **Monitor**: hit rate, Redis latency, cluster health
    9. **Alert**: high hit rate, Redis down, high latency
    10. **Test**: load testing, edge cases, Redis failures`,
    keyPoints: [
      'Token Bucket + Redis Cluster for distributed rate limiting at scale',
      'Lua scripts provide atomic operations to prevent race conditions across servers',
      'Use Redis TIME command to handle clock drift in distributed systems',
      'Implement multiple rate limit dimensions: per-user, per-IP, per-endpoint',
      'Fail open on Redis errors for better UX, but monitor and alert',
      'Redis Cluster (3 masters + 3 replicas) handles 1M requests/second',
    ],
  },
  {
    id: 'rate-limit-disc-2',
    question:
      'You have a public API with three tiers: Free (100 requests/day), Pro ($50/month, 10,000 requests/day), and Enterprise (custom pricing, unlimited). Design the rate limiting strategy including how to handle burst traffic, trial periods, overages, and billing integration. Discuss implementation, user experience, and edge cases.',
    sampleAnswer: `**Tiered Rate Limiting Strategy for Public API**

**1. Rate Limit Structure**

\`\`\`typescript
interface RateLimitTier {
  name: string;
  dailyLimit: number;
  burstLimit: number;
  overage: {
    allowed: boolean;
    maxOverage: number;
    costPerRequest: number;
  };
}

const tiers: Record<string, RateLimitTier> = {
  free: {
    name: 'Free',
    dailyLimit: 100,
    burstLimit: 10, // 10 requests per minute burst
    overage: {
      allowed: false,
      maxOverage: 0,
      costPerRequest: 0
    }
  },
  pro: {
    name: 'Pro',
    dailyLimit: 10_000,
    burstLimit: 100, // 100 requests per minute burst
    overage: {
      allowed: true,
      maxOverage: 1000, // Allow 10% overage
      costPerRequest: 0.01 // $0.01 per request over limit
    }
  },
  enterprise: {
    name: 'Enterprise',
    dailyLimit: Infinity,
    burstLimit: 1000, // 1000 requests per minute burst
    overage: {
      allowed: true,
      maxOverage: Infinity,
      costPerRequest: 0 // Custom billing
    }
  }
};
\`\`\`

**2. Multi-Dimensional Rate Limiting**

\`\`\`typescript
interface RateLimitKey {
  userId: string;
  tier: string;
  apiKey: string;
}

class TieredRateLimiter {
  private redis: Redis;
  
  async checkRateLimit(key: RateLimitKey): Promise<RateLimitResult> {
    const tier = tiers[key.tier];
    
    // Check daily limit
    const dailyUsage = await this.getDailyUsage(key);
    
    // Check burst limit (per minute)
    const burstUsage = await this.getBurstUsage(key);
    
    // Evaluate limits
    if (dailyUsage >= tier.dailyLimit) {
      // Over daily limit - check overage
      if (!tier.overage.allowed) {
        return {
          allowed: false,
          reason: 'daily_limit_exceeded',
          retryAfter: this.secondsUntilMidnight(),
          usage: {
            daily: dailyUsage,
            dailyLimit: tier.dailyLimit,
            burst: burstUsage,
            burstLimit: tier.burstLimit
          }
        };
      }
      
      // Overage allowed - check max overage
      const overage = dailyUsage - tier.dailyLimit;
      if (overage >= tier.overage.maxOverage) {
        return {
          allowed: false,
          reason: 'max_overage_exceeded',
          retryAfter: this.secondsUntilMidnight(),
          overageCost: overage * tier.overage.costPerRequest
        };
      }
      
      // Allow with overage charge
      await this.incrementUsage(key);
      await this.recordOverageCharge(key, tier.overage.costPerRequest);
      
      return {
        allowed: true,
        isOverage: true,
        overageCost: (overage + 1) * tier.overage.costPerRequest,
        usage: { daily: dailyUsage + 1, dailyLimit: tier.dailyLimit }
      };
    }
    
    // Check burst limit
    if (burstUsage >= tier.burstLimit) {
      return {
        allowed: false,
        reason: 'burst_limit_exceeded',
        retryAfter: 60, // 1 minute
        usage: { burst: burstUsage, burstLimit: tier.burstLimit }
      };
    }
    
    // Within limits
    await this.incrementUsage(key);
    
    return {
      allowed: true,
      isOverage: false,
      usage: {
        daily: dailyUsage + 1,
        dailyLimit: tier.dailyLimit,
        burst: burstUsage + 1,
        burstLimit: tier.burstLimit
      }
    };
  }
  
  private async getDailyUsage(key: RateLimitKey): Promise<number> {
    const dailyKey = \`rate_limit:daily:\${key.userId}:\${this.getToday()}\`;
    const count = await this.redis.get(dailyKey);
    return parseInt(count || '0');
  }
  
  private async getBurstUsage(key: RateLimitKey): Promise<number> {
    const burstKey = \`rate_limit:burst:\${key.userId}:\${this.getCurrentMinute()}\`;
    const count = await this.redis.get(burstKey);
    return parseInt(count || '0');
  }
  
  private async incrementUsage(key: RateLimitKey): Promise<void> {
    const dailyKey = \`rate_limit:daily:\${key.userId}:\${this.getToday()}\`;
    const burstKey = \`rate_limit:burst:\${key.userId}:\${this.getCurrentMinute()}\`;
    
    await this.redis
      .multi()
      .incr(dailyKey)
      .expire(dailyKey, 86400) // 24 hours
      .incr(burstKey)
      .expire(burstKey, 60) // 1 minute
      .exec();
  }
  
  private async recordOverageCharge(key: RateLimitKey, cost: number): Promise<void> {
    const chargeKey = \`overage_charges:\${key.userId}:\${this.getMonth()}\`;
    await this.redis.incrbyfloat(chargeKey, cost);
    
    // Add to billing queue
    await this.queueBillingEvent({
      userId: key.userId,
      amount: cost,
      description: 'API overage charge',
      timestamp: Date.now()
    });
  }
}
\`\`\`

**3. Trial Period Handling**

\`\`\`typescript
interface TrialConfig {
  duration: number; // days
  limits: RateLimitTier;
  upgradePrompt: boolean;
}

async function checkTrial(userId: string): Promise<TrialStatus> {
  const trial = await db.trials.findOne({ userId });
  
  if (!trial) {
    // No active trial
    return { active: false };
  }
  
  const now = Date.now();
  const endTime = trial.startTime + (trial.duration * 86400000);
  
  if (now > endTime) {
    // Trial expired
    await db.users.update({ userId }, { tier: 'free' });
    return {
      active: false,
      expired: true,
      message: 'Your trial has ended. Upgrade to continue with higher limits.'
    };
  }
  
  const daysRemaining = Math.ceil((endTime - now) / 86400000);
  
  // Prompt upgrade at 80% and 100% of usage
  const usage = await getDailyUsage({ userId, tier: 'trial', apiKey: '' });
  const usagePercent = (usage / trial.limits.dailyLimit) * 100;
  
  if (usagePercent >= 80 && trial.upgradePrompt) {
    return {
      active: true,
      daysRemaining,
      showUpgradePrompt: true,
      message: \`You've used \${usagePercent.toFixed(0)}% of your trial limits. Upgrade to Pro for 100x more requests.\`
    };
  }
  
  return {
    active: true,
    daysRemaining,
    showUpgradePrompt: false
  };
}
\`\`\`

**4. Overage Billing Integration**

\`\`\`typescript
// Monthly billing job
async function processMonthlyOverages() {
  const month = getCurrentMonth();
  
  // Get all users with overage charges
  const keys = await redis.keys(\`overage_charges:*:\${month}\`);
  
  for (const key of keys) {
    const userId = key.split(':')[1];
    const totalOverageCharge = parseFloat(await redis.get(key) || '0');
    
    if (totalOverageCharge > 0) {
      // Charge via Stripe
      const user = await db.users.findOne({ userId });
      
      try {
        await stripe.invoiceItems.create({
          customer: user.stripeCustomerId,
          amount: Math.round(totalOverageCharge * 100), // cents
          currency: 'usd',
          description: \`API overage charges for \${month}\`
        });
        
        // Send notification
        await sendEmail({
          to: user.email,
          subject: 'API Overage Invoice',
          body: \`You used \${totalOverageCharge.toFixed(2)} USD in API overages this month.\`
        });
        
        // Clear overage counter
        await redis.del(key);
        
      } catch (error) {
        // Log billing failure
        logger.error(\`Billing failed for user \${userId}\`, error);
        
        // Retry later
        await redis.lpush('billing_retry_queue', JSON.stringify({
          userId,
          amount: totalOverageCharge,
          month
        }));
      }
    }
  }
}
\`\`\`

**5. User Experience**

**Headers for Transparency**:

\`\`\`typescript
app.use((req, res, next) => {
  const result = await rateLimiter.checkRateLimit(req.user);
  
  // Set headers
  res.setHeader('X-RateLimit-Limit', result.usage.dailyLimit);
  res.setHeader('X-RateLimit-Remaining', 
    Math.max(0, result.usage.dailyLimit - result.usage.daily));
  res.setHeader('X-RateLimit-Reset', getMidnightTimestamp());
  
  if (result.isOverage) {
    res.setHeader('X-RateLimit-Overage', 'true');
    res.setHeader('X-RateLimit-Overage-Cost', result.overageCost.toFixed(2));
  }
  
  if (!result.allowed) {
    res.setHeader('Retry-After', result.retryAfter);
    return res.status(429).json({
      error: 'Rate limit exceeded',
      reason: result.reason,
      retryAfter: result.retryAfter,
      upgradeUrl: 'https://example.com/upgrade'
    });
  }
  
  next();
});
\`\`\`

**Dashboard**:

\`\`\`typescript
// Real-time usage dashboard
app.get('/api/usage', async (req, res) => {
  const userId = req.user.id;
  const tier = req.user.tier;
  
  const daily = await getDailyUsage({ userId, tier, apiKey: '' });
  const limits = tiers[tier];
  
  const overageCharges = await redis.get(\`overage_charges:\${userId}:\${getMonth()}\`);
  
  res.json({
    tier: tier,
    usage: {
      daily: daily,
      dailyLimit: limits.dailyLimit,
      percentUsed: (daily / limits.dailyLimit) * 100,
      remaining: Math.max(0, limits.dailyLimit - daily)
    },
    burst: {
      limit: limits.burstLimit,
      current: await getBurstUsage({ userId, tier, apiKey: '' })
    },
    overage: {
      allowed: limits.overage.allowed,
      current: Math.max(0, daily - limits.dailyLimit),
      cost: parseFloat(overageCharges || '0').toFixed(2)
    },
    resetTime: getMidnightTimestamp()
  });
});
\`\`\`

**6. Edge Cases**

**Timezone Handling**:
\`\`\`typescript
// Use UTC for daily resets to avoid confusion
function getToday(): string {
  return new Date().toISOString().split('T')[0]; // YYYY-MM-DD in UTC
}
\`\`\`

**Tier Upgrades Mid-Day**:
\`\`\`typescript
// When user upgrades, don't reset counter - just increase limit
async function handleTierUpgrade(userId: string, newTier: string) {
  await db.users.update({ userId }, { tier: newTier });
  
  // Usage counter persists - user immediately gets higher limit
  // No need to reset daily counter
  
  // Send confirmation
  await sendEmail({
    to: user.email,
    subject: 'Tier Upgraded',
    body: \`You now have \${tiers[newTier].dailyLimit} requests per day.\`
  });
}
\`\`\`

**Billing Failures**:
\`\`\`typescript
// If billing fails, don't immediately block user
// Grace period: 7 days
async function checkBillingGrace(userId: string): Promise<boolean> {
  const failedBillings = await db.billingFailures.find({
    userId,
    resolved: false
  });
  
  if (failedBillings.length === 0) {
    return true; // No issues
  }
  
  const oldestFailure = failedBillings[0].timestamp;
  const daysSinceFailure = (Date.now() - oldestFailure) / 86400000;
  
  if (daysSinceFailure > 7) {
    // Grace period expired - downgrade to free tier
    await db.users.update({ userId }, { tier: 'free' });
    
    await sendEmail({
      to: user.email,
      subject: 'Account Downgraded',
      body: 'Your payment failed. Update payment info to restore access.'
    });
    
    return false;
  }
  
  return true; // Still in grace period
}
\`\`\`

**Key Takeaways**:
1. **Multi-dimensional limits**: daily, burst, overage
2. **Transparent pricing**: Show usage and costs in headers and dashboard
3. **Graceful overage**: Allow Pro users to exceed limits with per-request charges
4. **Trial management**: Auto-downgrade after trial, prompt upgrades
5. **Billing integration**: Monthly overage invoices via Stripe
6. **UX-first**: Clear error messages with upgrade paths
7. **Edge cases**: Handle timezones (UTC), mid-day upgrades, billing failures with grace periods`,
    keyPoints: [
      'Multi-dimensional rate limiting: daily limits, burst limits, and overage allowances',
      'Tiered pricing: Free (hard limit), Pro (overage allowed), Enterprise (unlimited)',
      'Transparent UX: Show usage in headers (X-RateLimit-*) and real-time dashboard',
      'Overage billing: Track per-request charges, invoice monthly via Stripe',
      'Trial management: Auto-downgrade after expiration, prompt upgrades at 80% usage',
      'Edge cases: UTC for resets, preserve usage on tier upgrades, 7-day billing grace period',
    ],
  },
  {
    id: 'rate-limit-disc-3',
    question:
      'Compare Token Bucket, Leaky Bucket, Fixed Window Counter, and Sliding Window algorithms for rate limiting. For each algorithm, explain the implementation, pros/cons, memory requirements, and ideal use cases. Which would you choose for a high-traffic REST API and why?',
    sampleAnswer: `**Comprehensive Comparison of Rate Limiting Algorithms**

---

## **1. Token Bucket**

**How it Works**:
- Bucket holds tokens (max capacity = bucket size)
- Tokens added at constant rate (refill rate)
- Each request consumes 1 token
- If bucket empty, request denied

**Pseudocode**:
\`\`\`python
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity  # max tokens
        self.tokens = capacity    # current tokens
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.now()
    
    def allow_request(self):
        # Refill tokens based on time elapsed
        now = time.now()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
        
        # Check if we have tokens
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
\`\`\`

**Redis Implementation**:
\`\`\`lua
-- token_bucket.lua
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

-- Get current state
local state = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(state[1]) or capacity
local last_refill = tonumber(state[2]) or now

-- Refill tokens
local elapsed = now - last_refill
local tokens_to_add = elapsed * refill_rate
tokens = math.min(capacity, tokens + tokens_to_add)

-- Check if we can allow request
if tokens >= requested then
    tokens = tokens - requested
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    return 1  -- allowed
else
    return 0  -- denied
end
\`\`\`

**Pros**:
✅ Allows burst traffic (accumulate tokens when idle)
✅ Simple to understand and implement
✅ Constant memory (just stores token count)
✅ Works well for APIs with variable traffic

**Cons**:
❌ Can't prevent sustained bursts if bucket is large
❌ Requires timestamps (clock drift in distributed systems)

**Memory**: O(1) - stores 2 values (tokens, last_refill)

**Ideal Use Cases**:
- REST APIs with variable traffic
- Allow legitimate bursts (user loads page → multiple API calls)
- Most common choice for public APIs

---

## **2. Leaky Bucket**

**How it Works**:
- Requests enter bucket (queue)
- Requests "leak" out at constant rate
- If bucket full, new requests dropped

**Pseudocode**:
\`\`\`python
class LeakyBucket:
    def __init__(self, capacity, leak_rate):
        self.capacity = capacity
        self.queue = []
        self.leak_rate = leak_rate  # requests per second
        self.last_leak = time.now()
    
    def allow_request(self):
        # Leak requests
        now = time.now()
        elapsed = now - self.last_leak
        requests_to_leak = int(elapsed * self.leak_rate)
        
        for _ in range(min(requests_to_leak, len(self.queue))):
            self.queue.pop(0)
        
        self.last_leak = now
        
        # Add new request
        if len(self.queue) < self.capacity:
            self.queue.append(now)
            return True
        return False
\`\`\`

**Pros**:
✅ Smooths out traffic (enforces constant rate)
✅ Prevents bursts (good for protecting downstream)
✅ Fair queuing

**Cons**:
❌ No bursts allowed (bad UX for legitimate spikes)
❌ Requires queue (memory intensive)
❌ Complexity in distributed systems (shared queue)

**Memory**: O(n) - stores queue of requests

**Ideal Use Cases**:
- Protecting downstream services that can't handle bursts
- Network traffic shaping
- Message queue rate limiting

---

## **3. Fixed Window Counter**

**How it Works**:
- Divide time into fixed windows (e.g., 1-minute windows)
- Count requests in current window
- Reset counter at window boundary

**Pseudocode**:
\`\`\`python
class FixedWindowCounter:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size  # seconds
        self.counters = {}  # {window_id: count}
    
    def allow_request(self):
        now = time.now()
        window_id = int(now / self.window_size)
        
        count = self.counters.get(window_id, 0)
        
        if count < self.limit:
            self.counters[window_id] = count + 1
            return True
        return False
\`\`\`

**Redis Implementation**:
\`\`\`lua
-- fixed_window.lua
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])  -- window size in seconds

local current = redis.call('INCR', key)

if current == 1 then
    redis.call('EXPIRE', key, window)
end

if current <= limit then
    return 1  -- allowed
else
    return 0  -- denied
end
\`\`\`

**Pros**:
✅ Extremely simple to implement
✅ Very memory efficient (single counter)
✅ Fast (just increment)

**Cons**:
❌ **Boundary problem**: 2x limit possible at window boundary
  - Example: 100 req/min limit
  - User sends 100 requests at 12:00:59
  - User sends 100 requests at 12:01:00
  - Result: 200 requests in 1 second!
❌ Unfair (early requests in window have advantage)

**Memory**: O(1) - single counter per window

**Ideal Use Cases**:
- Simple rate limiting where boundary problem is acceptable
- Low-traffic APIs
- Internal APIs (not user-facing)

---

## **4. Sliding Window Log**

**How it Works**:
- Store timestamp of each request
- Count requests in last N seconds (sliding window)
- Remove old timestamps

**Pseudocode**:
\`\`\`python
class SlidingWindowLog:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.log = []  # list of timestamps
    
    def allow_request(self):
        now = time.now()
        cutoff = now - self.window_size
        
        # Remove old timestamps
        self.log = [ts for ts in self.log if ts > cutoff]
        
        if len(self.log) < self.limit:
            self.log.append(now)
            return True
        return False
\`\`\`

**Redis Implementation**:
\`\`\`lua
-- sliding_window_log.lua
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

-- Remove old entries
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- Count current entries
local count = redis.call('ZCARD', key)

if count < limit then
    -- Add new entry
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window)
    return 1  -- allowed
else
    return 0  -- denied
end
\`\`\`

**Pros**:
✅ Most accurate (no boundary problem)
✅ True sliding window
✅ Fair

**Cons**:
❌ Memory intensive (stores all timestamps)
❌ Slower (need to clean up old entries)
❌ O(n) operations

**Memory**: O(n) - stores timestamp for each request in window

**Ideal Use Cases**:
- When accuracy is critical
- Low request rates
- Compliance/auditing requirements

---

## **5. Sliding Window Counter** (Hybrid)

**How it Works**:
- Combines Fixed Window + Sliding Window
- Uses weighted average of current and previous window

**Formula**:
\`\`\`
count = previous_window_count * (1 - position_in_current_window) + current_window_count
\`\`\`

**Pseudocode**:
\`\`\`python
class SlidingWindowCounter:
    def __init__(self, limit, window_size):
        self.limit = limit
        self.window_size = window_size
        self.windows = {}  # {window_id: count}
    
    def allow_request(self):
        now = time.now()
        window_id = int(now / self.window_size)
        previous_window_id = window_id - 1
        
        # Calculate position in current window (0.0 to 1.0)
        position = (now % self.window_size) / self.window_size
        
        previous_count = self.windows.get(previous_window_id, 0)
        current_count = self.windows.get(window_id, 0)
        
        # Weighted count
        estimated_count = previous_count * (1 - position) + current_count
        
        if estimated_count < self.limit:
            self.windows[window_id] = current_count + 1
            return True
        return False
\`\`\`

**Pros**:
✅ Solves boundary problem
✅ Memory efficient (only 2 counters)
✅ Fast (just math)
✅ Good approximation of sliding window

**Cons**:
❌ Still an approximation (not exact)
❌ Slightly more complex than fixed window

**Memory**: O(1) - stores 2 counters

**Ideal Use Cases**:
- High-traffic APIs
- Need accuracy without memory overhead
- Best balance of accuracy and efficiency

---

## **Comparison Table**

| Algorithm | Accuracy | Memory | Speed | Bursts | Complexity |
|-----------|----------|--------|-------|--------|------------|
| **Token Bucket** | Good | O(1) | Fast | ✅ Yes | Low |
| **Leaky Bucket** | Excellent | O(n) | Slow | ❌ No | High |
| **Fixed Window** | Poor (boundary) | O(1) | Fastest | ✅ Yes | Lowest |
| **Sliding Log** | Perfect | O(n) | Slowest | ✅ Yes | Medium |
| **Sliding Counter** | Very Good | O(1) | Fast | ✅ Yes | Medium |

---

## **Recommendation for High-Traffic REST API**

**Choose: Token Bucket or Sliding Window Counter**

**Why Token Bucket**:
1. **Allows bursts** - users can accumulate tokens during idle periods
2. **Memory efficient** - O(1) memory per user
3. **Simple** - easy to implement and debug
4. **Industry standard** - AWS API Gateway, Stripe, GitHub all use it
5. **Good UX** - doesn't penalize legitimate burst patterns

**Example**: User loads dashboard → 5 API calls simultaneously
- Token Bucket: ✅ All 5 succeed (had 100 tokens saved up)
- Leaky Bucket: ❌ 4 of 5 queued/rejected
- Fixed Window: ✅ Depends on boundary
- Sliding Log: ✅ All succeed if under limit

**Why Sliding Window Counter as Alternative**:
- If boundary problem is critical
- Need more accurate limiting
- Still memory efficient (O(1))

**Implementation Choice**:
\`\`\`typescript
// Production: Use Token Bucket with Redis
const rateLimiter = new TokenBucketRateLimiter({
  capacity: 100,      // 100 requests
  refillRate: 10,     // 10 tokens per second = 600/minute
  redis: redisClient
});
\`\`\`

**Why NOT the others for high-traffic REST API**:
- **Leaky Bucket**: No bursts = bad UX
- **Fixed Window**: Boundary problem = potential abuse
- **Sliding Log**: O(n) memory = too expensive at scale`,
    keyPoints: [
      'Token Bucket: Allows bursts, O(1) memory, industry standard for REST APIs',
      'Leaky Bucket: Smooths traffic, no bursts, O(n) memory, good for downstream protection',
      'Fixed Window: Simple but has boundary problem (2x rate possible at window edge)',
      'Sliding Window Log: Most accurate but O(n) memory (stores all timestamps)',
      'Sliding Window Counter: Best hybrid - solves boundary problem with O(1) memory',
      'For high-traffic REST API: Choose Token Bucket (allows legitimate bursts, efficient)',
    ],
  },
];
