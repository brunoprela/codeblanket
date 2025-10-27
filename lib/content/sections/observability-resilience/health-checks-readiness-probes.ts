/**
 * Health Checks & Readiness Probes Section
 */

export const healthChecksSection = {
  id: 'health-checks-readiness-probes',
  title: 'Health Checks & Readiness Probes',
  content: `Health checks and readiness probes are the foundation of self-healing systems. They tell load balancers, orchestrators, and monitoring systems whether a service instance is healthy and ready to serve traffic. Without proper health checks, failing instances continue receiving requests, dead containers stay in rotation, and cascading failures occur. This section covers how to implement robust health checking.

## What are Health Checks?

**Health Check**: An endpoint or mechanism that reports whether a service instance is functioning correctly

**Purpose**:
- Route traffic only to healthy instances
- Remove failing instances from load balancer
- Trigger auto-healing (restart containers)
- Monitor system health

**Basic Example**:
\`\`\`
GET /health
→ 200 OK: Instance healthy
→ 503 Service Unavailable: Instance unhealthy
\`\`\`

---

## Types of Health Checks

### **1. Liveness Probe**

**Question**: Is the application running?

**Purpose**: Detect deadlocked or crashed applications

**If Fails**: Restart the container/instance

**Example Failures**:
- Application deadlocked (infinite loop)
- Out of memory crash
- Unrecoverable error state

**Implementation**:
\`\`\`javascript
app.get('/healthz/live', (req, res) => {
  // Simple check: Can the app respond?
  res.status(200).send('OK');
});
\`\`\`

**Kubernetes**:
\`\`\`yaml
livenessProbe:
  httpGet:
    path: /healthz/live
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
\`\`\`

### **2. Readiness Probe**

**Question**: Is the application ready to serve traffic?

**Purpose**: Determine if instance should receive requests

**If Fails**: Remove from load balancer (don't restart)

**Example Failures**:
- Database connection not established
- Cache warming in progress
- Dependency service unavailable
- Loading large dataset

**Implementation**:
\`\`\`javascript
app.get('/healthz/ready', async (req, res) => {
  try {
    // Check dependencies
    await database.ping();
    await redis.ping();
    
    // Check critical resources
    if (cacheWarmedUp && databaseConnected) {
      res.status(200).send('Ready');
    } else {
      res.status(503).send('Not Ready');
    }
  } catch (error) {
    res.status(503).send('Not Ready');
  }
});
\`\`\`

**Kubernetes**:
\`\`\`yaml
readinessProbe:
  httpGet:
    path: /healthz/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
\`\`\`

### **3. Startup Probe**

**Question**: Has the application finished starting up?

**Purpose**: Give slow-starting applications more time

**If Fails**: Restart the container (after long timeout)

**Use Case**: 
- Applications that take 60+ seconds to start
- Large data loading on startup
- JVM warm-up

**Implementation**:
\`\`\`javascript
let startupComplete = false;

app.get('/healthz/startup', (req, res) => {
  if (startupComplete) {
    res.status(200).send('Started');
  } else {
    res.status(503).send('Starting...');
  }
});

// After initialization
async function initialize() {
  await loadData();
  await warmCache();
  startupComplete = true;
}
\`\`\`

**Kubernetes**:
\`\`\`yaml
startupProbe:
  httpGet:
    path: /healthz/startup
    port: 8080
  initialDelaySeconds: 0
  periodSeconds: 10
  timeoutSeconds: 3
  failureThreshold: 30  # 30 * 10s = 5 minutes max startup
\`\`\`

---

## Liveness vs Readiness vs Startup

### **Comparison**

| Probe | Question | On Failure | When to Use |
|-------|----------|------------|-------------|
| **Liveness** | Is app running? | Restart | Detect deadlock/crash |
| **Readiness** | Can serve traffic? | Remove from LB | Check dependencies |
| **Startup** | Finished starting? | Restart (after timeout) | Slow startup apps |

### **Flow Diagram**

\`\`\`
Pod Starts
   ↓
[Startup Probe]
   │
   ├─ Fail → Wait → Retry → (30 failures) → Restart Pod
   │
   ↓ Pass
[Readiness Probe]
   │
   ├─ Fail → Don't route traffic
   │
   ↓ Pass → Route traffic
   ↓
[Liveness Probe]
   │
   ├─ Fail → Restart Pod
   │
   ↓ Pass → Keep running
\`\`\`

### **Example Scenario**

**Application**: API server that loads 1GB dataset on startup

**Probes**:
\`\`\`yaml
# Startup: Allow 10 minutes for data loading
startupProbe:
  httpGet:
    path: /healthz/startup
    port: 8080
  periodSeconds: 10
  failureThreshold: 60  # 60 * 10s = 10 minutes

# Readiness: Check dependencies
readinessProbe:
  httpGet:
    path: /healthz/ready
    port: 8080
  periodSeconds: 5
  failureThreshold: 3

# Liveness: Basic health
livenessProbe:
  httpGet:
    path: /healthz/live
    port: 8080
  periodSeconds: 10
  failureThreshold: 3
\`\`\`

---

## What to Check

### **Liveness Probe: Minimal Checks**

✅ **Do Check**:
- Application can respond (basic HTTP response)
- No deadlock (response within timeout)

❌ **Don't Check**:
- Database connectivity (use readiness)
- External dependencies (use readiness)
- Disk space (use readiness)

**Why**: Liveness failure triggers restart. Don't restart for dependency issues.

**Example**:
\`\`\`javascript
// ✅ Good liveness check
app.get('/healthz/live', (req, res) => {
  res.status(200).send('OK');
});

// ❌ Bad liveness check
app.get('/healthz/live', async (req, res) => {
  try {
    await database.ping(); // Don't check DB in liveness!
    res.status(200).send('OK');
  } catch (error) {
    res.status(503).send('Unhealthy');
  }
});
\`\`\`

### **Readiness Probe: Comprehensive Checks**

✅ **Do Check**:
- Database connectivity
- Cache connectivity
- Critical dependencies
- Required configuration loaded
- Sufficient resources (memory, disk)

❌ **Don't Check**:
- Non-critical dependencies
- External APIs (unless critical)
- Time-consuming operations

**Example**:
\`\`\`javascript
app.get('/healthz/ready', async (req, res) => {
  const checks = {
    database: false,
    redis: false,
    config: false
  };

  try {
    // Check database
    await database.ping();
    checks.database = true;

    // Check Redis
    await redis.ping();
    checks.redis = true;

    // Check configuration
    checks.config = config.loaded;

    // All critical checks passed
    if (checks.database && checks.redis && checks.config) {
      res.status(200).json({
        status: 'ready',
        checks
      });
    } else {
      res.status(503).json({
        status: 'not ready',
        checks
      });
    }
  } catch (error) {
    res.status(503).json({
      status: 'not ready',
      checks,
      error: error.message
    });
  }
});
\`\`\`

---

## Health Check Configuration

### **Timeouts**

**initialDelaySeconds**: Wait before first check
\`\`\`yaml
initialDelaySeconds: 30  # Give app 30s to start
\`\`\`

**periodSeconds**: Interval between checks
\`\`\`yaml
periodSeconds: 10  # Check every 10 seconds
\`\`\`

**timeoutSeconds**: How long to wait for response
\`\`\`yaml
timeoutSeconds: 5  # Response within 5 seconds
\`\`\`

**failureThreshold**: Consecutive failures before action
\`\`\`yaml
failureThreshold: 3  # 3 failures = unhealthy
\`\`\`

**successThreshold**: Consecutive successes to recover
\`\`\`yaml
successThreshold: 1  # 1 success = healthy again
\`\`\`

### **Tuning Guidelines**

**Liveness**:
\`\`\`yaml
initialDelaySeconds: 30      # Application startup time
periodSeconds: 10            # Not too frequent (overhead)
timeoutSeconds: 5            # Generous (avoid false positives)
failureThreshold: 3          # Multiple checks before restart
\`\`\`

**Readiness**:
\`\`\`yaml
initialDelaySeconds: 5       # Start checking quickly
periodSeconds: 5             # More frequent (routing decision)
timeoutSeconds: 3            # Response should be fast
failureThreshold: 3          # A few failures to avoid flapping
\`\`\`

**Startup**:
\`\`\`yaml
initialDelaySeconds: 0       # Start immediately
periodSeconds: 10            # Don't overwhelm starting app
timeoutSeconds: 3
failureThreshold: 30         # Long timeout (e.g., 5 minutes)
\`\`\`

---

## Health Check Patterns

### **1. Shallow Health Check**

Quick, lightweight check

\`\`\`javascript
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'UP' });
});
\`\`\`

**Use For**: Liveness probe
**Response Time**: < 10ms

### **2. Deep Health Check**

Comprehensive dependency checking

\`\`\`javascript
app.get('/health/deep', async (req, res) => {
  const checks = {
    database: await checkDatabase(),
    redis: await checkRedis(),
    s3: await checkS3(),
    externalAPI: await checkExternalAPI()
  };

  const healthy = Object.values (checks).every (c => c.healthy);

  res.status (healthy ? 200 : 503).json({
    status: healthy ? 'UP' : 'DOWN',
    checks
  });
});
\`\`\`

**Use For**: Readiness probe, manual diagnostics
**Response Time**: < 1s

### **3. Detailed Health Check**

Includes metrics and diagnostics

\`\`\`javascript
app.get('/health/detailed', async (req, res) => {
  const health = {
    status: 'UP',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: process.env.APP_VERSION,
    checks: {
      database: {
        status: 'UP',
        responseTime: '15ms',
        connections: 45
      },
      redis: {
        status: 'UP',
        responseTime: '3ms',
        memoryUsage: '1.2GB'
      },
      disk: {
        status: 'UP',
        usage: '65%'
      }
    },
    metrics: {
      requestsPerSecond: 150,
      errorRate: 0.002,
      p99Latency: '120ms'
    }
  };

  res.status(200).json (health);
});
\`\`\`

**Use For**: Monitoring, debugging
**Response Time**: < 2s

---

## Load Balancer Health Checks

### **AWS Application Load Balancer (ALB)**

**Configuration**:
\`\`\`json
{
  "HealthCheckPath": "/health",
  "HealthCheckIntervalSeconds": 30,
  "HealthCheckTimeoutSeconds": 5,
  "HealthyThresholdCount": 2,
  "UnhealthyThresholdCount": 2,
  "Matcher": {
    "HttpCode": "200"
  }
}
\`\`\`

**Behavior**:
- Check every 30 seconds
- 2 consecutive successes → Healthy
- 2 consecutive failures → Unhealthy
- Unhealthy targets don't receive traffic

### **NGINX**

**Configuration**:
\`\`\`nginx
upstream backend {
  server backend1:8080 max_fails=3 fail_timeout=30s;
  server backend2:8080 max_fails=3 fail_timeout=30s;
}

server {
  location / {
    proxy_pass http://backend;
    proxy_next_upstream error timeout http_503;
  }
}
\`\`\`

### **HAProxy**

**Configuration**:
\`\`\`
backend api_servers
  option httpchk GET /health
  http-check expect status 200
  server api1 10.0.1.10:8080 check inter 5s fall 3 rise 2
  server api2 10.0.1.11:8080 check inter 5s fall 3 rise 2
\`\`\`

---

## Common Pitfalls

### **1. Too Aggressive Health Checks**

❌ **Problem**:
\`\`\`yaml
periodSeconds: 1  # Every 1 second
timeoutSeconds: 10
\`\`\`

**Issue**: Overhead, false positives

✅ **Solution**:
\`\`\`yaml
periodSeconds: 10  # Every 10 seconds
timeoutSeconds: 5
\`\`\`

### **2. Health Check Dependencies**

❌ **Problem**:
\`\`\`javascript
// Health check calls another service
app.get('/health', async (req, res) => {
  const response = await fetch('http://other-service/health');
  res.status (response.ok ? 200 : 503).send('OK');
});
\`\`\`

**Issue**: Cascading failures, circular dependencies

✅ **Solution**:
\`\`\`javascript
// Health check only checks local state
app.get('/health', (req, res) => {
  const healthy = localStateOK();
  res.status (healthy ? 200 : 503).send('OK');
});
\`\`\`

### **3. Expensive Health Checks**

❌ **Problem**:
\`\`\`javascript
app.get('/health', async (req, res) => {
  // Expensive query
  const count = await db.query('SELECT COUNT(*) FROM users');
  res.status(200).send('OK');
});
\`\`\`

**Issue**: High database load

✅ **Solution**:
\`\`\`javascript
app.get('/health', async (req, res) => {
  // Lightweight ping
  await db.ping();
  res.status(200).send('OK');
});
\`\`\`

### **4. No Health Check**

❌ **Problem**: No health check endpoint

**Issue**: Load balancer can't detect failures

✅ **Solution**: Always implement health checks

### **5. Health Check Flapping**

❌ **Problem**:
\`\`\`
10:00 → Healthy
10:01 → Unhealthy
10:02 → Healthy
10:03 → Unhealthy
\`\`\`

**Issue**: Instance constantly added/removed from pool

✅ **Solution**:
\`\`\`yaml
failureThreshold: 3  # Require multiple failures
successThreshold: 2  # Require multiple successes
\`\`\`

---

## Advanced Patterns

### **Graceful Shutdown**

Handle SIGTERM gracefully:

\`\`\`javascript
let isShuttingDown = false;

// Readiness check
app.get('/healthz/ready', (req, res) => {
  if (isShuttingDown) {
    res.status(503).send('Shutting down');
  } else {
    res.status(200).send('Ready');
  }
});

// Handle shutdown signal
process.on('SIGTERM', () => {
  console.log('SIGTERM received, starting graceful shutdown');
  
  // Mark as not ready (stop receiving new requests)
  isShuttingDown = true;
  
  // Wait for existing requests to complete
  setTimeout(() => {
    console.log('Shutting down now');
    server.close(() => {
      process.exit(0);
    });
  }, 30000); // 30 second grace period
});
\`\`\`

### **Warming Up**

Cache warming before accepting traffic:

\`\`\`javascript
let cacheWarmed = false;

app.get('/healthz/ready', (req, res) => {
  if (cacheWarmed) {
    res.status(200).send('Ready');
  } else {
    res.status(503).send('Warming up cache');
  }
});

// Warm cache on startup
async function warmCache() {
  console.log('Warming cache...');
  await loadFrequentlyAccessedData();
  await precomputeExpensiveOperations();
  cacheWarmed = true;
  console.log('Cache warmed, ready for traffic');
}

warmCache();
\`\`\`

### **Health Check Caching**

Cache expensive checks:

\`\`\`javascript
let cachedHealthStatus = null;
let lastCheck = 0;
const CACHE_TTL = 5000; // 5 seconds

app.get('/healthz/ready', async (req, res) => {
  const now = Date.now();
  
  // Use cached result if fresh
  if (cachedHealthStatus && (now - lastCheck) < CACHE_TTL) {
    return res.status (cachedHealthStatus.code)
              .json (cachedHealthStatus.body);
  }
  
  // Perform checks
  const healthy = await performHealthChecks();
  
  // Cache result
  cachedHealthStatus = {
    code: healthy ? 200 : 503,
    body: { status: healthy ? 'UP' : 'DOWN' }
  };
  lastCheck = now;
  
  res.status (cachedHealthStatus.code).json (cachedHealthStatus.body);
});
\`\`\`

---

## Monitoring Health Checks

### **Metrics**

**Health Check Success Rate**:
\`\`\`
health_check_success_total / health_check_total
Target: > 99%
\`\`\`

**Health Check Duration**:
\`\`\`
health_check_duration_seconds
Target: < 100ms
\`\`\`

**Unhealthy Instances**:
\`\`\`
instances_unhealthy / instances_total
Alert if: > 50%
\`\`\`

### **Alerts**

**Multiple Instances Unhealthy**:
\`\`\`
ALERT: 50% of instances failing health checks
Service: api
Unhealthy: 5/10 instances
Action: Investigate common cause
\`\`\`

**Health Check Flapping**:
\`\`\`
WARNING: Instance health flapping
Instance: api-pod-123
Changes: 10 in last 5 minutes
Action: Check for transient issues
\`\`\`

---

## Best Practices

### **Do's**
✅ Implement both liveness and readiness probes
✅ Keep liveness checks simple (avoid dependencies)
✅ Check dependencies in readiness probes
✅ Return proper HTTP status codes (200, 503)
✅ Include version/build info in health response
✅ Monitor health check metrics
✅ Test health check behavior
✅ Implement graceful shutdown

### **Don'ts**
❌ Heavy operations in health checks
❌ Call other services from health checks
❌ Same endpoint for liveness and readiness
❌ Ignore health check failures
❌ Too aggressive timeouts
❌ No health checks at all
❌ Circular health check dependencies

---

## Interview Tips

### **Key Concepts**1. **Liveness**: Is app running? Restart if fails.
2. **Readiness**: Ready for traffic? Remove from LB if fails.
3. **Startup**: Finished starting? Long timeout for slow apps.
4. **Shallow vs Deep**: Quick check vs dependency checks.
5. **Graceful Shutdown**: Mark unready before shutdown.

### **Common Questions**

**Q: What\'s the difference between liveness and readiness?**
A: Liveness checks if app is running (restart if fails). Readiness checks if app is ready for traffic (remove from load balancer if fails). Liveness should be simple, readiness can check dependencies.

**Q: What should you check in a liveness probe?**
A: Minimal checks—just that the app can respond. Don't check dependencies (database, cache) because liveness failure triggers restart, which won't fix dependency issues.

**Q: How do you handle slow-starting applications?**
A: Use startup probe with long failureThreshold (e.g., 30 × 10s = 5 minutes). This prevents liveness probe from killing app during startup.

**Q: What causes health check flapping?**
A: Checks too aggressive, dependency instability, or intermittent issues. Solution: Increase failureThreshold and successThreshold to require multiple consistent results.

---

## Real-World Examples

### **Kubernetes**
- **Three Probes**: Liveness, readiness, startup
- **Best Practice**: All production pods should have probes
- **Lesson**: Probes are critical for self-healing

### **AWS**
- **ELB Health Checks**: Built into all load balancers
- **Auto-Scaling**: Replaces unhealthy instances
- **Lesson**: Health checks enable auto-healing

### **Netflix**
- **Eureka**: Service discovery with health checks
- **Chaos Monkey**: Tests health check behavior
- **Lesson**: Health checks must be robust

---

## Summary

Health checks and readiness probes enable self-healing systems:

1. **Liveness**: Is app running? Simple check, restart if fails.
2. **Readiness**: Ready for traffic? Check dependencies, remove from LB if fails.
3. **Startup**: Finished starting? Long timeout for slow apps.
4. **Configuration**: Tune timeouts, periods, and thresholds carefully.
5. **Best Practice**: Keep liveness simple, readiness comprehensive.
6. **Graceful Shutdown**: Mark unready before shutting down.
7. **Monitoring**: Track health check success rate and duration.

Health checks are the foundation of resilient distributed systems. Without them, failing instances stay in rotation, cascading failures occur, and manual intervention is required. With them, systems self-heal automatically.`,
};
