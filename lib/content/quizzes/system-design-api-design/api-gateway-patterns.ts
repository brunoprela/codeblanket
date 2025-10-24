/**
 * Quiz questions for API Gateway Patterns section
 */

export const apigatewaypatternsQuiz = [
  {
    id: 'gateway-d1',
    question:
      'Design an API gateway for a social media platform handling web, mobile, and third-party developers. Include authentication, rate limiting, and caching strategies.',
    sampleAnswer: `Comprehensive API gateway design for social media platform:

**Architecture Overview**:

\`\`\`
┌─────────────┐
│  Web Client │─────┐
└─────────────┘     │
                    v
┌──────────────┐  ┌──────────────────┐
│ Mobile Client│─>│  API Gateway      │
└──────────────┘  │  (Kong/Express)   │
                  └──────────────────┘
┌──────────────┐         │
│  3rd Party   │─────────┘
│  Developers  │
└──────────────┘

         │
         v
┌─────────────────────────────────┐
│  Backend Microservices          │
│  - User Service                 │
│  - Post Service                 │
│  - Feed Service                 │
│  - Notification Service         │
│  - Analytics Service            │
└─────────────────────────────────┘
\`\`\`

**1. Gateway Implementation (Express.js)**

\`\`\`javascript
const express = require('express');
const jwt = require('jsonwebtoken');
const redis = require('redis');
const rateLimit = require('express-rate-limit');
const CircuitBreaker = require('opossum');

const app = express();
const cache = redis.createClient();

// ============ Authentication ============

// Middleware: JWT validation
async function authenticate(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({ error: 'Authentication required' });
  }
  
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    
    // Add user context for downstream services
    req.headers['x-user-id',] = decoded.userId;
    req.headers['x-user-role',] = decoded.role;
    
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
}

// API Key validation for third-party developers
async function validateApiKey(req, res, next) {
  const apiKey = req.headers['x-api-key',];
  
  if (!apiKey) {
    return res.status(401).json({ error: 'API key required' });
  }
  
  // Check Redis cache first
  const cached = await cache.get(\`apikey:\${apiKey}\`);
  if (cached) {
    req.apiClient = JSON.parse(cached);
    return next();
  }
  
  // Validate with API key service
  const client = await fetch(\`http://auth-service/api-keys/\${apiKey}\`);
  
  if (!client.ok) {
    return res.status(401).json({ error: 'Invalid API key' });
  }
  
  const clientData = await client.json();
  
  // Cache for 1 hour
  await cache.setex(\`apikey:\${apiKey}\`, 3600, JSON.stringify(clientData));
  
  req.apiClient = clientData;
  next();
}

// ============ Rate Limiting ============

// User rate limiting (authenticated)
const userRateLimiter = rateLimit({
  store: new (require('rate-limit-redis'))({
    client: cache
  }),
  windowMs: 15 * 60 * 1000,  // 15 minutes
  max: 1000,                  // 1000 requests
  keyGenerator: (req) => req.user?.id || req.ip,
  handler: (req, res) => {
    res.status(429).json({
      error: 'Rate limit exceeded',
      retryAfter: req.rateLimit.resetTime
    });
  }
});

// Third-party API rate limiting (stricter)
const apiKeyRateLimiter = rateLimit({
  store: new (require('rate-limit-redis'))({
    client: cache
  }),
  windowMs: 60 * 60 * 1000,  // 1 hour
  max: (req) => {
    // Tiered pricing: different limits per plan
    const plan = req.apiClient?.plan || 'free';
    return {
      free: 100,
      basic: 1000,
      premium: 10000
    }[plan];
  },
  keyGenerator: (req) => req.apiClient.id,
  handler: (req, res) => {
    res.status(429).json({
      error: 'API quota exceeded',
      plan: req.apiClient.plan,
      upgradeUrl: 'https://example.com/pricing'
    });
  }
});

// ============ Caching ============

// Cache middleware
async function cacheMiddleware(ttl) {
  return async (req, res, next) => {
    // Only cache GET requests
    if (req.method !== 'GET') return next();
    
    const cacheKey = \`cache:\${req.originalUrl}:\${req.user?.id || 'public'}\`;
    
    // Check cache
    const cached = await cache.get(cacheKey);
    if (cached) {
      return res.json(JSON.parse(cached));
    }
    
    // Override res.json to cache response
    const originalJson = res.json;
    res.json = function(data) {
      cache.setex(cacheKey, ttl, JSON.stringify(data));
      originalJson.call(this, data);
    };
    
    next();
  };
}

// ============ Circuit Breakers ============

const serviceBreakers = {
  user: new CircuitBreaker(fetchUserService, {
    timeout: 3000,
    errorThreshold: 50,
    resetTimeout: 30000
  }),
  post: new CircuitBreaker(fetchPostService, {
    timeout: 3000,
    errorThreshold: 50,
    resetTimeout: 30000
  }),
  feed: new CircuitBreaker(fetchFeedService, {
    timeout: 5000,
    errorThreshold: 50,
    resetTimeout: 30000
  })
};

// Fallbacks
serviceBreakers.user.fallback(() => ({
  id: null,
  username: 'Unknown',
  avatar: '/default-avatar.png'
}));

// ============ Routes ============

// Public routes (no auth)
app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

// User routes (JWT auth)
app.get('/users/:id',
  authenticate,
  userRateLimiter,
  cacheMiddleware(300),  // Cache 5 min
  async (req, res) => {
    try {
      const user = await serviceBreakers.user.fire(req.params.id);
      res.json(user);
    } catch (error) {
      res.status(503).json({ error: 'Service unavailable' });
    }
  }
);

// Feed aggregation (mobile-optimized)
app.get('/feed',
  authenticate,
  userRateLimiter,
  cacheMiddleware(60),  // Cache 1 min
  async (req, res) => {
    try {
      // Parallel requests
      const [posts, trending, suggested] = await Promise.all([
        serviceBreakers.feed.fire(req.user.id),
        fetch('http://trending-service/trending'),
        fetch(\`http://recommendation-service/suggest/\${req.user.id}\`)
      ]);
      
      res.json({
        posts: await posts,
        trending: await trending.json(),
        suggested: await suggested.json()
      });
    } catch (error) {
      res.status(503).json({ error: 'Service unavailable' });
    }
  }
);

// Third-party API routes (API key auth)
app.get('/api/v1/posts',
  validateApiKey,
  apiKeyRateLimiter,
  cacheMiddleware(300),
  async (req, res) => {
    const posts = await fetch('http://post-service/posts', {
      headers: {
        'x-api-client': req.apiClient.id
      }
    });
    
    res.json(await posts.json());
  }
);

// Write operations (no caching)
app.post('/posts',
  authenticate,
  userRateLimiter,
  async (req, res) => {
    const post = await fetch('http://post-service/posts', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-user-id': req.user.id
      },
      body: JSON.stringify(req.body)
    });
    
    // Invalidate feed cache
    await cache.del(\`cache:/feed:\${req.user.id}\`);
    
    res.status(201).json(await post.json());
  }
);

// ============ Monitoring ============

const prometheus = require('prom-client');

const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration',
  labelNames: ['method', 'route', 'status',]
});

app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    httpRequestDuration.labels(
      req.method,
      req.route?.path || 'unknown',
      res.statusCode
    ).observe(duration);
  });
  
  next();
});

app.get('/metrics', async (req, res) => {
  res.set('Content-Type', prometheus.register.contentType);
  res.end(await prometheus.register.metrics());
});

// ============ Error Handling ============

app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({
    error: 'Internal server error',
    requestId: req.headers['x-request-id',]
  });
});

app.listen(80);
\`\`\`

**2. Kong Configuration Alternative**

\`\`\`yaml
services:
  - name: user-service
    url: http://user-service:3000
    routes:
      - name: user-routes
        paths:
          - /users
    plugins:
      - name: jwt
      - name: rate-limiting
        config:
          minute: 1000
      - name: proxy-cache
        config:
          response_code: [200]
          request_method: [GET]
          content_type: [application/json]
          cache_ttl: 300

  - name: post-service
    url: http://post-service:3001
    routes:
      - name: post-routes
        paths:
          - /posts
    plugins:
      - name: jwt
      - name: rate-limiting
        config:
          minute: 100
      - name: response-transformer
        config:
          remove:
            headers: [X-Internal-Token]

  - name: public-api
    url: http://api-service:3002
    routes:
      - name: api-routes
        paths:
          - /api
    plugins:
      - name: key-auth
      - name: rate-limiting
        config:
          hour: 1000
      - name: request-transformer
        config:
          add:
            headers: [X-API-Client:$(headers.x-api-key)]
\`\`\`

**3. High Availability Setup**

\`\`\`
             Load Balancer
                  │
       ┌──────────┼──────────┐
       v          v          v
   Gateway 1  Gateway 2  Gateway 3
   (Active)   (Active)   (Active)
       │          │          │
       └──────────┴──────────┘
                  │
            Redis Cluster
         (Shared State)
\`\`\`

**Key Design Decisions**:

1. **Authentication**: JWT for users, API keys for third parties
2. **Rate Limiting**: Redis-backed, tiered by plan
3. **Caching**: Redis, TTL varies by endpoint (1-5 min)
4. **Circuit Breaking**: Per-service, with fallbacks
5. **Monitoring**: Prometheus metrics, request tracing
6. **High Availability**: Multiple gateway instances, shared Redis

This design balances performance, security, and scalability for a social media platform.`,
    keyPoints: [
      'Separate authentication for users (JWT) vs third parties (API keys)',
      'Tiered rate limiting based on user plan or client type',
      'Aggressive caching for read-heavy operations with appropriate TTLs',
      'Circuit breakers per service with sensible fallbacks',
      'High availability with multiple gateway instances and shared state',
    ],
  },
  {
    id: 'gateway-d2',
    question:
      'Your API gateway is experiencing high latency and becoming a bottleneck. What strategies would you implement to optimize performance?',
    sampleAnswer: `Comprehensive strategies to optimize API gateway performance:

**1. Connection Pooling**

\`\`\`javascript
const http = require('http');
const https = require('https');

// Reuse HTTP connections
const httpAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 100,    // Max concurrent connections
  maxFreeSockets: 10  // Keep 10 idle connections
});

const httpsAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 100
});

// Use agent in fetch calls
async function fetchWithPool(url) {
  return fetch(url, {
    agent: url.startsWith('https') ? httpsAgent : httpAgent
  });
}
\`\`\`

**Impact**: Eliminates TCP/TLS handshake overhead for each request.

**2. Response Streaming**

Don't buffer entire responses in gateway:

\`\`\`javascript
const { pipeline } = require('stream');

app.get('/large-file/:id', (req, res) => {
  // Stream directly from backend to client
  const backendStream = request(\`http://file-service/files/\${req.params.id}\`);
  
  pipeline(
    backendStream,
    res,
    (err) => {
      if (err) console.error('Stream error:', err);
    }
  );
});
\`\`\`

**3. Parallel Request Execution**

\`\`\`javascript
// Bad: Sequential (slow)
const user = await fetch('http://user-service/users/123');
const posts = await fetch('http://post-service/posts?userId=123');
const comments = await fetch('http://comment-service/comments?userId=123');

// Good: Parallel (fast)
const [user, posts, comments] = await Promise.all([
  fetch('http://user-service/users/123'),
  fetch('http://post-service/posts?userId=123'),
  fetch('http://comment-service/comments?userId=123')
]);
\`\`\`

**4. Efficient Caching**

\`\`\`javascript
const redis = require('redis');
const { promisify } = require('util');

const client = redis.createClient();
const getAsync = promisify(client.get).bind(client);
const setexAsync = promisify(client.setex).bind(client);

// Cache with stale-while-revalidate
async function cacheWithSWR(key, fetchFn, ttl) {
  const cached = await getAsync(key);
  
  if (cached) {
    const data = JSON.parse(cached);
    
    // If within 80% of TTL, return cached
    if (data.cachedAt + (ttl * 0.8 * 1000) > Date.now()) {
      return data.value;
    }
    
    // Stale: return cached but refresh in background
    fetchFn().then(fresh => {
      setexAsync(key, ttl, JSON.stringify({
        value: fresh,
        cachedAt: Date.now()
      }));
    });
    
    return data.value;
  }
  
  // Cache miss: fetch and cache
  const fresh = await fetchFn();
  await setexAsync(key, ttl, JSON.stringify({
    value: fresh,
    cachedAt: Date.now()
  }));
  
  return fresh;
}
\`\`\`

**5. Request Coalescing**

Combine duplicate concurrent requests:

\`\`\`javascript
const pendingRequests = new Map();

async function coalescedFetch(url) {
  // Check if request already in flight
  if (pendingRequests.has(url)) {
    return pendingRequests.get(url);
  }
  
  // Make request
  const promise = fetch(url).then(res => res.json());
  pendingRequests.set(url, promise);
  
  // Clean up after response
  promise.finally(() => {
    pendingRequests.delete(url);
  });
  
  return promise;
}
\`\`\`

**6. Compression**

\`\`\`javascript
const compression = require('compression');

app.use(compression({
  level: 6,  // Compression level (1-9)
  threshold: 1024,  // Only compress responses > 1KB
  filter: (req, res) => {
    // Don't compress already compressed formats
    if (req.headers['x-no-compression',]) return false;
    return compression.filter(req, res);
  }
}));
\`\`\`

**7. Load Balancing**

Distribute across multiple backend instances:

\`\`\`javascript
const backends = [
  'http://user-service-1:3000',
  'http://user-service-2:3000',
  'http://user-service-3:3000'
];

let currentIndex = 0;

function roundRobinBackend() {
  const backend = backends[currentIndex];
  currentIndex = (currentIndex + 1) % backends.length;
  return backend;
}

app.get('/users/:id', async (req, res) => {
  const backend = roundRobinBackend();
  const user = await fetch(\`\${backend}/users/\${req.params.id}\`);
  res.json(await user.json());
});
\`\`\`

**8. Timeout Optimization**

Set aggressive timeouts to fail fast:

\`\`\`javascript
const AbortController = require('abort-controller');

async function fetchWithTimeout(url, timeout = 3000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      signal: controller.signal
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}
\`\`\`

**9. Selective Aggregation**

Don't always aggregate everything:

\`\`\`javascript
app.get('/dashboard', async (req, res) => {
  const { include } = req.query;  // ?include=orders,recommendations
  
  const requests = { profile: fetch(\`http://user-service/users/\${req.user.id}\`) };
  
  // Conditionally fetch based on client needs
  if (include?.includes('orders')) {
    requests.orders = fetch(\`http://order-service/orders?userId=\${req.user.id}\`);
  }
  
  if (include?.includes('recommendations')) {
    requests.recommendations = fetch(\`http://recommendation-service/recommend/\${req.user.id}\`);
  }
  
  const results = await Promise.all(Object.values(requests));
  
  res.json({
    profile: await results[0].json(),
    ...(requests.orders && { orders: await results[1].json() }),
    ...(requests.recommendations && { recommendations: await results[2].json() })
  });
});
\`\`\`

**10. Horizontal Scaling**

Deploy multiple gateway instances:

\`\`\`yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 5  # Multiple instances
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: gateway
        image: api-gateway:latest
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        env:
        - name: REDIS_URL
          value: redis://redis-cluster
---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway
spec:
  selector:
    app: api-gateway
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
\`\`\`

**11. Monitoring & Profiling**

Identify slow paths:

\`\`\`javascript
const prometheus = require('prom-client');

const gatewayDuration = new prometheus.Histogram({
  name: 'gateway_request_duration_seconds',
  help: 'Gateway request duration',
  labelNames: ['route', 'backend',],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
});

app.use(async (req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    gatewayDuration.labels(req.route?.path, 'gateway').observe(duration);
    
    // Alert on slow requests
    if (duration > 1) {
      console.warn('Slow request:', {
        route: req.route?.path,
        duration,
        method: req.method
      });
    }
  });
  
  next();
});
\`\`\`

**12. Service Mesh (Advanced)**

Offload traffic management to service mesh:

\`\`\`yaml
# Istio VirtualService for intelligent routing
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: user-service
spec:
  hosts:
  - user-service
  http:
  - match:
    - headers:
        x-user-role:
          exact: premium
    route:
    - destination:
        host: user-service
        subset: premium-pool  # Premium users → faster instances
  - route:
    - destination:
        host: user-service
        subset: standard-pool
\`\`\`

**Performance Impact Summary**:

| Strategy | Latency Reduction | Complexity |
|----------|-------------------|------------|
| Connection Pooling | 20-30% | Low |
| Response Streaming | 40-50% (large responses) | Low |
| Parallel Requests | 50-70% | Low |
| Caching | 80-95% (cache hit) | Medium |
| Request Coalescing | 30-50% (duplicate requests) | Medium |
| Compression | 10-20% | Low |
| Horizontal Scaling | Linear with instances | Low |
| Service Mesh | 10-20% | High |

**Recommended Implementation Order**:

1. Connection pooling (easy win)
2. Caching strategy
3. Parallel requests
4. Compression
5. Horizontal scaling
6. Advanced: Request coalescing, service mesh

This comprehensive approach can reduce gateway latency by 60-80% in typical scenarios.`,
    keyPoints: [
      'Connection pooling eliminates TCP handshake overhead',
      'Parallel request execution reduces aggregation latency',
      'Stale-while-revalidate caching serves cached data while refreshing',
      'Request coalescing combines duplicate concurrent requests',
      'Horizontal scaling distributes load across multiple gateway instances',
    ],
  },
  {
    id: 'gateway-d3',
    question:
      'Compare API Gateway vs Service Mesh for microservices architecture. When would you choose one over the other?',
    sampleAnswer: `Detailed comparison of API Gateway vs Service Mesh:

**API Gateway vs Service Mesh: Core Differences**

\`\`\`
API Gateway (North-South Traffic)
┌──────────┐
│  Client  │
└────┬─────┘
     │
     v
┌────────────────┐
│  API Gateway   │  ← Single entry point
└────────────────┘
     │
     v
┌─────────────────────────┐
│  Microservices          │
│  ┌────┐ ┌────┐ ┌────┐ │
│  │ A  │ │ B  │ │ C  │ │
│  └────┘ └────┘ └────┘ │
└─────────────────────────┘

Service Mesh (East-West Traffic)
┌──────────┐
│  Client  │
└────┬─────┘
     │
     v
┌─────────────────────────┐
│  Microservices          │
│  ┌────┐  ┌────┐  ┌────┐│
│  │ A  │->│ B  │->│ C  ││ ← Mesh handles service-to-service
│  └────┘  └────┘  └────┘│
│  Each service has sidecar proxy (Envoy)
└─────────────────────────┘
\`\`\`

**Comparison Table**:

| Feature | API Gateway | Service Mesh |
|---------|-------------|--------------|
| **Purpose** | External client → services | Service → service |
| **Traffic** | North-South (ingress) | East-West (internal) |
| **Deployment** | Centralized (gateway tier) | Decentralized (sidecar per service) |
| **Routing** | Path-based (/users → user-service) | Service-based (service A → service B) |
| **Auth** | Client authentication (OAuth, API keys) | mTLS between services |
| **Use case** | Public API, third-party integration | Microservices communication |
| **Complexity** | Low-Medium | High |
| **Latency** | +5-20ms | +1-5ms per hop |
| **Examples** | Kong, AWS API Gateway, Traefik | Istio, Linkerd, Consul Connect |

**Scenario 1: E-commerce with External API**

**Use API Gateway when**:
- Exposing APIs to external clients (web, mobile, partners)
- Need centralized authentication (OAuth, API keys)
- Want to aggregate multiple services
- API versioning and transformation needed
- Rate limiting per client

\`\`\`javascript
// API Gateway handles external traffic
External Client → API Gateway → [User Service, Order Service, Product Service]

// Configuration
{
  routes: [
    { path: '/users', target: 'user-service' },
    { path: '/products', target: 'product-service' },
    { path: '/orders', target: 'order-service' }
  ],
  auth: 'JWT',
  rateLimit: 1000,
  caching: true
}
\`\`\`

**Scenario 2: Internal Microservices (100+ services)**

**Use Service Mesh when**:
- Heavy service-to-service communication
- Need mTLS for zero-trust security
- Want automatic retry, circuit breaking, load balancing
- Require distributed tracing across services
- Service discovery and health checking

\`\`\`yaml
# Istio handles internal traffic
Order Service → (Envoy sidecar) → Payment Service
             → (Envoy sidecar) → Inventory Service
             → (Envoy sidecar) → Notification Service

# Istio configuration
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: payment-service
spec:
  hosts:
  - payment-service
  http:
  - route:
    - destination:
        host: payment-service
        subset: v2
      weight: 90
    - destination:
        host: payment-service
        subset: v1
      weight: 10  # Canary deployment
    retries:
      attempts: 3
      perTryTimeout: 2s
    timeout: 10s
\`\`\`

**Scenario 3: Hybrid (Best of Both Worlds)**

Most production systems use **both**:

\`\`\`
External Client
     │
     v
┌────────────────┐
│  API Gateway   │  ← Handles external traffic
└────────────────┘
     │
     v
┌─────────────────────────────────┐
│  Service Mesh (Istio)           │
│  ┌─────────────────────────┐   │
│  │  Order Service          │   │
│  │  ┌──────────────────┐   │   │
│  │  │ Envoy sidecar    │   │   │
│  │  └──────────────────┘   │   │
│  └─────────────────────────┘   │
│         │                       │
│         v                       │
│  ┌─────────────────────────┐   │
│  │  Payment Service        │   │
│  │  ┌──────────────────┐   │   │
│  │  │ Envoy sidecar    │   │   │
│  │  └──────────────────┘   │   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
\`\`\`

**Implementation Example**:

\`\`\`yaml
# 1. API Gateway (Kong) for external traffic
apiVersion: configuration.konghq.com/v1
kind: KongIngress
metadata:
  name: external-api
spec:
  route:
    protocols:
    - https
    methods:
    - GET
    - POST
  upstream:
    host: order-service.default.svc.cluster.local

---
# 2. Service Mesh (Istio) for internal traffic
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-to-payment
spec:
  hosts:
  - payment-service
  http:
  - match:
    - sourceLabels:
        app: order-service
    route:
    - destination:
        host: payment-service
    retries:
      attempts: 3
    timeout: 5s

---
# 3. mTLS between services (Istio)
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
spec:
  mtls:
    mode: STRICT  # Enforce mTLS for all services
\`\`\`

**Decision Framework**:

**Choose API Gateway when**:
1. ✅ Need external API exposure
2. ✅ Require API keys, OAuth, third-party auth
3. ✅ Want to aggregate multiple services
4. ✅ Simple architecture (< 20 services)
5. ✅ Need API versioning and transformation
6. ✅ Quick setup (hours/days)

**Choose Service Mesh when**:
1. ✅ Large microservices architecture (> 50 services)
2. ✅ Heavy internal service communication
3. ✅ Need zero-trust security (mTLS)
4. ✅ Want automatic retries, circuit breaking, load balancing
5. ✅ Require distributed tracing
6. ✅ Team comfortable with complexity (weeks/months setup)

**Choose Both when**:
1. ✅ Large system with external APIs
2. ✅ Complex microservices architecture
3. ✅ Need both external and internal traffic management
4. ✅ Security is critical (external auth + internal mTLS)

**Real-World Examples**:

**Uber**:
- API Gateway: Zuul (for mobile/web clients)
- Service Mesh: Custom solution (internal service communication)

**Lyft**:
- API Gateway: Envoy (also creator of Envoy)
- Service Mesh: Envoy-based mesh

**Netflix**:
- API Gateway: Zuul 2
- Service Mesh: Not using traditional mesh (custom solutions)

**Small Startup (< 10 services)**:
- API Gateway: Yes (Kong or AWS API Gateway)
- Service Mesh: No (overkill)

**Enterprise (> 100 services)**:
- API Gateway: Yes (external traffic)
- Service Mesh: Yes (internal traffic)

**Cost Comparison**:

| Aspect | API Gateway | Service Mesh |
|--------|-------------|--------------|
| Ops complexity | Low-Medium | High |
| Resource overhead | 1-2 gateway instances | Sidecar per service (2x pods) |
| Latency impact | 5-20ms (one hop) | 1-5ms per hop |
| Learning curve | Days-Weeks | Weeks-Months |
| Debugging | Easy | Difficult |

**Verdict**:

- **Start with API Gateway** for external traffic (always needed)
- **Add Service Mesh** when:
  - > 50 microservices
  - Internal traffic is complex
  - Security/compliance requires mTLS
  - Team has expertise

Don't prematurely add service mesh complexity. Many successful companies run 20-30 services with just an API gateway.`,
    keyPoints: [
      'API Gateway handles external (North-South) traffic; Service Mesh handles internal (East-West) traffic',
      'API Gateway provides client auth, rate limiting, aggregation; Service Mesh provides mTLS, retries, tracing',
      'Most large systems use both: API Gateway for external, Service Mesh for internal',
      'Start with API Gateway; add Service Mesh only when microservices grow (> 50 services)',
      'Service Mesh adds significant complexity; only adopt if benefits outweigh costs',
    ],
  },
];
