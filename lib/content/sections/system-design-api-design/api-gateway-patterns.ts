/**
 * API Gateway Patterns Section
 */

export const apigatewaypatternsSection = {
  id: 'api-gateway-patterns',
  title: 'API Gateway Patterns',
  content: `An API Gateway is a single entry point for all client requests, handling routing, authentication, rate limiting, and more. Understanding gateway patterns is crucial for microservices architectures.

## What is an API Gateway?

**API Gateway** acts as a reverse proxy that routes requests to appropriate microservices.

### **Core Responsibilities**1. **Routing**: Direct requests to backend services
2. **Authentication & Authorization**: Centralized security
3. **Rate Limiting**: Prevent abuse
4. **Load Balancing**: Distribute traffic
5. **Request/Response Transformation**: Adapt data formats
6. **Caching**: Improve performance
7. **Monitoring & Logging**: Centralized observability
8. **Protocol Translation**: REST ↔ gRPC, GraphQL

### **Benefits**

- Single entry point simplifies client code
- Centralized cross-cutting concerns
- Reduces microservice coupling
- Enables API versioning and migration
- Provides aggregation layer

### **Drawbacks**

- Single point of failure (mitigate with HA)
- Potential bottleneck
- Additional latency
- Complexity in configuration

## Gateway Patterns

### **1. Simple Gateway (Reverse Proxy)**

Routes requests based on path:

\`\`\`javascript
// NGINX configuration
server {
  listen 80;
  
  location /users {
    proxy_pass http://user-service:3000;
  }
  
  location /products {
    proxy_pass http://product-service:3001;
  }
  
  location /orders {
    proxy_pass http://order-service:3002;
  }
}
\`\`\`

**Express.js Gateway**:

\`\`\`javascript
const express = require('express');
const httpProxy = require('http-proxy-middleware');

const app = express();

// Route to user service
app.use('/users', httpProxy.createProxyMiddleware({
  target: 'http://user-service:3000',
  changeOrigin: true
}));

// Route to product service
app.use('/products', httpProxy.createProxyMiddleware({
  target: 'http://product-service:3001',
  changeOrigin: true
}));

app.listen(80);
\`\`\`

### **2. Aggregator Gateway**

Combines multiple backend calls into single response:

\`\`\`javascript
app.get('/dashboard/:userId', async (req, res) => {
  const { userId } = req.params;
  
  // Call multiple services in parallel
  const [user, orders, recommendations] = await Promise.all([
    fetch(\`http://user-service/users/\${userId}\`),
    fetch(\`http://order-service/orders?userId=\${userId}\`),
    fetch(\`http://recommendation-service/recommend/\${userId}\`)
  ]);
  
  // Aggregate response
  res.json({
    user: await user.json(),
    orders: await orders.json(),
    recommendations: await recommendations.json()
  });
});
\`\`\`

**Benefits**: Reduces client requests from 3 to 1

**Use case**: Mobile apps with limited bandwidth

### **3. Backend for Frontend (BFF)**

Separate gateway per client type:

\`\`\`
┌─────────────┐
│  Web Client │──────┐
└─────────────┘      │
                     v
┌─────────────┐  ┌──────────┐
│Mobile Client│─>│  Web BFF │──┐
└─────────────┘  └──────────┘  │
                                │  ┌──────────────┐
┌─────────────┐  ┌────────────┐│  │User Service  │
│  IoT Device │─>│ Mobile BFF ││─>│Product Service│
└─────────────┘  └────────────┘  │Order Service  │
                                  └──────────────┘
\`\`\`

**Web BFF** (returns full HTML):

\`\`\`javascript
app.get('/product/:id', async (req, res) => {
  const product = await fetch(\`http://product-service/products/\${req.params.id}\`);
  const reviews = await fetch(\`http://review-service/reviews?productId=\${req.params.id}\`);
  
  res.json({
    ...product,
    reviews: reviews,
    displayPrice: formatCurrency (product.price)  // Web-specific formatting
  });
});
\`\`\`

**Mobile BFF** (returns minimal data):

\`\`\`javascript
app.get('/product/:id', async (req, res) => {
  const product = await fetch(\`http://product-service/products/\${req.params.id}\`);
  
  // Mobile: don't include reviews by default (save bandwidth)
  res.json({
    id: product.id,
    name: product.name,
    price: product.price,  // Client formats
    imageUrl: product.thumbnailUrl  // Smaller image
  });
});
\`\`\`

### **4. GraphQL Gateway**

Single GraphQL endpoint federating multiple services:

\`\`\`javascript
const { ApolloServer } = require('apollo-server');
const { ApolloGateway } = require('@apollo/gateway');

const gateway = new ApolloGateway({
  serviceList: [
    { name: 'users', url: 'http://user-service/graphql' },
    { name: 'products', url: 'http://product-service/graphql' },
    { name: 'orders', url: 'http://order-service/graphql' }
  ]
});

const server = new ApolloServer({
  gateway,
  subscriptions: false
});

server.listen(4000);
\`\`\`

**Client Query** (single request):

\`\`\`graphql
query {
  user (id: "123") {
    name
    orders {           # From orders service
      product {        # From products service
        name
        price
      }
    }
  }
}
\`\`\`

Gateway automatically routes to appropriate services.

### **5. Service Mesh Gateway**

Gateway integrated with service mesh (Istio, Linkerd):

\`\`\`yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: api-gateway
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "api.example.com"

---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: routes
spec:
  hosts:
  - "api.example.com"
  gateways:
  - api-gateway
  http:
  - match:
    - uri:
        prefix: "/users"
    route:
    - destination:
        host: user-service
        port:
          number: 3000
  - match:
    - uri:
        prefix: "/products"
    route:
    - destination:
        host: product-service
        port:
          number: 3001
\`\`\`

**Benefits**: 
- Advanced traffic management (canary, A/B testing)
- mTLS between services
- Distributed tracing
- Circuit breaking

## Authentication & Authorization

### **JWT Validation at Gateway**

\`\`\`javascript
const jwt = require('jsonwebtoken');

// Middleware
async function authenticate (req, res, next) {
  const token = req.headers.authorization?.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({ error: 'No token provided' });
  }
  
  try {
    const decoded = jwt.verify (token, process.env.JWT_SECRET);
    req.user = decoded;
    
    // Add user context to downstream services
    req.headers['x-user-id'] = decoded.userId;
    req.headers['x-user-role'] = decoded.role;
    
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
}

// Apply to routes
app.use('/api', authenticate);
\`\`\`

### **OAuth 2.0 Integration**

\`\`\`javascript
app.get('/oauth/callback', async (req, res) => {
  const { code } = req.query;
  
  // Exchange code for token
  const tokenResponse = await fetch('https://oauth-provider.com/token', {
    method: 'POST',
    body: JSON.stringify({
      code,
      client_id: process.env.CLIENT_ID,
      client_secret: process.env.CLIENT_SECRET,
      grant_type: 'authorization_code'
    })
  });
  
  const { access_token } = await tokenResponse.json();
  
  // Store token (session, cookie, etc.)
  res.cookie('access_token', access_token, { httpOnly: true });
  res.redirect('/dashboard');
});
\`\`\`

## Rate Limiting at Gateway

\`\`\`javascript
const rateLimit = require('express-rate-limit');
const RedisStore = require('rate-limit-redis');

const limiter = rateLimit({
  store: new RedisStore({
    client: redisClient
  }),
  windowMs: 15 * 60 * 1000,  // 15 minutes
  max: 100,  // 100 requests per window
  keyGenerator: (req) => {
    // Rate limit by user ID if authenticated
    return req.user?.id || req.ip;
  },
  handler: (req, res) => {
    res.status(429).json({
      error: 'Too many requests',
      retryAfter: req.rateLimit.resetTime
    });
  }
});

app.use('/api', limiter);
\`\`\`

## Request Transformation

### **API Versioning**

\`\`\`javascript
app.use('/v1/users/:id', async (req, res) => {
  const user = await fetch(\`http://user-service/users/\${req.params.id}\`);
  const data = await user.json();
  
  // V1 format (legacy)
  res.json({
    id: data.id,
    full_name: data.name,  // Old field name
    email: data.email
  });
});

app.use('/v2/users/:id', async (req, res) => {
  const user = await fetch(\`http://user-service/users/\${req.params.id}\`);
  const data = await user.json();
  
  // V2 format (new)
  res.json({
    id: data.id,
    name: data.name,  // New field name
    email: data.email,
    createdAt: data.created_at  // Additional field
  });
});
\`\`\`

### **Protocol Translation**

\`\`\`javascript
const grpc = require('@grpc/grpc-js');

// REST endpoint that calls gRPC service
app.get('/users/:id', async (req, res) => {
  // Translate REST to gRPC
  const grpcClient = new UserServiceClient('user-service:50051');
  
  grpcClient.getUser({ id: req.params.id }, (err, response) => {
    if (err) {
      return res.status(500).json({ error: err.message });
    }
    
    // Return as JSON
    res.json (response);
  });
});
\`\`\`

## Caching at Gateway

\`\`\`javascript
const redis = require('redis');
const client = redis.createClient();

async function cacheMiddleware (req, res, next) {
  const cacheKey = \`cache:\${req.method}:\${req.url}\`;
  
  // Check cache
  const cached = await client.get (cacheKey);
  if (cached) {
    return res.json(JSON.parse (cached));
  }
  
  // Override res.json to cache response
  const originalJson = res.json;
  res.json = function (data) {
    // Cache for 5 minutes
    client.setex (cacheKey, 300, JSON.stringify (data));
    originalJson.call (this, data);
  };
  
  next();
}

// Cache GET requests
app.get('/products', cacheMiddleware, async (req, res) => {
  const products = await fetch('http://product-service/products');
  res.json (await products.json());
});
\`\`\`

## Circuit Breaking

\`\`\`javascript
const CircuitBreaker = require('opossum');

// Circuit breaker for user service
const breaker = new CircuitBreaker(
  async (userId) => {
    const response = await fetch(\`http://user-service/users/\${userId}\`);
    return response.json();
  },
  {
    timeout: 3000,      // 3s timeout
    errorThreshold: 50, // Open after 50% errors
    resetTimeout: 30000 // Try again after 30s
  }
);

// Fallback
breaker.fallback(() => ({
  id: null,
  name: 'Guest',
  email: null
}));

app.get('/users/:id', async (req, res) => {
  try {
    const user = await breaker.fire (req.params.id);
    res.json (user);
  } catch (error) {
    res.status(503).json({ error: 'Service unavailable' });
  }
});
\`\`\`

## Monitoring & Logging

\`\`\`javascript
const morgan = require('morgan');
const prometheus = require('prom-client');

// HTTP request logger
app.use (morgan('combined'));

// Prometheus metrics
const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration',
  labelNames: ['method', 'route', 'status']
});

app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    httpRequestDuration.labels(
      req.method,
      req.route?.path || req.path,
      res.statusCode
    ).observe (duration);
  });
  
  next();
});

// Metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', prometheus.register.contentType);
  res.end (await prometheus.register.metrics());
});
\`\`\`

## Popular Gateway Solutions

### **1. Kong**

Open-source, Lua-based, plugin architecture:

\`\`\`yaml
services:
  - name: user-service
    url: http://user-service:3000
    routes:
      - name: user-route
        paths:
          - /users
    plugins:
      - name: rate-limiting
        config:
          minute: 100
      - name: jwt
      - name: cors
\`\`\`

### **2. AWS API Gateway**

Managed service:
- Serverless (Lambda integration)
- Built-in authorization (Cognito, IAM)
- Usage plans and API keys
- Request/response transformation
- CloudWatch monitoring

### **3. Envoy**

High-performance proxy used by Istio:
- Service mesh integration
- Advanced load balancing
- Observability (Jaeger tracing)
- WebAssembly plugins

### **4. Traefik**

Kubernetes-native, automatic service discovery:

\`\`\`yaml
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: api-gateway
spec:
  entryPoints:
    - web
  routes:
    - match: PathPrefix(\`/users\`)
      kind: Rule
      services:
        - name: user-service
          port: 3000
    - match: PathPrefix(\`/products\`)
      kind: Rule
      services:
        - name: product-service
          port: 3001
\`\`\`

### **5. Apollo Gateway** (GraphQL)

Federation for GraphQL:
- Combines multiple GraphQL services
- Schema stitching
- Query planning

## Best Practices

1. **Keep gateway stateless**: Use Redis for shared state
2. **Implement health checks**: Monitor backend services
3. **Use connection pooling**: Reuse HTTP connections
4. **Set timeouts**: Prevent hanging requests
5. **Log request IDs**: Trace requests across services
6. **Cache aggressively**: Reduce backend load
7. **Rate limit by user**: Prevent abuse
8. **Use HTTPS**: Always encrypt traffic
9. **Implement circuit breakers**: Handle failures gracefully
10. **Monitor gateway metrics**: Latency, error rates, throughput

## Gateway Anti-Patterns

### **❌ Business Logic in Gateway**

Don't:
\`\`\`javascript
app.get('/orders/:id', async (req, res) => {
  const order = await fetch(\`http://order-service/orders/\${req.params.id}\`);
  
  // ❌ Complex business logic in gateway
  if (order.total > 1000 && !order.isPremiumUser) {
    return res.status(403).json({ error: 'Premium users only' });
  }
  
  res.json (order);
});
\`\`\`

Do: Push logic to backend services.

### **❌ Direct Database Access**

Gateway should never access databases directly. Always go through services.

### **❌ Gateway Chaining**

Avoid: Gateway → Gateway → Service (adds latency)

## Real-World Example: E-commerce

\`\`\`javascript
const express = require('express');
const app = express();

// Authentication
app.use (authenticate);

// Product search (cached, public)
app.get('/search', cache(60), rateLimit(1000), async (req, res) => {
  const results = await fetch(\`http://search-service/search?q=\${req.query.q}\`);
  res.json (await results.json());
});

// User dashboard (aggregated, authenticated)
app.get('/dashboard', async (req, res) => {
  const [profile, orders, recommendations] = await Promise.all([
    fetch(\`http://user-service/users/\${req.user.id}\`),
    fetch(\`http://order-service/orders?userId=\${req.user.id}\`),
    fetch(\`http://recommendation-service/recommend/\${req.user.id}\`)
  ]);
  
  res.json({
    profile: await profile.json(),
    recentOrders: await orders.json(),
    recommended: await recommendations.json()
  });
});

// Checkout (critical path, no caching)
app.post('/checkout', rateLimit(10), async (req, res) => {
  // Orchestrate checkout across services
  const payment = await fetch('http://payment-service/charge', {
    method: 'POST',
    body: JSON.stringify (req.body)
  });
  
  if (!payment.ok) {
    return res.status(402).json({ error: 'Payment failed' });
  }
  
  const order = await fetch('http://order-service/orders', {
    method: 'POST',
    body: JSON.stringify({ userId: req.user.id, ...req.body })
  });
  
  res.json (await order.json());
});

app.listen(80);
\`\`\`

## When to Use API Gateway

**Use API Gateway when**:
- Microservices architecture (many backend services)
- Need centralized auth, rate limiting, logging
- Multiple client types (web, mobile, IoT)
- API versioning and migration needed
- Aggregation of multiple calls

**Skip API Gateway when**:
- Monolithic architecture (direct access simpler)
- Very simple system (single service)
- Extremely low latency required (every hop adds latency)
- Internal-only services (service mesh might be better)`,
};
