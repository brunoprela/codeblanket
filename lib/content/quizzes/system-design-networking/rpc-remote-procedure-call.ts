/**
 * Quiz questions for RPC (Remote Procedure Call) section
 */

export const rpcremoteprocedurecallQuiz = [
  {
    id: 'rpc-microservices-migration',
    question:
      'Your company is migrating from a monolithic application to microservices. The architecture team is debating between using REST APIs and gRPC for inter-service communication. You have 15 services written in Java, Python, and Node.js. The system handles financial transactions requiring low latency (<50ms) and high reliability. Provide a detailed recommendation on which approach to use, including specific technical justifications, trade-offs, and a migration strategy.',
    sampleAnswer: `**Recommendation: Use gRPC for internal microservices, REST for public APIs**

**Technical Justification**:

1. **Performance Requirements**:
   - Financial transactions require <50ms latency
   - gRPC with Protocol Buffers: ~1-2ms serialization overhead
   - REST with JSON: ~5-10ms serialization overhead
   - gRPC uses HTTP/2 multiplexing → multiple calls over single connection
   - REST (HTTP/1.1) → separate connection per request (connection pooling helps but still overhead)
   - **Result**: gRPC provides 3-5x better latency

2. **Reliability Requirements**:
   - gRPC has built-in: deadlines, retries, health checking, load balancing
   - Strong typing via protobuf prevents serialization errors
   - Code generation ensures client/server contract matching
   - REST requires implementing these manually

3. **Polyglot Support** (Java, Python, Node.js):
   - gRPC has excellent support for all three languages
   - Protobuf definitions generate idiomatic code
   - Single .proto file defines contract for all services
   - REST requires maintaining separate client libraries or OpenAPI specs

**Trade-offs**:

| **Aspect** | **gRPC** | **REST** |
|------------|----------|----------|
| **Performance** | ⚡ Excellent | 🐢 Good |
| **Type Safety** | ✅ Strong | ❌ Weak |
| **Debugging** | ⚠️ Harder (binary) | ✅ Easy (curl, Postman) |
| **Browser Support** | ⚠️ Requires gRPC-Web | ✅ Native |
| **Learning Curve** | ⚠️ Steeper | ✅ Familiar |
| **Ecosystem** | ⚠️ Maturing | ✅ Mature |

**Migration Strategy**:

**Phase 1: Infrastructure Setup** (Week 1-2)
- Set up protobuf compilation in build pipelines
- Create shared proto repository for service contracts
- Implement gRPC server/client scaffolding in Java, Python, Node.js
- Set up Envoy proxy for load balancing and observability

**Phase 2: Pilot Service** (Week 3-4)
- Choose low-risk service (e.g., User Profile Service)
- Implement gRPC version alongside existing REST endpoints
- Run in parallel, measure latency improvements
- Train team on gRPC development and debugging

**Phase 3: Critical Path Services** (Week 5-8)
- Migrate transaction processing services (most latency-sensitive)
- Order Service → Payment Service → Inventory Service
- Keep REST API Gateway for external clients
- Use gRPC for internal service-to-service calls

**Phase 4: Remaining Services** (Week 9-12)
- Migrate remaining services in dependency order
- Retire old REST endpoints after validation
- Monitor error rates, latency, and throughput

**Architecture**:

\`\`\`
Public Clients (Web, Mobile)
         ↓
    API Gateway (REST)
         ↓
   [Internal Services use gRPC]
         ↓
   +--- Order Service (gRPC)
   |         ↓
   |    Payment Service (gRPC)
   |         ↓
   |    Inventory Service (gRPC)
   |
   +--- User Service (gRPC)
   |
   +--- Analytics Service (gRPC)
\`\`\`

**Observability**:
- Use Envoy for metrics (latency percentiles, error rates, QPS)
- Implement distributed tracing (Jaeger/Zipkin)
- Dashboard: gRPC vs REST latency comparison
- Alert on: p99 latency >50ms, error rate >1%

**Expected Results**:
- 3-5x latency improvement for service-to-service calls
- Reduced serialization errors (strong typing)
- Simpler client code (code generation)
- Better resource utilization (fewer connections, less CPU for serialization)

**Risks and Mitigations**:
- **Risk**: Team unfamiliar with gRPC → **Mitigation**: Training + pilot service
- **Risk**: Debugging more difficult → **Mitigation**: grpcurl, grpc-web devtools
- **Risk**: Breaking changes in protos → **Mitigation**: Versioning strategy, backwards compatibility
- **Risk**: Migration bugs → **Mitigation**: Parallel running, gradual rollout, feature flags

**Final Recommendation**: Use gRPC for internal services (performance + reliability), keep REST for public API (accessibility). This hybrid approach gives you the best of both worlds.`,
    keyPoints: [
      'Use gRPC for internal microservices (low latency, type safety)',
      'Keep REST for public APIs (accessibility, browser support)',
      'gRPC provides 3-5x better performance for inter-service communication',
      'Hybrid approach: gRPC internally, REST gateway for external clients',
      'Consider team expertise and tooling maturity in decision',
    ],
  },
  {
    id: 'rpc-error-handling-strategy',
    question:
      'Design a comprehensive error handling and retry strategy for a distributed system using gRPC. Your system has an Order Service that calls Inventory Service, Payment Service, and Shipping Service. Each downstream service has different SLAs and failure characteristics. Include specific gRPC status codes, retry policies, timeout configurations, circuit breaker patterns, and how to handle partial failures.',
    sampleAnswer: `**Comprehensive gRPC Error Handling Strategy**

**1. Service SLAs and Characteristics**:

| **Service** | **p99 Latency** | **Failure Rate** | **Failure Type** |
|-------------|-----------------|------------------|------------------|
| Inventory   | 50ms | 0.1% | DB overload (transient) |
| Payment     | 200ms | 0.5% | External API timeout |
| Shipping    | 100ms | 0.2% | Rate limiting |

**2. Timeout Configuration**:

\`\`\`javascript
const TIMEOUTS = {
  inventory: {
    deadline: 200,    // 4x p99 (50ms × 4)
    retryDeadline: 500 // Total time including retries
  },
  payment: {
    deadline: 800,    // 4x p99 (200ms × 4)
    retryDeadline: 2000
  },
  shipping: {
    deadline: 400,    // 4x p99 (100ms × 4)
    retryDeadline: 1000
  }
};
\`\`\`

**Rationale**: Set deadline to 4x p99 latency to allow for occasional slowness but prevent hanging.

**3. Retry Policy by Status Code**:

\`\`\`javascript
const RETRY_POLICY = {
  // Retry with exponential backoff
  [grpc.status.UNAVAILABLE]: {
    maxRetries: 3,
    backoff: 'exponential',
    initialDelay: 100,
    maxDelay: 1000
  },
  [grpc.status.DEADLINE_EXCEEDED]: {
    maxRetries: 2,
    backoff: 'exponential',
    initialDelay: 200,
    maxDelay: 1000
  },
  [grpc.status.RESOURCE_EXHAUSTED]: {
    maxRetries: 3,
    backoff: 'exponential',
    initialDelay: 500, // Longer initial delay
    maxDelay: 5000
  },
  
  // Don't retry
  [grpc.status.INVALID_ARGUMENT]: { maxRetries: 0 },
  [grpc.status.NOT_FOUND]: { maxRetries: 0 },
  [grpc.status.ALREADY_EXISTS]: { maxRetries: 0 },
  [grpc.status.PERMISSION_DENIED]: { maxRetries: 0 },
  [grpc.status.UNAUTHENTICATED]: { maxRetries: 0 },
  [grpc.status.FAILED_PRECONDITION]: { maxRetries: 0 }
};

async function callWithRetry(client, method, request, config) {
  const policy = RETRY_POLICY[config.statusCode] || { maxRetries: 0 };
  let attempts = 0;
  
  while (attempts <= policy.maxRetries) {
    try {
      const deadline = Date.now() + config.deadline;
      const result = await promisify(client[method])(request, { deadline });
      return result;
    } catch (error) {
      attempts++;
      
      // Check if error is retryable
      const retryConfig = RETRY_POLICY[error.code];
      if (!retryConfig || attempts > retryConfig.maxRetries) {
        throw error;
      }
      
      // Calculate backoff delay
      const delay = Math.min(
        retryConfig.initialDelay * Math.pow(2, attempts - 1),
        retryConfig.maxDelay
      );
      
      logger.warn(\`Retry \${attempts}/\${retryConfig.maxRetries} for \${method} after \${delay}ms\`, {
        error: error.code,
        message: error.message
      });
      
      await sleep(delay);
    }
  }
}
\`\`\`

**4. Circuit Breaker Pattern**:

\`\`\`javascript
class CircuitBreaker {
  constructor(options = {}) {
    this.failureThreshold = options.failureThreshold || 5;
    this.resetTimeout = options.resetTimeout || 60000; // 60s
    this.monitoringWindow = options.monitoringWindow || 10000; // 10s
    
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    this.failures = [];
    this.nextAttempt = null;
  }
  
  async execute(fn) {
    if (this.state === 'OPEN') {
      if (Date.now() < this.nextAttempt) {
        throw new Error('Circuit breaker is OPEN');
      }
      // Try one request (HALF_OPEN state)
      this.state = 'HALF_OPEN';
    }
    
    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }
  
  onSuccess() {
    if (this.state === 'HALF_OPEN') {
      this.state = 'CLOSED';
      this.failures = [];
    }
  }
  
  onFailure() {
    const now = Date.now();
    this.failures.push(now);
    
    // Remove old failures outside monitoring window
    this.failures = this.failures.filter(
      time => now - time < this.monitoringWindow
    );
    
    if (this.failures.length >= this.failureThreshold) {
      this.state = 'OPEN';
      this.nextAttempt = now + this.resetTimeout;
      logger.error(\`Circuit breaker opened. Next attempt at \${new Date(this.nextAttempt)}\`);
    }
  }
}

// Create circuit breakers for each service
const circuitBreakers = {
  inventory: new CircuitBreaker({ failureThreshold: 5, resetTimeout: 30000 }),
  payment: new CircuitBreaker({ failureThreshold: 3, resetTimeout: 60000 }),
  shipping: new CircuitBreaker({ failureThreshold: 5, resetTimeout: 30000 })
};
\`\`\`

**5. Handling Partial Failures in Order Service**:

\`\`\`javascript
async function createOrder(call, callback) {
  const { user_id, items, payment_info, shipping_address } = call.request;
  const orderId = generateOrderId();
  
  // Store rollback actions
  const rollbackActions = [];
  
  try {
    // Step 1: Check inventory (critical)
    const inventoryResult = await circuitBreakers.inventory.execute(() =>
      callWithRetry(inventoryClient, 'checkAvailability', { items }, {
        deadline: TIMEOUTS.inventory.deadline,
        statusCode: grpc.status.UNAVAILABLE
      })
    );
    
    if (!inventoryResult.available) {
      return callback({
        code: grpc.status.FAILED_PRECONDITION,
        message: 'Items not in stock'
      });
    }
    
    // Step 2: Reserve inventory (critical, must rollback)
    await circuitBreakers.inventory.execute(() =>
      callWithRetry(inventoryClient, 'reserveItems', { items, orderId }, {
        deadline: TIMEOUTS.inventory.deadline
      })
    );
    rollbackActions.push(() => 
      inventoryClient.releaseItems({ orderId })
    );
    
    // Step 3: Process payment (critical, must rollback)
    let paymentResult;
    try {
      paymentResult = await circuitBreakers.payment.execute(() =>
        callWithRetry(paymentClient, 'processPayment', {
          amount: inventoryResult.total,
          payment_info,
          idempotency_key: orderId
        }, {
          deadline: TIMEOUTS.payment.deadline
        })
      );
    } catch (error) {
      // Payment failed, rollback inventory
      await executeRollback(rollbackActions);
      
      return callback({
        code: grpc.status.FAILED_PRECONDITION,
        message: 'Payment failed',
        details: error.message
      });
    }
    
    rollbackActions.push(() =>
      paymentClient.refundPayment({ transaction_id: paymentResult.transactionId })
    );
    
    // Step 4: Create shipping label (optional, can retry later)
    let shippingResult;
    try {
      shippingResult = await circuitBreakers.shipping.execute(() =>
        callWithRetry(shippingClient, 'createLabel', {
          orderId,
          address: shipping_address,
          items
        }, {
          deadline: TIMEOUTS.shipping.deadline
        })
      );
    } catch (error) {
      // Shipping failed, but order can still be created
      // Queue for retry via async job
      logger.warn('Shipping label creation failed, queuing for retry', {
        orderId,
        error: error.message
      });
      await queueShippingRetry(orderId, shipping_address, items);
      shippingResult = { status: 'pending' };
    }
    
    // Step 5: Create order record
    const order = {
      order_id: orderId,
      user_id,
      items,
      status: 'confirmed',
      total: inventoryResult.total,
      payment_transaction_id: paymentResult.transactionId,
      shipping_status: shippingResult.status
    };
    
    await saveOrder(order);
    
    // Success!
    callback(null, order);
    
  } catch (error) {
    // Unexpected error, rollback everything
    logger.error('Order creation failed, rolling back', {
      orderId,
      error: error.message,
      stack: error.stack
    });
    
    await executeRollback(rollbackActions);
    
    // Determine appropriate error code
    if (error.code === grpc.status.UNAVAILABLE) {
      return callback({
        code: grpc.status.UNAVAILABLE,
        message: 'Service temporarily unavailable, please try again'
      });
    } else if (error.code === grpc.status.DEADLINE_EXCEEDED) {
      return callback({
        code: grpc.status.DEADLINE_EXCEEDED,
        message: 'Request timeout, please try again'
      });
    } else {
      return callback({
        code: grpc.status.INTERNAL,
        message: 'Internal error, please contact support',
        details: error.message
      });
    }
  }
}

async function executeRollback(actions) {
  for (const action of actions.reverse()) {
    try {
      await action();
    } catch (error) {
      // Log but don't throw (best-effort rollback)
      logger.error('Rollback action failed', { error: error.message });
    }
  }
}
\`\`\`

**6. Monitoring and Alerting**:

\`\`\`javascript
// Metrics to track
const metrics = {
  requestDuration: new Histogram('grpc_request_duration_seconds'),
  requestTotal: new Counter('grpc_requests_total'),
  circuitBreakerState: new Gauge('circuit_breaker_state'),
  retryTotal: new Counter('grpc_retries_total')
};

// Alert thresholds
const ALERTS = {
  errorRate: 0.01,        // 1% error rate
  p99Latency: 1000,       // 1 second p99
  circuitBreakerOpen: 1   // Any circuit breaker open
};
\`\`\`

**7. Idempotency**:

For payment and other critical operations, use idempotency keys:

\`\`\`javascript
// Client sends same idempotency key on retry
await paymentClient.processPayment({
  amount: 100,
  payment_info: {...},
  idempotency_key: orderId  // Same for all retries
});

// Server deduplicates by idempotency key
async function processPayment(call, callback) {
  const { idempotency_key } = call.request;
  
  // Check if already processed
  const existing = await getTransactionByIdempotencyKey(idempotency_key);
  if (existing) {
    return callback(null, existing); // Return cached result
  }
  
  // Process payment...
}
\`\`\`

**Key Takeaways**:

1. **Set timeouts to 4x p99 latency** to balance responsiveness and reliability
2. **Retry transient errors** (UNAVAILABLE, DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED)
3. **Never retry logical errors** (INVALID_ARGUMENT, NOT_FOUND, PERMISSION_DENIED)
4. **Use exponential backoff** to avoid thundering herd
5. **Circuit breakers prevent cascading failures** by failing fast when service is down
6. **Rollback on critical failures** (payment, inventory) but allow optional failures (shipping)
7. **Idempotency prevents duplicate operations** on retry
8. **Monitor circuit breaker state**, error rates, latency percentiles
9. **Partial failures**: Critical ops must succeed, optional ops can be queued for retry
10. **Always log** errors with context for debugging`,
    keyPoints: [
      'Set appropriate timeouts based on service SLAs (50ms-200ms)',
      'Implement retry with exponential backoff for transient errors',
      'Use circuit breakers to prevent cascading failures',
      'Handle partial failures: roll back critical ops, queue optional ones',
      'Use idempotency keys to prevent duplicate operations on retry',
    ],
  },
  {
    id: 'rpc-public-api-gateway',
    question:
      "You need to design an API Gateway that exposes your internal gRPC microservices to external clients (web browsers, mobile apps, third-party integrators). The gateway must support REST, GraphQL, and potentially gRPC-Web. Design the architecture, explain the trade-offs of each protocol option, describe how you'd handle authentication/authorization, rate limiting, and provide a specific implementation approach. Include diagrams or pseudo-code as needed.",
    sampleAnswer: `**API Gateway Architecture for gRPC Microservices**

**High-Level Architecture**:

\`\`\`
External Clients
      |
[Web Browser]  [Mobile App]  [3rd Party]
      |              |              |
   REST API     GraphQL API    gRPC-Web
      |              |              |
      +----------- API Gateway ------------+
                     |
         +-----------+-----------+
         |           |           |
    [REST] [GraphQL] [gRPC-Web] [Admin]
         |           |           |
      Transcoder  Resolver   gRPC-Web Proxy
         |           |           |
         +--------- gRPC ---------+
                     |
         +-----------+-----------+
         |           |           |
    User Service  Order Service  Inventory Service
     (gRPC)        (gRPC)         (gRPC)
\`\`\`

**1. Protocol Support and Trade-offs**:

| **Protocol** | **Use Case** | **Pros** | **Cons** |
|--------------|--------------|----------|----------|
| **REST** | Public API, 3rd parties | Familiar, cacheable, curl-testable | Slower, no streaming, over/under-fetching |
| **GraphQL** | Mobile apps, dashboards | Single endpoint, flexible queries, efficient | Complex caching, N+1 queries, learning curve |
| **gRPC-Web** | Internal web apps | Fast, streaming, type-safe | Requires grpc-web proxy, limited browser support |

**Recommendation**:
- **REST**: Default for public API (widest compatibility)
- **GraphQL**: For mobile/web apps (efficient, flexible)
- **gRPC-Web**: For internal dashboards (performance)

**2. API Gateway Implementation (Node.js + Envoy)**:

**Architecture Components**:
- **Envoy Proxy**: gRPC-Web proxy, load balancing, TLS termination
- **Node.js API Gateway**: REST/GraphQL → gRPC transcoding, auth, rate limiting
- **Redis**: Rate limiting, session storage
- **Auth Service**: JWT validation, OAuth

**Envoy Configuration (envoy.yaml)**:

\`\`\`yaml
static_resources:
  listeners:
    - name: main_listener
      address:
        socket_address:
          address: 0.0.0.0
          port_value: 8080
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                codec_type: AUTO
                stat_prefix: ingress_http
                
                # gRPC-Web support
                http_filters:
                  - name: envoy.filters.http.grpc_web
                  - name: envoy.filters.http.cors
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.cors.v3.Cors
                  - name: envoy.filters.http.router
                
                route_config:
                  name: local_route
                  virtual_hosts:
                    - name: backend
                      domains: ["*",]
                      routes:
                        # gRPC-Web routes
                        - match:
                            prefix: "/grpc"
                          route:
                            cluster: grpc_services
                            timeout: 30s
                        # REST routes (proxy to Node.js)
                        - match:
                            prefix: "/api"
                          route:
                            cluster: api_gateway
                        # GraphQL routes
                        - match:
                            prefix: "/graphql"
                          route:
                            cluster: api_gateway
  
  clusters:
    - name: grpc_services
      type: STRICT_DNS
      lb_policy: ROUND_ROBIN
      http2_protocol_options: {}
      load_assignment:
        cluster_name: grpc_services
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: user-service
                      port_value: 50051
    
    - name: api_gateway
      type: STRICT_DNS
      lb_policy: ROUND_ROBIN
      load_assignment:
        cluster_name: api_gateway
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: node-gateway
                      port_value: 3000
\`\`\`

**3. Node.js API Gateway Implementation**:

**REST → gRPC Transcoding**:

\`\`\`javascript
const express = require('express');
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const redis = require('redis');
const jwt = require('jsonwebtoken');

const app = express();
app.use(express.json());

// Load gRPC clients
const packageDefinition = protoLoader.loadSync('user.proto');
const userProto = grpc.loadPackageDefinition(packageDefinition).user;
const userClient = new userProto.UserService(
  'user-service:50051',
  grpc.credentials.createInsecure()
);

const redisClient = redis.createClient({ url: 'redis://redis:6379' });
await redisClient.connect();

// Middleware: Authentication
async function authenticate(req, res, next) {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or invalid token' });
  }
  
  const token = authHeader.substring(7);
  
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' });
  }
}

// Middleware: Rate Limiting (Token Bucket)
async function rateLimit(req, res, next) {
  const key = \`rate_limit:\${req.user.id}\`;
  const limit = 100; // requests per minute
  const window = 60; // seconds
  
  const current = await redisClient.incr(key);
  
  if (current === 1) {
    await redisClient.expire(key, window);
  }
  
  if (current > limit) {
    return res.status(429).json({
      error: 'Rate limit exceeded',
      retryAfter: await redisClient.ttl(key)
    });
  }
  
  res.setHeader('X-RateLimit-Limit', limit);
  res.setHeader('X-RateLimit-Remaining', limit - current);
  
  next();
}

// REST Endpoints

// GET /api/users/:id → userService.GetUser
app.get('/api/users/:id', authenticate, rateLimit, (req, res) => {
  const metadata = new grpc.Metadata();
  metadata.add('authorization', \`Bearer \${req.user.token}\`);
  metadata.add('request-id', req.id);
  
  userClient.getUser(
    { id: req.params.id },
    metadata,
    { deadline: Date.now() + 5000 },
    (error, user) => {
      if (error) {
        if (error.code === grpc.status.NOT_FOUND) {
          return res.status(404).json({ error: 'User not found' });
        } else if (error.code === grpc.status.PERMISSION_DENIED) {
          return res.status(403).json({ error: 'Permission denied' });
        } else {
          logger.error('gRPC error', { error });
          return res.status(500).json({ error: 'Internal server error' });
        }
      }
      
      res.json(user);
    }
  );
});

// POST /api/users → userService.CreateUser
app.post('/api/users', authenticate, rateLimit, (req, res) => {
  const { name, email, age } = req.body;
  
  // Validation
  if (!name || !email) {
    return res.status(400).json({ error: 'Name and email required' });
  }
  
  const metadata = new grpc.Metadata();
  metadata.add('authorization', \`Bearer \${req.user.token}\`);
  
  userClient.createUser(
    { name, email, age },
    metadata,
    { deadline: Date.now() + 5000 },
    (error, user) => {
      if (error) {
        if (error.code === grpc.status.ALREADY_EXISTS) {
          return res.status(409).json({ error: 'User already exists' });
        } else if (error.code === grpc.status.INVALID_ARGUMENT) {
          return res.status(400).json({ error: error.message });
        } else {
          logger.error('gRPC error', { error });
          return res.status(500).json({ error: 'Internal server error' });
        }
      }
      
      res.status(201).json(user);
    }
  );
});

// GET /api/users → userService.ListUsers (server streaming)
app.get('/api/users', authenticate, rateLimit, (req, res) => {
  const { page_size = 20, page_token } = req.query;
  
  const metadata = new grpc.Metadata();
  metadata.add('authorization', \`Bearer \${req.user.token}\`);
  
  const call = userClient.listUsers({ page_size, page_token }, metadata);
  
  const users = [];
  
  call.on('data', (user) => {
    users.push(user);
  });
  
  call.on('end', () => {
    res.json({ users, next_page_token: null }); // Simplified
  });
  
  call.on('error', (error) => {
    logger.error('gRPC streaming error', { error });
    if (!res.headersSent) {
      res.status(500).json({ error: 'Internal server error' });
    }
  });
});

app.listen(3000, () => {
  console.log('API Gateway listening on port 3000');
});
\`\`\`

**4. GraphQL Integration**:

\`\`\`javascript
const { ApolloServer, gql } = require('apollo-server-express');

// GraphQL Schema
const typeDefs = gql\`
  type User {
    id: ID!
    name: String!
    email: String!
    age: Int
  }
  
  type Query {
    user(id: ID!): User
    users(limit: Int = 20): [User!]!
  }
  
  type Mutation {
    createUser(name: String!, email: String!, age: Int): User!
  }
\`;

// Resolvers
const resolvers = {
  Query: {
    user: async (_, { id }, context) => {
      return new Promise((resolve, reject) => {
        const metadata = new grpc.Metadata();
        metadata.add('authorization', \`Bearer \${context.user.token}\`);
        
        userClient.getUser({ id }, metadata, (error, user) => {
          if (error) reject(error);
          else resolve(user);
        });
      });
    },
    
    users: async (_, { limit }, context) => {
      return new Promise((resolve, reject) => {
        const metadata = new grpc.Metadata();
        metadata.add('authorization', \`Bearer \${context.user.token}\`);
        
        const call = userClient.listUsers({ page_size: limit }, metadata);
        const users = [];
        
        call.on('data', (user) => users.push(user));
        call.on('end', () => resolve(users));
        call.on('error', (error) => reject(error));
      });
    }
  },
  
  Mutation: {
    createUser: async (_, { name, email, age }, context) => {
      return new Promise((resolve, reject) => {
        const metadata = new grpc.Metadata();
        metadata.add('authorization', \`Bearer \${context.user.token}\`);
        
        userClient.createUser(
          { name, email, age },
          metadata,
          (error, user) => {
            if (error) reject(error);
            else resolve(user);
          }
        );
      });
    }
  }
};

// Create Apollo Server
const server = new ApolloServer({
  typeDefs,
  resolvers,
  context: async ({ req }) => {
    // Extract user from JWT
    const token = req.headers.authorization?.substring(7);
    if (!token) throw new Error('Unauthorized');
    
    const user = jwt.verify(token, process.env.JWT_SECRET);
    return { user: { ...user, token } };
  }
});

server.applyMiddleware({ app, path: '/graphql' });
\`\`\`

**5. Authentication Flow**:

\`\`\`
1. Client sends: Authorization: Bearer <JWT>
2. API Gateway validates JWT
3. Gateway extracts user_id, roles from JWT
4. Gateway adds metadata to gRPC call:
   - authorization: Bearer <JWT>
   - user-id: 123
   - roles: admin,user
5. Microservice validates metadata (optional, for defense in depth)
6. Microservice performs authorization check
7. Response flows back through gateway
\`\`\`

**6. Key Takeaways**:

1. **Use Envoy for gRPC-Web** proxy and load balancing
2. **REST for public APIs**, GraphQL for mobile/web, gRPC-Web for internal
3. **Authenticate at gateway**, propagate user context via metadata
4. **Rate limit per user** using Redis (token bucket or sliding window)
5. **Map gRPC status codes to HTTP** (NOT_FOUND → 404, PERMISSION_DENIED → 403)
6. **Handle streaming** carefully in REST (buffer or SSE)
7. **Timeout all gRPC calls** to prevent hanging
8. **Monitor**: latency, error rate, rate limit hits
9. **Cache** REST responses at CDN/gateway when possible
10. **Defense in depth**: Validate auth at both gateway and service level`,
    keyPoints: [
      'Support multiple protocols: REST (public), GraphQL (mobile), gRPC-Web (internal)',
      'Use Envoy for gRPC-Web proxy, TLS termination, and load balancing',
      'Node.js API Gateway for REST/GraphQL → gRPC transcoding',
      'Authentication via JWT validation, propagate to microservices via gRPC metadata',
      'Rate limiting with Redis (token bucket or sliding window)',
      'Protocol trade-offs: REST (familiar), GraphQL (flexible), gRPC-Web (fast)',
      'Error mapping: gRPC status codes → HTTP status codes (NOT_FOUND → 404)',
      'Monitor latency, error rates, and rate limit hits',
    ],
  },
];
