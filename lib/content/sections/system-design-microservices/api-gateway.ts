/**
 * API Gateway Pattern Section
 */

export const apigatewaySection = {
  id: 'api-gateway',
  title: 'API Gateway Pattern',
  content: `The API Gateway pattern is a single entry point for all client requests in a microservices architecture. It acts as a reverse proxy, routing requests to appropriate microservices.

## The Problem

**Without API Gateway**:
\`\`\`
Mobile App ─────┬──→ Auth Service (port 8001)
                ├──→ Order Service (port 8002)
                ├──→ Payment Service (port 8003)
                ├──→ Product Service (port 8004)
                └──→ User Service (port 8005)
\`\`\`

**Problems**:
- Clients must know about all services and their locations
- Each service may have different authentication mechanisms
- Can't enforce rate limiting across services
- CORS issues with multiple origins
- Client makes multiple round trips (slow on mobile)
- Protocol translation (HTTP → gRPC) handled by each client

**With API Gateway**:
\`\`\`
Mobile App ───→ API Gateway:443 ───┬──→ Auth Service
Web App    ───→                     ├──→ Order Service
                                    ├──→ Payment Service
                                    ├──→ Product Service
                                    └──→ User Service
\`\`\`

Single entry point, handles routing, auth, rate limiting, etc.

---

## Core Responsibilities

### 1. **Request Routing**

Route requests to appropriate backend services.

**Example**:
\`\`\`
GET  /api/products      → Product Service
POST /api/orders        → Order Service
GET  /api/users/me      → User Service
POST /api/auth/login    → Auth Service
\`\`\`

**Path-Based Routing**:
\`\`\`nginx
# NGINX configuration
location /api/products {
    proxy_pass http://product-service:8080;
}

location /api/orders {
    proxy_pass http://order-service:8080;
}
\`\`\`

**Intelligent Routing** (based on headers, query params):
\`\`\`javascript
// Route to different service versions
if (request.headers['X-Client-Version'] === '2.0') {
    route to product-service-v2
} else {
    route to product-service-v1
}
\`\`\`

### 2. **Request Aggregation (Backend for Frontend - BFF)**

Combine multiple backend calls into one response.

**Problem**: Mobile app needs user profile + recent orders + notifications = 3 API calls

**Solution**: API Gateway makes 3 calls internally, returns combined response

**Example**:
\`\`\`javascript
// API Gateway endpoint: GET /api/home
async function getHomeData (userId) {
    // Make 3 parallel calls to backend services
    const [user, orders, notifications] = await Promise.all([
        userService.getUser (userId),
        orderService.getRecentOrders (userId),
        notificationService.getUnread (userId)
    ]);
    
    // Return combined response
    return {
        user,
        recentOrders: orders,
        unreadNotifications: notifications
    };
}
\`\`\`

**Benefit**: Mobile app makes 1 request instead of 3, reducing latency

### 3. **Authentication & Authorization**

Centralize auth logic instead of duplicating in each service.

**Flow**:
\`\`\`
1. Client sends request with JWT token to API Gateway
2. Gateway validates token (signature, expiration)
3. Gateway extracts user ID and roles
4. Gateway adds user context to request headers
5. Gateway forwards request to backend service
6. Backend service trusts headers (doesn't re-validate)
\`\`\`

**Example**:
\`\`\`javascript
// API Gateway middleware
async function authenticate (request) {
    const token = request.headers['Authorization'].replace('Bearer ', ');
    
    // Validate JWT
    const decoded = jwt.verify (token, JWT_SECRET);
    
    // Add user context to request
    request.headers['X-User-Id'] = decoded.userId;
    request.headers['X-User-Roles'] = decoded.roles.join(',');
    
    // Forward to backend
    return proxy (request);
}
\`\`\`

**Benefits**:
- Backend services don't need JWT validation logic
- Can enforce authorization policies centrally
- Easy to switch auth mechanisms

### 4. **Rate Limiting & Throttling**

Protect backend services from abuse.

**Example**:
\`\`\`javascript
// Rate limit: 100 requests per minute per user
const rateLimiter = new RateLimiter({
    windowMs: 60 * 1000, // 1 minute
    max: 100 // requests
});

app.use (async (req, res, next) => {
    const userId = req.headers['X-User-Id'];
    
    if (await rateLimiter.isBlocked (userId)) {
        return res.status(429).json({
            error: 'Too many requests'
        });
    }
    
    next();
});
\`\`\`

**Different limits for different tiers**:
\`\`\`
Free tier:    100 req/min
Pro tier:   1,000 req/min
Enterprise: 10,000 req/min
\`\`\`

### 5. **Protocol Translation**

Translate between different protocols.

**Example**: Client uses HTTP/REST, backend uses gRPC

\`\`\`javascript
// API Gateway
app.post('/api/orders', async (req, res) => {
    // Receive HTTP/JSON request
    const orderData = req.body;
    
    // Call backend gRPC service
    const grpcClient = getGrpcClient('order-service');
    const result = await grpcClient.createOrder (orderData);
    
    // Return HTTP/JSON response
    res.json (result);
});
\`\`\`

**Other translations**:
- GraphQL → multiple REST calls
- REST → message queue
- WebSocket → HTTP polling

### 6. **Response Transformation**

Modify responses before returning to client.

**Example**: Remove internal fields, add pagination metadata

\`\`\`javascript
async function getProducts (req, res) {
    // Call backend service
    const products = await productService.getAll();
    
    // Transform response
    const transformed = products.map (p => ({
        id: p.id,
        name: p.name,
        price: p.price,
        // Remove internal fields
        // internalCost: p.internalCost  ❌
        // supplierId: p.supplierId      ❌
    }));
    
    // Add metadata
    res.json({
        data: transformed,
        total: transformed.length,
        page: 1
    });
}
\`\`\`

### 7. **Caching**

Cache responses to reduce backend load.

**Example**:
\`\`\`javascript
const redis = require('redis').createClient();

app.get('/api/products/:id', async (req, res) => {
    const cacheKey = \`product:\${req.params.id}\`;
    
    // Check cache
    const cached = await redis.get (cacheKey);
    if (cached) {
        return res.json(JSON.parse (cached));
    }
    
    // Cache miss - call backend
    const product = await productService.get (req.params.id);
    
    // Store in cache (5 minutes)
    await redis.setex (cacheKey, 300, JSON.stringify (product));
    
    res.json (product);
});
\`\`\`

### 8. **Load Balancing**

Distribute requests across service instances.

**Example: Round Robin**
\`\`\`javascript
const instances = [
    'http://order-service-1:8080',
    'http://order-service-2:8080',
    'http://order-service-3:8080'
];
let currentIndex = 0;

function getNextInstance() {
    const instance = instances[currentIndex];
    currentIndex = (currentIndex + 1) % instances.length;
    return instance;
}
\`\`\`

---

## API Gateway Implementations

### 1. **Kong**

Open-source API Gateway with extensive plugin ecosystem.

**Features**:
- Load balancing
- Authentication (JWT, OAuth 2.0, API Keys)
- Rate limiting
- Request/response transformation
- Logging and monitoring

**Example**:
\`\`\`bash
# Add service
curl -i -X POST http://localhost:8001/services \\
  --data name=order-service \\
  --data url=http://order-service:8080

# Add route
curl -i -X POST http://localhost:8001/services/order-service/routes \\
  --data 'paths[]=/api/orders'

# Add JWT auth plugin
curl -X POST http://localhost:8001/services/order-service/plugins \\
  --data name=jwt
\`\`\`

### 2. **AWS API Gateway**

Fully managed service for creating, publishing, and managing APIs.

**Features**:
- Automatic scaling
- Integration with AWS Lambda
- API versioning
- Usage plans and API keys
- CloudWatch logging

**Example**:
\`\`\`yaml
# SAM template
Resources:
  MyApi:
    Type: AWS::Serverless::Api
    Properties:
      StageName: prod
      Auth:
        ApiKeyRequired: true
      
  GetProduct:
    Type: AWS::Serverless::Function
    Properties:
      Handler: index.handler
      Runtime: nodejs18.x
      Events:
        GetProductApi:
          Type: Api
          Properties:
            RestApiId: !Ref MyApi
            Path: /products/{id}
            Method: get
\`\`\`

### 3. **NGINX / NGINX Plus**

High-performance web server that can act as API Gateway.

**Example**:
\`\`\`nginx
upstream product_service {
    server product-1:8080;
    server product-2:8080;
}

server {
    listen 443 ssl;
    
    location /api/products {
        # Authentication
        auth_request /auth;
        
        # Rate limiting
        limit_req zone=api_limit burst=10;
        
        # Proxy to backend
        proxy_pass http://product_service;
        proxy_set_header X-User-Id $http_x_user_id;
    }
    
    location = /auth {
        internal;
        proxy_pass http://auth-service/validate;
    }
}
\`\`\`

### 4. **Spring Cloud Gateway**

Built on Spring Boot, reactive gateway for microservices.

**Example**:
\`\`\`java
@Configuration
public class GatewayConfig {
    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()
            .route("products", r -> r.path("/api/products/**")
                .filters (f -> f
                    .stripPrefix(2)
                    .addRequestHeader("X-Gateway", "true")
                    .circuitBreaker (config -> config
                        .setName("productsCircuitBreaker")
                        .setFallbackUri("/fallback/products")))
                .uri("lb://product-service"))
            .build();
    }
}
\`\`\`

### 5. **Traefik**

Cloud-native reverse proxy and load balancer.

**Features**:
- Automatic service discovery
- Let\'s Encrypt SSL
- Middleware (auth, rate limit, retry)
- Kubernetes ingress controller

**Example**:
\`\`\`yaml
# docker-compose.yml
services:
  traefik:
    image: traefik:v2.9
    command:
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
    labels:
      - "traefik.http.routers.api.rule=Host(\`api.example.com\`)"
      - "traefik.http.routers.api.middlewares=auth,ratelimit"
  
  product-service:
    image: product-service:latest
    labels:
      - "traefik.http.routers.products.rule=PathPrefix(\`/api/products\`)"
      - "traefik.http.routers.products.middlewares=auth"
\`\`\`

---

## Backend for Frontend (BFF) Pattern

Different clients (mobile, web, IoT) have different needs. BFF creates dedicated gateways for each.

**Architecture**:
\`\`\`
Mobile App → Mobile BFF ─┐
Web App    → Web BFF    ─┼──→ Microservices
IoT Device → IoT BFF    ─┘
\`\`\`

**Example**:

**Mobile BFF** (optimized for bandwidth):
\`\`\`javascript
// Returns minimal data
GET /api/products
{
    products: [
        {id: 1, name: "iPhone", price: 999, thumb: "url"}
    ]
}
\`\`\`

**Web BFF** (more details):
\`\`\`javascript
// Returns full data
GET /api/products
{
    products: [
        {
            id: 1,
            name: "iPhone 15 Pro",
            description: "...",
            price: 999,
            images: ["url1", "url2", "url3"],
            specs: {...},
            reviews: [...]
        }
    ]
}
\`\`\`

**Advantages**:
✅ Optimized for each client type
✅ Can evolve independently
✅ Better separation of concerns

**Disadvantages**:
❌ More code to maintain
❌ Potential duplication

---

## API Composition Pattern

Gateway composes data from multiple services.

**Example: Order Details Page**

Needs:
1. Order info (Order Service)
2. Product details (Product Service)
3. Shipping status (Shipping Service)
4. Payment status (Payment Service)

**API Gateway implementation**:
\`\`\`javascript
async function getOrderDetails (orderId) {
    // Get order
    const order = await orderService.getOrder (orderId);
    
    // Get related data in parallel
    const [products, shipping, payment] = await Promise.all([
        productService.getByIds (order.productIds),
        shippingService.getStatus (orderId),
        paymentService.getStatus (order.paymentId)
    ]);
    
    // Compose response
    return {
        order: {
            id: order.id,
            status: order.status,
            total: order.total,
            items: order.items.map (item => ({
                ...item,
                product: products.find (p => p.id === item.productId)
            }))
        },
        shipping: {
            status: shipping.status,
            estimatedDelivery: shipping.estimatedDelivery,
            trackingNumber: shipping.trackingNumber
        },
        payment: {
            status: payment.status,
            method: payment.method
        }
    };
}
\`\`\`

---

## API Gateway Anti-Patterns

### ❌ Smart Gateway, Dumb Services

**Problem**: Gateway contains too much business logic

**Example**:
\`\`\`javascript
// BAD: Business logic in gateway
app.post('/api/orders', async (req, res) => {
    // Gateway calculates tax, applies discounts, validates inventory
    const tax = calculateTax (req.body.items, req.body.shippingAddress);
    const discount = applyPromotions (req.body.items, req.user.id);
    const total = calculateTotal (req.body.items, tax, discount);
    
    // Then calls service
    await orderService.create({...req.body, total});
});
\`\`\`

**Why it's bad**: Logic duplicated if you add another gateway, hard to test

**Better**: Gateway only routes, service handles logic
\`\`\`javascript
// GOOD: Gateway just routes
app.post('/api/orders', async (req, res) => {
    // Service handles all business logic
    const result = await orderService.create (req.body, req.user.id);
    res.json (result);
});
\`\`\`

### ❌ Chatty Gateway

**Problem**: Gateway makes too many backend calls for single request

**Example**:
\`\`\`javascript
// BAD: 10 sequential calls
for (const productId of order.productIds) {
    const product = await productService.get (productId); // 1 call per product
}
\`\`\`

**Better**: Batch requests
\`\`\`javascript
// GOOD: 1 call
const products = await productService.getByIds (order.productIds);
\`\`\`

### ❌ God Gateway

**Problem**: Single gateway for everything (becomes bottleneck)

**Better**: Multiple gateways per domain (e.g., public API gateway, internal gateway, admin gateway)

---

## Security Considerations

### SSL Termination

Gateway terminates SSL, talks to backend over HTTP.

\`\`\`
Client (HTTPS) → Gateway (HTTPS) → Services (HTTP)
\`\`\`

**Advantage**: Services don't need SSL certificates

**Risk**: Unencrypted internal traffic (use VPC/private network)

### API Keys

\`\`\`javascript
app.use((req, res, next) => {
    const apiKey = req.headers['X-API-Key'];
    
    if (!isValidApiKey (apiKey)) {
        return res.status(401).json({error: 'Invalid API key'});
    }
    
    next();
});
\`\`\`

### CORS Handling

\`\`\`javascript
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', 'https://example.com');
    res.header('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE');
    res.header('Access-Control-Allow-Headers', 'Content-Type,Authorization');
    next();
});
\`\`\`

---

## Performance Optimization

### Connection Pooling

Reuse connections to backend services.

\`\`\`javascript
const axios = require('axios');
const http = require('http');
const https = require('https');

const httpAgent = new http.Agent({
    keepAlive: true,
    maxSockets: 50
});

const httpsAgent = new https.Agent({
    keepAlive: true,
    maxSockets: 50
});

const client = axios.create({
    httpAgent,
    httpsAgent
});
\`\`\`

### Request Coalescing

Combine duplicate requests.

\`\`\`javascript
const cache = new Map();

async function getProduct (id) {
    // If request in flight, wait for it
    if (cache.has (id)) {
        return cache.get (id);
    }
    
    // Start new request
    const promise = productService.get (id);
    cache.set (id, promise);
    
    // Clean up when done
    const result = await promise;
    setTimeout(() => cache.delete (id), 1000);
    
    return result;
}
\`\`\`

---

## Decision Framework

**Use API Gateway When**:
✅ Multiple client types (mobile, web, partners)
✅ Need centralized auth/rate limiting
✅ Want to hide backend complexity
✅ Microservices architecture

**Skip API Gateway When**:
❌ Simple monolith application
❌ Only one client type
❌ Team too small to maintain it
❌ Latency-critical (extra hop)

---

## Interview Tips

**Red Flags**:
❌ Saying "API Gateway is required for microservices"
❌ Putting business logic in gateway
❌ Not mentioning trade-offs

**Good Responses**:
✅ Explain core responsibilities (routing, auth, aggregation)
✅ Mention specific tools (Kong, AWS API Gateway, NGINX)
✅ Discuss BFF pattern for different clients
✅ Acknowledge trade-offs (single point of failure, latency)

**Sample Answer**:
*"I'd use an API Gateway as the single entry point for all client requests. It would handle authentication, rate limiting, and request routing. For our mobile app, I'd implement the BFF pattern with request aggregation to minimize round trips. We'd use Kong for its plugin ecosystem or AWS API Gateway if we're fully on AWS. The gateway would be stateless and horizontally scalable. We'd monitor latency carefully since it adds an extra hop."*

---

## Key Takeaways

1. **API Gateway** is a single entry point for all clients
2. **Core functions**: routing, auth, rate limiting, aggregation
3. **BFF pattern**: Dedicated gateway per client type
4. **API composition**: Combine multiple service calls
5. **Don't** put business logic in gateway
6. **Tools**: Kong, AWS API Gateway, NGINX, Traefik
7. **Trade-off**: Simplifies clients but adds latency and single point of failure
8. **Scale horizontally** and use health checks`,
};
