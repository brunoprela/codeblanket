/**
 * API Gateway Section
 */

export const apigatewaySection = {
  id: 'api-gateway',
  title: 'API Gateway',
  content: `An API Gateway is a server that acts as a single entry point for all client requests, routing them to appropriate backend services while providing cross-cutting functionality like authentication, rate limiting, and monitoring.

## What is an API Gateway?

**Definition**: An API Gateway sits between clients and backend microservices, providing a unified interface and handling common tasks centrally.

### **Why Use an API Gateway?**

**Without API Gateway:**
- Clients call microservices directly (tight coupling)
- Each service implements authentication separately (duplication)
- No centralized rate limiting or logging
- Complex client logic (knows all service endpoints)
- Security risks (services exposed directly)

**With API Gateway:**
- Single entry point for all requests
- Centralized authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- Load balancing and service discovery
- Simplified client code

**Real-world**: Netflix uses Zuul API Gateway to handle billions of requests per day.

---

## Core Responsibilities

### **1. Request Routing**

Route requests to appropriate backend services based on URL path.

**Example:**
- GET /api/users/123 → User Service
- GET /api/orders/456 → Order Service
- GET /api/products/789 → Product Service

**Benefits:** Clients only know one endpoint (gateway), not all microservices.

---

### **2. Authentication & Authorization**

Verify user identity and permissions before forwarding requests.

**Flow:**1. Client sends request with JWT token
2. Gateway validates token
3. If valid: Extract user info, forward to backend
4. If invalid: Return 401 Unauthorized

**Benefits:** Authentication logic centralized (not in every microservice).

---

### **3. Rate Limiting & Throttling**

Limit number of requests per user/API key to prevent abuse.

**Example:**
- Free tier: 1000 requests/day
- Premium tier: 100,000 requests/day

**Implementation:** Track request count in Redis, reject if limit exceeded.

**Benefits:** Protect backend from overload, enforce fair usage.

---

### **4. Request/Response Transformation**

Modify requests or responses (format conversion, header manipulation).

**Example:**
- Client sends XML → Gateway converts to JSON for backend
- Backend returns v2 API → Gateway adapts to v1 for legacy clients

**Benefits:** Backward compatibility, protocol translation.

---

### **5. Load Balancing**

Distribute requests across multiple backend instances.

**Example:**
- User Service has 5 instances
- Gateway load balances requests across all 5

**Benefits:** Horizontal scaling, high availability.

---

### **6. Caching**

Cache responses to reduce backend load.

**Example:**
- GET /api/products/123 → Cache for 60 seconds
- Subsequent requests served from cache

**Benefits:** Lower latency, reduced backend load.

---

### **7. Logging & Monitoring**

Centralized logging for all API requests.

**Metrics:**
- Request count per endpoint
- Response time (latency)
- Error rate (4xx, 5xx)
- Top users by request count

**Benefits:** Single place to monitor all API traffic.

---

## API Gateway vs Load Balancer

**Comparison:**

**Load Balancer:**
- Layer 4 or Layer 7
- Simple traffic distribution
- No business logic
- Routes to same service type

**API Gateway:**
- Layer 7 only (HTTP/HTTPS)
- Intelligent routing (different services per path)
- Business logic (auth, rate limiting, transformation)
- Routes to different services

**When to use both:** Gateway handles application logic, Load Balancer handles traffic distribution.

---

## API Gateway Patterns

### **1. Backend for Frontend (BFF)**

Separate gateway per client type (web, mobile, IoT).

**Architecture:**
- Web BFF: Optimized for web browsers
- Mobile BFF: Optimized for mobile apps (smaller payloads)
- IoT BFF: Optimized for IoT devices (minimal data)

**Benefits:** Tailored responses per client, better performance.

---

### **2. Aggregation**

Gateway fetches data from multiple services, aggregates, returns to client.

**Example:**
- Client requests user profile
- Gateway calls: User Service, Order Service, Recommendation Service
- Gateway aggregates responses into single response

**Benefits:** Reduced client-side complexity, fewer round trips.

---

### **3. GraphQL Gateway**

Gateway exposes GraphQL API, translates to REST calls for backends.

**Benefits:** Clients query exactly what they need, no over-fetching.

---

## Popular API Gateway Solutions

### **AWS API Gateway**
- Fully managed (serverless)
- Integrates with Lambda, EC2, other AWS services
- Built-in authentication (IAM, Cognito)
- Pay-per-request pricing

**Best for:** AWS-native applications

---

### **Kong**
- Open-source (also has enterprise version)
- Plugin architecture (auth, rate limiting, logging)
- Built on NGINX (high performance)
- Self-hosted or cloud-managed

**Best for:** On-premise or hybrid deployments

---

### **NGINX**
- Not a dedicated API gateway, but can be configured as one
- High performance (handles 10K+ requests/sec)
- Reverse proxy + load balancer + API gateway

**Best for:** Simple use cases, high performance needs

---

### **Apigee (Google)**
- Enterprise API management platform
- Advanced analytics and monetization
- Developer portal
- Expensive

**Best for:** Large enterprises, API monetization

---

## API Gateway Challenges

### **1. Single Point of Failure**

**Problem:** If gateway down, all services unavailable.

**Solution:** 
- Deploy multiple gateway instances behind load balancer
- Health checks and auto-scaling
- Circuit breaker patterns

---

### **2. Performance Bottleneck**

**Problem:** All traffic goes through gateway (latency increase).

**Solution:**
- Scale gateway horizontally (add more instances)
- Optimize gateway code (avoid heavy processing)
- Cache aggressively

---

### **3. Complexity**

**Problem:** Gateway becomes complex with too many responsibilities.

**Solution:**
- Keep gateway thin (essential logic only)
- Push business logic to services
- Use plugins/middleware for modularity

---

## Security Best Practices

**1. Authentication:** Verify user identity (JWT, OAuth)

**2. Authorization:** Check user permissions (RBAC, ABAC)

**3. Rate Limiting:** Prevent DDoS and abuse

**4. Input Validation:** Sanitize requests (prevent injection attacks)

**5. HTTPS Only:** Encrypt all traffic (TLS/SSL)

**6. IP Whitelisting:** Restrict access by IP (for admin APIs)

**7. API Keys:** Require keys for programmatic access

---

## Real-World Examples

### **Netflix (Zuul)**
- Handles billions of requests/day
- Dynamic routing (A/B testing, canary deployments)
- Resilience patterns (circuit breaker, retries)

### **Uber**
- API Gateway for mobile apps
- Authentication and rate limiting
- Request logging and analytics

### **Shopify**
- API Gateway for merchants
- Rate limiting per shop
- Request transformation (API versioning)

---

## Key Takeaways

1. **API Gateway = single entry point** for all client requests
2. **Centralized cross-cutting concerns:** auth, rate limiting, logging
3. **Request routing:** Route to appropriate backend service by URL path
4. **Not a replacement for load balancer:** Use both together
5. **Security:** Authenticate, authorize, rate limit at gateway
6. **High availability:** Deploy multiple gateway instances`,
};
