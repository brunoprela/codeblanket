/**
 * Microservices Anti-Patterns Section
 */

export const microservicesantipatternsSection = {
  id: 'microservices-anti-patterns',
  title: 'Microservices Anti-Patterns',
  content: `Learning from mistakes is crucial. These anti-patterns represent common pitfalls when adopting microservices.

## 1. Distributed Monolith

**Problem**: Microservices that are tightly coupled, defeating the purpose.

**Symptoms**:
- Services must be deployed together
- Changes in one service break others
- Shared database
- Synchronous chains of calls
- No independent deployment

**Example**:
\`\`\`
Order Service → Payment Service → Fraud Service → Bank Service
(All must be deployed together, changes ripple through all services)
\`\`\`

**Why it's bad**: All complexity of microservices (distributed, network calls) without benefits (independent deployment, scaling).

**Solution**:
- Database per service
- Asynchronous communication
- Backward-compatible APIs
- Independent deployment pipelines

**Real example**: Teams split monolith by technical layers (UI layer service, business logic service, data access service) instead of business domains → distributed monolith.

---

## 2. Nanoservices

**Problem**: Services that are too small (excessive granularity).

**Symptoms**:
- Service has 1-2 endpoints
- Service called by only one other service
- Too much network overhead
- Debugging nightmare (request touches 20+ services)

**Example**:
\`\`\`
User Service
├─ GetUser Service
├─ UpdateUser Service
├─ DeleteUser Service
└─ CreateUser Service

Why? These should be ONE service!
\`\`\`

**Why it's bad**: Network latency, operational overhead (20+ services to deploy/monitor), no clear boundaries.

**Solution**:
- Follow business capabilities (DDD)
- Service should have cohesive responsibility
- Rule of thumb: 5-15 services for most systems (not 100+)

**Quote**: *"If services are too fine-grained, you'll end up with a distributed mess." - Sam Newman*

---

## 3. Shared Database

**Problem**: Multiple services accessing same database.

**Example**:
\`\`\`
Order Service  ─┐
User Service   ─┼─→ Shared Database
Product Service─┘
\`\`\`

**Why it's bad**:
- Tight coupling (schema change breaks multiple services)
- Can't scale independently
- Can't choose different database types
- No clear ownership
- Transaction boundaries unclear

**Solution**:
- Database per service
- Services communicate via APIs or events
- Data duplication is okay (trade-off for independence)

**Exception**: Migration from monolith (Strangler Fig pattern starts with shared DB)

---

## 4. Lack of API Versioning

**Problem**: Breaking API changes without versioning.

**Bad**:
\`\`\`javascript
// Version 1
POST /orders
{
    "userId": "123",
    "productId": "456"
}

// Version 2 (BREAKING CHANGE)
POST /orders
{
    "userId": "123",
    "items": [{ "productId": "456", "quantity": 1 }]  // Changed format!
}
\`\`\`

Old clients break when deployed.

**Solution**:
\`\`\`javascript
// Version 1 (still works)
POST /v1/orders

// Version 2 (new format)
POST /v2/orders

// OR use header versioning
POST /orders
Headers: Accept: application/vnd.api.v2+json
\`\`\`

**Best practice**: Support old version for 6-12 months while clients migrate.

---

## 5. Synchronous Coupling Chains

**Problem**: Long chains of synchronous calls.

**Example**:
\`\`\`
API Gateway → Order Service → Payment Service → Fraud Service → Bank API
              ↓
         Inventory Service → Warehouse API
              ↓
         Shipping Service → Carrier API

Total latency: 50ms + 100ms + 200ms + 150ms + 80ms + 120ms = 700ms
\`\`\`

**Why it's bad**:
- High latency (sum of all latencies)
- Low availability (product of all availabilities)
- Cascading failures

**Availability math**:
\`\`\`
If each service has 99.9% uptime:
6 services in chain = 0.999^6 = 99.4% uptime

43 minutes downtime/month → 4.3 hours downtime/month!
\`\`\`

**Solution**:
- Asynchronous communication where possible
- Circuit breakers
- Cache responses
- Parallelize calls

**Better**:
\`\`\`javascript
// Parallel calls
const [payment, inventory, shipping] = await Promise.all([
    paymentService.authorize (order),
    inventoryService.reserve (order.items),
    shippingService.calculate (order.address)
]);
\`\`\`

---

## 6. No Monitoring/Observability

**Problem**: Deploying microservices without proper monitoring.

**Symptoms**:
- Don't know which service is slow
- Can't trace requests across services
- No alerts when services fail
- Debugging takes hours

**Solution**:
- Metrics (Prometheus + Grafana)
- Distributed tracing (Jaeger)
- Centralized logging (ELK)
- Correlation IDs
- Health checks
- Dashboards

**Real story**: Company deployed 20 microservices, production issue took 8 hours to debug because no distributed tracing. After adding Jaeger, similar issues resolved in minutes.

---

## 7. Trying to Do ACID Transactions

**Problem**: Distributed transactions across services.

**Bad**:
\`\`\`javascript
BEGIN TRANSACTION;
  await orderService.createOrder (order);
  await paymentService.charge (payment);
  await inventoryService.reserve (items);
COMMIT;
\`\`\`

**Why it's bad**: Doesn't work! Each service has its own database. 2PC (Two-Phase Commit) is problematic in microservices.

**Solution**: Saga pattern with eventual consistency

\`\`\`javascript
// Choreography or Orchestration
1. Create order (local transaction)
2. Publish OrderCreated event
3. Payment service charges (local transaction)
4. Publish PaymentCompleted event
5. Inventory service reserves (local transaction)

// If payment fails, compensating transactions rollback
\`\`\`

---

## 8. Microservices First

**Problem**: Starting greenfield project with microservices.

**Why it's bad**:
- Don't know boundaries yet
- Over-engineering
- More complexity upfront
- Team may lack expertise

**Better**: Start with modular monolith

**Evolution**:
\`\`\`
1. Modular Monolith (clear module boundaries)
   ↓
2. Extract most painful service (e.g., report generation)
   ↓
3. Gradually extract more services (Strangler Fig)
   ↓
4. Mature microservices architecture
\`\`\`

**Quote**: *"Almost every successful microservices story has started with a monolith that became too big and was broken up." - Martin Fowler*

---

## 9. Ignoring Conway\'s Law

**Problem**: Microservices architecture doesn't match team structure.

**Conway's Law**: *"Organizations design systems that mirror their communication structure."*

**Bad**:
\`\`\`
Architecture: Order Service, Payment Service, Shipping Service
Team structure: Frontend Team, Backend Team, DBA Team

Result: Each service touched by all teams → coordination nightmare
\`\`\`

**Good**:
\`\`\`
Team 1: Owns Order Service (frontend, backend, database)
Team 2: Owns Payment Service (frontend, backend, database)
Team 3: Owns Shipping Service (frontend, backend, database)

Result: Teams can deploy independently
\`\`\`

**Solution**: Align team ownership with service boundaries

---

## 10. Hardcoded Config

**Problem**: Configuration in code or environment variables in Dockerfile.

**Bad**:
\`\`\`javascript
// Hardcoded
const PAYMENT_SERVICE_URL = 'http://payment-service:8080';

// Or in Dockerfile
ENV PAYMENT_SERVICE_URL=http://payment-service:8080
\`\`\`

**Why it's bad**: Can't change config without rebuilding image, different per environment.

**Solution**: ConfigMaps, Secrets, or config service

\`\`\`yaml
# Kubernetes ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: order-service-config
data:
  PAYMENT_SERVICE_URL: "http://payment-service"
  DATABASE_HOST: "postgres.prod.example.com"
\`\`\`

---

## 11. Smart Gateway, Dumb Services

**Problem**: Business logic in API Gateway.

**Bad**:
\`\`\`javascript
// API Gateway
app.post('/orders', async (req, res) => {
    // Business logic in gateway!
    const tax = calculateTax (req.body.items);
    const discount = applyPromotions (req.body.items);
    const total = calculateTotal (req.body.items, tax, discount);
    
    // Then call service
    await orderService.create({ ...req.body, total });
});
\`\`\`

**Why it's bad**: Logic duplicated if you add another gateway, hard to test, gateway becomes bottleneck.

**Solution**: Gateway only routes/authenticates

\`\`\`javascript
// API Gateway (routing only)
app.post('/orders', async (req, res) => {
    const result = await orderService.create (req.body);
    res.json (result);
});

// Order Service (business logic)
async function create (orderData) {
    const tax = calculateTax (orderData.items);
    const discount = applyPromotions (orderData.items);
    const total = calculateTotal (orderData.items, tax, discount);
    // ...
}
\`\`\`

---

## 12. Not Automating Deployment

**Problem**: Manual deployment of microservices.

**Why it's bad**: With 10+ services, manual deployment is error-prone, slow, doesn't scale.

**Solution**: CI/CD pipeline

\`\`\`yaml
# Example
git push → Run tests → Build container → Deploy to staging → Deploy to prod
\`\`\`

**Essential**:
- Automated tests
- Containerization (Docker)
- Orchestration (Kubernetes)
- Monitoring
- Rollback capability

---

## How to Avoid Anti-Patterns

**Checklist**:
- ✅ **Start with monolith** (or modular monolith)
- ✅ **Database per service** (no shared databases)
- ✅ **Clear service boundaries** (business capabilities, not technical layers)
- ✅ **API versioning** (backward compatibility)
- ✅ **Async communication** where possible
- ✅ **Comprehensive monitoring** (metrics, logs, traces)
- ✅ **Saga pattern** (not distributed transactions)
- ✅ **Team ownership** aligned with services (Conway\'s Law)
- ✅ **Automated CI/CD** (cannot do microservices manually)
- ✅ **Service mesh** for cross-cutting concerns (at scale)

---

## Interview Tips

**Red Flags**:
❌ Jumping straight to microservices
❌ Not mentioning trade-offs
❌ Ignoring operational complexity

**Good Responses**:
✅ Acknowledge anti-patterns
✅ Explain trade-offs (microservices aren't always better)
✅ Start with monolith recommendation
✅ Mention specific anti-patterns (distributed monolith, nanoservices)

**Sample Answer**:
*"Common microservices anti-patterns: (1) Distributed monolith - services tightly coupled, must deploy together, negating benefits, (2) Nanoservices - too granular, excessive network overhead, (3) Shared database - defeats purpose of independence, (4) Synchronous coupling chains - high latency and cascading failures, (5) No monitoring - can't debug distributed systems. To avoid: start with modular monolith, decompose by business domains (DDD), database per service, async communication, comprehensive observability, automated CI/CD. Microservices add complexity - only worth it at scale with mature team."*

---

## Key Takeaways

1. **Distributed monolith**: Worst of both worlds (complexity without benefits)
2. **Nanoservices**: Too small is as bad as too big
3. **Start with monolith**: Extract services gradually
4. **Database per service**: Mandatory for true independence
5. **Async communication**: Reduce coupling and cascading failures
6. **Monitor everything**: Can't operate microservices blind
7. **No ACID across services**: Use Saga pattern
8. **Align teams with services**: Conway\'s Law
9. **Automate deployment**: Essential at scale
10. **Microservices aren't always better**: Trade-offs matter`,
};
