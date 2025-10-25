/**
 * Inter-Service Communication Section
 */

export const interservicecommunicationSection = {
  id: 'inter-service-communication',
  title: 'Inter-Service Communication',
  content: `One of the biggest challenges in microservices is how services communicate with each other. The choices you make here fundamentally impact performance, reliability, and complexity.

## Communication Patterns Overview

Microservices can communicate in two fundamental ways:

**1. Synchronous Communication**
- Request/response
- Client waits for response
- Examples: HTTP/REST, gRPC

**2. Asynchronous Communication**
- Fire-and-forget or publish/subscribe
- Client doesn't wait
- Examples: Message queues, event streaming

---

## Synchronous Communication

### HTTP/REST

**Most common pattern** for microservices communication.

**How it works**:
\`\`\`
Order Service ---HTTP POST---> Payment Service
              <--- Response ---
\`\`\`

**Advantages**:
✅ Simple and intuitive
✅ Wide tooling support
✅ Easy to debug (request/response visible)
✅ Strong typing with OpenAPI/Swagger
✅ Immediate feedback

**Disadvantages**:
❌ Tight coupling (Order Service blocks waiting for Payment)
❌ Cascading failures (if Payment down, Order fails)
❌ Synchronous chain latency adds up
❌ Less resilient to network issues

**Example: Order Checkout**
\`\`\`javascript
// Order Service
async function createOrder (orderData) {
  // 1. Validate inventory (synchronous call)
  const inventory = await inventoryService.checkStock (orderData.items);
  if (!inventory.available) {
    throw new Error('Out of stock');
  }
  
  // 2. Process payment (synchronous call)
  const payment = await paymentService.charge({
    amount: orderData.total,
    customerId: orderData.customerId
  });
  
  if (!payment.success) {
    throw new Error('Payment failed');
  }
  
  // 3. Create order
  const order = await db.orders.create (orderData);
  
  // 4. Send notification (synchronous call)
  await notificationService.sendEmail({
    to: orderData.customerEmail,
    template: 'order_confirmation',
    data: order
  });
  
  return order;
}
\`\`\`

**Problems with this approach**:
- If notification service is down, entire order creation fails
- If payment service is slow, user waits
- If any service times out, need retry logic
- Network latency: 3 services × 10ms = 30ms minimum

**When to use REST**:
✅ Need immediate response (user waiting)
✅ Simple request/response
✅ CRUD operations
✅ Low latency requirements
✅ Small number of services

---

### gRPC

**Modern RPC framework** using HTTP/2 and Protocol Buffers.

**Advantages over REST**:
✅ **Performance**: Binary protocol (protobuf) faster than JSON
✅ **Streaming**: Bidirectional streaming support
✅ **Type safety**: Strong schema enforcement
✅ **Efficient**: HTTP/2 multiplexing

**Disadvantages**:
❌ Less human-readable (binary)
❌ Steeper learning curve
❌ Browser support limited
❌ Debugging harder

**Example Service Definition**:
\`\`\`protobuf
// payment.proto
service PaymentService {
  rpc ProcessPayment(PaymentRequest) returns (PaymentResponse);
  rpc GetPaymentStatus(PaymentId) returns (PaymentStatus);
  rpc RefundPayment(RefundRequest) returns (RefundResponse);
}

message PaymentRequest {
  string customer_id = 1;
  double amount = 2;
  string currency = 3;
  string payment_method = 4;
}

message PaymentResponse {
  string payment_id = 1;
  PaymentStatus status = 2;
  string message = 3;
}
\`\`\`

**When to use gRPC**:
✅ Internal microservices (not public API)
✅ High-performance requirements
✅ Streaming data (real-time updates)
✅ Strong typing needed
✅ Polyglot environments (generate client libraries)

**Real-world**: Google uses gRPC extensively for internal services.

---

## Asynchronous Communication

### Message Queues (Point-to-Point)

**Pattern**: Producer sends message to queue, single consumer processes it.

\`\`\`
Producer → [Queue] → Consumer
\`\`\`

**Example: Order Processing**
\`\`\`javascript
// Order Service (Producer)
async function createOrder (orderData) {
  // 1. Create order immediately
  const order = await db.orders.create (orderData);
  
  // 2. Publish to queue (non-blocking)
  await queue.publish('order.created', {
    orderId: order.id,
    customerId: order.customerId,
    items: order.items,
    total: order.total
  });
  
  // 3. Return immediately (don't wait for processing)
  return order;
}

// Payment Service (Consumer)
queue.subscribe('order.created', async (message) => {
  try {
    await processPayment (message.orderId, message.total);
  } catch (error) {
    // Retry logic handled by queue
    throw error;
  }
});
\`\`\`

**Advantages**:
✅ **Decoupling**: Producer doesn't know/care about consumer
✅ **Async**: Producer doesn't wait
✅ **Reliability**: Message persisted, retries handled
✅ **Load leveling**: Consumers process at their own pace
✅ **Fault tolerance**: If consumer down, messages queued

**Disadvantages**:
❌ **No immediate response**: Producer doesn't know result
❌ **Complexity**: Need to handle eventual consistency
❌ **Debugging**: Harder to trace async flows
❌ **Message ordering**: Can be challenging
❌ **Duplicate processing**: Need idempotency

**Technologies**: RabbitMQ, AWS SQS, Azure Service Bus

**When to use**:
✅ Operations can be async (emails, notifications)
✅ Load leveling needed (burst traffic)
✅ Retry logic important
✅ Decoupling services

---

### Pub/Sub (Publish/Subscribe)

**Pattern**: Producer publishes event, multiple consumers can subscribe.

\`\`\`
Producer → [Topic] → Consumer 1
                   → Consumer 2
                   → Consumer 3
\`\`\`

**Example: Order Placed Event**
\`\`\`javascript
// Order Service (Publisher)
async function createOrder (orderData) {
  const order = await db.orders.create (orderData);
  
  // Publish event to topic (multiple subscribers)
  await eventBus.publish('order.placed', {
    orderId: order.id,
    customerId: order.customerId,
    items: order.items,
    total: order.total,
    timestamp: new Date()
  });
  
  return order;
}

// Payment Service (Subscriber 1)
eventBus.subscribe('order.placed', async (event) => {
  await processPayment (event.orderId, event.total);
});

// Inventory Service (Subscriber 2)
eventBus.subscribe('order.placed', async (event) => {
  await reserveInventory (event.items);
});

// Notification Service (Subscriber 3)
eventBus.subscribe('order.placed', async (event) => {
  await sendOrderConfirmation (event.customerId, event.orderId);
});

// Analytics Service (Subscriber 4)
eventBus.subscribe('order.placed', async (event) => {
  await trackOrderEvent (event);
});
\`\`\`

**Advantages**:
✅ **Loose coupling**: Publisher doesn't know subscribers
✅ **Extensibility**: Add new subscribers without changing publisher
✅ **Parallel processing**: All subscribers process simultaneously
✅ **Event-driven architecture**: Natural fit for domain events

**Disadvantages**:
❌ **No response**: Publisher doesn't know if subscribers succeeded
❌ **Eventual consistency**: Data may be temporarily inconsistent
❌ **Debugging**: Hard to trace event flows
❌ **Message ordering**: Subscribers may process out of order

**Technologies**: Apache Kafka, AWS SNS, Google Pub/Sub, RabbitMQ (with topic exchanges)

**When to use**:
✅ Multiple services need same event
✅ Event-driven architecture
✅ Auditability (event log)
✅ Decoupling critical

---

## Hybrid Pattern: Synchronous + Asynchronous

**Best practice**: Mix both patterns based on requirements.

**Example: E-commerce Checkout**
\`\`\`javascript
async function checkout (orderData) {
  // SYNCHRONOUS: Need immediate result
  // 1. Validate inventory (must succeed before proceeding)
  const inventoryCheck = await inventoryService.reserve (orderData.items);
  if (!inventoryCheck.success) {
    return { error: 'Out of stock' };
  }
  
  // 2. Process payment (must succeed before confirming)
  const payment = await paymentService.charge (orderData.amount);
  if (!payment.success) {
    // Compensate: release inventory
    await inventoryService.release (orderData.items);
    return { error: 'Payment failed' };
  }
  
  // 3. Create order
  const order = await db.orders.create({
    ...orderData,
    status: 'confirmed',
    paymentId: payment.id
  });
  
  // ASYNCHRONOUS: Fire and forget
  // 4. Send notification (don't wait)
  await queue.publish('order.confirmed', order);
  
  // 5. Update analytics (don't wait)
  await eventBus.publish('order.placed', order);
  
  // 6. Trigger fulfillment (don't wait)
  await queue.publish('order.fulfillment', order);
  
  // Return immediately to user
  return { success: true, orderId: order.id };
}
\`\`\`

**Decision Tree**:
\`\`\`
Does user need immediate response?
├─ Yes → Synchronous (REST/gRPC)
└─ No → Asynchronous (Queue/Event)

Is operation critical to request success?
├─ Yes → Synchronous (payment, inventory check)
└─ No → Asynchronous (notification, analytics)

Do multiple services need this data?
├─ Yes → Pub/Sub (event)
└─ No → Queue (single consumer)
\`\`\`

---

## Service Mesh

**Problem**: With many microservices, managing communication becomes complex:
- Service discovery
- Load balancing
- Retries and timeouts
- Circuit breakers
- Mutual TLS
- Distributed tracing

**Solution: Service Mesh** (like Istio, Linkerd)

**Architecture**:
\`\`\`
Service A → [Sidecar Proxy A] → [Sidecar Proxy B] → Service B
               ↓                      ↓
           Control Plane (manages all proxies)
\`\`\`

**Key Features**:

**1. Traffic Management**
- Load balancing (round robin, least request)
- Retry logic (automatic retries)
- Timeouts (prevent hanging requests)
- Circuit breaking (fail fast when service unhealthy)

**2. Security**
- Mutual TLS (mTLS) - automatic encryption
- Authentication/authorization between services
- Certificate management

**3. Observability**
- Automatic distributed tracing
- Metrics collection (latency, error rate)
- Service topology mapping

**Example: Istio Configuration**
\`\`\`yaml
# Retry configuration
apiVersion: networking.istio.io/v1alpha3
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
    retries:
      attempts: 3
      perTryTimeout: 2s
      retryOn: 5xx,reset,connect-failure
    timeout: 10s
\`\`\`

**When to use Service Mesh**:
✅ Many microservices (> 10)
✅ Need sophisticated traffic management
✅ Security requirements (mTLS)
✅ Polyglot environment
✅ Using Kubernetes

**When NOT to use**:
❌ Few services (< 5) - overhead not worth it
❌ Team lacks operational maturity
❌ Simple communication patterns

---

## API Contracts and Backward Compatibility

**Problem**: Service A depends on Service B's API. Service B changes API. Service A breaks.

**Solution: Versioning and Backward Compatibility**

### Strategy 1: URL Versioning
\`\`\`
/v1/payments
/v2/payments
\`\`\`

**Pros**: Clear, explicit
**Cons**: Must maintain multiple versions

### Strategy 2: Header Versioning
\`\`\`
GET /payments
Header: API-Version: 2
\`\`\`

**Pros**: Single endpoint
**Cons**: Less visible, harder to route

### Strategy 3: Backward-Compatible Changes
\`\`\`javascript
// Version 1: Original API
{
  "amount": 100.00,
  "currency": "USD"
}

// Version 2: Add fields (backward compatible)
{
  "amount": 100.00,
  "currency": "USD",
  "tax": 10.00,      // NEW: optional field
  "discount": 5.00   // NEW: optional field
}

// Consumers on v1 API ignore new fields (works fine)
// Consumers on v2 API can use new fields
\`\`\`

**Backward-Compatible Changes**:
✅ Add optional fields
✅ Add new endpoints
✅ Deprecate (but don't remove) fields
✅ Make required fields optional

**Breaking Changes** (require versioning):
❌ Remove fields
❌ Rename fields
❌ Change field types
❌ Change validation rules
❌ Remove endpoints

---

## Communication Anti-Patterns

### Anti-Pattern 1: Chatty Services
\`\`\`
❌ API Gateway → User Service
               → Product Service (needs user data)
                 → User Service (again!)
                   → Order Service (needs user data)
                     → User Service (again!)
\`\`\`

**Problem**: 3 calls to User Service for same data. Network overhead.

**Solution**: Pass data, don't fetch repeatedly
\`\`\`
✅ API Gateway → User Service (get user)
               → Product Service (pass user data)
                 → Order Service (pass user data)
\`\`\`

Or use API Gateway to aggregate:
\`\`\`
✅ API Gateway → User, Product, Order (parallel calls)
               → Aggregate response
               → Return to client
\`\`\`

### Anti-Pattern 2: Synchronous Chains
\`\`\`
❌ Service A → Service B → Service C → Service D
\`\`\`

**Problem**: 
- If any service down, entire chain fails
- Latency adds up (4 × 10ms = 40ms)
- Tight coupling

**Solution**: Use events
\`\`\`
✅ Service A → Publish Event
             ↓
    [Event Bus]
             ↓ (parallel subscriptions)
    Service B, Service C, Service D
\`\`\`

### Anti-Pattern 3: Distributed Monolith via Communication
\`\`\`
❌ Every operation requires calling 5+ services
   Services can't operate independently
   Synchronous coupling everywhere
\`\`\`

**Solution**: Redesign service boundaries (see Section 2)

---

## Real-World Example: Netflix

**Netflix Communication Patterns**:

1. **User requests video**
   - API Gateway → User Service (REST)
   - API Gateway → Video Service (REST)
   - Response to user (synchronous)

2. **Video playback starts**
   - Video Service → Publish event: "PlaybackStarted"
   - Analytics Service subscribes (for reporting)
   - Recommendation Service subscribes (for ML)
   - Billing Service subscribes (for usage tracking)
   (asynchronous, pub/sub)

3. **Encoding new video**
   - Upload Service → Queue: "VideoEncoding"
   - Encoding Workers pull from queue (async, load leveling)
   - Progress updates via events

**Key Insight**: Netflix uses BOTH sync and async based on requirements.

---

## Decision Framework

**Use Synchronous (REST/gRPC) When**:
✅ Need immediate response
✅ User is waiting
✅ Operation must complete before proceeding
✅ Simple request/response
✅ Low latency critical

**Use Asynchronous (Queue) When**:
✅ Operation can be delayed
✅ Load leveling needed
✅ Retry logic important
✅ Single consumer

**Use Pub/Sub (Events) When**:
✅ Multiple services need same data
✅ Audit trail needed
✅ Event-driven architecture
✅ Loose coupling critical

**Use Service Mesh When**:
✅ Many services (> 10)
✅ Need traffic management (retries, circuit breakers)
✅ Security requirements (mTLS)
✅ Kubernetes-based

---

## Interview Tips

### Red Flags:
❌ "Always use REST" or "Always use events"
❌ No mention of trade-offs
❌ Ignoring latency implications
❌ Not discussing failure scenarios

### Good Responses:
✅ Explain both sync and async patterns
✅ Justify choice based on requirements
✅ Discuss retry logic and failure handling
✅ Mention backward compatibility
✅ Reference real-world examples (Netflix, Uber)

### Sample Answer:
*"For the checkout flow, I'd use synchronous REST calls for inventory check and payment processing since we need immediate results and these are critical for order confirmation. However, for notifications, analytics, and fulfillment, I'd use asynchronous message queues since users don't need to wait for these to complete. I'd implement retry logic with exponential backoff for async operations and circuit breakers for synchronous calls to prevent cascading failures. For service-to-service authentication, I'd use mutual TLS via a service mesh like Istio."*

---

## Key Takeaways

1. **No one-size-fits-all**: Mix sync and async based on requirements
2. **Synchronous**: Simple but tight coupling, use when response needed
3. **Asynchronous**: Complex but resilient, use when operations can be delayed
4. **Pub/Sub**: Best for events that multiple services need
5. **Service Mesh**: Solves cross-cutting concerns (security, observability, traffic management)
6. **Backward compatibility**: Critical for independent deployment
7. **Avoid chatty services**: Minimize network calls
8. **Design for failure**: Timeouts, retries, circuit breakers essential

The choice of communication pattern fundamentally impacts your microservices architecture's performance, reliability, and operational complexity.`,
};
