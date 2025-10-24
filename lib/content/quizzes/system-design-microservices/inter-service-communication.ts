/**
 * Quiz questions for Inter-Service Communication section
 */

export const interservicecommunicationQuiz = [
  {
    id: 'q1-communication',
    question:
      'Design the communication strategy for an e-commerce checkout flow involving Order Service, Inventory Service, Payment Service, and Notification Service. Explain which communications should be synchronous vs asynchronous, how you would handle failures, implement idempotency, and ensure data consistency. Include specific examples of message formats and error handling.',
    sampleAnswer: `**E-Commerce Checkout Communication Strategy**

**1. Service Flow Overview**

\`\`\`
User Checkout
    ↓
Order Service (orchestrator)
    ├→ Inventory Service (sync)
    ├→ Payment Service (sync)
    ├→ Fulfillment Service (async)
    ├→ Notification Service (async)
    └→ Analytics Service (async)
\`\`\`

**2. Synchronous Communication**

**Inventory Check** (Synchronous - REST/gRPC):

\`\`\`javascript
// Order Service → Inventory Service
async function reserveInventory(orderId, items) {
  try {
    const response = await httpClient.post(
      'http://inventory-service/api/reserve',
      {
        orderId: orderId,
        items: items.map(item => ({
          productId: item.productId,
          quantity: item.quantity
        })),
        reservationTimeout: 300 // 5 minutes
      },
      {
        timeout: 2000, // 2 second timeout
        headers: {
          'X-Request-ID': generateRequestId(),
          'X-Idempotency-Key': \`reserve-\${orderId}\`
        }
      }
    );
    
    return {
      success: true,
      reservationId: response.data.reservationId
    };
    
  } catch (error) {
    if (error.code === 'OUT_OF_STOCK') {
      throw new OutOfStockError(error.message);
    } else if (error.code === 'TIMEOUT') {
      // Retry once
      return await reserveInventory(orderId, items);
    }
    throw error;
  }
}
\`\`\`

**Why Synchronous**: Need immediate confirmation before proceeding to payment.

**Payment Processing** (Synchronous - REST):

\`\`\`javascript
async function processPayment(orderId, paymentDetails) {
  const idempotencyKey = \`payment-\${orderId}\`;
  
  try {
    const response = await httpClient.post(
      'http://payment-service/api/charge',
      {
        orderId: orderId,
        amount: paymentDetails.amount,
        currency: 'USD',
        paymentMethod: paymentDetails.paymentMethodId
      },
      {
        timeout: 10000, // 10 seconds (payment can take time)
        headers: {
          'X-Request-ID': generateRequestId(),
          'X-Idempotency-Key': idempotencyKey
        }
      }
    );
    
    return {
      success: true,
      transactionId: response.data.transactionId
    };
    
  } catch (error) {
    if (error.code === 'PAYMENT_DECLINED') {
      // Release inventory reservation
      await releaseInventory(orderId);
      throw new PaymentDeclinedError();
    }
    throw error;
  }
}
\`\`\`

**Why Synchronous**: User must know immediately if payment succeeded.

**3. Asynchronous Communication**

**Order Confirmation** (Async - Message Queue):

\`\`\`javascript
// Order Service publishes event
async function publishOrderCreatedEvent(order) {
  const event = {
    eventId: generateUUID(),
    eventType: 'OrderCreated',
    timestamp: new Date().toISOString(),
    aggregateId: order.id,
    version: 1,
    data: {
      orderId: order.id,
      userId: order.userId,
      items: order.items,
      totalAmount: order.totalAmount,
      paymentTransactionId: order.transactionId
    }
  };
  
  await messageQueue.publish('orders.created', event, {
    messageId: event.eventId,
    persistent: true
  });
}

// Notification Service subscribes
messageQueue.subscribe('orders.created', async (message) => {
  const order = message.data;
  
  // Send email
  await emailService.send({
    to: order.userEmail,
    subject: 'Order Confirmation',
    body: \`Your order #\${order.orderId} has been confirmed\`
  });
  
  // Acknowledge message
  message.ack();
});
\`\`\`

**Why Asynchronous**: User doesn't need to wait for email to be sent.

**4. Complete Checkout Implementation**

\`\`\`javascript
class OrderService {
  async createOrder(userId, items, paymentDetails) {
    const orderId = generateOrderId();
    let reservationId = null;
    let transactionId = null;
    
    try {
      // Step 1: Reserve inventory (SYNC)
      console.log('Reserving inventory...');
      const inventoryResult = await this.reserveInventory(orderId, items);
      reservationId = inventoryResult.reservationId;
      
      // Step 2: Process payment (SYNC)
      console.log('Processing payment...');
      const paymentResult = await this.processPayment(orderId, paymentDetails);
      transactionId = paymentResult.transactionId;
      
      // Step 3: Confirm inventory (SYNC)
      console.log('Confirming inventory...');
      await this.confirmInventory(reservationId);
      
      // Step 4: Create order in database
      const order = await this.orderRepository.create({
        id: orderId,
        userId: userId,
        items: items,
        status: 'CONFIRMED',
        paymentTransactionId: transactionId,
        createdAt: new Date()
      });
      
      // Step 5: Publish events (ASYNC)
      await this.publishOrderCreatedEvent(order);
      
      return {
        success: true,
        orderId: orderId,
        message: 'Order created successfully'
      };
      
    } catch (error) {
      // Compensating transactions
      console.error('Order creation failed:', error);
      
      if (reservationId && !transactionId) {
        // Payment failed - release inventory
        await this.releaseInventory(reservationId);
      }
      
      if (transactionId) {
        // Rare case: payment succeeded but confirmation failed
        // Mark for manual review
        await this.createManualReviewTask({
          orderId,
          transactionId,
          reservationId,
          error: error.message
        });
      }
      
      throw error;
    }
  }
}
\`\`\`

**5. Idempotency Implementation**

\`\`\`javascript
// Idempotency middleware
async function idempotencyMiddleware(req, res, next) {
  const idempotencyKey = req.headers['x-idempotency-key',];
  
  if (!idempotencyKey) {
    return res.status(400).json({ error: 'Idempotency-Key header required' });
  }
  
  // Check if we've seen this key before
  const cached = await redis.get(\`idempotency:\${idempotencyKey}\`);
  
  if (cached) {
    // Return cached response
    return res.status(200).json(JSON.parse(cached));
  }
  
  // Store original response
  const originalSend = res.send;
  res.send = function(data) {
    // Cache for 24 hours
    redis.setex(\`idempotency:\${idempotencyKey}\`, 86400, data);
    originalSend.call(this, data);
  };
  
  next();
}

app.post('/api/orders', idempotencyMiddleware, async (req, res) => {
  // Order creation logic
});
\`\`\`

**6. Error Handling Strategy**

**Retry Logic with Exponential Backoff**:

\`\`\`javascript
async function retryWithBackoff(fn, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      
      // Retry on transient errors
      if (['TIMEOUT', 'CONNECTION_ERROR', 'SERVICE_UNAVAILABLE',].includes(error.code)) {
        const delay = Math.pow(2, i) * 1000; // 1s, 2s, 4s
        await sleep(delay);
        continue;
      }
      
      // Don't retry on client errors
      throw error;
    }
  }
}
\`\`\`

**Circuit Breaker**:

\`\`\`javascript
class CircuitBreaker {
  constructor(threshold = 5, timeout = 60000) {
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.threshold = threshold;
    this.timeout = timeout;
  }
  
  async execute(fn) {
    if (this.state === 'OPEN') {
      throw new Error('Circuit breaker is OPEN');
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
    this.failureCount = 0;
    this.state = 'CLOSED';
  }
  
  onFailure() {
    this.failureCount++;
    if (this.failureCount >= this.threshold) {
      this.state = 'OPEN';
      setTimeout(() => {
        this.state = 'HALF_OPEN';
        this.failureCount = 0;
      }, this.timeout);
    }
  }
}
\`\`\`

**7. Data Consistency**

**Event-Driven Updates**:

\`\`\`javascript
// Notification Service maintains read model
messageQueue.subscribe('orders.created', async (message) => {
  const order = message.data;
  
  // Update local read model
  await notificationDB.orders.upsert({
    orderId: order.orderId,
    userId: order.userId,
    status: 'CONFIRMED',
    lastUpdated: new Date()
  });
  
  message.ack();
});

messageQueue.subscribe('orders.shipped', async (message) => {
  // Update read model
  await notificationDB.orders.update(
    { orderId: message.data.orderId },
    { status: 'SHIPPED' }
  );
  
  message.ack();
});
\`\`\`

**Key Takeaways**:

1. **Sync for critical path**: Inventory + Payment (user must know result)
2. **Async for non-blocking**: Notifications, analytics, fulfillment
3. **Idempotency**: Use idempotency keys for all mutations
4. **Compensating transactions**: Release inventory if payment fails
5. **Retry with backoff**: Automatic retry for transient errors
6. **Circuit breakers**: Prevent cascading failures
7. **Event sourcing**: Publish events for async consumers
8. **Eventual consistency**: Accept that notifications may be delayed`,
    keyPoints: [
      'Synchronous (REST/gRPC): Inventory check and payment processing (need immediate response)',
      'Asynchronous (message queue): Notifications, analytics, fulfillment (user can wait)',
      'Idempotency keys: Prevent duplicate orders on retry (store in Redis for 24h)',
      'Compensating transactions: Release inventory if payment fails',
      'Circuit breakers + retry with exponential backoff for fault tolerance',
      'Event-driven: Publish OrderCreated event for async processing by multiple services',
    ],
  },
  {
    id: 'q2-communication',
    question:
      'Compare REST, gRPC, and message queues for microservice communication. For each approach, discuss performance characteristics, use cases, error handling, backward compatibility, and operational complexity. Which would you choose for different scenarios and why?',
    sampleAnswer: `**Comprehensive Comparison: REST vs gRPC vs Message Queues**

---

## **1. REST (Representational State Transfer)**

**Protocol**: HTTP/HTTPS with JSON/XML payloads

**Example**:
\`\`\`javascript
// REST API call
const response = await fetch('https://api.example.com/users/123', {
  method: 'GET',
  headers: {
    'Authorization': 'Bearer token',
    'Accept': 'application/json'
  }
});
const user = await response.json();
\`\`\`

### **Performance**

**Throughput**: ~1,000-5,000 requests/second per instance
- Text-based (JSON) → larger payloads
- HTTP/1.1 → one request per connection (unless HTTP/2)
- Parsing overhead for JSON

**Latency**: ~10-50ms typical
- Network + JSON serialization/deserialization

**Bandwidth**: ~500 bytes - 2KB per request (JSON overhead)

### **Use Cases**

✅ **Public APIs** (RESTful, cacheable, curl-testable)
✅ **Browser clients** (native fetch support)
✅ **CRUD operations** (GET, POST, PUT, DELETE map naturally)
✅ **Third-party integrations** (universal support)

### **Error Handling**

\`\`\`javascript
try {
  const response = await fetch('/api/orders', {
    method: 'POST',
    body: JSON.stringify(order)
  });
  
  if (!response.ok) {
    if (response.status === 429) {
      // Rate limited - retry after delay
      throw new RateLimitError();
    } else if (response.status === 503) {
      // Service unavailable - transient error
      throw new TransientError();
    }
  }
  
  return await response.json();
} catch (error) {
  // Handle network errors
  throw error;
}
\`\`\`

### **Backward Compatibility**

✅ Easy with versioning (/api/v1/, /api/v2/)
✅ Additive changes safe (new fields don't break old clients)
❌ Removing fields breaks clients

### **Operational Complexity**

**Low** - Well-understood, simple debugging, standard tooling

---

## **2. gRPC (Google Remote Procedure Call)**

**Protocol**: HTTP/2 with Protocol Buffers (protobuf)

**Example**:
\`\`\`protobuf
// user.proto
service UserService {
  rpc GetUser (GetUserRequest) returns (User);
  rpc StreamUsers (StreamUsersRequest) returns (stream User);
}

message User {
  int64 id = 1;
  string name = 2;
  string email = 3;
}
\`\`\`

\`\`\`javascript
// gRPC client
const client = new UserServiceClient('localhost:50051');

client.getUser({ id: 123 }, (error, user) => {
  if (error) {
    console.error(error);
    return;
  }
  console.log(user);
});
\`\`\`

### **Performance**

**Throughput**: ~10,000-50,000 requests/second per instance
- Binary protocol (protobuf) → ~5-10x smaller than JSON
- HTTP/2 multiplexing → multiple streams per connection
- No parsing overhead

**Latency**: ~1-10ms typical
- Faster serialization than JSON

**Bandwidth**: ~50-200 bytes per request (protobuf efficiency)

### **Use Cases**

✅ **Internal microservices** (high performance needed)
✅ **Real-time streaming** (bi-directional streams)
✅ **Polyglot environments** (protobuf generates code for all languages)
✅ **Mobile backends** (bandwidth efficiency)

### **Error Handling**

\`\`\`javascript
client.getUser({ id: 123 }, (error, user) => {
  if (error) {
    switch (error.code) {
      case grpc.status.NOT_FOUND:
        // User not found
        break;
      case grpc.status.UNAVAILABLE:
        // Service unavailable - retry
        break;
      case grpc.status.DEADLINE_EXCEEDED:
        // Timeout
        break;
      default:
        // Unknown error
    }
  }
});
\`\`\`

### **Backward Compatibility**

✅ Excellent - protobuf designed for evolution
✅ Adding fields safe (field numbers never reused)
✅ Removing fields safe (clients ignore unknown fields)
✅ No versioning needed (schema evolution built-in)

### **Operational Complexity**

**Medium** - Requires protobuf compilation, harder to debug (binary), limited browser support

---

## **3. Message Queues (RabbitMQ, Kafka, SQS)**

**Protocol**: AMQP, Kafka protocol, or cloud-specific

**Example**:
\`\`\`javascript
// Publisher
await messageQueue.publish('orders.created', {
  orderId: '123',
  userId: 'user-456',
  items: [...]
});

// Subscriber
messageQueue.subscribe('orders.created', async (message) => {
  const order = message.data;
  await processOrder(order);
  message.ack();
});
\`\`\`

### **Performance**

**Throughput**: ~10,000-1,000,000 messages/second (Kafka)
- Kafka: highest throughput (log-based)
- RabbitMQ: ~10,000-50,000 messages/second
- SQS: ~3,000 messages/second per queue

**Latency**: ~5-50ms
- Depends on queue implementation
- Kafka: batch processing → higher throughput, slightly higher latency

**Bandwidth**: Variable (depends on message size)

### **Use Cases**

✅ **Asynchronous processing** (don't need immediate response)
✅ **Event-driven architecture** (pub/sub patterns)
✅ **Decoupling services** (temporal decoupling)
✅ **Load leveling** (buffer spikes in traffic)
✅ **Guaranteed delivery** (at-least-once with acknowledgments)

### **Error Handling**

\`\`\`javascript
messageQueue.subscribe('orders.created', async (message) => {
  try {
    await processOrder(message.data);
    message.ack(); // Success - remove from queue
  } catch (error) {
    if (error.isTransient) {
      // Retry after delay
      message.nack({ requeue: true, delay: 5000 });
    } else {
      // Permanent failure - send to dead letter queue
      message.nack({ requeue: false });
      await deadLetterQueue.publish('orders.failed', {
        originalMessage: message.data,
        error: error.message
      });
    }
  }
});
\`\`\`

### **Backward Compatibility**

✅ Good - consumer reads only fields it knows
⚠️ Need schema registry for Kafka (Avro/Protobuf)
⚠️ Old consumers might ignore new event types

### **Operational Complexity**

**High** - Requires queue infrastructure, monitoring, dead letter queues, handling duplicates

---

## **Comparison Table**

| Aspect | REST | gRPC | Message Queue |
|--------|------|------|---------------|
| **Performance** | Moderate | High (5-10x faster) | High (async) |
| **Latency** | 10-50ms | 1-10ms | 5-50ms + processing |
| **Throughput** | 1K-5K req/s | 10K-50K req/s | 10K-1M msg/s |
| **Coupling** | Tight (sync) | Tight (sync) | Loose (async) |
| **Use Case** | Public APIs, CRUD | Internal services | Events, async |
| **Debugging** | Easy (text) | Hard (binary) | Medium |
| **Browser Support** | Native | Limited (gRPC-Web) | None (backend only) |
| **Backward Compat** | Versioning | Protobuf evolution | Schema registry |
| **Failure Handling** | Retry | Retry | DLQ + requeue |
| **Ops Complexity** | Low | Medium | High |

---

## **Decision Framework**

### **Choose REST when:**
- **Public-facing APIs** (third-party developers)
- **Browser clients** (fetch API works natively)
- **Simple CRUD** (GET, POST, PUT, DELETE)
- **Team familiarity** (most devs know REST)
- **Debugging ease** (curl-testable)

### **Choose gRPC when:**
- **Internal microservices** (backend-to-backend)
- **High performance required** (low latency, high throughput)
- **Streaming** (server streaming, client streaming, bidirectional)
- **Polyglot** (multiple languages)
- **Mobile apps** (bandwidth efficiency)

### **Choose Message Queues when:**
- **Asynchronous processing** (user doesn't wait)
- **Event-driven architecture** (multiple consumers)
- **Decoupling** (services don't call each other directly)
- **Load leveling** (buffer traffic spikes)
- **Guaranteed delivery** (at-least-once semantics)

---

## **Real-World Example: E-Commerce Platform**

\`\`\`
┌─────────────────────────────────────────────┐
│           Client (Browser/Mobile)           │
└──────────────────┬──────────────────────────┘
                   │
                   │ REST (public API)
                   ↓
          ┌────────────────┐
          │  API Gateway   │
          └────────────────┘
                   │
                   ├─ gRPC ─→ User Service
                   ├─ gRPC ─→ Product Service  
                   ├─ gRPC ─→ Order Service
                   └─ gRPC ─→ Payment Service
                          │
                          │ Publish events
                          ↓
                   ┌─────────────┐
                   │ Message Bus │
                   └─────────────┘
                          │
                          ├─→ Notification Service (email)
                          ├─→ Analytics Service (metrics)
                          ├─→ Fulfillment Service (shipping)
                          └─→ Inventory Service (stock update)
\`\`\`

**Reasoning**:
1. **REST**: Browser → API Gateway (universal support)
2. **gRPC**: Internal services (performance)
3. **Message Queue**: Notifications, analytics (async, decoupled)

---

## **My Recommendation**

For most production microservices:

**Frontend ↔ Backend**: REST (simplicity, browser support)
**Backend ↔ Backend**: gRPC (performance, type safety)
**Async/Events**: Message Queue (decoupling, reliability)

This hybrid approach gives you the best of all worlds!`,
    keyPoints: [
      'REST: Best for public APIs and browser clients (easy to use, universally supported)',
      'gRPC: Best for internal microservices (5-10x faster, type-safe, streaming support)',
      'Message Queues: Best for async processing and event-driven architecture (decoupling, guaranteed delivery)',
      'Performance: gRPC fastest (1-10ms), REST moderate (10-50ms), MQ async but high throughput',
      'Operational complexity: REST (low), gRPC (medium), Message Queue (high)',
      'Hybrid approach recommended: REST for public, gRPC internal, MQ for events',
    ],
  },
  {
    id: 'q3-communication',
    question:
      'Design a request tracing system for microservices that allows debugging distributed transactions across 20+ services. Explain how you would propagate trace context, collect spans, handle sampling, store trace data, and build a query interface. Include specific implementation details for instrumentation and visualization.',
    sampleAnswer: `**Distributed Tracing System Design**

**1. Architecture Overview**

\`\`\`
Client Request
    ↓
API Gateway (generates trace_id)
    ├→ Service A (adds span)
    │   ├→ Service B (adds span)
    │   └→ Service C (adds span)
    │       └→ Service D (adds span)
    └→ Service E (adds span)
         │
         └ All spans sent to Collector
              ↓
         Trace Storage (Cassandra/Elasticsearch)
              ↓
         Query API & UI (Jaeger/Zipkin)
\`\`\`

**Key Components**:
1. **Trace Context Propagation** (W3C Trace Context standard)
2. **Span Collection** (OpenTelemetry)
3. **Sampling** (Head-based and tail-based)
4. **Storage** (Cassandra for traces, Elasticsearch for search)
5. **Query Interface** (Jaeger UI)

---

**2. Trace Context Propagation**

**W3C Trace Context Format**:

\`\`\`
traceparent: 00-{trace-id}-{span-id}-{flags}
tracestate: vendor1=value1,vendor2=value2
\`\`\`

**Implementation**:

\`\`\`javascript
// API Gateway (Entry Point)
const tracer = require('@opentelemetry/api').trace.getTracer('api-gateway');

app.use((req, res, next) => {
  // Start new trace or continue existing
  const span = tracer.startSpan('http_request', {
    kind: SpanKind.SERVER,
    attributes: {
      'http.method': req.method,
      'http.url': req.url,
      'http.target': req.path,
      'http.host': req.hostname
    }
  });
  
  // Extract or generate trace context
  const traceId = req.headers['traceparent',] 
    ? extractTraceId(req.headers['traceparent',])
    : generateTraceId();
  
  // Inject into request context
  req.traceId = traceId;
  req.spanId = span.spanContext().spanId;
  
  // Set response header
  res.setHeader('X-Trace-ID', traceId);
  
  // Continue
  span.end();
  next();
});

// Service-to-Service propagation
async function callServiceA(data) {
  const span = tracer.startSpan('call_service_a');
  
  try {
    const response = await httpClient.post('http://service-a/api/process', data, {
      headers: {
        'traceparent': \`00-\${traceId}-\${span.spanContext().spanId}-01\`,
        'X-Request-ID': generateRequestId()
      }
    });
    
    span.setStatus({ code: SpanStatusCode.OK });
    return response.data;
  } catch (error) {
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: error.message
    });
    span.recordException(error);
    throw error;
  } finally {
    span.end();
  }
}
\`\`\`

---

**3. Span Collection with OpenTelemetry**

**Instrumentation**:

\`\`\`javascript
const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');
const { BatchSpanProcessor } = require('@opentelemetry/sdk-trace-base');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');
const { Resource } = require('@opentelemetry/resources');
const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');

// Initialize tracer
const provider = new NodeTracerProvider({
  resource: new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: 'order-service',
    [SemanticResourceAttributes.SERVICE_VERSION]: '1.2.3',
    [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: 'production'
  })
});

// Configure exporter
const exporter = new JaegerExporter({
  endpoint: 'http://jaeger-collector:14268/api/traces',
  maxPacketSize: 65000
});

// Batch spans for efficiency
provider.addSpanProcessor(new BatchSpanProcessor(exporter, {
  maxQueueSize: 2048,
  maxExportBatchSize: 512,
  scheduledDelayMillis: 5000
}));

provider.register();

// Auto-instrumentation
const { registerInstrumentations } = require('@opentelemetry/instrumentation');
const { HttpInstrumentation } = require('@opentelemetry/instrumentation-http');
const { ExpressInstrumentation } = require('@opentelemetry/instrumentation-express');
const { PgInstrumentation } = require('@opentelemetry/instrumentation-pg');

registerInstrumentations({
  instrumentations: [
    new HttpInstrumentation(),
    new ExpressInstrumentation(),
    new PgInstrumentation()
  ]
});

// Manual instrumentation example
const tracer = require('@opentelemetry/api').trace.getTracer('order-service');

async function processOrder(orderId) {
  const span = tracer.startSpan('process_order', {
    attributes: {
      'order.id': orderId,
      'operation': 'process_order'
    }
  });
  
  try {
    // Business logic
    await reserveInventory(orderId);
    await processPayment(orderId);
    await confirmOrder(orderId);
    
    span.addEvent('order_processed_successfully');
    span.setStatus({ code: SpanStatusCode.OK });
    
  } catch (error) {
    span.recordException(error);
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: error.message
    });
    throw error;
  } finally {
    span.end();
  }
}
\`\`\`

---

**4. Sampling Strategy**

**Head-Based Sampling** (at trace start):

\`\`\`javascript
const { TraceIdRatioBasedSampler, ParentBasedSampler } = require('@opentelemetry/sdk-trace-base');

// Sample 10% of traces
const sampler = new ParentBasedSampler({
  root: new TraceIdRatioBasedSampler(0.1) // 10%
});

const provider = new NodeTracerProvider({
  sampler: sampler
});
\`\`\`

**Tail-Based Sampling** (after trace completes):

\`\`\`javascript
// Collector config (tail-based sampling)
processors:
  tail_sampling:
    decision_wait: 10s
    num_traces: 100000
    expected_new_traces_per_sec: 1000
    policies:
      - name: errors-policy
        type: status_code
        status_code: {status_codes: [ERROR]}
      - name: slow-traces-policy
        type: latency
        latency: {threshold_ms: 1000}
      - name: random-policy
        type: probabilistic
        probabilistic: {sampling_percentage: 1}
\`\`\`

**Adaptive Sampling**:

\`\`\`javascript
class AdaptiveSampler {
  constructor(targetRate = 100) { // 100 traces/sec
    this.targetRate = targetRate;
    this.currentRate = 0;
    this.samplingProbability = 1.0;
  }
  
  shouldSample(traceId) {
    // Always sample errors
    if (this.hasError(traceId)) {
      return true;
    }
    
    // Always sample slow traces
    if (this.isSlow(traceId)) {
      return true;
    }
    
    // Probabilistic sampling for others
    return Math.random() < this.samplingProbability;
  }
  
  adjustRate() {
    // Adjust sampling rate every minute
    if (this.currentRate > this.targetRate) {
      this.samplingProbability *= 0.9;
    } else if (this.currentRate < this.targetRate * 0.8) {
      this.samplingProbability = Math.min(1.0, this.samplingProbability * 1.1);
    }
  }
}
\`\`\`

---

**5. Trace Storage**

**Schema**:

\`\`\`sql
-- Cassandra schema
CREATE TABLE traces (
    trace_id text,
    span_id text,
    parent_span_id text,
    service_name text,
    operation_name text,
    start_time timestamp,
    duration bigint,
    tags map<text, text>,
    logs list<frozen<log_entry>>,
    PRIMARY KEY (trace_id, start_time, span_id)
) WITH CLUSTERING ORDER BY (start_time DESC);

-- Index for queries
CREATE INDEX ON traces (service_name);
CREATE INDEX ON traces (operation_name);
CREATE INDEX ON traces (duration);
\`\`\`

**Elasticsearch for Search**:

\`\`\`json
{
  "mappings": {
    "properties": {
      "traceId": { "type": "keyword" },
      "spanId": { "type": "keyword" },
      "serviceName": { "type": "keyword" },
      "operationName": { "type": "keyword" },
      "startTime": { "type": "date" },
      "duration": { "type": "long" },
      "tags": {
        "type": "nested",
        "properties": {
          "key": { "type": "keyword" },
          "value": { "type": "text" }
        }
      },
      "status": { "type": "keyword" }
    }
  }
}
\`\`\`

---

**6. Query Interface**

**API**:

\`\`\`javascript
// Query API
app.get('/api/traces', async (req, res) => {
  const {
    service,
    operation,
    minDuration,
    maxDuration,
    tags,
    startTime,
    endTime,
    limit = 20
  } = req.query;
  
  const query = {
    bool: {
      must: []
    }
  };
  
  if (service) {
    query.bool.must.push({ term: { serviceName: service } });
  }
  
  if (operation) {
    query.bool.must.push({ term: { operationName: operation } });
  }
  
  if (minDuration || maxDuration) {
    query.bool.must.push({
      range: {
        duration: {
          gte: minDuration || 0,
          lte: maxDuration || Infinity
        }
      }
    });
  }
  
  const traces = await elasticsearch.search({
    index: 'traces',
    body: { query },
    size: limit
  });
  
  res.json(traces.hits.hits);
});

// Get trace by ID
app.get('/api/traces/:traceId', async (req, res) => {
  const spans = await cassandra.execute(
    'SELECT * FROM traces WHERE trace_id = ?',
    [req.params.traceId]
  );
  
  // Build trace tree
  const trace = buildTraceTree(spans.rows);
  res.json(trace);
});
\`\`\`

**7. Visualization**

Use **Jaeger UI** for visualization:
- Gantt chart of spans (timeline view)
- Service dependency graph
- Span details (tags, logs, events)
- Trace comparison

---

**Key Takeaways**:

1. **W3C Trace Context** for standardized propagation
2. **OpenTelemetry** for vendor-neutral instrumentation
3. **Sampling**: Head-based (simple) or tail-based (smarter)
4. **Storage**: Cassandra for traces, Elasticsearch for search
5. **Always trace errors and slow requests**
6. **Jaeger/Zipkin** for visualization
7. **Auto-instrumentation** for frameworks, manual for business logic
8. **Batch span exports** for efficiency (5-second batches)`,
    keyPoints: [
      'Propagate trace context using W3C Trace Context standard (traceparent header)',
      'Instrument with OpenTelemetry (vendor-neutral, auto-instrumentation for HTTP/DB)',
      'Sampling: Always sample errors and slow requests, probabilistic for others',
      'Storage: Cassandra for traces (scalable writes), Elasticsearch for search queries',
      'Batch span exports every 5 seconds for efficiency (reduce network calls)',
      'Visualization: Jaeger UI for Gantt charts, service graphs, and trace comparison',
    ],
  },
];
