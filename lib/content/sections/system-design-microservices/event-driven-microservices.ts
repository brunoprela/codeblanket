/**
 * Event-Driven Microservices Section
 */

export const eventdrivenmicroservicesSection = {
  id: 'event-driven-microservices',
  title: 'Event-Driven Microservices',
  content: `Event-driven architecture enables loose coupling, scalability, and resilience in microservices. Services communicate through events rather than direct calls.

## What is Event-Driven Architecture?

**Synchronous** (request-response):
\`\`\`
Order Service → calls → Payment Service (waits for response)
                         ↓
                      Returns result
\`\`\`

**Asynchronous** (event-driven):
\`\`\`
Order Service → publishes OrderCreated event → Message Bus
                                                    ↓
                                          Payment Service subscribes
\`\`\`

**Key difference**: Order Service doesn't wait for Payment Service.

---

## Events vs Commands

### Commands

**Intent**: Tell service to do something

**Example**: \`ChargePayment\`, \`ReserveInventory\`

**Sent to**: Specific service

**Response**: Success or failure

### Events

**Intent**: Announce something happened

**Example**: \`OrderCreated\`, \`PaymentCompleted\`

**Sent to**: Anyone interested (pub/sub)

**Response**: None (fire and forget)

**Comparison**:
\`\`\`javascript
// Command (synchronous)
const result = await paymentService.chargePayment({
    orderId: '123',
    amount: 99.99
});

if (result.success) {
    // Continue
} else {
    // Handle failure
}

// Event (asynchronous)
await eventBus.publish('OrderCreated', {
    orderId: '123',
    userId: 'user-456',
    total: 99.99
});
// Don't wait for response, continue immediately
\`\`\`

---

## Message Brokers

**Tools**: RabbitMQ, Apache Kafka, AWS SQS/SNS, Google Pub/Sub

### RabbitMQ

**Good for**: Task queues, RPC, routing patterns

**Pattern**: Exchanges + Queues

\`\`\`javascript
const amqp = require('amqplib');

// Publisher
async function publishOrderCreated (order) {
    const connection = await amqp.connect('amqp://localhost');
    const channel = await connection.createChannel();
    
    const exchange = 'orders';
    await channel.assertExchange (exchange, 'topic', { durable: true });
    
    channel.publish(
        exchange,
        'order.created',
        Buffer.from(JSON.stringify (order)),
        { persistent: true }
    );
    
    console.log('Published OrderCreated event');
}

// Subscriber
async function subscribeToOrderEvents() {
    const connection = await amqp.connect('amqp://localhost');
    const channel = await connection.createChannel();
    
    const exchange = 'orders';
    const queue = 'payment-service-orders';
    
    await channel.assertExchange (exchange, 'topic', { durable: true });
    await channel.assertQueue (queue, { durable: true });
    await channel.bindQueue (queue, exchange, 'order.*');
    
    channel.consume (queue, (msg) => {
        const event = JSON.parse (msg.content.toString());
        console.log('Received:', event);
        
        // Process event
        processOrderCreated (event);
        
        // Acknowledge
        channel.ack (msg);
    });
}
\`\`\`

### Apache Kafka

**Good for**: High throughput, event streaming, event sourcing

**Pattern**: Topics + Partitions

\`\`\`javascript
const { Kafka } = require('kafkajs');

const kafka = new Kafka({
    clientId: 'order-service',
    brokers: ['localhost:9092']
});

// Producer
async function publishOrderCreated (order) {
    const producer = kafka.producer();
    await producer.connect();
    
    await producer.send({
        topic: 'orders',
        messages: [
            {
                key: order.id,  // Partition key
                value: JSON.stringify({
                    eventType: 'OrderCreated',
                    data: order,
                    timestamp: new Date().toISOString()
                })
            }
        ]
    });
    
    await producer.disconnect();
}

// Consumer
async function subscribeToOrderEvents() {
    const consumer = kafka.consumer({ groupId: 'payment-service' });
    await consumer.connect();
    await consumer.subscribe({ topic: 'orders', fromBeginning: false });
    
    await consumer.run({
        eachMessage: async ({ topic, partition, message }) => {
            const event = JSON.parse (message.value.toString());
            
            if (event.eventType === 'OrderCreated') {
                await processOrderCreated (event.data);
            }
        }
    });
}
\`\`\`

---

## Event Design

### Event Structure

**Good event**:
\`\`\`javascript
{
    eventId: "evt_abc123",           // Unique ID
    eventType: "OrderCreated",        // Event type
    eventVersion: "1.0",              // Schema version
    timestamp: "2024-01-15T10:30:00Z", // When it happened
    source: "order-service",          // Who published it
    data: {                           // Event payload
        orderId: "order-456",
        userId: "user-789",
        items: [
            { productId: "prod-1", quantity: 2, price: 29.99 }
        ],
        total: 59.98,
        currency: "USD"
    },
    metadata: {                       // Context
        correlationId: "req_xyz",
        causationId: "evt_previous",
        userId: "user-789"
    }
}
\`\`\`

### Event Versioning

**Problem**: Event schema changes

**Solution**: Version events

\`\`\`javascript
// Version 1.0
{
    eventType: "OrderCreated",
    eventVersion: "1.0",
    data: {
        orderId: "123",
        total: 99.99
    }
}

// Version 2.0 (added currency)
{
    eventType: "OrderCreated",
    eventVersion: "2.0",
    data: {
        orderId: "123",
        total: 99.99,
        currency: "USD"  // NEW
    }
}

// Consumer handles both versions
function handleOrderCreated (event) {
    const { data, eventVersion } = event;
    
    let currency = 'USD';  // Default
    
    if (eventVersion === '2.0') {
        currency = data.currency;
    }
    
    // Process with currency
}
\`\`\`

---

## Event Patterns

### 1. Event Notification

**Notify** other services something happened.

**Example**: Order created → notify email service

\`\`\`javascript
// Order Service
await eventBus.publish('OrderCreated', { orderId, userId, total });

// Email Service
eventBus.on('OrderCreated', async (event) => {
    await sendOrderConfirmationEmail (event.userId, event.orderId);
});

// Analytics Service
eventBus.on('OrderCreated', async (event) => {
    await trackOrderMetric (event);
});
\`\`\`

**Benefits**:
✅ Services decoupled
✅ Easy to add new subscribers
✅ Publisher doesn't know subscribers

### 2. Event-Carried State Transfer

**Include** all necessary data in event (reduce queries).

\`\`\`javascript
// ❌ Bad: Minimal data
{
    eventType: "OrderCreated",
    data: {
        orderId: "123"
    }
}
// Subscriber must call Order Service to get details

// ✅ Good: Include all data
{
    eventType: "OrderCreated",
    data: {
        orderId: "123",
        userId: "user-456",
        items: [{ productId: "prod-1", quantity: 2 }],
        total: 59.98,
        shippingAddress: {...}
    }
}
// Subscriber has everything it needs
\`\`\`

### 3. Event Sourcing

**Store** all changes as events (event log is source of truth).

**Traditional**:
\`\`\`sql
-- Current state only
SELECT * FROM orders WHERE id = 123;
-- Result: { id: 123, status: "SHIPPED", total: 99.99 }
-- Lost history: When was it created? When shipped?
\`\`\`

**Event Sourcing**:
\`\`\`javascript
// Event log
[
    { eventType: "OrderCreated", timestamp: "10:00:00", data: {...} },
    { eventType: "PaymentReceived", timestamp: "10:00:05", data: {...} },
    { eventType: "OrderShipped", timestamp: "10:30:00", data: {...} }
]

// Reconstruct current state by replaying events
function getOrderState (orderId) {
    const events = getEvents (orderId);
    let state = {};
    
    for (const event of events) {
        state = applyEvent (state, event);
    }
    
    return state;
}

function applyEvent (state, event) {
    switch (event.eventType) {
        case 'OrderCreated':
            return { ...event.data, status: 'PENDING' };
        case 'PaymentReceived':
            return { ...state, status: 'PAID' };
        case 'OrderShipped':
            return { ...state, status: 'SHIPPED', shippedAt: event.timestamp };
        default:
            return state;
    }
}
\`\`\`

**Benefits**:
✅ Full audit trail
✅ Time travel (state at any point)
✅ Debugging (replay events)
✅ Can add new projections

**Drawbacks**:
❌ Complexity
❌ Query performance (must replay)
❌ Event versioning

### 4. CQRS (Command Query Responsibility Segregation)

**Separate** read and write models.

**Architecture**:
\`\`\`
Write Side (Commands):
  Order Service → Event Store

Events:
  OrderCreated, PaymentReceived, OrderShipped

Read Side (Queries):
  Event Handler → Read Database (denormalized views)
  
Query:
  API → Read Database (fast queries)
\`\`\`

**Implementation**:
\`\`\`javascript
// Write side (commands)
async function createOrder (orderData) {
    const order = { id: generateId(), ...orderData, status: 'PENDING' };
    
    // Store event
    await eventStore.append('OrderCreated', order);
    
    // Publish event
    await eventBus.publish('OrderCreated', order);
    
    return order;
}

// Read side (projections)
eventBus.on('OrderCreated', async (event) => {
    // Update read model
    await orderReadDB.insert({
        orderId: event.data.id,
        userId: event.data.userId,
        total: event.data.total,
        status: 'PENDING',
        createdAt: event.timestamp
    });
});

eventBus.on('OrderShipped', async (event) => {
    // Update read model
    await orderReadDB.update (event.data.orderId, {
        status: 'SHIPPED',
        shippedAt: event.timestamp
    });
});

// Query (fast!)
async function getOrdersByUser (userId) {
    return await orderReadDB.query({ userId });
}
\`\`\`

---

## Handling Failures

### Idempotency

**Problem**: Event processed twice

**Solution**: Idempotent event handlers

\`\`\`javascript
// ❌ Not idempotent
eventBus.on('OrderCreated', async (event) => {
    await database.query(
        'UPDATE inventory SET quantity = quantity - $1 WHERE productId = $2',
        [event.quantity, event.productId]
    );
});
// If event processed twice, quantity decremented twice!

// ✅ Idempotent
eventBus.on('OrderCreated', async (event) => {
    // Check if already processed
    const existing = await database.query(
        'SELECT * FROM processed_events WHERE eventId = $1',
        [event.eventId]
    );
    
    if (existing.length > 0) {
        return; // Already processed
    }
    
    // Process (in transaction)
    await database.transaction (async (tx) => {
        // Update inventory
        await tx.query(
            'UPDATE inventory SET quantity = quantity - $1 WHERE productId = $2',
            [event.quantity, event.productId]
        );
        
        // Mark as processed
        await tx.query(
            'INSERT INTO processed_events (eventId, processedAt) VALUES ($1, $2)',
            [event.eventId, new Date()]
        );
    });
});
\`\`\`

### Dead Letter Queue

**Problem**: Event processing fails repeatedly

**Solution**: Move to dead letter queue after N retries

\`\`\`javascript
eventBus.on('OrderCreated', async (event) => {
    try {
        await processOrderCreated (event);
    } catch (error) {
        const retryCount = event.retryCount || 0;
        
        if (retryCount < 3) {
            // Retry with exponential backoff
            const delay = Math.pow(2, retryCount) * 1000;
            setTimeout(() => {
                eventBus.publish('OrderCreated', {
                    ...event,
                    retryCount: retryCount + 1
                });
            }, delay);
        } else {
            // Move to dead letter queue
            await deadLetterQueue.send (event);
            await alerting.notify('Event processing failed after 3 retries', event);
        }
    }
});
\`\`\`

### Eventual Consistency

**Accept** that data is temporarily inconsistent.

**Example**:
\`\`\`
Time 10:00:00 - Order created
Time 10:00:01 - Email service receives event
Time 10:00:02 - Email sent

Between 10:00:00 and 10:00:02, user hasn't received email yet (eventually consistent)
\`\`\`

**How to handle**:
1. **Don't show** intermediate states to users
2. **Status fields**: "Processing...", "Pending..."
3. **Eventual consistency UI**: "Email will be sent shortly"

---

## Event-Driven vs Request-Response

| Aspect | Request-Response | Event-Driven |
|--------|-----------------|--------------|
| **Coupling** | Tight (caller knows callee) | Loose (publisher doesn't know subscribers) |
| **Failure** | Immediate failure | Eventual processing (resilient) |
| **Performance** | Blocking (synchronous) | Non-blocking (asynchronous) |
| **Consistency** | Immediate | Eventual |
| **Scalability** | Limited by slowest service | Highly scalable |
| **Debugging** | Easier (call stack) | Harder (distributed) |
| **Use case** | Real-time responses needed | Background processing, notifications |

**When to use each**:

**Request-Response**:
- User needs immediate response (login, search)
- Simple workflows
- Strong consistency required

**Event-Driven**:
- Background tasks (emails, analytics)
- Multiple services interested
- High scalability needed
- Loose coupling preferred

---

## Best Practices

1. **Events are immutable** (never change published event)
2. **Include all data** (event-carried state transfer)
3. **Version events** for schema evolution
4. **Idempotent handlers** (safe to process twice)
5. **Dead letter queues** for failed events
6. **Correlation IDs** for tracing
7. **Event store** for audit trail
8. **Monitor event lag** (time between publish and process)

---

## Interview Tips

**Red Flags**:
❌ Using events for everything (including synchronous operations)
❌ Not handling failures
❌ Ignoring eventual consistency

**Good Responses**:
✅ Explain events vs commands
✅ Discuss trade-offs (loose coupling vs complexity)
✅ Mention idempotency
✅ Talk about specific tools (Kafka, RabbitMQ)
✅ Discuss eventual consistency

**Sample Answer**:
*"For event-driven microservices, I'd use Kafka for high-throughput event streaming or RabbitMQ for simpler pub/sub. Services publish events (OrderCreated, PaymentCompleted) to a message broker, and interested services subscribe. This provides loose coupling - services don't know about each other. Trade-offs: eventual consistency (Order Service doesn't wait for Email Service), complexity (distributed debugging), need for idempotent handlers. I'd use events for background tasks (emails, analytics) and request-response for operations needing immediate feedback (login, search). Include correlation IDs for tracing, implement dead letter queues for failures, and monitor event processing lag."*

---

## Key Takeaways

1. **Events** announce something happened; **Commands** tell service to do something
2. **Message brokers**: Kafka (high throughput), RabbitMQ (flexible routing)
3. **Event-carried state transfer** includes all data in event
4. **Event sourcing** stores all changes as events (full audit trail)
5. **CQRS** separates read and write models
6. **Idempotency** makes handlers safe to execute multiple times
7. **Eventual consistency** is acceptable trade-off for loose coupling
8. **Use events** for background tasks, **request-response** for immediate feedback`,
};
