/**
 * Event-Driven Architecture Section
 */

export const eventdrivenarchitectureSection = {
  id: 'event-driven-architecture',
  title: 'Event-Driven Architecture',
  content: `Event-Driven Architecture (EDA) is a software design pattern where services communicate through events rather than direct calls. It enables loose coupling, scalability, and temporal decoupling in distributed systems.

## What is Event-Driven Architecture?

**EDA** = Services react to events + Asynchronous communication + Event bus/broker + Loose coupling

### **Traditional vs Event-Driven:**

\`\`\`
Traditional (Synchronous):
Order Service → Payment Service (HTTP)
             → Inventory Service (HTTP)
             → Email Service (HTTP)

❌ Tight coupling (knows all services)
❌ Synchronous (must wait for all)
❌ Cascading failures (if one service down)

Event-Driven (Asynchronous):
Order Service → Event: "OrderCreated" → Event Bus
                                          ↓ Subscribe
                                     Payment Service (reacts)
                                     Inventory Service (reacts)
                                     Email Service (reacts)

✅ Loose coupling (publishes event, doesn't know subscribers)
✅ Asynchronous (doesn't wait)
✅ Resilient (services fail independently)
✅ Extensible (add subscribers without changing producer)
\`\`\`

---

## Core Concepts

### **1. Event**

An **event** is a notification that something happened (past tense).

\`\`\`
Event: "OrderCreated"
{
  "eventId": "evt_123",
  "eventType": "OrderCreated",
  "timestamp": "2023-06-15T10:30:00Z",
  "aggregateId": "order_456",
  "data": {
    "orderId": "order_456",
    "customerId": "cust_789",
    "items": [...],
    "total": 99.99
  },
  "metadata": {
    "userId": "user_001",
    "correlationId": "req_abc"
  }
}

Event characteristics:
- Immutable (cannot be changed)
- Represents fact (something that happened)
- Past tense naming ("OrderCreated", not "CreateOrder")
- Contains all necessary data
- Identified by unique ID
\`\`\`

### **2. Event Producer**

Publishes events when something notable happens:

\`\`\`
Order Service (Producer):
1. Receives order request
2. Validates order
3. Saves to database
4. Publishes "OrderCreated" event
5. Returns immediately

✅ Doesn't know who consumes event
✅ Doesn't wait for consumers
✅ Just publishes and continues
\`\`\`

### **3. Event Consumer**

Subscribes to events and reacts:

\`\`\`
Payment Service (Consumer):
1. Subscribes to "OrderCreated" events
2. Receives event notification
3. Processes payment
4. Publishes "PaymentProcessed" event

Email Service (Consumer):
1. Subscribes to "OrderCreated" events
2. Receives event notification
3. Sends confirmation email
4. Done

✅ Independent of other consumers
✅ Can fail/retry independently
✅ Processes at its own pace
\`\`\`

### **4. Event Bus/Broker**

Routes events from producers to consumers:

\`\`\`
Technologies:
- Kafka (distributed log)
- RabbitMQ (message broker)
- AWS EventBridge (managed event bus)
- Azure Event Grid
- Google Cloud Pub/Sub

Responsibilities:
- Receive events from producers
- Store events (persistence)
- Deliver to subscribers
- Handle failures/retries
\`\`\`

---

## Event Types and Patterns

### **1. Event Notification**

Minimal data, consumer fetches details if needed:

\`\`\`
Event: "OrderCreated"
{
  "orderId": "order_456",
  "customerId": "cust_789"
}

Consumer (Email Service):
1. Receives event
2. Calls Order API to get full order details
3. Sends email

Pros:
✅ Small events (low bandwidth)
✅ Consumer gets latest data
✅ Easy to understand

Cons:
❌ Consumer depends on Order API (coupling)
❌ Additional API call (latency)
❌ Order API must be available
\`\`\`

### **2. Event-Carried State Transfer**

Complete data in event, no need to fetch:

\`\`\`
Event: "OrderCreated"
{
  "orderId": "order_456",
  "customerId": "cust_789",
  "customerName": "John Doe",
  "customerEmail": "john@example.com",
  "items": [
    {"productId": "prod_1", "name": "Widget", "quantity": 2, "price": 49.99}
  ],
  "total": 99.99,
  "shippingAddress": {...}
}

Consumer (Email Service):
1. Receives event
2. Has all data needed
3. Sends email immediately

Pros:
✅ No additional API calls
✅ Consumer fully decoupled
✅ Works even if Order Service down
✅ Lower latency

Cons:
❌ Larger events (more bandwidth)
❌ Data duplication
❌ Consumers may have stale data (if source updates)
\`\`\`

### **3. Event Sourcing**

Store all changes as sequence of events (append-only log):

\`\`\`
Traditional (State-Based):
Database: orders table
order_id | status  | total
---------|---------|------
456      | shipped | 99.99

Only current state, history lost

Event Sourcing (Event-Based):
Event Store (append-only):
1. OrderCreated   {orderId: 456, total: 99.99}
2. OrderPaid      {orderId: 456, paymentId: 789}
3. OrderShipped   {orderId: 456, trackingNumber: "ABC123"}

Current state derived by replaying events
Full history preserved ✅

Pros:
✅ Complete audit trail (who, when, what)
✅ Time travel (reconstruct past state)
✅ Debugging (replay events)
✅ New features (replay old events with new logic)

Cons:
❌ Complex to implement
❌ Event schema evolution
❌ Snapshot required for performance
❌ Eventual consistency

Use case: Banking, audit-critical systems, complex domains
\`\`\`

### **4. CQRS (Command Query Responsibility Segregation)**

Separate read and write models:

\`\`\`
Traditional:
┌─────────────┐
│  Database   │
│ (Read/Write)│
└─────────────┘
     ↑ ↓
   Read/Write

CQRS:
Commands (Write) → Write Model → Event Store
                                      ↓ Events
                                 Read Model 1 (optimized for queries)
                                 Read Model 2 (different view)
                                 Read Model 3 (reporting)

Write Model:
- Handles commands (CreateOrder, UpdateInventory)
- Enforces business rules
- Publishes events
- Normalized schema

Read Model:
- Handles queries (GetOrderHistory, GetInventory)
- Optimized for specific queries
- Denormalized (fast reads)
- Eventually consistent

Pros:
✅ Optimized writes (normalized)
✅ Optimized reads (denormalized, multiple views)
✅ Scale reads/writes independently
✅ Different data stores (write: SQL, read: NoSQL)

Cons:
❌ Complexity (two models)
❌ Eventual consistency
❌ Synchronization overhead

Use case: High read/write ratio, complex queries, audit requirements
\`\`\`

---

## Event-Driven Patterns

### **1. Saga Pattern (Distributed Transactions)**

Manage long-running transactions across services:

\`\`\`
Problem: Place order (requires payment, inventory, shipping)
- Can't use 2PC (slow, not available in microservices)
- Need coordinated transaction across services

Solution: Saga Pattern

Choreography (Event-Based):
1. Order Service: Creates order → Publishes "OrderCreated"
2. Payment Service: Listens → Processes payment → Publishes "PaymentSucceeded"
3. Inventory Service: Listens → Reserves items → Publishes "InventoryReserved"
4. Shipping Service: Listens → Arranges shipping → Publishes "ShippingScheduled"

If failure (e.g., payment fails):
1. Payment Service: Publishes "PaymentFailed"
2. Order Service: Listens → Cancels order → Publishes "OrderCancelled"
3. Inventory Service: Listens → Releases reservation

✅ No distributed transaction
✅ Each service autonomous
✅ Compensating transactions for failures

Orchestration (Coordinator):
Order Saga Orchestrator:
1. Send "ProcessPayment" command → Payment Service
2. Wait for "PaymentProcessed" event
3. Send "ReserveInventory" command → Inventory Service
4. Wait for "InventoryReserved" event
5. Send "ScheduleShipping" command → Shipping Service
6. If any step fails → Trigger compensating actions

✅ Centralized control
✅ Easier to understand flow
❌ Orchestrator is single point of knowledge
\`\`\`

### **2. Event Stream Processing**

Process continuous streams of events:

\`\`\`
Use case: Real-time fraud detection

Event Stream: Credit card transactions
Event 1: {cardId: "1234", amount: 50, location: "NYC"}
Event 2: {cardId: "1234", amount: 100, location: "NYC"}
Event 3: {cardId: "1234", amount: 500, location: "LA"}
                                            ↑
                                      Suspicious! (location jump)

Stream Processor (Kafka Streams, Flink):
- Window: Last 5 minutes
- Aggregate: Count transactions, track locations
- Detect: >3 transactions or location change >1000 miles
- Alert: Publish "FraudDetected" event

✅ Real-time processing
✅ Stateful aggregations
✅ Complex event processing (CEP)
\`\`\`

### **3. Event Replay**

Reprocess historical events:

\`\`\`
Scenario: New analytics service needs historical data

Traditional:
- Bulk data migration
- Complex ETL

Event-Driven:
- New service subscribes to event stream
- Seeks to beginning (offset 0)
- Processes all historical events
- Catches up to real-time

✅ No special migration logic
✅ Same code for historical and real-time
✅ Can test with production data

Kafka: Retention allows replay (7 days, 30 days)
Event Store: All events retained forever
\`\`\`

---

## Event Schema Design

### **Best Practices:**

**1. Use Semantic Versioning:**
\`\`\`
Event: "OrderCreatedV1", "OrderCreatedV2"

Or:
{
  "eventType": "OrderCreated",
  "schemaVersion": "2.0.0"
}
\`\`\`

**2. Backward Compatible Changes:**
\`\`\`
✅ Add optional fields (consumers ignore unknown fields)
✅ Add new event types
❌ Remove fields (breaks old consumers)
❌ Change field types (string → number)
❌ Rename fields

Schema Evolution:
V1: {orderId, customerId, total}
V2: {orderId, customerId, total, discount}  ✅ Optional field added

Old consumers: Ignore discount (still works)
New consumers: Use discount field
\`\`\`

**3. Include Metadata:**
\`\`\`
{
  "eventId": "uuid",
  "eventType": "OrderCreated",
  "timestamp": "ISO-8601",
  "schemaVersion": "1.0.0",
  "correlationId": "request-id",  // Track request across services
  "causationId": "parent-event-id",  // Event that caused this event
  "aggregateId": "order-123",
  "aggregateType": "Order",
  "userId": "user who triggered",
  "data": {...}
}
\`\`\`

**4. Use Schema Registry:**
\`\`\`
Confluent Schema Registry (Kafka)
- Centralized schema storage
- Schema validation
- Version management
- Compatibility checking

Producer:
1. Register schema
2. Validate event against schema
3. Publish event with schema ID

Consumer:
1. Receive event
2. Fetch schema by ID
3. Deserialize and validate
\`\`\`

---

## Event-Driven Architecture in Practice

### **Example: E-Commerce Platform**

\`\`\`
Services and Events:

Order Service:
- Publishes: OrderCreated, OrderCancelled, OrderCompleted
- Subscribes: PaymentSucceeded, PaymentFailed, InventoryReserved

Payment Service:
- Publishes: PaymentSucceeded, PaymentFailed, RefundIssued
- Subscribes: OrderCreated, OrderCancelled

Inventory Service:
- Publishes: InventoryReserved, InventoryReleased, StockLow
- Subscribes: OrderCreated, OrderCancelled, OrderCompleted

Email Service:
- Publishes: EmailSent
- Subscribes: OrderCreated, PaymentSucceeded, OrderShipped

Analytics Service:
- Publishes: None
- Subscribes: All events (for reporting)

Event Flow (Happy Path):
1. User places order
2. Order Service → "OrderCreated" event
3. Payment Service processes payment → "PaymentSucceeded"
4. Inventory Service reserves items → "InventoryReserved"
5. Shipping Service schedules → "ShippingScheduled"
6. Email Service sends confirmation → "EmailSent"
7. Order Service marks complete → "OrderCompleted"

Event Flow (Failure):
1. User places order
2. Order Service → "OrderCreated"
3. Payment Service fails → "PaymentFailed"
4. Order Service compensates → "OrderCancelled"
5. Email Service sends cancellation → "EmailSent"

✅ Each service autonomous
✅ Failure handled gracefully
✅ Easy to add new services (just subscribe to events)
\`\`\`

---

## Benefits and Challenges

### **Benefits:**

**1. Loose Coupling:**
- Services don't know about each other
- Easy to add/remove services
- Change independently

**2. Scalability:**
- Scale services independently
- Async processing (buffer traffic spikes)
- Parallel processing

**3. Resilience:**
- Services fail independently
- Event bus provides persistence
- Retry mechanisms

**4. Extensibility:**
- Add new functionality without changing existing services
- New service subscribes to events

**5. Audit Trail:**
- All events logged
- Complete history
- Replay capability

### **Challenges:**

**1. Complexity:**
- Distributed system complexity
- Harder to debug (async, multiple services)
- Eventual consistency

**2. Event Schema Evolution:**
- Backward compatibility required
- Version management
- Migration complexity

**3. Monitoring:**
- Need distributed tracing (correlation IDs)
- Visualize event flows
- Debug async failures

**4. Testing:**
- Integration testing harder
- Need to test event flows
- Eventual consistency complicates assertions

**5. Data Consistency:**
- Eventual consistency (not immediate)
- Need compensating transactions
- Handle duplicate events (idempotency)

---

## Event-Driven in System Design Interviews

### **When to Propose:**

✅ **Microservices architecture** (loose coupling needed)
✅ **High scalability required** (independent scaling)
✅ **Async workflows** (don't need immediate response)
✅ **Audit trail important** (compliance, debugging)
✅ **Multiple services react to same event** (fan-out)

### **Example Discussion:**

\`\`\`
Interviewer: "Design a ride-sharing platform (Uber-like)"

You:
"I'll use event-driven architecture for key workflows:

Events:
- RideRequested {riderId, pickup, destination}
- DriverAssigned {rideId, driverId}
- DriverArrived {rideId, location}
- RideStarted {rideId, startTime}
- RideCompleted {rideId, endTime, distance}
- PaymentProcessed {rideId, amount}
- RatingSubmitted {rideId, rating}

Services:
1. Ride Service:
   - Publishes: RideRequested, RideStarted, RideCompleted
   - Subscribes: DriverAssigned, PaymentProcessed

2. Matching Service:
   - Subscribes: RideRequested
   - Publishes: DriverAssigned
   - Logic: Find nearby available driver

3. Payment Service:
   - Subscribes: RideCompleted
   - Publishes: PaymentProcessed
   - Logic: Calculate fare, charge rider

4. Notification Service:
   - Subscribes: All events
   - Publishes: NotificationSent
   - Logic: Send push notifications

5. Analytics Service:
   - Subscribes: All events
   - No publishes
   - Logic: Real-time dashboards, reporting

Event Bus: Apache Kafka
- High throughput (millions of rides/day)
- Event replay (rebuild analytics)
- Partitioning by rideId (ordering per ride)

Benefits:
✅ Matching Service scales independently (complex algorithm)
✅ Add new services without changing existing ones
✅ Real-time notifications (event-driven)
✅ Complete audit trail (replay events)
✅ Resilient (service failures isolated)

Handling Failures (Saga):
If payment fails:
1. Payment Service → PaymentFailed event
2. Ride Service → Retry payment or cancel ride
3. Notification Service → Alert rider

Monitoring:
- Correlation ID per ride (trace across services)
- Distributed tracing (Jaeger)
- Event flow visualization
- Alert on stuck rides (no RideCompleted after 2 hours)
"
\`\`\`

---

## Key Takeaways

1. **Event-driven = React to events, not direct calls** → Loose coupling, async
2. **Event types: Notification, state transfer, sourcing** → Choose based on needs
3. **Event sourcing = Append-only log** → Complete history, audit trail
4. **CQRS = Separate read/write** → Optimized for each
5. **Saga pattern for distributed transactions** → Choreography or orchestration
6. **Schema evolution critical** → Backward compatibility, versioning
7. **Benefits: Loose coupling, scalability, resilience** → Cost: Complexity
8. **Use Kafka/EventBridge/RabbitMQ as event bus** → Reliable delivery
9. **Idempotent consumers handle duplicates** → At-least-once delivery
10. **In interviews: Discuss events, services, saga** → Show architecture skills

---

**Next:** We'll explore **Stream Processing**—real-time data processing, windowing, late data, and stream processing frameworks like Kafka Streams and Flink.`,
};
