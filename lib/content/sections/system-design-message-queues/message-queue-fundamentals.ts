/**
 * Message Queue Fundamentals Section
 */

export const messagequeuefundamentalsSection = {
  id: 'message-queue-fundamentals',
  title: 'Message Queue Fundamentals',
  content: `Message queues are a fundamental building block for asynchronous communication in distributed systems. They enable decoupling of services, load smoothing, and reliable message delivery.

## What is a Message Queue?

A **message queue** is a form of asynchronous service-to-service communication used in serverless and microservices architectures. Messages are stored in the queue until they are processed and deleted. Each message is processed only once, by a single consumer.

### **Key Concept:**
Producer → Queue → Consumer

- **Producer**: Creates and sends messages to the queue
- **Queue**: Temporarily stores messages until consumed
- **Consumer**: Retrieves, processes, and deletes messages

---

## Why Message Queues Matter

### **1. Decoupling**

Services don't need to know about each other:

\`\`\`
Without Queue:
Order Service → [HTTP Call] → Inventory Service (must be available)
                ↓ [HTTP Call] → Email Service (must be available)
                ↓ [HTTP Call] → Analytics Service (must be available)

❌ If any service is down, order processing fails
❌ Services are tightly coupled
❌ All services must scale together

With Queue:
Order Service → [Queue] ← Inventory Service (processes independently)
                         ← Email Service (processes independently)
                         ← Analytics Service (processes independently)

✅ Services operate independently
✅ No synchronous dependencies
✅ Services can scale independently
\`\`\`

### **2. Load Smoothing**

Handle traffic spikes gracefully:

\`\`\`
Traffic Pattern:
- Normal: 100 requests/second
- Black Friday: 10,000 requests/second

Without Queue:
❌ Must scale to handle peak (expensive)
❌ Requests fail if system overwhelmed
❌ Over-provisioned during normal periods

With Queue:
✅ Queue absorbs burst traffic
✅ Consumers process at their own pace
✅ Cost-effective scaling
✅ No dropped requests
\`\`\`

### **3. Reliability**

Messages persist until successfully processed:

\`\`\`
Scenario: Processing payment
1. Producer sends message to queue
2. Message persisted to disk
3. Consumer crashes during processing
4. Message remains in queue (not acknowledged)
5. Another consumer picks up message
6. Payment processed successfully

✅ At-least-once delivery guarantee
✅ No lost messages
✅ Automatic retry on failure
\`\`\`

### **4. Asynchronous Processing**

Non-blocking operations:

\`\`\`
E-commerce Order Flow:

Synchronous (Slow):
User clicks "Place Order"
  ↓ Wait for payment processing (2s)
  ↓ Wait for inventory update (1s)
  ↓ Wait for email sending (3s)
  ↓ Wait for analytics logging (1s)
User sees confirmation (7 seconds total) ❌

Asynchronous (Fast):
User clicks "Place Order"
  ↓ Write to order queue (50ms)
User sees confirmation (50ms total) ✅
  ↓ Background: Process payment
  ↓ Background: Update inventory
  ↓ Background: Send emails
  ↓ Background: Log analytics

User experience: Instant!
\`\`\`

---

## Queue vs Topic (Point-to-Point vs Pub/Sub)

### **1. Queue (Point-to-Point)**

One message → One consumer

\`\`\`
Producer → [Queue] → Consumer 1 (processes message)
                   → Consumer 2 (waits for next message)
                   → Consumer 3 (waits for next message)

Each message delivered to exactly ONE consumer
Use case: Task distribution, work queues
Example: Job processing, order fulfillment
\`\`\`

**Characteristics:**
- Load balancing across consumers
- Competitive consumption (consumers compete for messages)
- Message deleted after successful processing
- Scaling: Add more consumers to process faster

**Real-World Example:**
\`\`\`
Image Processing Queue:
- 1000 images uploaded
- 10 worker processes
- Each worker processes 100 images
- Load automatically balanced
\`\`\`

### **2. Topic (Publish/Subscribe)**

One message → Multiple subscribers

\`\`\`
Publisher → [Topic] → Subscriber 1 (receives copy)
                    → Subscriber 2 (receives copy)
                    → Subscriber 3 (receives copy)

Each message delivered to ALL subscribers
Use case: Event broadcasting, notifications
Example: User activity, system events
\`\`\`

**Characteristics:**
- Fan-out: One message → Many consumers
- Each subscriber gets a copy
- Subscribers don't affect each other
- Scaling: Each subscriber scales independently

**Real-World Example:**
\`\`\`
User Registration Event:
User registers → [Topic]
                  ↓ Email Service (sends welcome email)
                  ↓ Analytics Service (tracks signup)
                  ↓ CRM Service (creates lead)
                  ↓ Notification Service (sends push notification)

All services receive the same event
\`\`\`

---

## Message Delivery Guarantees

### **1. At-Most-Once (Fire and Forget)**

**Behavior:** Message delivered 0 or 1 times

\`\`\`
Producer → Queue → Consumer
                     ↓ Process
                     ↓ [Network failure before ACK]
Message lost ❌

Guarantee: Message won't be redelivered
Risk: Messages can be lost
\`\`\`

**When to Use:**
- Metrics, logging (occasional loss acceptable)
- Real-time analytics (slightly stale data OK)
- System health checks

**Example:**
\`\`\`
Temperature sensor readings:
- 1000 readings/second
- If 5 readings lost, not critical
- Next reading arrives in 1ms
\`\`\`

### **2. At-Least-Once (Most Common)**

**Behavior:** Message delivered 1 or more times

\`\`\`
Producer → Queue → Consumer
                     ↓ Process
                     ↓ [Crash before ACK]
                     ↓ Message redelivered
                     ↓ Process again (duplicate!)

Guarantee: Message won't be lost
Risk: Messages can be duplicated
\`\`\`

**When to Use:**
- Most production systems
- Order processing, payments (with idempotency)
- Email sending (duplicate email OK)

**Handling Duplicates (Idempotency):**
\`\`\`
Message: { order_id: "12345", amount: 100 }

Consumer Logic:
1. Check if order_id already processed
2. If yes: Skip (idempotent)
3. If no: Process and mark as processed

Database:
processed_orders = {
  "12345": { processed: true, timestamp: ... }
}

✅ Duplicate messages handled safely
\`\`\`

### **3. Exactly-Once (Holy Grail)**

**Behavior:** Message delivered exactly 1 time (no duplicates, no losses)

\`\`\`
Producer → Queue → Consumer
                     ↓ Process
                     ↓ [Transactional guarantee]
                     ✅ Processed once and only once

Guarantee: Perfect delivery
Cost: Performance overhead, complexity
\`\`\`

**When to Use:**
- Financial transactions (payments, transfers)
- Inventory management
- Any scenario where duplicates are unacceptable

**How It Works (Kafka Exactly-Once):**
\`\`\`
1. Transactional producer
2. Idempotent writes
3. Transactional reads
4. Two-phase commit

Trade-off: Slower but guaranteed
\`\`\`

---

## Message Ordering

### **1. No Ordering Guarantee**

Messages processed in arbitrary order:

\`\`\`
Producer sends:    Message 1 → Message 2 → Message 3
Consumer receives: Message 2 → Message 3 → Message 1

Why? Distributed system, multiple consumers, network delays
\`\`\`

**When It\'s OK:**
- Independent events (user registrations)
- Stateless operations (send email)
- Metrics aggregation

### **2. Partial Ordering (Per-Partition)**

Messages within same partition ordered:

\`\`\`
User A actions:    Login → View → Purchase (same partition)
                   ✅ Processed in order

User B actions:    Register → Update Profile (different partition)
                   ✅ Processed in order

User A and User B: No ordering guarantee between them
                   ✅ OK - independent users
\`\`\`

**Implementation (Kafka):**
\`\`\`
Partition Key: user_id

message = {
  key: "user_123",  // Partition key
  value: { action: "purchase", ... }
}

All messages for user_123 → Same partition → Ordered
Messages for user_456 → Different partition → Independent ordering
\`\`\`

### **3. Total Ordering (Global)**

All messages processed in strict order:

\`\`\`
Producer sends:    Msg 1 → Msg 2 → Msg 3
Consumer receives: Msg 1 → Msg 2 → Msg 3 (always)

How? Single partition, single consumer

Trade-off: No parallelism, limits throughput
\`\`\`

**When Required:**
- Financial ledger (transaction sequence matters)
- Audit logs (chronological order required)
- Replicated state machines

---

## Durable vs Transient Queues

### **Durable Queues**

Messages persist to disk:

\`\`\`
Producer → [Queue - Disk] → Consumer

Message lifecycle:
1. Received by queue → Written to disk
2. Queue crashes → Message survives
3. Queue restarts → Message still available
4. Consumer processes → Message deleted

✅ Survives queue restarts
✅ Survives server failures
❌ Slower (disk I/O)
\`\`\`

**Use Cases:**
- Payment processing
- Order fulfillment
- Critical notifications

### **Transient Queues**

Messages stored in memory only:

\`\`\`
Producer → [Queue - RAM] → Consumer

Message lifecycle:
1. Received by queue → Stored in memory
2. Queue crashes → Messages lost ❌
3. Queue restarts → No messages

✅ Extremely fast (no disk I/O)
❌ Messages lost on crash
\`\`\`

**Use Cases:**
- Real-time metrics (brief loss OK)
- Temporary caching
- Non-critical notifications

---

## Common Message Queue Patterns

### **1. Work Queue (Task Distribution)**

\`\`\`
                 ┌→ Worker 1 (processes 1/3)
Producer → Queue ├→ Worker 2 (processes 1/3)
                 └→ Worker 3 (processes 1/3)

Use case: Parallel task processing
Example: Video transcoding, image processing
\`\`\`

### **2. Priority Queue**

\`\`\`
Producer → Priority Queue → Consumer
             High (P0): Process first
             Medium (P1): Process next
             Low (P2): Process last

Use case: Critical tasks first
Example: Premium user requests prioritized
\`\`\`

### **3. Delayed/Scheduled Queue**

\`\`\`
Producer → Delayed Queue → Consumer (after delay)
             Message 1: Process in 5 minutes
             Message 2: Process tomorrow

Use case: Scheduled tasks
Example: Send reminder email after 24 hours
\`\`\`

### **4. Dead Letter Queue (DLQ)**

\`\`\`
Main Queue → Consumer (fails 3 times) → Dead Letter Queue

Purpose: Handle poison messages that repeatedly fail
Action: Manual investigation, reprocess, or discard
\`\`\`

**Example:**
\`\`\`
Message: { order_id: "invalid", customer_id: null }
                                                    ↑
                                            Malformed data

Retry 1: Fail (customer_id null)
Retry 2: Fail (customer_id null)
Retry 3: Fail (customer_id null)
→ Move to DLQ for manual inspection

✅ Prevents blocking the entire queue
\`\`\`

---

## When to Use Message Queues

### **Use Message Queues When:**

✅ **Services need to be decoupled**
- Microservices architecture
- Independent deployment and scaling

✅ **Traffic is bursty**
- E-commerce (sale events)
- Social media (viral posts)

✅ **Long-running operations**
- Video processing (minutes)
- Report generation (hours)
- ML model training (days)

✅ **Reliability is critical**
- Financial transactions
- Order processing
- Inventory updates

✅ **Work can be asynchronous**
- Email sending
- Notifications
- Background jobs

### **Don't Use Message Queues When:**

❌ **Need immediate response**
- User authentication (must be synchronous)
- Real-time chat (WebSockets better)
- Synchronous API calls

❌ **Complexity not justified**
- Simple CRUD operations
- Small monolithic apps
- Direct service calls sufficient

❌ **Strong consistency required across services**
- Distributed transactions
- Two-phase commit needed
- (Consider saga pattern with compensation)

---

## Popular Message Queue Technologies

### **1. RabbitMQ**
- Full-featured message broker
- Supports multiple protocols (AMQP, MQTT, STOMP)
- Complex routing capabilities
- Good for: Traditional enterprise, flexible routing

### **2. Apache Kafka**
- Distributed streaming platform
- High throughput (millions of messages/sec)
- Durable, replicated, partitioned
- Good for: Event streaming, high scale, replay capability

### **3. AWS SQS (Simple Queue Service)**
- Fully managed
- Standard (best-effort ordering) and FIFO (ordered)
- Easy integration with AWS ecosystem
- Good for: AWS-based systems, low maintenance

### **4. AWS SNS (Simple Notification Service)**
- Pub/sub messaging
- Fan-out to multiple subscribers
- Push-based delivery
- Good for: Event broadcasting, notifications

### **5. Redis (Pub/Sub, Streams)**
- In-memory, extremely fast
- Pub/sub and streams support
- Good for: Real-time use cases, low latency

### **6. Google Cloud Pub/Sub**
- Fully managed, global
- At-least-once delivery
- Auto-scaling
- Good for: GCP-based systems, global scale

---

## Message Queue in System Design Interviews

### **How to Discuss:**1. **Identify async opportunities:**
   - "User doesn't need to wait for email sending"
   - "Processing can happen in background"

2. **Justify the choice:**
   - "We need decoupling for independent scaling"
   - "Traffic is bursty (10× during sales)"
   - "Long-running video processing (5 minutes)"

3. **Choose appropriate guarantee:**
   - "At-least-once with idempotency keys"
   - "Exactly-once for payment processing"

4. **Address failure scenarios:**
   - "Dead letter queue for poison messages"
   - "Retry with exponential backoff"

5. **Discuss trade-offs:**
   - "Added complexity (must manage queue)"
   - "Eventual consistency (not immediate)"
   - "Debugging harder (async flow)"

### **Example (E-commerce Order):**

\`\`\`
Interviewer: "Design an order processing system"

You: 
"I'll use message queues for several steps:

1. Order Placement (Synchronous):
   - User clicks "Buy" → Write to order DB → Return order_id
   - Response time: <100ms
   
2. Async Processing (Via Queues):
   - Payment Queue: Process payment (at-least-once + idempotency)
   - Inventory Queue: Update stock (exactly-once, inventory critical)
   - Email Queue: Send confirmation (at-most-once, duplicates OK)
   - Analytics Queue: Log event (at-most-once, loss acceptable)

Benefits:
- Fast user response (100ms vs 5+ seconds)
- Independent scaling (payment workers vs email workers)
- Reliability (queues persist messages)
- Graceful degradation (if email slow, doesn't block order)

Trade-offs:
- Eventual consistency (email sent 5 seconds later)
- More complex (must manage queues, monitor)
- Harder debugging (distributed async flow)

For this use case, trade-offs worth it due to scale (millions of orders)."
\`\`\`

---

## Key Takeaways

1. **Message queues enable asynchronous communication** → Decoupling, reliability, scalability
2. **Queue (point-to-point) vs Topic (pub/sub)** → One consumer vs many subscribers
3. **Delivery guarantees matter** → At-most-once, at-least-once, exactly-once
4. **Ordering is complex in distributed systems** → Per-partition ordering most common
5. **Idempotency handles duplicates** → Design consumers to handle reprocessing safely
6. **Durable queues persist to disk** → Reliability vs performance trade-off
7. **Dead letter queues handle failures** → Prevent poison messages from blocking queue
8. **Use queues for async, decoupled, reliable communication** → Not for real-time sync operations
9. **Popular options: Kafka (streaming), RabbitMQ (routing), SQS (managed)** → Choose based on needs
10. **In interviews: Justify async design, discuss guarantees, address failures** → Show deep understanding

---

**Next:** We'll dive deep into **Apache Kafka Architecture**, understanding how it achieves high throughput, durability, and scalability as a distributed streaming platform.`,
};
