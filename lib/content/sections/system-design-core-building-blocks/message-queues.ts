/**
 * Message Queues & Async Processing Section
 */

export const messagequeuesSection = {
  id: 'message-queues',
  title: 'Message Queues & Async Processing',
  content: `Message queues enable asynchronous communication between services, decoupling producers and consumers for better scalability and reliability.

## What is a Message Queue?

**Definition**: A message queue is a form of asynchronous service-to-service communication where messages are stored in a queue until the consumer is ready to process them.

### **Why Use Message Queues?**

**Without Message Queues:**
- Services tightly coupled (caller waits for response)
- If consumer is slow, caller is blocked
- If consumer is down, caller fails
- No buffering (spike in traffic overwhelms consumer)
- Difficult to scale independently

**With Message Queues:**
- Services decoupled (producer sends message and continues)
- Asynchronous processing (consumer processes when ready)
- Buffering (queue absorbs traffic spikes)
- Resilience (retry failed messages automatically)
- Independent scaling (scale producers and consumers separately)

**Real-world**: Amazon uses SQS (Simple Queue Service) to process millions of orders per day asynchronously.

---

## Core Concepts

### **Producer**

**Role**: Creates and sends messages to the queue.

**Example**: Web server receives user signup request, sends message to queue: "New user: email@example.com"

**Characteristics:**
- Fire and forget (doesn't wait for processing)
- Fast response to user
- Can continue handling other requests

---

### **Consumer**

**Role**: Reads messages from queue and processes them.

**Example**: Background worker reads "New user" message, sends welcome email, creates database record.

**Characteristics:**
- Pulls messages from queue (or queue pushes to consumer)
- Processes at its own pace
- Can scale independently (add more consumers for high load)

---

### **Queue**

**Role**: Stores messages between producer and consumer.

**Characteristics:**
- FIFO (First In First Out) - typically
- Persistent (messages not lost if consumer crashes)
- Durable (survives broker restarts)
- Configurable retention (messages expire after N days)

---

### **Message**

**Structure:**
- **Body**: Actual data (JSON, XML, binary)
- **Attributes**: Metadata (timestamp, message ID, priority)
- **Headers**: Routing information

**Example message:**
- Message ID: "abc123"
- Timestamp: "2023-10-15T10:30:00Z"  
- Body: userId=12345, email="user@example.com", action="signup"

---

## Common Use Cases

### **1. Asynchronous Processing**

**Problem**: User uploads video (takes 10 minutes to transcode). Can't make user wait.

**Solution**: 
1. User uploads video → Server responds immediately "Upload successful"
2. Server sends message to queue: "Transcode video 12345"
3. Background worker processes transcoding
4. Notify user when done (via email or websocket)

**Benefit**: Fast user experience, heavy processing in background.

---

### **2. Load Leveling (Traffic Spike Handling)**

**Problem**: Black Friday sale → 100K orders/sec, but order processing service only handles 10K/sec.

**Solution**:
1. Orders sent to queue (accepts 100K/sec)
2. Queue buffers 100K messages
3. Order processing service consumes at 10K/sec
4. Queue drains over time (takes ~10 seconds)

**Benefit**: System doesn't crash, orders processed eventually.

---

### **3. Service Decoupling**

**Problem**: Payment service needs to notify: Email service, Analytics service, Inventory service. If any service is down, payment fails.

**Solution**:
1. Payment service sends "Payment completed" message to queue
2. Email, Analytics, Inventory each subscribe to queue
3. Each service processes message independently
4. If one service is down, others still work

**Benefit**: Services independent, no single point of failure.

---

### **4. Task Distribution (Work Queue)**

**Problem**: 1000 background jobs need processing (e.g., send 1000 emails). Single worker is slow.

**Solution**:
1. Producer sends 1000 messages to queue
2. 10 workers consume messages in parallel
3. Each worker processes 100 messages
4. Total time reduced 10×

**Benefit**: Horizontal scaling, faster processing.

---

## Queue vs Topic (Pub/Sub)

### **Queue (Point-to-Point)**

**Model**: One producer → Queue → One consumer (or consumer group)

**Characteristics:**
- Each message consumed by ONE consumer only
- Once consumed, message deleted from queue
- Used for task distribution

**Example**: Order processing queue
- Producer: Web server sends "Process order 123"
- Consumers: 5 workers compete for messages
- Each order processed by exactly one worker

**Use case**: Background jobs, task queues.

---

### **Topic (Publish/Subscribe)**

**Model**: One producer → Topic → Multiple subscribers

**Characteristics:**
- Each message delivered to ALL subscribers
- Message not deleted until all subscribers consume it
- Used for event broadcasting

**Example**: Payment completed event
- Producer: Payment service publishes "Payment completed for order 123"
- Subscribers: Email service, Analytics service, Inventory service
- All three receive and process the message independently

**Use case**: Event notifications, fan-out scenarios.

---

## Message Delivery Guarantees

### **At-Most-Once**

**Guarantee**: Message delivered 0 or 1 times (may be lost, never duplicated).

**How it works:**
1. Producer sends message (no acknowledgment required)
2. Message may be lost in transit
3. Consumer receives message, processes it (no acknowledgment)

**Use case**: Metrics, logs (occasional loss acceptable).

**Trade-off**: Fast, but unreliable.

---

### **At-Least-Once**

**Guarantee**: Message delivered 1 or more times (never lost, may be duplicated).

**How it works:**
1. Producer sends message, waits for acknowledgment
2. If no ack, producer retries (may result in duplicate)
3. Consumer receives message, processes it, sends acknowledgment
4. If consumer crashes before ack, message redelivered

**Use case**: Most common (order processing, emails).

**Trade-off**: Reliable, but consumer must handle duplicates (idempotency).

**Idempotency**: Processing same message multiple times has same effect as processing once.

**Example**: Email service tracks sent emails by message ID, skips if already sent.

---

### **Exactly-Once**

**Guarantee**: Message delivered exactly 1 time (never lost, never duplicated).

**How it works:**
- Complex: Requires distributed transactions (2-phase commit) or deduplication
- Kafka: Transactional producer + idempotent consumer
- Very expensive in terms of performance

**Use case**: Financial transactions (critical correctness).

**Trade-off**: Slow, complex, but guaranteed correctness.

---

## Message Ordering

### **FIFO (First In First Out)**

**Guarantee**: Messages processed in order sent.

**Use case**: Order status updates (must process "Order created" before "Order shipped").

**Implementation:**
- Single consumer (parallel consumers break ordering)
- Or: Partition by key (messages with same key go to same consumer)

**Example**: User actions
- Message 1: User created account
- Message 2: User updated profile
- Message 3: User deleted account
- Must process in order!

**Challenge**: Single consumer = no parallelism (slower).

---

### **Unordered**

**Guarantee**: No order guarantee (messages may be processed out of order).

**Use case**: Independent tasks (send emails, no order required).

**Benefit**: High parallelism (many consumers).

**Example**: Email notifications
- 1000 "Welcome email" messages
- Order doesn't matter
- 10 workers process in parallel

---

## Handling Failures

### **Dead Letter Queue (DLQ)**

**Problem**: Message processing fails repeatedly (bad data, bug, external service down).

**Solution**: After N retries, move message to Dead Letter Queue.

**Flow:**
1. Consumer tries to process message
2. Processing fails (exception)
3. Message returned to queue
4. Retry 3 times (configurable)
5. After 3 failures, move to DLQ
6. Alert operations team
7. Manually inspect and fix

**Example**: Email service fails to send (invalid email address). After 3 retries, move to DLQ for manual review.

**Configuration:**
- Max retries: 3
- Retry backoff: Exponential (1s, 2s, 4s)
- DLQ retention: 14 days

---

### **Exponential Backoff**

**Problem**: Service temporarily down. Retrying immediately overwhelms it.

**Solution**: Increase retry delay exponentially.

**Example:**
- Retry 1: Wait 1 second
- Retry 2: Wait 2 seconds
- Retry 3: Wait 4 seconds
- Retry 4: Wait 8 seconds
- Max: 60 seconds

**Benefit**: Gives service time to recover, reduces load.

---

## Popular Message Queue Systems

### **RabbitMQ**

**Type**: Traditional message broker

**Features:**
- Supports queues and topics
- AMQP protocol
- Good performance (~50K messages/sec)
- Easy to set up
- Complex routing (exchange types: direct, topic, fanout)

**Use case**: General-purpose message queue.

**Pros:**
- Mature, reliable
- Good monitoring (management UI)
- Flexible routing

**Cons:**
- Single point of failure (needs clustering)
- Lower throughput than Kafka

---

### **Apache Kafka**

**Type**: Distributed streaming platform

**Features:**
- Extremely high throughput (millions of messages/sec)
- Distributed (scales horizontally)
- Persistent log (messages retained for days/weeks)
- Replay messages (consumers can rewind)
- Partitioning for parallelism

**Use case**: High-throughput event streaming, log aggregation.

**Pros:**
- Massive scale
- Durability (messages persisted to disk)
- Replay capability

**Cons:**
- Complex to set up and operate
- Overkill for simple use cases

---

### **AWS SQS**

**Type**: Managed cloud message queue

**Features:**
- Fully managed (no servers)
- Auto-scaling
- At-least-once delivery
- FIFO queues available

**Use case**: Cloud-native applications.

**Pros:**
- Zero maintenance
- Pay-per-use
- Integrates with AWS services

**Cons:**
- Vendor lock-in
- Limited features vs RabbitMQ/Kafka

---

### **Redis (with Pub/Sub or Streams)**

**Type**: In-memory data store with messaging

**Features:**
- Very fast (in-memory)
- Simple pub/sub
- Streams (similar to Kafka, added in Redis 5.0)

**Use case**: Real-time messaging, lightweight queues.

**Pros:**
- Extremely fast
- Simple to set up
- Multi-purpose (cache + queue)

**Cons:**
- Not persistent by default (messages may be lost)
- Limited features vs dedicated message queues

---

## Comparison Table

**RabbitMQ / Kafka / AWS SQS / Redis:**

**Throughput**: 50K msg/sec / Millions msg/sec / 300K msg/sec / Very high

**Durability**: High / Very high / High / Low (unless configured)

**Ordering**: FIFO per queue / Per partition / FIFO queues / Pub/sub unordered

**Setup**: Medium / Hard / Easy (managed) / Easy

**Use case**: General / High-throughput streaming / Cloud-native / Real-time, lightweight

---

## Best Practices

**1. Idempotent Consumers**

Consumer must handle duplicate messages gracefully.

**Example**: Email service checks if email already sent before sending again.

**Implementation**: Track processed message IDs in database or cache.

---

**2. Monitor Queue Depth**

Track number of messages in queue.

**Alert if**: Queue depth growing (consumers too slow or down).

**Metrics**: Messages in queue, consumer lag, processing rate.

---

**3. Set Message TTL**

Messages expire after N hours/days.

**Why**: Prevent old messages from being processed (e.g., "Send email about sale" after sale ended).

**Configuration**: TTL = 24 hours for time-sensitive tasks.

---

**4. Use Dead Letter Queues**

Move failed messages to DLQ after retries.

**Why**: Prevent bad messages from blocking queue forever.

**Operations**: Monitor DLQ size, alert if growing.

---

**5. Partition for Parallelism**

Use multiple queues or partitions for parallel processing.

**Example**: 10 queues, 10 consumers (one per queue) for 10× throughput.

**Trade-off**: Ordering only guaranteed within partition.

---

## Real-World Examples

### **Uber**

**Use case**: Ride matching, surge pricing calculations.

**System**: Kafka for high-throughput event streaming.

**Scale**: Millions of events per second (GPS updates, ride requests).

---

### **Netflix**

**Use case**: Video encoding, recommendations.

**System**: AWS SQS for task distribution.

**Scale**: Thousands of workers processing millions of encoding jobs.

---

### **Slack**

**Use case**: Message delivery, notifications.

**System**: Custom message queue (similar to Kafka).

**Scale**: Billions of messages per day.

---

## Interview Tips

### **Common Questions:**

**Q: "Design a system to send 1 million emails asynchronously."**

✅ Good answer: "Use message queue:
1. Web server receives request to send emails
2. Produce 1M messages to queue (each message = one email)
3. 100 worker instances consume messages in parallel
4. Each worker sends 10K emails (1M / 100)
5. Use at-least-once delivery (retry failures)
6. Implement idempotency (track sent emails)
7. DLQ for emails that fail after retries (invalid addresses)
8. Monitor queue depth and worker health"

**Q: "What happens if message queue goes down?"**

✅ Good answer: "Impact:
- Producers can't send messages (fail or buffer locally)
- Consumers can't receive messages (processing stops)

Mitigation:
1. Use managed service (AWS SQS) with built-in HA
2. Run RabbitMQ/Kafka in cluster (multiple brokers)
3. Producer-side buffering (local disk queue)
4. Fallback to synchronous processing temporarily
5. Monitor queue health, alert on failures"

**Q: "Queue vs Database for task storage?"**

✅ Good answer: "Queue advantages:
- Optimized for FIFO, fast enqueue/dequeue
- Built-in retry, DLQ
- At-least-once delivery guarantees
- Better for high-throughput (millions/sec)

Database advantages:
- Queryable (find tasks by status, date)
- Transactions (atomicity)
- Complex filtering
- Persistent (never lose tasks)

Use queue for: Simple task distribution, high throughput.
Use database for: Complex querying, audit requirements."

---

## Key Takeaways

1. **Message queues decouple services** for better scalability and resilience
2. **Asynchronous processing** improves user experience (fast responses)
3. **Load leveling** absorbs traffic spikes, prevents system overload
4. **At-least-once delivery** is most common (consumers must be idempotent)
5. **Dead Letter Queue** handles failed messages after retries
6. **Kafka** for high throughput, **RabbitMQ** for general use, **SQS** for cloud-native
7. **Monitor queue depth** to detect consumer issues early`,
};
