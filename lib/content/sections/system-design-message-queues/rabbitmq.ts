/**
 * RabbitMQ Section
 */

export const rabbitmqSection = {
  id: 'rabbitmq',
  title: 'RabbitMQ',
  content: `RabbitMQ is a mature, feature-rich message broker that implements the Advanced Message Queuing Protocol (AMQP). It\'s known for flexible routing, reliable delivery, and support for multiple messaging patterns.

## What is RabbitMQ?

**RabbitMQ** = Message Broker + AMQP Protocol + Flexible Routing + Multi-Protocol Support

### **Key Characteristics:**1. **Message broker**: Routes messages between producers and consumers
2. **AMQP protocol**: Standard wire protocol for messaging
3. **Flexible routing**: Complex routing via exchanges and bindings
4. **Reliability**: Message persistence, acknowledgments, clustering
5. **Management UI**: Built-in web interface for monitoring
6. **Plugins**: Extensive plugin ecosystem

### **RabbitMQ vs Kafka:**

\`\`\`
RabbitMQ:
- Message broker (smart broker, dumb consumer)
- Low latency (microseconds)
- Complex routing (exchanges, bindings)
- Traditional queue model
- Message deleted after consumption
- Good for: Task queues, RPC, complex routing

Kafka:
- Distributed commit log (dumb broker, smart consumer)
- Low latency (milliseconds)
- Simple routing (topics, partitions)
- Event streaming model
- Messages retained (days/weeks)
- Good for: Event streaming, high throughput, replay

Choose RabbitMQ when:
✅ Need complex routing (topic-based, header-based)
✅ Traditional task queue pattern
✅ RPC (request-reply) needed
✅ Lower latency critical (<10ms)
✅ Messages small and transient
✅ Moderate throughput (< 100K msg/sec)

Choose Kafka when:
✅ High throughput (millions msg/sec)
✅ Event streaming
✅ Message replay needed
✅ Multiple consumers, independent consumption
✅ Large messages, long retention
\`\`\`

---

## RabbitMQ Architecture

### **Core Components:**

\`\`\`
Producer → Exchange → Queue → Consumer
            ↓
         Binding
         (routing rules)

Producer: Publishes messages to exchanges
Exchange: Routes messages based on rules
Queue: Stores messages until consumed
Consumer: Receives and processes messages
Binding: Links exchange to queue with routing key
\`\`\`

### **Message Flow:**

\`\`\`
1. Producer sends message to Exchange
   Message: {
     routing_key: "order.created",
     body: { order_id: "12345", ... }
   }

2. Exchange evaluates routing key
   - Checks bindings
   - Determines target queues

3. Message copied to matching queues
   Queue 1: order.created (matches!)
   Queue 2: order.* (matches!)
   Queue 3: inventory.* (no match)

4. Consumers receive from queues
   Consumer A ← Queue 1
   Consumer B ← Queue 2

5. Consumers acknowledge (ACK)
   → Message deleted from queue
\`\`\`

---

## Exchanges

An **exchange** receives messages from producers and routes them to queues based on rules.

### **1. Direct Exchange**

Routes messages with **exact routing key match**:

\`\`\`
Exchange: "direct-exchange" (type: direct)

Bindings:
- Queue "errors" bound with routing_key="error"
- Queue "warnings" bound with routing_key="warning"
- Queue "info" bound with routing_key="info"

Messages:
publish (routing_key="error", body="Fatal error")
  → Routed to "errors" queue

publish (routing_key="warning", body="Disk space low")
  → Routed to "warnings" queue

publish (routing_key="debug", body="Debug info")
  → Not routed (no matching binding) → Dropped

Use case: Severity-based routing (errors, warnings, info)
\`\`\`

**Example:**

\`\`\`python
import pika

connection = pika.BlockingConnection (pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare exchange
channel.exchange_declare (exchange='logs_direct', exchange_type='direct')

# Declare queues
channel.queue_declare (queue='error_queue')
channel.queue_declare (queue='warning_queue')

# Bind queues to exchange
channel.queue_bind (exchange='logs_direct', queue='error_queue', routing_key='error')
channel.queue_bind (exchange='logs_direct', queue='warning_queue', routing_key='warning')

# Publish messages
channel.basic_publish (exchange='logs_direct', routing_key='error', body='Fatal error occurred')
channel.basic_publish (exchange='logs_direct', routing_key='warning', body='Disk space low')
\`\`\`

### **2. Topic Exchange**

Routes messages with **pattern matching** on routing key:

\`\`\`
Exchange: "topic-exchange" (type: topic)

Routing key format: word.word.word (dot-separated)
Wildcards:
  * (star) = exactly one word
  # (hash) = zero or more words

Bindings:
- Queue "orders" bound with pattern="order.*"
- Queue "all_logs" bound with pattern="#"
- Queue "critical" bound with pattern="*.critical"

Messages:
publish (routing_key="order.created", ...)
  → Matches "order.*" → orders queue
  → Matches "#" → all_logs queue

publish (routing_key="payment.critical", ...)
  → Matches "*.critical" → critical queue
  → Matches "#" → all_logs queue

publish (routing_key="user.login.success", ...)
  → Matches "#" → all_logs queue only

Use case: Flexible pub/sub (logging, monitoring, events)
\`\`\`

**Example:**

\`\`\`python
# Declare topic exchange
channel.exchange_declare (exchange='events', exchange_type='topic')

# Bind queues with patterns
channel.queue_bind (exchange='events', queue='order_queue', routing_key='order.*')
channel.queue_bind (exchange='events', queue='payment_queue', routing_key='payment.#')
channel.queue_bind (exchange='events', queue='critical_queue', routing_key='*.critical')

# Publish
channel.basic_publish (exchange='events', routing_key='order.created', body='New order')
channel.basic_publish (exchange='events', routing_key='payment.credit.card', body='Payment')
channel.basic_publish (exchange='events', routing_key='system.critical', body='CPU high')
\`\`\`

### **3. Fanout Exchange**

Routes messages to **all bound queues** (broadcast):

\`\`\`
Exchange: "fanout-exchange" (type: fanout)

Bindings (routing key ignored):
- Queue "analytics"
- Queue "archive"
- Queue "monitoring"

Message:
publish (routing_key="anything", body="User registered")
  → Routed to ALL queues:
    - analytics queue
    - archive queue
    - monitoring queue

Use case: Broadcasting events (user actions, system events)
\`\`\`

**Example:**

\`\`\`python
# Declare fanout exchange
channel.exchange_declare (exchange='user_events', exchange_type='fanout')

# Multiple queues bound (routing key irrelevant)
channel.queue_bind (exchange='user_events', queue='analytics_queue')
channel.queue_bind (exchange='user_events', queue='email_queue')
channel.queue_bind (exchange='user_events', queue='notification_queue')

# Publish once, delivered to all queues
channel.basic_publish (exchange='user_events', routing_key='', body='User signed up')

# All three queues receive the message
\`\`\`

### **4. Headers Exchange**

Routes based on **message headers** instead of routing key:

\`\`\`
Exchange: "headers-exchange" (type: headers)

Bindings:
- Queue "jpg_queue" bound with headers={format: "jpg", x-match: "any"}
- Queue "large_queue" bound with headers={size: "large", x-match: "all"}

Message 1:
headers={format: "jpg", size: "small"}
  → Routed to "jpg_queue" (format matches, x-match: "any")
  → NOT routed to "large_queue" (size doesn't match, x-match: "all")

Message 2:
headers={format: "png", size: "large"}
  → Routed to "large_queue" (size matches)
  → NOT routed to "jpg_queue" (format doesn't match)

x-match:
  "any": Match if any header matches
  "all": Match only if all headers match

Use case: Complex routing based on metadata
\`\`\`

---

## Queues

A **queue** stores messages until consumed.

### **Queue Properties:**

\`\`\`python
channel.queue_declare(
    queue='my_queue',
    durable=True,        # Survive broker restart
    exclusive=False,     # Can be accessed by multiple connections
    auto_delete=False,   # Don't delete when last consumer disconnects
    arguments={
        'x-message-ttl': 60000,         # Message TTL: 60 seconds
        'x-max-length': 10000,          # Max 10K messages
        'x-max-length-bytes': 10485760, # Max 10MB
        'x-dead-letter-exchange': 'dlx', # Dead letter exchange
        'x-max-priority': 10            # Priority queue (0-10)
    }
)
\`\`\`

### **Queue Types:**

**1. Classic Queue:**
\`\`\`
Traditional queue implementation
✅ Reliable, well-tested
✅ Supports all features
❌ Limited throughput
❌ Not distributed
\`\`\`

**2. Quorum Queue (Recommended):**
\`\`\`
Replicated queue using Raft consensus
✅ High availability (replicated)
✅ Data safety (majority quorum)
✅ Handles node failures
❌ Higher latency (replication overhead)

Declare:
channel.queue_declare(
    queue='my_quorum_queue',
    durable=True,
    arguments={'x-queue-type': 'quorum'}
)

Use case: Critical messages (orders, payments)
\`\`\`

**3. Stream Queue:**
\`\`\`
Kafka-like append-only log
✅ High throughput
✅ Message replay
✅ Non-destructive reads
❌ No priority
❌ Limited filtering

Use case: Event streams, audit logs
\`\`\`

---

## Message Acknowledgments

### **Consumer Acknowledgments:**

\`\`\`python
# Auto-acknowledge (not recommended)
channel.basic_consume (queue='my_queue', on_message_callback=callback, auto_ack=True)
# ❌ Message deleted before processing
# ❌ Lost if consumer crashes

# Manual acknowledge (recommended)
def callback (ch, method, properties, body):
    try:
        process_message (body)
        ch.basic_ack (delivery_tag=method.delivery_tag)  # Success
    except RecoverableError:
        ch.basic_nack (delivery_tag=method.delivery_tag, requeue=True)  # Retry
    except PermanentError:
        ch.basic_nack (delivery_tag=method.delivery_tag, requeue=False)  # Discard

channel.basic_consume (queue='my_queue', on_message_callback=callback, auto_ack=False)

✅ Message only deleted after successful processing
✅ Requeue on failure
✅ At-least-once delivery
\`\`\`

### **Publisher Confirms:**

\`\`\`python
# Enable publisher confirms
channel.confirm_delivery()

try:
    channel.basic_publish (exchange='', routing_key='my_queue', body='Hello')
    # If we get here, message was confirmed by broker
    print("Message delivered")
except pika.exceptions.UnroutableError:
    print("Message could not be routed")

✅ Guaranteed delivery to broker
✅ Detect routing failures
\`\`\`

---

## Message Patterns

### **1. Work Queue (Load Balancing)**

\`\`\`
Producer → Queue → Worker 1
                 → Worker 2
                 → Worker 3

Each message delivered to ONE worker
Load balanced across workers

Use case: Task distribution, job processing
\`\`\`

**Example:**

\`\`\`python
# Producer
for i in range(100):
    channel.basic_publish (exchange='', routing_key='tasks', body=f'Task {i}')

# Workers (multiple instances)
def callback (ch, method, properties, body):
    print(f"Processing {body}")
    time.sleep(2)  # Simulate work
    ch.basic_ack (delivery_tag=method.delivery_tag)

channel.basic_qos (prefetch_count=1)  # Fair dispatch (one task per worker)
channel.basic_consume (queue='tasks', on_message_callback=callback)
channel.start_consuming()

# Worker 1 gets Task 0, Task 3, Task 6, ...
# Worker 2 gets Task 1, Task 4, Task 7, ...
# Worker 3 gets Task 2, Task 5, Task 8, ...
\`\`\`

### **2. Publish/Subscribe (Fanout)**

\`\`\`
Producer → Exchange (fanout) → Queue 1 → Consumer 1
                              → Queue 2 → Consumer 2
                              → Queue 3 → Consumer 3

Each consumer receives ALL messages

Use case: Event broadcasting, logging
\`\`\`

### **3. Routing (Direct/Topic)**

\`\`\`
Producer → Exchange (topic) → Queue "error" → Consumer 1
                            → Queue "info" → Consumer 2

Selective delivery based on routing key

Use case: Log levels, event types
\`\`\`

### **4. RPC (Request-Reply)**

\`\`\`
Client → Request Queue → Server
       ← Reply Queue   ←

RPC over message queue

Implementation:
1. Client sends request to "rpc_queue"
2. Request includes reply_to queue and correlation_id
3. Server processes and sends response to reply_to queue
4. Client receives response with matching correlation_id
\`\`\`

**Example:**

\`\`\`python
# RPC Client
class RpcClient:
    def call (self, n):
        self.response = None
        self.corr_id = str (uuid.uuid4())
        
        # Send request
        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id
            ),
            body=str (n)
        )
        
        # Wait for response
        while self.response is None:
            self.connection.process_data_events()
        
        return int (self.response)

# RPC Server
def on_request (ch, method, props, body):
    n = int (body)
    response = fibonacci (n)  # Calculate
    
    # Send response
    ch.basic_publish(
        exchange='',
        routing_key=props.reply_to,
        properties=pika.BasicProperties (correlation_id=props.correlation_id),
        body=str (response)
    )
    ch.basic_ack (delivery_tag=method.delivery_tag)
\`\`\`

---

## Dead Letter Exchanges (DLX)

Handle messages that cannot be delivered:

\`\`\`python
# Declare DLX
channel.exchange_declare (exchange='dead_letters', exchange_type='direct')

# Declare DLQ
channel.queue_declare (queue='failed_messages')
channel.queue_bind (exchange='dead_letters', queue='failed_messages', routing_key='failed')

# Declare main queue with DLX
channel.queue_declare(
    queue='main_queue',
    arguments={
        'x-dead-letter-exchange': 'dead_letters',
        'x-dead-letter-routing-key': 'failed'
    }
)

# Messages sent to DLX when:
# - Consumer rejects with requeue=False
# - Message TTL expires
# - Queue length limit exceeded

# Use case: Failed message inspection, retry logic
\`\`\`

---

## RabbitMQ Clustering and High Availability

### **Clustering:**

\`\`\`
Cluster: 3 nodes (rabbit@node1, rabbit@node2, rabbit@node3)

Shared:
- Exchanges
- Queue metadata
- Bindings

NOT Shared (by default):
- Queue contents (stored on one node)

Failover:
- If node fails, queues on that node unavailable ❌
- Need replication for HA

Benefits:
- Share load across nodes
- Single management interface
- Horizontal scaling (throughput)
\`\`\`

### **Quorum Queues (HA):**

\`\`\`
Quorum Queue: Replicated across nodes using Raft

Node 1 (Leader): [Msg1, Msg2, Msg3]
Node 2 (Follower): [Msg1, Msg2, Msg3]
Node 3 (Follower): [Msg1, Msg2, Msg3]

If Node 1 fails:
- Node 2 or Node 3 elected as leader
- Queue remains available ✅
- No message loss ✅

Tradeoff: Higher latency (replication overhead)

Declare:
arguments={'x-queue-type': 'quorum', 'x-quorum-initial-group-size': 3}
\`\`\`

---

## Performance Tuning

### **1. Prefetch Count:**

\`\`\`python
# Without prefetch limit
channel.basic_qos (prefetch_count=0)  # Default
# RabbitMQ sends all messages to consumer immediately
# ❌ Consumer overwhelmed
# ❌ Uneven load distribution

# With prefetch limit
channel.basic_qos (prefetch_count=10)
# RabbitMQ sends max 10 messages to consumer
# Consumer acknowledges → RabbitMQ sends more
# ✅ Backpressure
# ✅ Fair distribution
\`\`\`

### **2. Lazy Queues:**

\`\`\`python
# Regular queue: Messages in RAM (fast but limited)
# Lazy queue: Messages on disk (slower but scalable)

channel.queue_declare(
    queue='large_queue',
    arguments={'x-queue-mode': 'lazy'}
)

# Use case: Millions of messages, consumers slow
\`\`\`

### **3. Message TTL:**

\`\`\`python
# Per-queue TTL
channel.queue_declare (queue='temp_queue', arguments={'x-message-ttl': 60000})  # 60 sec

# Per-message TTL
channel.basic_publish(
    exchange='',
    routing_key='temp_queue',
    body='Expires soon',
    properties=pika.BasicProperties (expiration='60000')  # 60 sec
)

# Use case: Temporary data, cleanup
\`\`\`

---

## Monitoring and Management

### **Management UI:**

\`\`\`
http://localhost:15672
Username: guest
Password: guest

Dashboard shows:
- Overview: Message rates, queue depths
- Connections: Active connections
- Channels: Active channels
- Exchanges: Exchange list
- Queues: Queue list, stats
- Admin: Users, virtual hosts, policies

✅ Real-time monitoring
✅ Manual queue management
✅ Policy configuration
\`\`\`

### **Key Metrics:**

\`\`\`
1. Queue Depth
   - Number of messages in queue
   - Alert if > threshold (backlog)

2. Message Rates
   - Publish rate, deliver rate
   - Alert if deliver < publish (consumers slow)

3. Consumer Count
   - Number of active consumers
   - Alert if 0 (no consumers)

4. Memory Usage
   - RabbitMQ memory consumption
   - Alert if > 80% (scale or optimize)

5. Disk Space
   - For persistent messages
   - Alert if low

6. Connection Count
   - Active connections
   - Monitor for leaks
\`\`\`

---

## RabbitMQ Best Practices

### **1. Use Quorum Queues for Critical Data:**

\`\`\`python
arguments={'x-queue-type': 'quorum'}
# High availability, no data loss
\`\`\`

### **2. Enable Publisher Confirms:**

\`\`\`python
channel.confirm_delivery()
# Ensure message delivered to broker
\`\`\`

### **3. Manual Acknowledgments:**

\`\`\`python
auto_ack=False
# Control when messages deleted
\`\`\`

### **4. Set Prefetch Count:**

\`\`\`python
channel.basic_qos (prefetch_count=10)
# Prevent consumer overload
\`\`\`

### **5. Use Dead Letter Exchanges:**

\`\`\`python
'x-dead-letter-exchange': 'dlx'
# Handle failed messages
\`\`\`

### **6. Monitor Queue Depth:**

\`\`\`
Alert if queue depth > 10K messages
Scale consumers or investigate slow processing
\`\`\`

### **7. Idempotent Consumers:**

\`\`\`python
# Handle message redelivery
if not already_processed (message_id):
    process (message)
\`\`\`

---

## RabbitMQ in System Design Interviews

### **When to Propose RabbitMQ:**

✅ **Task queues** (background jobs)
✅ **RPC patterns** (microservices communication)
✅ **Complex routing** (topic-based, header-based)
✅ **Low latency** (<10ms)
✅ **Traditional messaging patterns**
✅ **Moderate scale** (<100K msg/sec)

### **Example Interview Discussion:**

\`\`\`
Interviewer: "Design an email notification system for e-commerce"

You:
"I'll use RabbitMQ for reliable email delivery:

Architecture:
Order Service → RabbitMQ → Email Workers → SMTP Server

Exchange Setup:
- Exchange: "notifications" (type: topic)
- Routing keys: "email.order", "email.shipping", "email.marketing"

Queues:
- "email_queue" (quorum queue for HA)
- Bound to "email.*" (all email notifications)
- DLX configured for failed emails

Publisher (Order Service):
- Publish to "notifications" exchange
- Routing key: "email.order"
- Publisher confirms enabled (ensure delivery)
- Message: { user_id, email, order_id, template }

Consumers (Email Workers):
- 5 worker instances (horizontal scaling)
- Prefetch count: 10 (fair dispatch)
- Manual ACK after email sent
- Retry logic: 3 attempts with exponential backoff
- Failed emails → DLQ for investigation

Benefits:
- Decoupling: Order service doesn't wait for email
- Reliability: Quorum queue, publisher confirms, ACKs
- Scalability: Add workers as needed
- Error handling: DLQ for failed emails
- Monitoring: Queue depth, delivery rate

Scale:
- 10K orders/hour = 2.8 orders/sec
- Email sending: 500ms/email
- Single worker: 2 emails/sec
- Need 2 workers (5 for headroom)

Why RabbitMQ over Kafka:
- Traditional task queue pattern (perfect fit)
- RPC if needed (order confirmation)
- Lower latency (<10ms)
- Simpler for this use case
- Moderate throughput (2.8 msg/sec << 100K limit)
"
\`\`\`

---

## Key Takeaways

1. **RabbitMQ = Message broker with flexible routing** → Exchanges, queues, bindings
2. **Four exchange types** → Direct (exact), Topic (pattern), Fanout (broadcast), Headers (metadata)
3. **Quorum queues for HA** → Replicated, fault-tolerant
4. **Manual ACKs recommended** → At-least-once delivery
5. **Publisher confirms ensure delivery** → No message loss
6. **Dead letter exchanges handle failures** → Failed message recovery
7. **Prefetch count prevents overload** → Backpressure, fair dispatch
8. **RPC pattern for request-reply** → Microservices communication
9. **Choose RabbitMQ for task queues, low latency** → Not for high-throughput streaming
10. **In interviews: Discuss routing, reliability, scaling** → Show messaging knowledge

---

**Next:** We'll explore **AWS SQS & SNS**—managed message queue and pub/sub services in AWS, and when to use managed vs self-hosted solutions.`,
};
