/**
 * Discussion Questions for RabbitMQ
 */

import { QuizQuestion } from '../../../types';

export const rabbitmqQuiz: QuizQuestion[] = [
  {
    id: 'rabbitmq-dq-1',
    question:
      'Explain RabbitMQ exchange types (direct, topic, fanout, headers) with real-world examples. Design a logging system where different services send logs, and consumers subscribe to specific log levels and services. Which exchange type(s) would you use and why?',
    hint: 'Consider routing flexibility, filtering capabilities, and performance implications of each exchange type.',
    sampleAnswer: `RabbitMQ exchanges route messages to queues based on routing rules. Each exchange type offers different routing capabilities.

**Exchange Types:**

**1. Direct Exchange:**
- Routes based on exact routing key match
- Use case: Task distribution by priority

**2. Topic Exchange:**
- Routes based on wildcard pattern matching
- Routing key: word.word.word
- Wildcards: * (one word), # (zero or more words)
- Use case: Flexible pub/sub with filtering

**3. Fanout Exchange:**
- Routes to all bound queues (broadcast)
- Ignores routing key
- Use case: Event broadcasting

**4. Headers Exchange:**
- Routes based on message headers (not routing key)
- Use case: Complex routing logic

**Logging System Design:**

**Requirements:**
- Services: api-service, db-service, auth-service
- Log levels: DEBUG, INFO, WARN, ERROR
- Consumers subscribe to specific combinations

**Solution: Topic Exchange ✅**

\`\`\`python
# Exchange
channel.exchange_declare(exchange='logs', exchange_type='topic')

# Producers (services send logs)
# api-service
channel.basic_publish(
    exchange='logs',
    routing_key='api.ERROR',  # service.level
    body='API error occurred'
)

# db-service
channel.basic_publish(
    exchange='logs',
    routing_key='db.WARN',
    body='Slow query detected'
)

# Consumers (subscribe to patterns)
# Consumer 1: All ERROR logs
channel.queue_bind(exchange='logs', queue='errors', routing_key='*.ERROR')
# Receives: api.ERROR, db.ERROR, auth.ERROR

# Consumer 2: All api-service logs
channel.queue_bind(exchange='logs', queue='api_logs', routing_key='api.*')
# Receives: api.DEBUG, api.INFO, api.WARN, api.ERROR

# Consumer 3: WARN and ERROR from all services
channel.queue_bind(exchange='logs', queue='alerts', routing_key='*.WARN')
channel.queue_bind(exchange='logs', queue='alerts', routing_key='*.ERROR')

# Consumer 4: Everything (monitoring)
channel.queue_bind(exchange='logs', queue='monitoring', routing_key='#')
\`\`\`

**Why Topic Exchange:**
✅ Flexible routing (patterns)
✅ Multiple bindings per queue
✅ Subscribers choose what they receive
✅ Easy to add new services/levels`,
    keyPoints: [
      'Direct: Exact match routing (task queues, priorities)',
      'Topic: Pattern matching with wildcards (logs, events)',
      'Fanout: Broadcast to all queues (event broadcasting)',
      'Headers: Route by message headers (complex logic)',
      'Choose topic exchange for flexible filtering (*.ERROR, api.*, #)',
      'Multiple bindings allow consumers to subscribe to multiple patterns',
    ],
  },
  {
    id: 'rabbitmq-dq-2',
    question:
      'Design a RabbitMQ architecture for an e-commerce order processing system that handles order placement, payment, inventory, and email notifications. Include exchange topology, queue configuration, acknowledgments, and dead letter queues. How would you ensure reliability and handle failures?',
    hint: 'Consider fanout for event broadcasting, DLQ for failures, acknowledgments for reliability, and quorum queues for HA.',
    sampleAnswer: `E-commerce order processing requires reliable message delivery with proper failure handling.

**Architecture:**

\`\`\`
Order API
  ↓
Exchange: "orders" (fanout)
  ├→ Queue: payment-queue → Payment Service
  ├→ Queue: inventory-queue → Inventory Service
  ├→ Queue: email-queue → Email Service
  └→ Queue: analytics-queue → Analytics Service

Each queue:
- Durable (survives restarts)
- Quorum queue (replicated)
- DLQ configured (failure handling)
- Manual acknowledgments
\`\`\`

**Implementation:**

\`\`\`python
# Declare fanout exchange
channel.exchange_declare(exchange='orders', exchange_type='fanout', durable=True)

# Declare DLX for failed messages
channel.exchange_declare(exchange='orders-dlx', exchange_type='direct', durable=True)

# Declare queues with DLQ
for queue_name in ['payment-queue', 'inventory-queue', 'email-queue']:
    # Main queue
    channel.queue_declare(
        queue=queue_name,
        durable=True,
        arguments={
            'x-queue-type': 'quorum',  # Replicated
            'x-dead-letter-exchange': 'orders-dlx',
            'x-dead-letter-routing-key': queue_name + '-failed',
            'x-message-ttl': 3600000  # 1 hour TTL
        }
    )
    
    # Bind to exchange
    channel.queue_bind(exchange='orders', queue=queue_name)
    
    # DLQ
    channel.queue_declare(queue=queue_name + '-dlq', durable=True)
    channel.queue_bind(
        exchange='orders-dlx',
        queue=queue_name + '-dlq',
        routing_key=queue_name + '-failed'
    )

# Producer: Publish order
def publish_order(order):
    channel.basic_publish(
        exchange='orders',
        routing_key='',  # Ignored by fanout
        body=json.dumps(order),
        properties=pika.BasicProperties(
            delivery_mode=2,  # Persistent
            content_type='application/json'
        )
    )

# Consumer: Payment service
def process_payment(ch, method, properties, body):
    try:
        order = json.loads(body)
        charge_card(order)
        ch.basic_ack(delivery_tag=method.delivery_tag)  # Success
    except PaymentError as e:
        logger.error(f"Payment failed: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)  # → DLQ
    except Exception as e:
        logger.error(f"Transient error: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)  # Retry

channel.basic_consume(queue='payment-queue', on_message_callback=process_payment)
\`\`\`

**Reliability Features:**
✅ Quorum queues (replicated, fault-tolerant)
✅ Persistent messages (survive restarts)
✅ Manual ACKs (process then acknowledge)
✅ DLQ (failed messages isolated)
✅ Fanout (all services receive order)`,
    keyPoints: [
      'Fanout exchange broadcasts order to all service queues',
      'Quorum queues provide replication and high availability',
      'Manual acknowledgments ensure message not lost (process then ACK)',
      'DLQ isolates failed messages after retries',
      'Persistent messages and durable queues survive broker restarts',
      'Separate queues per service enable independent scaling',
    ],
  },
  {
    id: 'rabbitmq-dq-3',
    question:
      'Compare RabbitMQ to Kafka for different use cases. For each scenario, which would you choose: (1) Background job processing, (2) Real-time event streaming with replay, (3) RPC between microservices, (4) Low-latency notifications. Explain your reasoning.',
    hint: 'Consider latency, throughput, message retention, replay capability, and operational complexity.',
    sampleAnswer: `RabbitMQ and Kafka have different strengths suited for different use cases.

**Comparison:**

| Aspect | RabbitMQ | Kafka |
|--------|----------|-------|
| **Model** | Message broker | Distributed log |
| **Latency** | Very low (μs-ms) | Low (ms) |
| **Throughput** | Moderate (10K/s) | Very high (1M/s) |
| **Retention** | Until consumed | Days/weeks |
| **Replay** | No | Yes |
| **Routing** | Flexible (exchanges) | Simple (topics) |

**Scenario Analysis:**

**1. Background Job Processing:**

**Choice: RabbitMQ ✅**

**Why:**
- Perfect fit for task queue pattern
- Work distribution automatic (competing consumers)
- Acknowledgments ensure reliability
- Priority queues (urgent jobs first)
- Simpler than Kafka for this use case

\`\`\`python
# Producer: Enqueue job
channel.basic_publish(
    exchange='',
    routing_key='jobs',
    body=json.dumps(job),
    properties=pika.BasicProperties(
        priority=job.priority,  # 0-10
        delivery_mode=2
    )
)

# Consumers: Workers compete for jobs
def process_job(ch, method, properties, body):
    job = json.loads(body)
    execute_job(job)
    ch.basic_ack(delivery_tag=method.delivery_tag)
\`\`\`

**2. Real-Time Event Streaming with Replay:**

**Choice: Kafka ✅**

**Why:**
- Retention allows replay (days/weeks)
- Multiple consumer groups read independently
- High throughput (millions of events)
- Replay for ML training, debugging, recovery

**Use case**: Click stream analytics
- Raw events retained 30 days
- Analytics service reads real-time
- ML service replays historical data
- Data warehouse batch loads

**3. RPC Between Microservices:**

**Choice: RabbitMQ ✅**

**Why:**
- Request-reply pattern built-in
- Lower latency (<10ms vs Kafka 50ms+)
- Reply-to queue and correlation ID
- Simpler for synchronous communication

\`\`\`python
# RPC Client
def call_rpc(request):
    corr_id = str(uuid.uuid4())
    channel.basic_publish(
        exchange='',
        routing_key='rpc_queue',
        properties=pika.BasicProperties(
            reply_to=callback_queue,
            correlation_id=corr_id
        ),
        body=request
    )
    # Wait for response...
    return response

# RPC Server
def on_request(ch, method, props, body):
    response = process_request(body)
    ch.basic_publish(
        exchange='',
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id=props.correlation_id),
        body=response
    )
\`\`\`

**4. Low-Latency Notifications:**

**Choice: RabbitMQ ✅**

**Why:**
- Sub-millisecond latency
- Direct routing (faster than Kafka)
- Push to connected clients immediately
- Transient messages (no persistence overhead)

**Use case**: Chat notifications
- Message sent → RabbitMQ → WebSocket servers → Users
- Latency: <5ms end-to-end

**Summary:**

- **RabbitMQ**: Task queues, RPC, low latency, flexible routing
- **Kafka**: Event streaming, high throughput, replay, analytics

Both can work for many scenarios; choice depends on priorities.`,
    keyPoints: [
      'RabbitMQ: Task queues (background jobs), RPC, low latency',
      'Kafka: Event streaming, high throughput, replay capability',
      'RabbitMQ lower latency (<10ms), Kafka higher throughput (1M+ msg/s)',
      'Kafka retains messages for replay, RabbitMQ deletes after consumption',
      'Choose based on requirements: latency vs throughput vs replay',
      'RabbitMQ simpler for traditional messaging, Kafka for event streaming',
    ],
  },
];
