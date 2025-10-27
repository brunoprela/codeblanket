/**
 * Discussion Questions for Message Queue Fundamentals
 */

import { QuizQuestion } from '../../../types';

export const messagequeuefundamentalsQuiz: QuizQuestion[] = [
  {
    id: 'mq-fundamentals-dq-1',
    question:
      'Compare point-to-point (queue) and publish-subscribe (topic) messaging patterns. For each pattern, provide specific real-world use cases where it excels, explain the trade-offs, and discuss scenarios where choosing the wrong pattern could lead to system problems.',
    hint: 'Think about message distribution, consumer independence, scaling characteristics, and failure isolation.',
    sampleAnswer: `Point-to-point (Queue) and Publish-Subscribe (Topic) represent fundamentally different messaging patterns with distinct characteristics and use cases.

**Point-to-Point (Queue) Pattern:**

In the queue pattern, each message is consumed by exactly one consumer from a group of competing consumers. Messages are distributed across consumers for load balancing.

**Architecture:**
- Producer → Queue → Consumer 1, Consumer 2, Consumer 3 (one receives message)
- Each message delivered to ONE consumer
- Consumers compete for messages
- Message deleted after successful consumption

**Use Cases:**1. **Task Distribution/Work Queues:**
   - Image processing pipeline: 1000 images → 10 worker processes
   - Each worker processes ~100 images (automatic load balancing)
   - Example: Video transcoding, report generation, batch email sending

2. **Job Processing:**
   - Background job execution where only one worker should handle each job
   - Example: Payment processing, order fulfillment
   - Critical that job executes exactly once (not multiple times)

3. **Rate Limiting/Throttling:**
   - Control processing rate by limiting consumer count
   - Example: API calls to third-party service (rate-limited to 100/sec)
   - Queue buffers burst traffic, consumers process at controlled rate

**Trade-offs:**
✅ Pros:
- Load balancing automatic (work distributed across consumers)
- Horizontal scaling (add consumers → higher throughput)
- Work persistence (messages buffered if consumers slow/down)
- Simple semantics (one processor per message)

❌ Cons:
- No message fan-out (can't send same message to multiple consumers)
- Coupling through shared queue (all consumers must understand message format)
- Ordering complex with multiple consumers (messages processed in parallel)

**Publish-Subscribe (Topic) Pattern:**

In pub/sub, each message is broadcast to all subscribed consumers. Each subscriber gets its own copy.

**Architecture:**
- Publisher → Topic → Subscriber 1 (gets copy)
                    → Subscriber 2 (gets copy)
                    → Subscriber 3 (gets copy)
- Each message delivered to ALL subscribers
- Subscribers independent of each other
- Fan-out: One message → Many recipients

**Use Cases:**1. **Event Broadcasting:**
   - User registration event → Email service, Analytics service, CRM service
   - Each service gets same event, processes independently
   - Example: E-commerce order placed → Payment, Inventory, Email, Analytics

2. **Real-time Notifications:**
   - Price updates → Multiple trading systems, dashboards, alert systems
   - Example: Stock price changes → All subscribed traders notified

3. **Cache Invalidation:**
   - Data update → Invalidate caches across multiple application instances
   - Example: User profile updated → All 100 API servers invalidate cache

**Trade-offs:**
✅ Pros:
- Fan-out (one message → many consumers)
- Loose coupling (subscribers don't know about each other)
- Extensibility (add subscribers without changing publisher)
- Independent scaling (each subscriber scales separately)

❌ Cons:
- No load balancing per subscriber (each gets all messages)
- Message duplication across subscribers (storage/bandwidth cost)
- Harder to ensure all subscribers processed message
- Backpressure complex (slow subscriber doesn't affect others)

**Choosing Wrong Pattern - Problems:**

**Problem 1: Using Queue for Event Broadcasting**
Scenario: User signup event
- Requirement: Email service, Analytics service, and CRM all need event
- Mistake: Use queue (only one consumer gets message)
- Result: Only one service processes event, others miss it ❌
- Fix: Use topic (pub/sub) so all services receive event

**Problem 2: Using Topic for Task Distribution**
Scenario: Image processing
- Requirement: 1000 images, process each exactly once
- Mistake: Use topic with multiple subscribers
- Result: Each subscriber processes ALL 1000 images (duplication!) ❌
- Waste: 10 workers × 1000 images = 10,000 processing operations (10× waste)
- Fix: Use queue for work distribution, one consumer per image

**Problem 3: Using Single Queue for Multiple Event Types**
Scenario: Mixed events (orders, payments, emails)
- Mistake: One queue, consumers filter by event type
- Result: Consumers receive irrelevant messages, waste processing
- Fix: Separate queues per event type, or use topic with filtering

**Hybrid Patterns:**

Many systems combine both:

**Example: E-commerce Order Processing**
\`\`\`
Order Service:
  ↓ Publishes to Topic: "orders" (pub/sub)
  
Subscribers:
  ↓ Payment Service → Queue: "payment-tasks" (queue)
      - Multiple workers process payments (load balanced)
  ↓ Email Service → Queue: "email-tasks" (queue)
      - Multiple workers send emails (load balanced)
  ↓ Analytics Service → Consumes directly (no queue needed)
      - Single instance, processes all events
\`\`\`

Topic for fan-out, queues for work distribution within each service.

**Real-World Example: Uber**

Ride request event uses pub/sub:
- Topic: "ride-requested"
- Subscribers:
  1. Matching Service (finds driver)
  2. Pricing Service (calculates surge)
  3. Analytics Service (tracks demand)
  4. Notification Service (alerts nearby drivers)

Each subscriber gets event, processes independently, scales separately.

Within Matching Service, use queue for work distribution:
- Queue: "matching-tasks"
- Multiple matching workers compete for tasks (load balanced)

**Decision Framework:**

Use Queue (Point-to-Point) when:
- Need load balancing across consumers
- Each message should be processed exactly once
- Work distribution required
- Example: Task processing, job queues, work pools

Use Topic (Pub/Sub) when:
- Need message fan-out (multiple recipients)
- Consumers independent of each other
- Extensibility important (add consumers without changes)
- Example: Event broadcasting, notifications, real-time updates

Use Both when:
- Fan-out needed (topic), then load balancing per service (queue)
- Example: Event-driven microservices

**Conclusion:**

The choice between queue and topic fundamentally impacts system architecture. Queues excel at load-balanced task distribution with single processing guarantee. Topics excel at event broadcasting with fan-out to multiple independent consumers. Many production systems use hybrid approaches: topics for service-to-service communication, queues within services for work distribution. Understanding these patterns deeply prevents architectural mistakes and enables building scalable, reliable distributed systems.`,
    keyPoints: [
      'Queue: One message → One consumer (load balancing, work distribution)',
      'Topic: One message → All subscribers (fan-out, event broadcasting)',
      'Queue use cases: Task processing, job queues, rate limiting',
      'Topic use cases: Event broadcasting, notifications, cache invalidation',
      'Wrong pattern causes problems: Missed events (queue instead of topic), duplicate work (topic instead of queue)',
      'Hybrid approach common: Topic for fan-out, queues for work distribution',
    ],
  },
  {
    id: 'mq-fundamentals-dq-2',
    question:
      'Explain the three message delivery guarantees (at-most-once, at-least-once, exactly-once) with concrete examples. For a payment processing system, analyze which guarantee is appropriate and how you would implement it in practice, including handling failures and ensuring idempotency.',
    hint: 'Consider the trade-offs between performance, reliability, and complexity. Think about failure scenarios and recovery mechanisms.',
    sampleAnswer: `Message delivery guarantees define how messaging systems handle message delivery in the presence of failures. Each guarantee has different semantics, trade-offs, and implementation complexity.

**1. At-Most-Once Delivery:**

**Definition:** Message delivered zero or one time (never duplicated, but may be lost)

**Mechanism:**
\`\`\`
Producer → Queue → Consumer
                     ↓ Process message
                     ↓ [Network failure before ACK]
Message lost ❌ (not redelivered)

Flow:
1. Producer sends message
2. Queue receives message
3. Consumer receives message
4. Consumer processes message
5. [Consumer crashes before ACK]
6. Queue assumes delivery successful (message deleted)
7. Message never reprocessed

Result: Message may be lost
\`\`\`

**Characteristics:**
- No retries after initial delivery attempt
- Lowest latency (no acknowledgment wait)
- Highest throughput (fire-and-forget)
- Data loss possible

**Use Cases:**
- Metrics/telemetry (occasional loss acceptable)
  Example: Temperature sensor readings (1000/sec, losing 5 not critical)
- Real-time analytics (approximate results acceptable)
  Example: Website traffic dashboard (99.9% accuracy sufficient)
- Logging (some log loss tolerable)
  Example: Debug logs (not business-critical)

**Implementation:**
\`\`\`python
# Producer: Fire and forget
queue.send (message, ack=False)

# Consumer: Auto-acknowledge before processing
message = queue.receive (auto_ack=True)
process (message)  # If crashes here, message lost
\`\`\`

**2. At-Least-Once Delivery:**

**Definition:** Message delivered one or more times (never lost, but may be duplicated)

**Mechanism:**
\`\`\`
Producer → Queue → Consumer
                     ↓ Process message
                     ↓ [Consumer crashes before ACK]
                     ↓ Message redelivered
                     ↓ Process again (duplicate!)

Flow:
1. Producer sends message
2. Queue receives message
3. Consumer receives message
4. Consumer processes message
5. [Consumer crashes before ACK]
6. Queue doesn't receive ACK → Marks message as unprocessed
7. Queue redelivers message
8. New consumer processes message again (duplicate)

Result: Message delivered at least once (duplicates possible)
\`\`\`

**Characteristics:**
- Messages never lost (retried until acknowledged)
- Duplicates possible (redelivery on failure)
- Higher throughput than exactly-once
- Requires idempotent processing

**Use Cases:**
- Most production systems (with idempotency)
- Email notifications (duplicate email acceptable)
- Order processing (with idempotency checks)
- Data pipelines (with deduplication)

**Implementation:**
\`\`\`python
# Producer: Wait for acknowledgment
queue.send (message)
queue.wait_for_ack()  # Retry if no ACK

# Consumer: Acknowledge after processing
message = queue.receive (auto_ack=False)
process (message)  # Process first
queue.acknowledge (message)  # Then acknowledge

# If crashes before acknowledge, message redelivered
\`\`\`

**Handling Duplicates (Idempotency):**
\`\`\`python
# Idempotent processing using message ID
def process_message (message):
    message_id = message.id
    
    # Check if already processed
    if redis.exists (f"processed:{message_id}"):
        return  # Skip duplicate
    
    # Process message
    result = do_processing (message)
    
    # Mark as processed atomically
    with transaction:
        save_result (result)
        redis.set (f"processed:{message_id}", "true", ex=86400)  # 24h TTL
    
    # Acknowledge
    queue.acknowledge (message)
\`\`\`

**3. Exactly-Once Delivery:**

**Definition:** Message delivered exactly one time (never lost, never duplicated)

**Mechanism:**
\`\`\`
Producer → Queue → Consumer
                     ↓ Process message
                     ↓ [Transactional processing]
                     ↓ Atomic: Write result + ACK

Flow:
1. Producer sends message with unique ID
2. Queue stores message with ID
3. Consumer receives message
4. Consumer processes within transaction:
   - Write result to database
   - Write ACK to queue
   - Commit transaction (atomic)
5. If crashes before commit → Transaction rolls back
6. Message redelivered, but:
   - Unique ID already exists in result DB
   - Transaction fails (duplicate detected)
   - Message acknowledged

Result: Processed exactly once (no loss, no duplicates)
\`\`\`

**Characteristics:**
- Guaranteed exactly-once semantics
- Highest reliability (no loss, no duplicates)
- Lowest throughput (transaction overhead)
- Highest latency (two-phase commit)
- Complex implementation

**Use Cases:**
- Financial transactions (payments, transfers)
- Inventory management (stock updates)
- Accounting/audit systems
- Any scenario where duplicates unacceptable

**Implementation (Kafka):**
\`\`\`java
// Enable exactly-once
props.put("enable.idempotence", "true");
props.put("transactional.id", "payment-processor-1");

producer.initTransactions();

try {
    producer.beginTransaction();
    
    // Process message
    PaymentResult result = processPayment (message);
    
    // Write result and commit offset atomically
    producer.send (new ProducerRecord<>("payment-results", result));
    producer.sendOffsetsToTransaction (offsets, consumerGroupId);
    
    producer.commitTransaction();  // Atomic commit
    
} catch (Exception e) {
    producer.abortTransaction();  // Rollback
}
\`\`\`

**Payment Processing System - Detailed Analysis:**

**Requirements:**
- Process credit card payments
- Charge customer exactly once (no double-charging)
- Never lose payment request
- High reliability, audit trail

**Chosen Guarantee: At-Least-Once + Idempotency**

**Why Not Exactly-Once?**
- Exactly-once requires transactional database + queue coordination
- Payment gateway (Stripe, PayPal) is external (can't include in transaction)
- Performance overhead too high for high-volume payments
- At-least-once with idempotency more practical

**Architecture:**
\`\`\`
Order Service → Queue: "payment-requests" → Payment Processor
                                                  ↓
                                            Payment Gateway (Stripe)
                                                  ↓
                                            Database: processed_payments
\`\`\`

**Implementation:**

**Step 1: Producer (Order Service)**
\`\`\`python
import uuid

def place_order (order):
    # Generate idempotency key
    payment_id = str (uuid.uuid4())
    
    payment_request = {
        "payment_id": payment_id,  # Unique ID
        "order_id": order.id,
        "amount": order.total,
        "customer_id": order.customer_id,
        "timestamp": datetime.now()
    }
    
    # Send to queue (at-least-once)
    queue.send("payment-requests", payment_request)
    
    return payment_id
\`\`\`

**Step 2: Consumer (Payment Processor)**
\`\`\`python
def process_payments():
    while True:
        # Receive message (at-least-once delivery)
        message = queue.receive("payment-requests", auto_ack=False)
        
        try:
            process_payment_idempotent (message)
            queue.acknowledge (message)  # ACK after processing
        except Exception as e:
            logger.error (f"Payment failed: {e}")
            # Don't acknowledge → Message redelivered
            # Will retry automatically

def process_payment_idempotent (message):
    payment_id = message["payment_id"]
    
    # Step 1: Check if already processed (idempotency check)
    if is_already_processed (payment_id):
        logger.info (f"Payment {payment_id} already processed, skipping")
        return
    
    # Step 2: Check if payment gateway already received this payment
    # Use idempotency key with payment gateway
    stripe_idempotency_key = f"payment_{payment_id}"
    
    try:
        # Step 3: Process payment with payment gateway
        charge = stripe.Charge.create(
            amount=message["amount"],
            currency="usd",
            customer=message["customer_id"],
            idempotency_key=stripe_idempotency_key,  # Stripe deduplicates
            metadata={"payment_id": payment_id}
        )
        
        # Step 4: Record payment in database (atomic with idempotency)
        with database.transaction():
            # Insert payment record
            database.execute("""
                INSERT INTO processed_payments 
                (payment_id, charge_id, amount, status, timestamp)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (payment_id) DO NOTHING
            """, (payment_id, charge.id, message["amount"], "success", datetime.now()))
            
            # If already exists (duplicate), INSERT ignored, no error
            
        logger.info (f"Payment {payment_id} processed successfully")
        
    except stripe.error.CardError as e:
        # Card declined (non-retriable)
        record_failed_payment (payment_id, str (e))
        # Still acknowledge message (don't retry card declines)
    
    except Exception as e:
        # Retriable error (network, timeout)
        logger.error (f"Payment {payment_id} failed (will retry): {e}")
        raise  # Don't acknowledge, message will be redelivered

def is_already_processed (payment_id):
    # Check database
    result = database.execute(
        "SELECT 1 FROM processed_payments WHERE payment_id = %s",
        (payment_id,)
    )
    return result.rowcount > 0
\`\`\`

**Handling Failure Scenarios:**

**Scenario 1: Consumer crashes after charging card but before ACK**
\`\`\`
1. Charge card (success) → $100 charged
2. Consumer crashes before ACK
3. Message redelivered
4. is_already_processed (payment_id) checks database
5. Database has no record (crashed before insert)
6. Charge card again with same idempotency_key
7. Stripe detects duplicate (same idempotency_key within 24 hours)
8. Stripe returns original charge (no double-charge) ✅
9. Insert into database (succeeds)
10. Acknowledge message

Result: Customer charged once, payment recorded once ✅
\`\`\`

**Scenario 2: Database insert succeeds, ACK fails**
\`\`\`
1. Charge card (success)
2. Insert into database (success)
3. ACK fails (network issue)
4. Message redelivered
5. is_already_processed (payment_id) returns True
6. Skip processing (already done)
7. Acknowledge message

Result: Customer charged once, payment recorded once ✅
\`\`\`

**Scenario 3: Card declined**
\`\`\`
1. Charge card → Card declined
2. Record failure in database
3. Acknowledge message (don't retry)
4. Notify customer (card declined)

Result: Customer not charged, failure recorded, no retries ✅
\`\`\`

**Why This Works:**1. **At-Least-Once Delivery:** Queue guarantees message not lost
2. **Idempotency Keys:** Payment gateway (Stripe) deduplicates charges
3. **Database Unique Constraint:** Prevents duplicate payment records
4. **Check Before Processing:** Skip if already processed

**Trade-offs:**

✅ Advantages:
- No double-charging (idempotency)
- No lost payments (at-least-once)
- Good performance (no distributed transactions)
- Simple to implement and understand

⚠️ Considerations:
- Requires idempotency key support from payment gateway
- Requires unique constraint in database
- Small window for duplicate API calls to gateway (mitigated by gateway dedup)
- Must handle all failure modes correctly

**Comparison:**

| Guarantee | Payment System | Why/Why Not |
|-----------|---------------|-------------|
| At-Most-Once | ❌ Not suitable | Payments may be lost (unacceptable) |
| At-Least-Once + Idempotency | ✅ Recommended | Reliable + performant + practical |
| Exactly-Once | ⚠️ Overkill | Complex, lower performance, external gateway can't participate in transaction |

**Conclusion:**

For payment processing, at-least-once delivery with idempotency is the practical choice. It combines reliability (no lost payments) with performance (no distributed transactions) and simplicity (straightforward implementation). Idempotency keys prevent double-charging, unique database constraints prevent duplicate records, and payment gateway deduplication provides additional safety. This approach is battle-tested at scale by companies like Stripe, Shopify, and Amazon.`,
    keyPoints: [
      'At-most-once: May lose messages, lowest latency, use for metrics/logs',
      'At-least-once: No loss but duplicates possible, use with idempotency',
      'Exactly-once: No loss, no duplicates, highest reliability but complex',
      'Payment processing: Use at-least-once + idempotency (practical choice)',
      'Idempotency implementation: Unique message ID + database check + payment gateway dedup',
      'Handle failures correctly: Retry retriable errors, record non-retriable failures',
    ],
  },
  {
    id: 'mq-fundamentals-dq-3',
    question:
      'Design a message queue architecture for an e-commerce platform that handles order processing, inventory management, email notifications, and analytics. Discuss queue vs topic choices, failure handling with dead letter queues, scaling strategies, and monitoring. How would you ensure reliability while maintaining high throughput?',
    hint: 'Consider the different requirements for each subsystem, event flow, decoupling strategies, and production best practices.',
    sampleAnswer: `Designing a message queue architecture for e-commerce requires careful consideration of different subsystem requirements, failure modes, scaling characteristics, and operational concerns. Here\'s a comprehensive architecture:

**System Requirements:**
- Scale: 10,000 orders/hour (peak: 30,000/hour during sales)
- Reliability: No lost orders, exactly-once payment processing
- Performance: User sees confirmation within 200ms
- Failure Recovery: Automatic retries, dead letter handling
- Monitoring: Real-time visibility into queue health

**Architecture Overview:**

\`\`\`
┌─────────────┐
│ Order API   │
└──────┬──────┘
       │ 1. Publish
       ↓
┌──────────────────────────────────┐
│ SNS Topic: "order-events"        │
│ (Pub/Sub - Fan-out)              │
└────┬────┬────┬────┬──────────────┘
     │    │    │    │
     │    │    │    └→ SQS: analytics-queue → Analytics Service
     │    │    └────→ SQS: email-queue → Email Service
     │    └─────────→ SQS: inventory-queue → Inventory Service
     └──────────────→ SQS: payment-queue → Payment Service
                             ↓ DLQ
                      payment-dlq (Dead Letter Queue)
\`\`\`

**Design Decisions:**

**1. Queue vs Topic Choice:**

**SNS Topic for Order Events (Pub/Sub):**
- Reasoning: Order event needs fan-out to multiple services
- Each service (payment, inventory, email, analytics) needs same event
- Services process independently, scale independently
- Easy to add new services (just subscribe to topic)

**SQS Queues per Service (Point-to-Point):**
- Reasoning: Work distribution within each service
- Multiple payment workers process payments (load balanced)
- Multiple email workers send emails (load balanced)
- Each message processed by one worker per queue

**Why Not Single Queue?**
❌ Single queue for all: Services compete for messages, filtering needed (wasteful)
❌ Tight coupling: All services know about all event types
✅ Separate queues: Clean separation, independent scaling, simple filtering

**2. Detailed Flow:**

**Order Placement:**

\`\`\`python
# Order API
@app.route('/orders', methods=['POST'])
def create_order():
    # 1. Validate order
    order = validate_order (request.json)
    
    # 2. Save to database (primary record)
    order_id = db.save_order (order)
    
    # 3. Publish to SNS (fan-out)
    event = {
        "event_type": "OrderCreated",
        "order_id": order_id,
        "customer_id": order.customer_id,
        "items": order.items,
        "total": order.total,
        "timestamp": datetime.now().isoformat(),
        "correlation_id": str (uuid.uuid4())  # Trace across services
    }
    
    sns.publish(
        TopicArn=order_events_topic,
        Message=json.dumps (event),
        MessageAttributes={
            'event_type': {'DataType': 'String', 'StringValue': 'OrderCreated'},
            'priority': {'DataType': 'String', 'StringValue': 'high'}
        }
    )
    
    # 4. Return immediately (don't wait for processing)
    return {'order_id': order_id, 'status': 'pending'}, 202

# Total latency: ~100ms (DB write + SNS publish)
# User sees response quickly ✅
\`\`\`

**SNS to SQS Fan-out:**
- SNS automatically delivers to all subscribed SQS queues
- Each queue receives a copy of the event
- Queues buffer messages (handle traffic spikes)

**3. Payment Processing (Critical Path):**

\`\`\`python
# Payment Service
def process_payments():
    while True:
        # Receive from payment queue
        messages = sqs.receive_message(
            QueueUrl=payment_queue_url,
            MaxNumberOfMessages=10,  # Batch for efficiency
            WaitTimeSeconds=20,  # Long polling
            VisibilityTimeout=300  # 5 min (payment processing time)
        )
        
        for message in messages['Messages']:
            try:
                event = json.loads (message['Body'])
                process_payment_idempotent (event)
                
                # Delete message (acknowledge)
                sqs.delete_message(
                    QueueUrl=payment_queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )
                
            except PaymentGatewayError as e:
                # Retriable error (network, timeout)
                logger.error (f"Payment failed (will retry): {e}")
                # Don't delete → Message becomes visible again after timeout
                # Automatic retry
                
            except CardDeclinedError as e:
                # Non-retriable error
                logger.info (f"Card declined: {e}")
                # Delete message (don't retry)
                sqs.delete_message(...)
                # Publish OrderPaymentFailed event
                publish_payment_failed (event['order_id'])

def process_payment_idempotent (event):
    payment_id = f"payment_{event['order_id']}"
    
    # Idempotency check
    if redis.exists (f"processed:{payment_id}"):
        return  # Already processed
    
    # Process payment with idempotency key
    charge = stripe.Charge.create(
        amount=event['total'],
        currency="usd",
        customer=event['customer_id'],
        idempotency_key=payment_id,  # Prevents double-charging
        metadata={"order_id": event['order_id']}
    )
    
    # Record in database + Redis atomically
    with db.transaction():
        db.insert_payment (payment_id, charge.id, event['total'])
        redis.setex (f"processed:{payment_id}", 86400, "true")
    
    # Publish PaymentSucceeded event (for order service)
    publish_payment_succeeded (event['order_id'], charge.id)
\`\`\`

**4. Dead Letter Queue (DLQ) Configuration:**

\`\`\`python
# Configure DLQ for payment queue
sqs.set_queue_attributes(
    QueueUrl=payment_queue_url,
    Attributes={
        'RedrivePolicy': json.dumps({
            'deadLetterTargetArn': payment_dlq_arn,
            'maxReceiveCount': 3  # After 3 failures → DLQ
        }),
        'VisibilityTimeout': '300',  # 5 minutes
        'MessageRetentionPeriod': '345600'  # 4 days
    }
)

# DLQ Processing (separate worker)
def process_dlq():
    while True:
        messages = sqs.receive_message(QueueUrl=payment_dlq_url, ...)
        
        for message in messages:
            event = json.loads (message['Body'])
            
            # 1. Log to monitoring system
            logger.critical (f"Payment DLQ: {event['order_id']}")
            
            # 2. Store for manual investigation
            db.insert_failed_payment (event)
            
            # 3. Alert on-call engineer
            pagerduty.trigger(
                title=f"Payment failed 3 times: Order {event['order_id']}",
                details=event
            )
            
            # 4. Optionally: Manual reprocessing workflow
            # Engineers can review, fix data, resubmit to main queue
            
            # 5. Delete from DLQ (acknowledged)
            sqs.delete_message(...)
\`\`\`

**5. Scaling Strategies:**

**Horizontal Scaling (Auto-scaling):**

\`\`\`python
# CloudWatch alarms trigger auto-scaling

# Scale up if queue depth high
alarm_scale_up = cloudwatch.put_metric_alarm(
    AlarmName='payment-queue-high-depth',
    MetricName='ApproximateNumberOfMessagesVisible',
    Namespace='AWS/SQS',
    Statistic='Average',
    Period=60,
    EvaluationPeriods=2,
    Threshold=1000,  # > 1000 messages
    ComparisonOperator='GreaterThanThreshold',
    AlarmActions=[scale_up_policy]
)

# Scale down if queue empty
alarm_scale_down = cloudwatch.put_metric_alarm(
    AlarmName='payment-queue-low-depth',
    MetricName='ApproximateNumberOfMessagesVisible',
    Statistic='Average',
    Period=300,
    EvaluationPeriods=2,
    Threshold=100,  # < 100 messages
    ComparisonOperator='LessThanThreshold',
    AlarmActions=[scale_down_policy]
)

# Auto-scaling configuration
autoscaling.put_scaling_policy(
    PolicyName='payment-workers-scale-up',
    ServiceNamespace='ecs',
    ResourceId='service/payment-service',
    ScalableDimension='ecs:service:DesiredCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SQSQueueApproximateNumberOfMessagesVisible'
        },
        'TargetValue': 500.0,  # Aim for 500 messages per instance
        'ScaleInCooldown': 300,  # Wait 5 min before scaling down
        'ScaleOutCooldown': 60   # Scale up quickly (1 min)
    }
)
\`\`\`

**Scaling Characteristics:**

| Service | Normal Load | Peak Load | Scaling Strategy |
|---------|-------------|-----------|------------------|
| Payment | 5 workers | 15 workers | CPU + Queue depth |
| Inventory | 3 workers | 10 workers | Queue depth |
| Email | 10 workers | 30 workers | Queue depth (high volume) |
| Analytics | 2 workers | 2 workers | No scaling (batch processing) |

**6. Monitoring and Observability:**

\`\`\`python
# CloudWatch Metrics

# Queue metrics
metrics = [
    'ApproximateNumberOfMessagesVisible',  # Queue depth
    'ApproximateAgeOfOldestMessage',       # Staleness
    'NumberOfMessagesSent',                # Throughput (in)
    'NumberOfMessagesDeleted',             # Throughput (out)
    'NumberOfMessagesReceived',            # Polling rate
]

# Custom application metrics
cloudwatch.put_metric_data(
    Namespace='ECommerce/PaymentService',
    MetricData=[
        {
            'MetricName': 'PaymentSuccessRate',
            'Value': success_count / total_count,
            'Unit': 'Percent'
        },
        {
            'MetricName': 'PaymentProcessingLatency',
            'Value': latency_ms,
            'Unit': 'Milliseconds',
            'StorageResolution': 1  # High-resolution (1-second)
        }
    ]
)

# Alerting
alerts = {
    'QueueDepthHigh': 'Queue > 5000 messages for 5 minutes',
    'OldMessageAlert': 'Oldest message > 30 minutes',
    'DLQNonEmpty': 'Messages in DLQ (immediate alert)',
    'LowThroughput': 'Messages deleted < 50% of sent (consumers slow)',
    'HighErrorRate': 'Error rate > 1% (payment failures)'
}

# Dashboard
grafana_dashboard = {
    'panels': [
        'Queue depth (real-time)',
        'Message age (p50, p95, p99)',
        'Throughput (in/out)',
        'Processing latency (p50, p95, p99)',
        'Error rate',
        'DLQ depth',
        'Worker count (auto-scaling)'
    ]
}

# Distributed Tracing (X-Ray)
# Trace order flow across services
with xray_recorder.capture('process_order'):
    # Trace ID propagated through events (correlation_id)
    # Visualize: Order API → SNS → SQS → Payment Service → Stripe
    pass
\`\`\`

**7. Reliability Guarantees:**

**No Lost Orders:**
\`\`\`
1. Order saved to database (primary record) ✅
2. SNS publish with retry (3 attempts) ✅
3. SQS persistence (replicated across AZs) ✅
4. DLQ captures failed messages ✅

If SNS publish fails → Order status remains "pending"
Background job retries SNS publish for pending orders
\`\`\`

**Exactly-Once Payment:**
\`\`\`
1. Idempotency key prevents double-charging ✅
2. Redis/DB check prevents reprocessing ✅
3. Stripe\'s idempotency (24-hour window) ✅
\`\`\`

**Graceful Degradation:**
\`\`\`
If Payment Service down:
- Messages accumulate in payment queue (buffered)
- Queue depth triggers auto-scaling (more workers)
- DLQ captures persistent failures
- Order status remains "pending" (visible to customer)
- Background reconciliation detects stuck orders

If Email Service down:
- Email queue buffers messages
- Doesn't affect payment/inventory processing ✅
- Emails sent when service recovers
\`\`\`

**8. Capacity Planning:**

\`\`\`
Peak Load: 30,000 orders/hour = 8.33 orders/sec

SNS throughput: 30,000/sec (no limit) ✅

SQS throughput:
- Standard queue: Unlimited ✅
- Payment queue: 8.33 msg/sec (easily handled)

Payment workers:
- Processing time: 2 seconds/payment (Stripe API call)
- Throughput needed: 8.33 payments/sec
- Workers needed: 8.33 × 2 = 16.66 → 20 workers (headroom)

Email workers:
- Processing time: 500ms/email (SMTP)
- Throughput needed: 8.33 emails/sec
- Workers needed: 8.33 × 0.5 = 4.16 → 10 workers (headroom for retries)

Inventory workers:
- Processing time: 100ms/update (DB update)
- Throughput needed: 8.33 updates/sec
- Workers needed: 8.33 × 0.1 = 0.83 → 3 workers (headroom)

Total: ~35 workers peak, ~15 workers normal
Cost: ~$2,000/month (t3.medium instances)
\`\`\`

**9. Cost Optimization:**

\`\`\`
SQS costs (10,000 orders/day):
- Requests: 40,000/day (4 queues × 10,000 messages)
- Cost: $0.016/day = $0.48/month (after free tier)

SNS costs:
- Publishes: 10,000/day
- Cost: $0.005/day = $0.15/month

Total messaging: ~$0.63/month ✅ (nearly free!)

Worker costs dominate:
- 15 workers × t3.medium ($0.0416/hour) × 730 hours = $455/month

Optimization:
- Use Lambda for email/analytics (pay per invocation)
- Reserve instances for payment/inventory (always running)
- Spot instances for analytics (interruptible OK)
\`\`\`

**10. Production Best Practices:**

✅ **Idempotency:** All consumers handle duplicates
✅ **DLQ:** Capture failed messages, alert on-call
✅ **Monitoring:** Queue depth, age, throughput, errors
✅ **Auto-scaling:** Scale workers based on queue depth
✅ **Long Polling:** Reduce costs, improve latency
✅ **Batching:** Process 10 messages per poll (efficiency)
✅ **Graceful Shutdown:** Drain in-flight messages before restart
✅ **Distributed Tracing:** Correlation IDs, X-Ray
✅ **Alerting:** PagerDuty for critical failures
✅ **Testing:** Chaos engineering (kill services, validate recovery)

**Conclusion:**

This architecture provides:
- **Reliability:** No lost orders, exactly-once payments, automatic retries
- **Scalability:** Auto-scaling workers, unlimited queue capacity
- **Performance:** Sub-200ms user response, async processing
- **Observability:** Comprehensive monitoring, distributed tracing
- **Cost-effectiveness:** Pay-per-use messaging, ~$0.60/month

The SNS + SQS fanout pattern is battle-tested at scale (AWS uses it internally). Combined with idempotent processing, DLQs, and auto-scaling, it provides production-grade reliability while maintaining simplicity and low cost. This architecture can scale to millions of orders/day with minimal changes.`,
    keyPoints: [
      'SNS topic for fan-out (pub/sub), SQS queues for work distribution (point-to-point)',
      'Separate queues per service (payment, inventory, email, analytics) for independence',
      'Dead letter queues (DLQ) capture failed messages after 3 retries',
      'Idempotent processing prevents double-charging (Redis + payment gateway dedup)',
      'Auto-scaling based on queue depth (scale up/down workers automatically)',
      'Comprehensive monitoring: Queue depth, message age, throughput, error rate, DLQ depth',
      'Graceful degradation: Services fail independently, queues buffer messages',
    ],
  },
];
