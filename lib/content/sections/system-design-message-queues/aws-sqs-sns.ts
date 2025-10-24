/**
 * AWS SQS & SNS Section
 */

export const awssqssnsSection = {
  id: 'aws-sqs-sns',
  title: 'AWS SQS & SNS',
  content: `AWS Simple Queue Service (SQS) and Simple Notification Service (SNS) are fully managed messaging services that eliminate the operational overhead of running message brokers like RabbitMQ or Kafka.

## AWS SQS (Simple Queue Service)

**SQS** = Fully Managed Message Queue + Serverless + Pay-per-use + Auto-scaling

### **Key Characteristics:**

1. **Fully managed**: No servers to provision or manage
2. **Serverless**: Automatic scaling, no capacity planning
3. **Highly available**: Redundant across multiple AZs
4. **Pay-per-use**: Charge per request, no idle costs
5. **Integration**: Seamless with AWS services (Lambda, EC2, ECS)

### **SQS Queue Types:**

**1. Standard Queue:**

\`\`\`
Characteristics:
- Unlimited throughput (truly unlimited!)
- At-least-once delivery (possible duplicates)
- Best-effort ordering (not guaranteed)
- Nearly unlimited messages in flight

Use case: High throughput, order not critical

Example: Log aggregation, metrics collection
\`\`\`

**2. FIFO Queue (First-In-First-Out):**

\`\`\`
Characteristics:
- Limited throughput: 3,000 msg/sec (with batching)
- Exactly-once processing (deduplication)
- Strict ordering (FIFO within message group)
- Message groups for parallel processing

Use case: Order matters, no duplicates

Example: Financial transactions, order processing
\`\`\`

---

## SQS Standard Queue

### **Message Flow:**

\`\`\`
Producer → SQS Standard Queue → Consumer
                ↓
           Distributed across
           multiple servers
           (for high availability)

Characteristics:
1. Message appears at least once
   (could appear multiple times - duplicates possible)

2. Message order not guaranteed
   Sent: A, B, C
   Received: B, A, C (possible)

3. Unlimited throughput
   Send 1M messages/sec? No problem!

4. Visibility timeout
   Message invisible to other consumers while being processed
\`\`\`

### **Producer Example:**

\`\`\`python
import boto3

sqs = boto3.client('sqs', region_name='us-east-1')
queue_url = 'https://sqs.us-east-1.amazonaws.com/123456789/my-queue'

# Send single message
response = sqs.send_message(
    QueueUrl=queue_url,
    MessageBody='Order placed',
    MessageAttributes={
        'OrderId': {'StringValue': '12345', 'DataType': 'String'},
        'Amount': {'StringValue': '99.99', 'DataType': 'Number'}
    }
)

print(f"Message ID: {response['MessageId']}")

# Send batch (up to 10 messages)
entries = [
    {'Id': '1', 'MessageBody': 'Order 1'},
    {'Id': '2', 'MessageBody': 'Order 2'},
    {'Id': '3', 'MessageBody': 'Order 3'}
]

response = sqs.send_message_batch(
    QueueUrl=queue_url,
    Entries=entries
)

print(f"Successful: {len(response['Successful'])}")
print(f"Failed: {len(response.get('Failed', []))}")
\`\`\`

### **Consumer Example:**

\`\`\`python
# Receive messages (long polling)
response = sqs.receive_message(
    QueueUrl=queue_url,
    MaxNumberOfMessages=10,  # Receive up to 10
    WaitTimeSeconds=20,      # Long polling (0-20 seconds)
    MessageAttributeNames=['All'],
    VisibilityTimeout=30     # Hide for 30 sec while processing
)

messages = response.get('Messages', [])

for message in messages:
    # Process message
    print(f"Processing: {message['Body']}")
    
    try:
        process_order(message['Body'])
        
        # Delete message (acknowledge)
        sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=message['ReceiptHandle']
        )
        
    except Exception as e:
        print(f"Error processing message: {e}")
        # Don't delete - message becomes visible again after timeout
        # Retry automatically
\`\`\`

### **Visibility Timeout:**

\`\`\`
Timeline:
0s:  Consumer receives message
     Message hidden from other consumers (visibility timeout = 30s)
     
5s:  Consumer processing...

10s: Consumer crashes ❌

30s: Visibility timeout expires
     Message becomes visible again
     Another consumer receives it ✅

Prevents:
- Multiple consumers processing same message simultaneously
- Message loss if consumer crashes

Configuration:
- Queue default: 30 seconds (configurable 0-12 hours)
- Per-message override in ReceiveMessage
- Can extend with ChangeMessageVisibility
\`\`\`

---

## SQS FIFO Queue

### **Characteristics:**

\`\`\`
FIFO Queue guarantees:
1. Exactly-once processing (5-minute deduplication window)
2. FIFO ordering (within message group)
3. Message groups for parallel processing

Naming:
- Queue name must end with .fifo
- Example: orders.fifo

Throughput:
- 300 msg/sec (default)
- 3,000 msg/sec (with batching, 10 messages per batch)
- 3,000 msg/sec × 10 msg/batch = 30,000 messages/sec
\`\`\`

### **Message Groups:**

\`\`\`
Message Groups enable parallel processing while maintaining order:

Queue: orders.fifo

Messages:
Msg1: Group=user_123, Body="Order A"
Msg2: Group=user_456, Body="Order X"
Msg3: Group=user_123, Body="Order B"
Msg4: Group=user_456, Body="Order Y"

Processing:
Consumer 1: Processes Group user_123
  - Receives Msg1 (Order A)
  - Then receives Msg3 (Order B)
  - FIFO maintained for user_123 ✅

Consumer 2: Processes Group user_456
  - Receives Msg2 (Order X)
  - Then receives Msg4 (Order Y)
  - FIFO maintained for user_456 ✅

✅ Ordering per group
✅ Parallelism across groups
\`\`\`

### **Deduplication:**

\`\`\`python
# Content-based deduplication
sqs.send_message(
    QueueUrl='https://sqs.us-east-1.amazonaws.com/123456789/orders.fifo',
    MessageBody='Order 12345',
    MessageGroupId='user_123',
    # SQS computes SHA-256 hash of MessageBody
    # Duplicate if same hash within 5 minutes
)

# Explicit deduplication ID
sqs.send_message(
    QueueUrl='https://sqs.us-east-1.amazonaws.com/123456789/orders.fifo',
    MessageBody='Order 12345',
    MessageGroupId='user_123',
    MessageDeduplicationId='order_12345_20230615'
    # Duplicate if same ID within 5 minutes
)

Result:
- Send same message twice within 5 min → Only delivered once
- Prevents accidental duplicates (network retries, etc.)
\`\`\`

---

## Dead Letter Queue (DLQ)

Handle messages that fail repeatedly:

\`\`\`python
# Create DLQ
dlq_response = sqs.create_queue(QueueName='failed-orders-dlq')
dlq_url = dlq_response['QueueUrl']
dlq_arn = sqs.get_queue_attributes(
    QueueUrl=dlq_url,
    AttributeNames=['QueueArn']
)['Attributes']['QueueArn']

# Configure main queue with DLQ
sqs.set_queue_attributes(
    QueueUrl=queue_url,
    Attributes={
        'RedrivePolicy': json.dumps({
            'deadLetterTargetArn': dlq_arn,
            'maxReceiveCount': 3  # After 3 failed attempts → DLQ
        })
    }
)

# Message lifecycle:
# 1. Receive message
# 2. Process fails, don't delete
# 3. Message becomes visible again (attempt 1)
# 4. Receive message again
# 5. Process fails again (attempt 2)
# 6. Receive message again
# 7. Process fails again (attempt 3)
# 8. Message moved to DLQ automatically

# Inspect DLQ
dlq_messages = sqs.receive_message(QueueUrl=dlq_url, MaxNumberOfMessages=10)
# Manually investigate, fix, reprocess
\`\`\`

---

## AWS SNS (Simple Notification Service)

**SNS** = Fully Managed Pub/Sub + Fan-out + Multiple protocols

### **Key Characteristics:**

1. **Pub/Sub model**: One message → Many subscribers
2. **Fan-out**: Deliver to multiple endpoints
3. **Multiple protocols**: SQS, HTTP, Lambda, Email, SMS
4. **Message filtering**: Subscribers receive subset based on attributes
5. **Push-based**: SNS pushes to subscribers (vs SQS pull)

### **SNS Architecture:**

\`\`\`
Publisher → SNS Topic → Subscriber 1 (SQS)
                      → Subscriber 2 (Lambda)
                      → Subscriber 3 (HTTP endpoint)
                      → Subscriber 4 (Email)
                      → Subscriber 5 (SMS)

Each subscriber receives a copy of the message
\`\`\`

---

## SNS + SQS Fanout Pattern

**Most common pattern**: SNS for fanout, SQS for reliable delivery

\`\`\`
Publisher → SNS Topic → SQS Queue 1 → Consumer 1 (Analytics)
                      → SQS Queue 2 → Consumer 2 (Email)
                      → SQS Queue 3 → Consumer 3 (Warehouse)

Benefits:
✅ Decouple publisher from consumers
✅ Reliable delivery (SQS persistence)
✅ Independent scaling per consumer
✅ Add/remove consumers without code changes
\`\`\`

### **Setup:**

\`\`\`python
import boto3

sns = boto3.client('sns')
sqs = boto3.client('sqs')

# Create SNS topic
topic_response = sns.create_topic(Name='user-events')
topic_arn = topic_response['TopicArn']

# Create SQS queues
analytics_queue = sqs.create_queue(QueueName='analytics-queue')['QueueUrl']
email_queue = sqs.create_queue(QueueName='email-queue')['QueueUrl']

# Get queue ARNs
analytics_arn = sqs.get_queue_attributes(
    QueueUrl=analytics_queue,
    AttributeNames=['QueueArn']
)['Attributes']['QueueArn']

email_arn = sqs.get_queue_attributes(
    QueueUrl=email_queue,
    AttributeNames=['QueueArn']
)['Attributes']['QueueArn']

# Subscribe queues to SNS topic
sns.subscribe(
    TopicArn=topic_arn,
    Protocol='sqs',
    Endpoint=analytics_arn
)

sns.subscribe(
    TopicArn=topic_arn,
    Protocol='sqs',
    Endpoint=email_arn
)

# Set queue policy (allow SNS to send messages)
policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sns.amazonaws.com"},
        "Action": "sqs:SendMessage",
        "Resource": analytics_arn,
        "Condition": {
            "ArnEquals": {"aws:SourceArn": topic_arn}
        }
    }]
}

sqs.set_queue_attributes(
    QueueUrl=analytics_queue,
    Attributes={'Policy': json.dumps(policy)}
)

# Publish to SNS
sns.publish(
    TopicArn=topic_arn,
    Message='User signed up',
    Subject='User Event',
    MessageAttributes={
        'event_type': {'DataType': 'String', 'StringValue': 'signup'},
        'user_id': {'DataType': 'String', 'StringValue': '12345'}
    }
)

# Message delivered to BOTH queues
# Analytics consumer processes independently
# Email consumer processes independently
\`\`\`

---

## Message Filtering

Filter messages at SNS level (reduce unnecessary delivery):

\`\`\`python
# Subscription filter policy
filter_policy = {
    "event_type": ["signup", "purchase"],  # Only these events
    "amount": [{"numeric": [">", 100]}]    # Amount > 100
}

sns.set_subscription_attributes(
    SubscriptionArn=subscription_arn,
    AttributeName='FilterPolicy',
    AttributeValue=json.dumps(filter_policy)
)

# Publish message
sns.publish(
    TopicArn=topic_arn,
    Message='User purchased',
    MessageAttributes={
        'event_type': {'DataType': 'String', 'StringValue': 'purchase'},
        'amount': {'DataType': 'Number', 'StringValue': '150'}
    }
)

# Result:
# - Matches filter (event_type=purchase, amount=150 > 100)
# - Delivered to subscriber ✅

# Another message:
sns.publish(
    TopicArn=topic_arn,
    Message='User viewed',
    MessageAttributes={
        'event_type': {'DataType': 'String', 'StringValue': 'view'},
        'amount': {'DataType': 'Number', 'StringValue': '0'}
    }
)

# Result:
# - Doesn't match filter (event_type=view)
# - NOT delivered to subscriber ❌

Benefits:
✅ Reduce unnecessary processing
✅ Save SQS costs (fewer messages)
✅ Simplify consumer logic
\`\`\`

---

## Lambda Integration

Trigger Lambda functions directly from SQS or SNS:

### **SQS → Lambda:**

\`\`\`python
# Lambda function triggered by SQS
def lambda_handler(event, context):
    # event['Records'] contains SQS messages
    for record in event['Records']:
        message_body = record['body']
        process_message(message_body)
    
    # No need to delete messages manually
    # Lambda deletes automatically on success
    return {'statusCode': 200}

# Configure SQS as event source
# Lambda polls SQS automatically
# Batch size: 1-10 messages per invocation
# Automatic retry on failure
# Moves to DLQ after maxReceiveCount
\`\`\`

### **SNS → Lambda:**

\`\`\`python
# Lambda function triggered by SNS
def lambda_handler(event, context):
    # event['Records'] contains SNS messages
    for record in event['Records']:
        sns_message = record['Sns']['Message']
        process_notification(sns_message)
    
    return {'statusCode': 200}

# Configure SNS subscription
# SNS invokes Lambda directly (push)
# Automatic retry on failure (3 retries)
\`\`\`

---

## Cost Optimization

### **SQS Pricing (Simplified):**

\`\`\`
Standard Queue:
- $0.40 per million requests (after free tier)
- First 1M requests/month free

Example:
- 10M requests/month
- Cost: (10M - 1M) × $0.40 / 1M = $3.60/month

FIFO Queue:
- $0.50 per million requests (25% more expensive)

Data transfer:
- Free within AWS
- Internet transfer charged ($0.09/GB)

Cost optimization:
✅ Use batching (10 messages = 1 request)
✅ Long polling (reduce empty receives)
✅ SNS filter policies (reduce SQS writes)
\`\`\`

### **SNS Pricing:**

\`\`\`
Publish:
- $0.50 per million requests

Delivery:
- HTTP/HTTPS: $0.06 per 100,000 notifications
- Email: $2.00 per 100,000 notifications
- SMS: $0.00645 per SMS (US)
- SQS, Lambda: Free (data transfer only)

Example:
- 10M publishes/month: $5.00
- 10M SQS deliveries: Free
- Total: $5.00/month
\`\`\`

---

## SQS/SNS vs Kafka/RabbitMQ

| Feature | SQS/SNS | Kafka/RabbitMQ |
|---------|---------|----------------|
| **Management** | Fully managed | Self-hosted |
| **Scaling** | Automatic | Manual |
| **Cost Model** | Pay-per-use | Fixed (instance cost) |
| **Throughput** | Unlimited (Standard) | High (millions/sec) |
| **Latency** | 10-100ms | <10ms |
| **Ordering** | FIFO queue | Per-partition |
| **Replay** | No | Yes (Kafka) |
| **Ops Overhead** | Zero | High |
| **AWS Integration** | Native | Manual |

**Choose SQS/SNS when:**
✅ Using AWS already
✅ Want zero ops overhead
✅ Variable/unpredictable load
✅ Cost-effective at moderate scale
✅ Don't need sub-10ms latency
✅ Don't need message replay

**Choose Kafka/RabbitMQ when:**
✅ Need message replay (Kafka)
✅ Sub-10ms latency required
✅ Very high throughput (millions/sec sustained)
✅ Multi-cloud or on-premise
✅ Already have Kafka/RabbitMQ expertise

---

## SQS/SNS Best Practices

### **1. Enable Long Polling:**

\`\`\`python
# Reduce empty receives, save costs
response = sqs.receive_message(
    QueueUrl=queue_url,
    WaitTimeSeconds=20  # Long poll up to 20 seconds
)

# vs Short polling (WaitTimeSeconds=0)
# Returns immediately even if no messages → Wasteful
\`\`\`

### **2. Use Batching:**

\`\`\`python
# Send 10 messages in one request (save 90% cost)
entries = [{'Id': str(i), 'MessageBody': f'Message {i}'} for i in range(10)]
sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)

# vs Sending individually (10 requests)
\`\`\`

### **3. Set Appropriate Visibility Timeout:**

\`\`\`python
# Timeout should be longer than max processing time
# Too short: Message processed multiple times
# Too long: Delayed retry on failure

# For 5-minute processing:
VisibilityTimeout=360  # 6 minutes (buffer)
\`\`\`

### **4. Use DLQ:**

\`\`\`python
# Catch poison messages
RedrivePolicy={'maxReceiveCount': 3}
# After 3 failures → DLQ
\`\`\`

### **5. Monitor Metrics:**

\`\`\`
CloudWatch metrics:
- ApproximateNumberOfMessagesVisible (queue depth)
- ApproximateAgeOfOldestMessage (staleness)
- NumberOfMessagesSent
- NumberOfMessagesReceived
- NumberOfMessagesDeleted

Alert if:
- Queue depth > threshold (consumers slow)
- Age > threshold (messages stuck)
\`\`\`

### **6. Idempotent Processing:**

\`\`\`python
# SQS Standard can deliver duplicates
# Design consumers to handle reprocessing

def process_message(message):
    message_id = extract_id(message)
    if already_processed(message_id):
        return  # Skip duplicate
    
    # Process
    save_result(message)
    mark_as_processed(message_id)
\`\`\`

---

## Real-World Example: E-Commerce Order Processing

\`\`\`
Architecture:

Order API
    ↓
SNS Topic: "orders"
    ├→ SQS Queue: "payment-queue" → Lambda: Process Payment
    ├→ SQS Queue: "inventory-queue" → ECS: Update Inventory
    ├→ SQS Queue: "email-queue" → Lambda: Send Confirmation
    ├→ SQS Queue: "analytics-queue" → Kinesis Firehose → S3

Benefits:
✅ Decoupled services (API doesn't wait)
✅ Resilient (SQS persists messages)
✅ Scalable (each consumer scales independently)
✅ Flexible (add new consumers without API changes)

Implementation:
1. Order API publishes to SNS
2. SNS fans out to 4 SQS queues
3. Each queue has dedicated consumers
4. DLQ configured for each queue
5. CloudWatch alarms on queue depth

Cost (10K orders/day):
- SNS: 10K publishes × $0.50 / 1M = $0.005/day
- SQS: 40K messages × $0.40 / 1M = $0.016/day
- Total: $0.021/day = $0.63/month

Compare to running RabbitMQ:
- t3.medium instance: ~$30/month
- SQS/SNS: $0.63/month (48× cheaper!)
\`\`\`

---

## SQS/SNS in System Design Interviews

### **When to Propose:**

✅ AWS-based architecture
✅ Variable/unpredictable load
✅ Want zero ops overhead
✅ Moderate throughput (<100K msg/sec)
✅ Fan-out required (SNS)

### **Example Discussion:**

\`\`\`
Interviewer: "Design a notification system for a social media app on AWS"

You:
"I'll use SNS + SQS for reliable, scalable notifications:

Architecture:
User Action → SNS Topic → SQS Queue 1 (Push Notifications)
                        → SQS Queue 2 (Email Notifications)
                        → SQS Queue 3 (SMS Notifications)
                        → SQS Queue 4 (In-App Notifications)

SNS Topics:
- "user-follows" (user follows another user)
- "user-likes" (user likes a post)
- "user-comments" (user comments on post)

Message Filtering:
- Push queue: Only high-priority (comments, mentions)
- Email queue: Digest notifications (daily summary)
- SMS queue: Critical notifications (security alerts)

Implementation:
1. User action triggers SNS publish
2. SNS fans out to subscribed queues (filter policies)
3. Lambda consumers process notifications
4. Batch processing for efficiency
5. DLQ for failed notifications

Scaling:
- SNS: Unlimited throughput
- SQS Standard: Unlimited throughput
- Lambda: Auto-scales with queue depth
- No manual scaling needed ✅

Cost (1M notifications/day):
- SNS: 1M × $0.50 / 1M = $0.50/day
- SQS: 4M (4 queues) × $0.40 / 1M = $1.60/day
- Lambda: $0.20/day (estimate)
- Total: ~$2.30/day = $69/month

vs Self-hosted RabbitMQ:
- Instance costs: $100+/month
- Ops overhead: Significant
- SQS/SNS: Zero ops, lower cost

Why SQS/SNS:
- Fully managed (zero ops)
- Perfect for AWS workloads
- Cost-effective at this scale
- Native Lambda integration
- High availability built-in
"
\`\`\`

---

## Key Takeaways

1. **SQS = Fully managed message queue** → No servers, auto-scaling
2. **Standard vs FIFO** → Throughput vs ordering
3. **SNS = Pub/sub for fan-out** → One message, many subscribers
4. **SNS + SQS fanout = Best practice** → Decouple producers from consumers
5. **Message filtering reduces costs** → Deliver only relevant messages
6. **Lambda integration = Serverless processing** → Auto-scaling, pay-per-use
7. **DLQ handles failures** → Poison message handling
8. **Long polling + batching = Cost optimization** → 10× cost reduction
9. **Choose for AWS workloads** → Native integration, zero ops
10. **In interviews: Discuss fanout, scaling, costs** → Show cloud-native thinking

---

**Next:** We'll explore **Event-Driven Architecture**—designing systems around events, event sourcing, CQRS, and domain events.`,
};
