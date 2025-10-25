/**
 * Discussion Questions for AWS SQS & SNS
 */

import { QuizQuestion } from '../../../types';

export const awssqssnsQuiz: QuizQuestion[] = [
  {
    id: 'aws-sqs-sns-dq-1',
    question:
      'Design a serverless order processing system using SNS and SQS. Orders should trigger multiple services (payment, inventory, email). Include fanout pattern, FIFO guarantees for payments, dead letter queues, and cost optimization strategies. How would you handle order cancellations that arrive after processing starts?',
    hint: 'Consider SNS topic → multiple SQS queues, FIFO for payment ordering, visibility timeout for cancellations, and batching for cost.',
    sampleAnswer: `AWS SNS and SQS enable serverless event-driven architectures with managed infrastructure.

**Architecture:**

\`\`\`
Order API (Lambda)
  ↓
SNS Topic: "orders" (fanout)
  ├→ SQS: payment-queue (FIFO) → Payment Lambda
  ├→ SQS: inventory-queue (Standard) → Inventory Lambda
  ├→ SQS: email-queue (Standard) → Email Lambda
  └→ SQS: analytics-queue (Standard) → Analytics Lambda

Dead Letter Queues:
  - payment-dlq (FIFO)
  - inventory-dlq (Standard)
  - email-dlq (Standard)
\`\`\`

**Implementation:**

\`\`\`python
import boto3
import json

sns = boto3.client('sns')
sqs = boto3.client('sqs')

# 1. Create SNS topic
topic_response = sns.create_topic(Name='orders')
topic_arn = topic_response['TopicArn']

# 2. Create FIFO queue for payments (ordering required)
payment_dlq = sqs.create_queue(
    QueueName='payment-dlq.fifo',
    Attributes={
        'FifoQueue': 'true',
        'ContentBasedDeduplication': 'true'
    }
)

payment_queue = sqs.create_queue(
    QueueName='payment-queue.fifo',
    Attributes={
        'FifoQueue': 'true',
        'ContentBasedDeduplication': 'true',
        'VisibilityTimeout': '300',  # 5 minutes
        'MessageRetentionPeriod': '1209600',  # 14 days
        'ReceiveMessageWaitTimeSeconds': '20',  # Long polling
        'RedrivePolicy': json.dumps({
            'deadLetterTargetArn': payment_dlq['QueueUrl'].split('/')[-1],
            'maxReceiveCount': '3'
        })
    }
)

# 3. Create standard queues for inventory, email (no ordering needed)
inventory_queue = sqs.create_queue(
    QueueName='inventory-queue',
    Attributes={
        'VisibilityTimeout': '30',
        'MessageRetentionPeriod': '86400',  # 1 day
        'ReceiveMessageWaitTimeSeconds': '20'
    }
)

# 4. Subscribe queues to SNS topic
sns.subscribe(
    TopicArn=topic_arn,
    Protocol='sqs',
    Endpoint=payment_queue['QueueUrl'],
    Attributes={
        'RawMessageDelivery': 'true',  # Receive message directly (not wrapped)
        'FilterPolicy': json.dumps({
            'event_type': ['order_placed', 'order_updated']
        })
    }
)

# 5. Publish order to SNS (fanout to all queues)
def publish_order (order):
    sns.publish(
        TopicArn=topic_arn,
        Message=json.dumps (order),
        MessageGroupId=order['customer_id'],  # FIFO: group by customer
        MessageDeduplicationId=order['order_id'],
        MessageAttributes={
            'event_type': {
                'DataType': 'String',
                'StringValue': 'order_placed'
            }
        }
    )

# 6. Consumer: Payment Lambda
def lambda_handler (event, context):
    for record in event['Records']:
        order = json.loads (record['body'])
        
        try:
            # Check for cancellation
            if check_cancellation (order['order_id']):
                print(f"Order {order['order_id']} cancelled, skipping")
                continue
            
            # Process payment
            process_payment (order)
            
        except Exception as e:
            print(f"Payment failed: {e}")
            # After 3 retries, message goes to DLQ automatically
            raise
\`\`\`

**Order Cancellation Handling:**

**Problem:** Order placed → Processing starts → Cancellation received

**Solution: Check cancellation before processing**

\`\`\`python
# Cancellation flow:
# 1. Order API receives cancellation
def cancel_order (order_id):
    # Write to DynamoDB cancellation table
    dynamodb.put_item(
        TableName='order_cancellations',
        Item={'order_id': order_id, 'cancelled_at': timestamp()}
    )
    
    # Publish cancellation event
    sns.publish(
        TopicArn=topic_arn,
        Message=json.dumps({'order_id': order_id}),
        MessageAttributes={
            'event_type': {'DataType': 'String', 'StringValue': 'order_cancelled'}
        }
    )

# 2. Consumers check before processing
def process_payment (order):
    # Check DynamoDB for cancellation
    response = dynamodb.get_item(
        TableName='order_cancellations',
        Key={'order_id': order['order_id']}
    )
    
    if 'Item' in response:
        print(f"Order cancelled, skipping payment")
        return  # Skip processing
    
    # Process payment...
    charge_card (order)
\`\`\`

**Race Condition:**
- Order processing starts → Cancellation arrives → Payment already processed
- Solution: **Idempotent refund** (if cancelled after payment, refund)

**Cost Optimization:**

\`\`\`python
# 1. Long polling (reduce empty receives)
Attributes={'ReceiveMessageWaitTimeSeconds': '20'}
# Cost: $0.40 per million requests
# Benefit: 10× fewer API calls vs short polling

# 2. Batch processing (up to 10 messages)
response = sqs.receive_message(
    QueueUrl=queue_url,
    MaxNumberOfMessages=10,  # Batch receive
    WaitTimeSeconds=20
)

for msg in response.get('Messages', []):
    process (msg)

# Delete batch (single API call)
sqs.delete_message_batch(
    QueueUrl=queue_url,
    Entries=[
        {'Id': msg['MessageId'], 'ReceiptHandle': msg['ReceiptHandle']}
        for msg in messages
    ]
)

# 3. Right-size retention period
# Standard queue: 1 day (not 14 days) if fast processing
Attributes={'MessageRetentionPeriod': '86400'}
# No cost savings, but best practice

# 4. Use Standard over FIFO when ordering not needed
# FIFO: Higher latency, lower throughput
# Standard: Lower latency, higher throughput, cheaper

# 5. Filter messages at SNS subscription
# Only receive relevant messages (reduce SQS costs)
FilterPolicy={
    'event_type': ['order_placed'],
    'region': ['us-east-1']
}

# Estimated costs:
# SNS: $0.50 per million publishes
# SQS Standard: $0.40 per million requests
# SQS FIFO: $0.50 per million requests
# Lambda: $0.20 per million requests + compute time
\`\`\`

**FIFO Guarantees:**

**Payment Queue (FIFO):**
\`\`\`
Customer A places order 1 → order 2 → order 3

MessageGroupId = customer_id

Processing order:
1. Order 1 (Group: customer_A)
2. Order 2 (Group: customer_A) waits
3. Order 3 (Group: customer_A) waits

Order 1 completes → Order 2 starts → Order 3 starts

✅ Strict ordering per customer
✅ No duplicate charges (ContentBasedDeduplication)
\`\`\`

**Dead Letter Queue Monitoring:**

\`\`\`python
# CloudWatch alarm on DLQ depth
cloudwatch.put_metric_alarm(
    AlarmName='payment-dlq-alert',
    MetricName='ApproximateNumberOfMessagesVisible',
    Namespace='AWS/SQS',
    Dimensions=[{'Name': 'QueueName', 'Value': 'payment-dlq.fifo'}],
    Statistic='Average',
    Period=300,
    EvaluationPeriods=1,
    Threshold=1,  # Alert on any message in DLQ
    ComparisonOperator='GreaterThanThreshold',
    AlarmActions=['arn:aws:sns:us-east-1:123456789:alerts']
)

# Reprocess from DLQ
def reprocess_dlq():
    while True:
        response = sqs.receive_message(
            QueueUrl=dlq_url,
            MaxNumberOfMessages=10
        )
        
        messages = response.get('Messages', [])
        if not messages:
            break
        
        for msg in messages:
            order = json.loads (msg['Body'])
            
            try:
                # Attempt reprocessing
                process_payment (order)
                
                # Success: Delete from DLQ
                sqs.delete_message(
                    QueueUrl=dlq_url,
                    ReceiptHandle=msg['ReceiptHandle']
                )
            except Exception as e:
                print(f"Reprocessing failed: {e}")
                # Leave in DLQ for manual review
\`\`\`

**Key Takeaways:**
✅ SNS fanout to multiple SQS queues (parallel processing)
✅ FIFO queues for payment ordering (per customer)
✅ Standard queues for inventory/email (no ordering needed)
✅ Long polling + batching for cost optimization
✅ DLQ for failed messages with CloudWatch alarms
✅ Cancellation checking before processing (DynamoDB)`,
    keyPoints: [
      'SNS fanout pattern broadcasts to multiple SQS queues (parallel processing)',
      'FIFO queue for payments ensures ordering per customer (MessageGroupId)',
      'Standard queues for inventory/email (cheaper, faster, no ordering)',
      'Long polling (20s) + batching (10 messages) reduces API costs',
      'DLQ with maxReceiveCount=3 isolates failed messages',
      'Cancellation handling: Check DynamoDB before processing',
    ],
  },
  {
    id: 'aws-sqs-sns-dq-2',
    question:
      'Compare SQS Standard vs FIFO queues. For a high-volume image processing pipeline (10K images/sec) with batch processing, which would you choose? Include throughput limits, deduplication, visibility timeout strategy, and scaling considerations.',
    hint: 'Consider throughput (3000 vs unlimited), deduplication overhead, batch processing, and cost-throughput trade-offs.',
    sampleAnswer: `SQS Standard and FIFO queues offer different guarantees with different throughput characteristics.

**Standard vs FIFO:**

| Feature | Standard | FIFO |
|---------|----------|------|
| **Throughput** | Unlimited | 3000 msg/sec (300 batch) |
| **Ordering** | Best-effort | Strict |
| **Duplicates** | Possible | Exactly-once |
| **Latency** | Very low | Low |
| **Cost** | $0.40/M | $0.50/M |

**Image Processing Pipeline:**

**Requirements:**
- Volume: 10K images/sec
- Processing: Resize, watermark, upload to S3
- No ordering requirement (images independent)
- Idempotent processing (same image processed twice = same result)

**Choice: Standard Queue ✅**

**Why:**
- Throughput: 10K images/sec exceeds FIFO limit (3000/sec)
- No ordering needed (images independent)
- Idempotent processing handles duplicates
- Lower cost

**Architecture:**

\`\`\`python
import boto3
import json

sqs = boto3.client('sqs')

# Create Standard queue (unlimited throughput)
queue = sqs.create_queue(
    QueueName='image-processing',
    Attributes={
        'VisibilityTimeout': '300',  # 5 minutes (processing time)
        'MessageRetentionPeriod': '86400',  # 1 day
        'ReceiveMessageWaitTimeSeconds': '20',  # Long polling
        'RedrivePolicy': json.dumps({
            'deadLetterTargetArn': dlq_arn,
            'maxReceiveCount': '3'
        })
    }
)

# Producer: Enqueue images
def enqueue_image (image_url, image_id):
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps({
            'image_url': image_url,
            'image_id': image_id,
            'enqueued_at': time.time()
        }),
        MessageDeduplicationId=image_id,  # Dedup (not enforced in Standard)
        MessageAttributes={
            'priority': {
                'DataType': 'String',
                'StringValue': 'normal'
            }
        }
    )

# Consumer: Lambda (batch processing)
def lambda_handler (event, context):
    processed = []
    
    for record in event['Records']:
        message = json.loads (record['body'])
        image_id = message['image_id']
        
        # Idempotency check (S3 or DynamoDB)
        if s3.head_object(Bucket='processed-images', Key=f"{image_id}.jpg"):
            print(f"Image {image_id} already processed, skipping")
            continue
        
        try:
            # Process image
            image_data = download_image (message['image_url'])
            processed_image = resize_and_watermark (image_data)
            
            # Upload to S3 (idempotent)
            s3.put_object(
                Bucket='processed-images',
                Key=f"{image_id}.jpg",
                Body=processed_image
            )
            
            processed.append (image_id)
            
        except Exception as e:
            print(f"Processing failed for {image_id}: {e}")
            # Will retry up to 3 times, then DLQ
            raise
    
    return {'processed': len (processed)}

# Lambda configuration:
# - Batch size: 10 images (process in parallel)
# - Reserved concurrency: 1000 (10K images/sec / 10 per batch)
# - Memory: 1024 MB (image processing)
# - Timeout: 300 seconds (5 minutes)
\`\`\`

**Throughput Analysis:**

\`\`\`
Standard Queue:
- Throughput: Unlimited (10K+ msg/sec) ✅
- Processing: 10 images per Lambda invocation
- Lambda concurrency: 10,000 images/sec / 10 per batch = 1000 concurrent
- Processing time: ~10 seconds per image
- Total capacity: 1000 concurrent × 10 images × 1/10 sec = 1000 images/sec per Lambda

Wait, that's only 1K images/sec!

Need more Lambda concurrency:
10,000 images/sec / 10 images per batch / (10 sec / 60 sec) = 
10,000 / 10 / 0.167 = 6000 concurrent Lambda invocations

With 6000 concurrent Lambdas processing 10 images each in 10 seconds:
6000 × 10 images × 6 batches/min = 360K images/min = 6K images/sec ❌

Better: Faster processing (5 sec per image):
6000 × 10 × 12 batches/min = 720K/min = 12K images/sec ✅

Standard queue handles this easily.

FIFO Queue:
- Max throughput: 3000 msg/sec ❌
- Can't handle 10K images/sec
- Would need to partition into multiple FIFO queues (complex)
\`\`\`

**Deduplication Strategy:**

\`\`\`python
# Standard queue doesn't guarantee deduplication
# Implement application-level deduplication

# Option 1: S3 check (as above)
try:
    s3.head_object(Bucket='processed', Key=f"{image_id}.jpg")
    return  # Already processed
except s3.exceptions.NoSuchKey:
    # Not processed, continue

# Option 2: DynamoDB
def is_processed (image_id):
    response = dynamodb.get_item(
        TableName='processed_images',
        Key={'image_id': image_id}
    )
    return 'Item' in response

# Option 3: ElastiCache Redis (fastest)
if redis.exists (f"processed:{image_id}"):
    return  # Already processed

# All options make processing idempotent
✅ Handles Standard queue duplicates
✅ Handles retry duplicates
\`\`\`

**Visibility Timeout Strategy:**

\`\`\`
Visibility Timeout = Expected processing time + buffer

Image processing: ~10 seconds per image
Batch of 10: 10 × 10 = 100 seconds
Buffer: 2× for safety
Visibility timeout: 200 seconds ✅

If processing takes longer:
- Extend visibility timeout programmatically

def process_with_extension (message):
    receipt_handle = message['ReceiptHandle']
    
    for i in range(10):
        process_image (i)
        
        # Extend visibility every 30 seconds
        if i % 3 == 0:
            sqs.change_message_visibility(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=200
            )
\`\`\`

**Scaling Considerations:**

\`\`\`
# Lambda event source mapping
lambda_client.create_event_source_mapping(
    EventSourceArn=queue_arn,
    FunctionName='image-processor',
    BatchSize=10,  # Process 10 images per invocation
    MaximumBatchingWindowInSeconds=5,  # Wait up to 5 sec to fill batch
    ScalingConfig={
        'MaximumConcurrency': 6000  # Max concurrent Lambda invocations
    }
)

# Auto-scaling based on queue depth
# If queue depth > 1000: Scale up Lambda concurrency
# If queue depth < 100: Scale down

# CloudWatch alarm
cloudwatch.put_metric_alarm(
    AlarmName='image-queue-depth-high',
    MetricName='ApproximateNumberOfMessagesVisible',
    Namespace='AWS/SQS',
    Dimensions=[{'Name': 'QueueName', 'Value': 'image-processing'}],
    Statistic='Average',
    Period=60,
    Threshold=1000,
    ComparisonOperator='GreaterThanThreshold',
    AlarmActions=['arn:aws:sns:us-east-1:123456789:scale-up']
)
\`\`\`

**Cost Analysis:**

\`\`\`
Volume: 10K images/sec = 864M images/day

SQS Standard:
- Enqueue: 864M requests × $0.40/M = $345.60/day
- Dequeue: 864M requests × $0.40/M = $345.60/day
- Total SQS: $691.20/day

Lambda:
- Invocations: 86.4M invocations × $0.20/M = $17.28/day
- Compute: 86.4M × 10 sec × $0.0000166667/GB-sec × 1 GB = $14,400/day
- Total Lambda: $14,417/day

Total: ~$15,100/day

SQS FIFO (if it could handle 10K/sec):
- Same costs but can't handle throughput ❌
\`\`\`

**Conclusion:**
Standard queue is the only viable option for 10K images/sec due to unlimited throughput, with idempotent processing handling duplicates.`,
    keyPoints: [
      'Standard queue: Unlimited throughput (10K+ msg/sec)',
      'FIFO queue: Limited to 3000 msg/sec (300 with batching)',
      'Choose Standard for high throughput when ordering not required',
      'Implement application-level deduplication (S3/DynamoDB/Redis)',
      'Visibility timeout = processing time + buffer (200 sec for batch)',
      'Scale Lambda concurrency based on queue depth (6000+ concurrent)',
    ],
  },
  {
    id: 'aws-sqs-sns-dq-3',
    question:
      'Design a cross-region disaster recovery setup for a critical order processing system using SNS and SQS. Include multi-region active-active architecture, message replication, failover strategy, and data consistency considerations. How would you ensure no duplicate orders during region failover?',
    hint: 'Consider SNS cross-region subscriptions, global deduplication, DynamoDB global tables, and health checks.',
    sampleAnswer: `Cross-region disaster recovery ensures business continuity during regional outages.

**Multi-Region Architecture:**

\`\`\`
Region 1 (us-east-1):
  API Gateway → Lambda → SNS Topic (orders-us-east-1)
    ↓
  SQS Queue → Processing Lambda → DynamoDB (orders-table)

Region 2 (us-west-2):
  API Gateway → Lambda → SNS Topic (orders-us-west-2)
    ↓
  SQS Queue → Processing Lambda → DynamoDB (orders-table)

Global:
  - Route 53 (health check + failover routing)
  - DynamoDB Global Tables (replication)
  - Global deduplication layer (ElastiCache Global Datastore)
\`\`\`

**Implementation:**

\`\`\`python
# 1. Cross-region SNS subscriptions
# Region 1 SNS → Region 2 SQS (backup)
sns_us_east.subscribe(
    TopicArn='arn:aws:sns:us-east-1:123:orders',
    Protocol='sqs',
    Endpoint='arn:aws:sqs:us-west-2:123:orders-backup-queue'
)

# Region 2 SNS → Region 1 SQS (backup)
sns_us_west.subscribe(
    TopicArn='arn:aws:sns:us-west-2:123:orders',
    Protocol='sqs',
    Endpoint='arn:aws:sqs:us-east-1:123:orders-backup-queue'
)

# 2. DynamoDB Global Table (automatic replication)
dynamodb.create_global_table(
    GlobalTableName='orders',
    ReplicationGroup=[
        {'RegionName': 'us-east-1'},
        {'RegionName': 'us-west-2'}
    ]
)

# 3. Order processing with global deduplication
def process_order (order):
    order_id = order['order_id']
    
    # Check global deduplication (ElastiCache Global Datastore)
    if redis_global.exists (f"processed:{order_id}"):
        print(f"Order {order_id} already processed globally")
        return
    
    # Check DynamoDB (replicated globally)
    response = dynamodb.get_item(
        TableName='orders',
        Key={'order_id': order_id}
    )
    
    if 'Item' in response:
        print(f"Order {order_id} exists in database")
        redis_global.setex (f"processed:{order_id}", 3600, "true")
        return
    
    # Process order
    try:
        result = charge_card (order)
        
        # Save to DynamoDB (replicated to other region)
        dynamodb.put_item(
            TableName='orders',
            Item={
                'order_id': order_id,
                'status': 'completed',
                'processed_region': os.environ['AWS_REGION'],
                'timestamp': int (time.time())
            },
            ConditionExpression='attribute_not_exists (order_id)'  # Prevent duplicates
        )
        
        # Cache globally (fast dedup)
        redis_global.setex (f"processed:{order_id}", 3600, "true")
        
    except dynamodb.exceptions.ConditionalCheckFailedException:
        print(f"Order {order_id} already exists (race condition)")
        return

# 4. Route 53 health check and failover
route53.change_resource_record_sets(
    HostedZoneId='Z123',
    ChangeBatch={
        'Changes': [{
            'Action': 'UPSERT',
            'ResourceRecordSet': {
                'Name': 'api.example.com',
                'Type': 'A',
                'SetIdentifier': 'us-east-1',
                'Failover': 'PRIMARY',
                'AliasTarget': {
                    'HostedZoneId': 'Z123',
                    'DNSName': 'api-us-east-1.execute-api.amazonaws.com',
                    'EvaluateTargetHealth': True
                },
                'HealthCheckId': 'health-check-us-east-1'
            }
        }, {
            'Action': 'UPSERT',
            'ResourceRecordSet': {
                'Name': 'api.example.com',
                'Type': 'A',
                'SetIdentifier': 'us-west-2',
                'Failover': 'SECONDARY',
                'AliasTarget': {
                    'HostedZoneId': 'Z456',
                    'DNSName': 'api-us-west-2.execute-api.amazonaws.com',
                    'EvaluateTargetHealth': True
                }
            }
        }]
    }
)
\`\`\`

**Failover Scenario:**

\`\`\`
Normal Operation (Region 1 primary):
T0: Client → Route 53 → Region 1 API
T1: Order published to SNS (Region 1)
T2: SQS Queue (Region 1) receives message
T3: Lambda (Region 1) processes
T4: DynamoDB write (Region 1) → Replicate to Region 2
T5: ElastiCache Global write → Replicate to Region 2

Region 1 Outage:
T6: Health check fails in Region 1
T7: Route 53 fails over to Region 2 (DNS propagation: 60 seconds)
T8: New orders → Region 2 API
T9: Processing continues in Region 2

Messages in-flight during failover:
- Messages in Region 1 SQS (not yet processed)
- Messages in cross-region backup queue (Region 2)

Processing:
- Region 2 processes from local queue (new orders)
- Region 2 processes from backup queue (Region 1 orders)
- Deduplication prevents duplicates ✅

Recovery:
T10: Region 1 recovers
T11: Process backlog from backup queue (Region 1)
T12: Deduplication prevents reprocessing ✅
T13: Route 53 fails back to Region 1 (or stay in Region 2)
\`\`\`

**Duplicate Prevention:**

\`\`\`
Layer 1: Client-side deduplication
- Generate order_id client-side (UUID)
- Retry with same order_id

Layer 2: API Gateway idempotency
- Cache requests for 5 minutes
- Same request = same response

Layer 3: ElastiCache Global Datastore
- Fast deduplication (sub-millisecond)
- Replicated cross-region
- TTL 1 hour (recent orders)

Layer 4: DynamoDB Global Table
- Conditional writes (attribute_not_exists)
- Replicated cross-region
- Permanent record

Scenario: Order arrives in both regions
- Region 1: Checks ElastiCache → Not found → Process
- Region 1: Writes DynamoDB → Success
- Region 1: Writes ElastiCache → Success
- DynamoDB replicates to Region 2 (1-2 seconds)
- Region 2: Receives duplicate order
- Region 2: Checks ElastiCache → Found! (replicated)
- Region 2: Skips processing ✅

If ElastiCache replication delayed:
- Region 2: Checks ElastiCache → Not found
- Region 2: Checks DynamoDB → Found! (replicated)
- Region 2: Skips processing ✅

If DynamoDB write race:
- Both regions write simultaneously
- DynamoDB last-writer-wins (timestamp-based)
- One write succeeds, one fails conditionally
- Payment gateway idempotency key prevents double charge
\`\`\`

**Consistency Considerations:**

\`\`\`
DynamoDB Global Tables:
- Replication lag: 1-2 seconds (typical)
- Conflict resolution: Last-writer-wins
- Consistency: Eventual

ElastiCache Global Datastore:
- Replication lag: <1 second
- Consistency: Eventual

Strong consistency not guaranteed across regions
Use application-level deduplication (idempotency keys)

For critical operations:
- Payment gateway idempotency key
- Inventory checks with distributed locks
- Order IDs generated client-side
\`\`\`

**Monitoring:**

\`\`\`python
# CloudWatch metrics
cloudwatch.put_metric_data(
    Namespace='OrderProcessing',
    MetricData=[
        {
            'MetricName': 'OrdersProcessed',
            'Value': 1,
            'Unit': 'Count',
            'Dimensions': [
                {'Name': 'Region', 'Value': 'us-east-1'}
            ]
        },
        {
            'MetricName': 'DuplicatesDetected',
            'Value': 1 if duplicate else 0,
            'Unit': 'Count'
        }
    ]
)

# Alarms
- Health check failures (immediate failover)
- Replication lag > 5 seconds (data consistency)
- Duplicate rate > 1% (deduplication issues)
- Cross-region queue depth > 1000 (failover processing backlog)
\`\`\`

**Key Takeaways:**
✅ Multi-region active-active with Route 53 failover
✅ Cross-region SNS → SQS subscriptions (backup queues)
✅ DynamoDB Global Tables (automatic replication)
✅ ElastiCache Global Datastore (fast deduplication)
✅ Multiple deduplication layers (cache, database, payment gateway)
✅ Idempotent processing handles eventual consistency`,
    keyPoints: [
      'Multi-region active-active with Route 53 health check failover',
      'Cross-region SNS subscriptions send to backup SQS queues',
      'DynamoDB Global Tables replicate orders across regions (1-2s lag)',
      'ElastiCache Global Datastore for fast cross-region deduplication',
      'Multiple deduplication layers prevent duplicate orders during failover',
      'Idempotent processing handles eventual consistency and race conditions',
    ],
  },
];
