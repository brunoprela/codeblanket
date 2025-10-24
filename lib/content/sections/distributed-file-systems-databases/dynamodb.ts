/**
 * DynamoDB Section
 */

export const dynamodbSection = {
  id: 'dynamodb',
  title: 'Amazon DynamoDB',
  content: `Amazon DynamoDB is a fully managed, serverless NoSQL database providing single-digit millisecond performance at any scale with built-in security, backup, and multi-region replication.

## Overview

**DynamoDB** = AWS managed key-value and document database

**Launched**: 2012 (inspired by Dynamo paper)

**Key features**:
- Fully managed (no servers to provision)
- Serverless scaling
- Single-digit millisecond latency
- Built-in replication across 3 AZs
- Global tables (multi-region)
- Pay per request or provisioned capacity

**Used by**:
- Amazon.com (shopping cart, session data)
- Lyft (trip data)
- Duolingo (user progress)
- Samsung (IoT data)

**Scale**:
- 10 trillion requests/day at peak
- Automatic scaling to any workload
- Petabytes of storage

---

## Data Model

### Core Concepts

**1. Table**:
- Collection of items
- No fixed schema
- Scalable

**2. Item**:
- Single data record (like a row)
- Up to 400 KB
- Collection of attributes

**3. Attributes**:
- Key-value pairs (like columns)
- Each item can have different attributes

**4. Primary Key**:
- **Partition Key** (required)
- **Sort Key** (optional)

### Primary Key Types

**Simple Primary Key** (Partition Key only):
\`\`\`
Table: Users
Primary Key: user_id

user_id (partition key) | name    | email
123                     | Alice   | alice@example.com
456                     | Bob     | bob@example.com
\`\`\`

**Composite Primary Key** (Partition Key + Sort Key):
\`\`\`
Table: Orders
Primary Key: (user_id, order_date)

user_id (PK) | order_date (SK)  | amount  | items
123          | 2024-01-15       | 99.99   | [...]
123          | 2024-01-20       | 50.00   | [...]
456          | 2024-01-16       | 120.00  | [...]
\`\`\`

**Partition key** determines data location (shard)
**Sort key** enables range queries within partition

---

## Partitioning

### How Partitioning Works

\`\`\`
hash(partition_key) → Partition
\`\`\`

**Example**:
\`\`\`
Table: Orders
Partition Key: user_id

user_id=123 → hash → Partition A
user_id=456 → hash → Partition B
user_id=789 → hash → Partition C
\`\`\`

**Each partition**:
- Max 10 GB storage
- Max 3000 RCU or 1000 WCU
- Data automatically split when limits exceeded

### Hot Partitions

**Problem**: Uneven access patterns

\`\`\`
Bad design:
Table: Analytics
Partition Key: date

date=2024-01-15 → ALL writes go to same partition!
  ↓
Hot partition (throttling)
\`\`\`

**Solution**: Add randomness
\`\`\`
Good design:
Partition Key: date#shard
  (e.g., "2024-01-15#1", "2024-01-15#2", ...)

Distribute writes across shards
Query: Scatter-gather from all shards
\`\`\`

---

## Indexes

### Local Secondary Index (LSI)

**Alternative sort key** on same partition key

\`\`\`
Base Table: Orders
Primary Key: (user_id, order_date)

LSI: OrdersByAmount
Primary Key: (user_id, amount)
\`\`\`

**Characteristics**:
- Same partition key as base table
- Different sort key
- Must be defined at table creation
- Max 5 LSIs per table
- Shares RCU/WCU with base table

**Use case**: Query same partition with different sort order

### Global Secondary Index (GSI)

**Completely different primary key**

\`\`\`
Base Table: Orders
Primary Key: (user_id, order_date)

GSI: OrdersByStatus
Primary Key: (status, order_date)
\`\`\`

**Characteristics**:
- Different partition key (and optional sort key)
- Can be added after table creation
- Max 20 GSIs per table
- Separate RCU/WCU provisioning
- Eventually consistent with base table

**Use case**: Query by different attributes

### Index Design Patterns

**1. Inverted Index**:
\`\`\`
Base: user_id → posts
GSI: post_id → user_id
\`\`\`

**2. Sparse Index** (save space):
\`\`\`
Only index items where attribute exists
Example: Only index "premium_user" = true
\`\`\`

**3. Overloading GSI** (single table design):
\`\`\`
GSI with generic PK/SK:
  gsi_pk: "USER#123"
  gsi_sk: "POST#456"
\`\`\`

---

## Read Operations

### GetItem

**Read single item** by primary key:

\`\`\`python
response = table.get_item(
    Key={'user_id': '123', 'order_date': '2024-01-15'}
)
\`\`\`

**Consistency**:
- **Eventually consistent** (default): May read stale data
- **Strongly consistent**: Always latest data (costs 2x RCU)

### BatchGetItem

**Read up to 100 items** (up to 16 MB):

\`\`\`python
response = dynamodb.batch_get_item(
    RequestItems={
        'Users': {
            'Keys': [
                {'user_id': '123'},
                {'user_id': '456'},
                {'user_id': '789'}
            ]
        }
    }
)
\`\`\`

### Query

**Read multiple items** with same partition key:

\`\`\`python
response = table.query(
    KeyConditionExpression=Key('user_id').eq('123') & 
                          Key('order_date').between('2024-01-01', '2024-01-31')
)
\`\`\`

**Query operators** on sort key:
- \`=\`, \`<\`, \`<=\`, \`>\`, \`>=\`
- \`between(low, high)\`
- \`begins_with(prefix)\`

**Returns**: Items in sort key order (ascending by default)

### Scan

**Read entire table** (avoid if possible!):

\`\`\`python
response = table.scan(
    FilterExpression=Attr('age').gt(25)
)
\`\`\`

**Problems**:
- ❌ Reads ALL data (expensive)
- ❌ Consumes RCU for entire table
- ❌ Slow for large tables

**Use scan only**:
- Analytics/ETL jobs
- Exporting data
- Small tables

---

## Write Operations

### PutItem

**Insert or replace** entire item:

\`\`\`python
table.put_item(
    Item={
        'user_id': '123',
        'name': 'Alice',
        'email': 'alice@example.com',
        'created_at': '2024-01-15'
    }
)
\`\`\`

### UpdateItem

**Update specific attributes**:

\`\`\`python
table.update_item(
    Key={'user_id': '123'},
    UpdateExpression='SET #n = :name, #e = :email',
    ExpressionAttributeNames={'#n': 'name', '#e': 'email'},
    ExpressionAttributeValues={
        ':name': 'Alice Smith',
        ':email': 'alice.smith@example.com'
    }
)
\`\`\`

**Atomic operations**:
- \`SET\`: Set attribute
- \`REMOVE\`: Delete attribute
- \`ADD\`: Increment number / add to set
- \`DELETE\`: Remove from set

### DeleteItem

**Delete single item**:

\`\`\`python
table.delete_item(
    Key={'user_id': '123', 'order_date': '2024-01-15'}
)
\`\`\`

### BatchWriteItem

**Write up to 25 items** (up to 16 MB):

\`\`\`python
dynamodb.batch_write_item(
    RequestItems={
        'Users': [
            {'PutRequest': {'Item': {'user_id': '123', 'name': 'Alice'}}},
            {'PutRequest': {'Item': {'user_id': '456', 'name': 'Bob'}}},
            {'DeleteRequest': {'Key': {'user_id': '789'}}}
        ]
    }
)
\`\`\`

---

## Conditional Writes

### Prevent Race Conditions

\`\`\`python
# Only update if version matches (optimistic locking)
table.update_item(
    Key={'user_id': '123'},
    UpdateExpression='SET balance = balance - :amount, version = version + :inc',
    ConditionExpression='version = :expected_version',
    ExpressionAttributeValues={
        ':amount': 100,
        ':inc': 1,
        ':expected_version': 5
    }
)
# Raises ConditionalCheckFailedException if version mismatch
\`\`\`

**Conditions**:
- \`attribute_exists(path)\`
- \`attribute_not_exists(path)\`
- \`attribute_type(path, type)\`
- Comparisons: \`= \`, \`<>\`, \` < \`, \` <= \`, \` > \`, \` >= \`
- Logical: \`AND\`, \`OR\`, \`NOT\`

---

## Capacity Modes

### 1. Provisioned Capacity

**Manual provisioning**:

\`\`\`
Read Capacity Units (RCU):
- 1 RCU = 1 strongly consistent read/sec (up to 4 KB)
- 1 RCU = 2 eventually consistent reads/sec (up to 4 KB)

Write Capacity Units (WCU):
- 1 WCU = 1 write/sec (up to 1 KB)
\`\`\`

**Example calculations**:
\`\`\`
Read 10 items/sec, each 8 KB, strongly consistent:
= 10 * (8/4) = 20 RCU

Write 5 items/sec, each 2 KB:
= 5 * (2/1) = 10 WCU
\`\`\`

**Auto scaling**:
- Set target utilization (e.g., 70%)
- DynamoDB scales up/down automatically

### 2. On-Demand Capacity

**Pay per request**:
- No capacity planning needed
- Scales automatically
- More expensive per request
- Ideal for unpredictable workloads

**Pricing**:
- Read: $0.25 per million requests
- Write: $1.25 per million requests

**When to use on-demand**:
- Unpredictable traffic
- New applications (unknown load)
- Spiky workloads

**When to use provisioned**:
- Predictable traffic
- Steady state workloads
- Cost optimization (cheaper at scale)

---

## Consistency Model

### Eventually Consistent (Default)

\`\`\`
Write to Node A (primary)
  ↓ (async replication)
Node B and Node C (replicas)

Read from Node B → Might get old data!
\`\`\`

**Characteristics**:
- Lower latency
- Higher throughput
- May read stale data (< 1 second typically)

### Strongly Consistent

\`\`\`
Write to Node A (primary)
  ↓ (wait for acknowledgment from replicas)
All replicas updated
  ↓
Read from any node → Latest data
\`\`\`

**Characteristics**:
- Higher latency
- Lower throughput (2x RCU cost)
- Always returns latest data

**Specify in request**:
\`\`\`python
response = table.get_item(
    Key={'user_id': '123'},
    ConsistentRead=True  # Strong consistency
)
\`\`\`

---

## Transactions

### ACID Transactions Across Items

**TransactWriteItems** (up to 25 operations):

\`\`\`python
dynamodb.transact_write_items(
    TransactItems=[
        {
            'Update': {
                'TableName': 'Accounts',
                'Key': {'account_id': '123'},
                'UpdateExpression': 'SET balance = balance - :amount',
                'ConditionExpression': 'balance >= :amount',
                'ExpressionAttributeValues': {':amount': 100}
            }
        },
        {
            'Update': {
                'TableName': 'Accounts',
                'Key': {'account_id': '456'},
                'UpdateExpression': 'SET balance = balance + :amount',
                'ExpressionAttributeValues': {':amount': 100}
            }
        }
    ]
)
# All-or-nothing: Both succeed or both fail
\`\`\`

**TransactGetItems** (up to 25 operations):
- Strongly consistent
- Snapshot isolation

**Cost**: 2x normal operation cost

---

## Streams

### Change Data Capture

**DynamoDB Streams** captures item-level changes:

\`\`\`
Table: Orders
  ↓ (insert/update/delete)
DynamoDB Stream
  ↓
Lambda Function (triggered)
  ↓
Process change (e.g., send notification)
\`\`\`

**Stream views**:
- \`KEYS_ONLY\`: Only keys
- \`NEW_IMAGE\`: New item
- \`OLD_IMAGE\`: Old item
- \`NEW_AND_OLD_IMAGES\`: Both

**Use cases**:
- Real-time notifications
- Replication to other systems
- Audit logging
- Materialized views

---

## Global Tables

### Multi-Region Replication

\`\`\`
Region: us-east-1              Region: eu-west-1
  Table: Users                   Table: Users (replica)
       ↓                              ↓
    Write                           Write
       ↓ (async replication)          ↓
    Eventually consistent sync
\`\`\`

**Characteristics**:
- Active-active (writes in any region)
- Automatic replication
- Conflict resolution (last-writer-wins)
- Sub-second replication latency

**Use cases**:
- Global applications
- Disaster recovery
- Low-latency access worldwide

**Setup**:
\`\`\`python
table.create_global_table(
    ReplicationGroup=[
        {'RegionName': 'us-east-1'},
        {'RegionName': 'eu-west-1'},
        {'RegionName': 'ap-southeast-1'}
    ]
)
\`\`\`

---

## Single-Table Design

### One Table for Entire Application

**Concept**: Store multiple entity types in one table

\`\`\`
PK                 | SK                | attributes...
-------------------+-------------------+-------------
USER#123           | METADATA          | name, email
USER#123           | POST#001          | content, date
USER#123           | POST#002          | content, date
USER#456           | METADATA          | name, email
POST#001           | COMMENT#001       | text, author
POST#001           | COMMENT#002       | text, author
\`\`\`

**Benefits**:
- ✅ Fewer tables to manage
- ✅ Related data in same partition (efficient queries)
- ✅ Transactions within partition

**Design patterns**:
- Hierarchical data (USER → POSTS → COMMENTS)
- Many-to-many relationships
- Adjacency lists

**When to use**:
- Complex access patterns
- Need for transactions across entities
- Cost optimization (fewer tables)

**When NOT to use**:
- Simple access patterns
- Team prefers clarity over optimization
- Learning curve too steep

---

## Best Practices

### 1. Choose Right Primary Key

**Bad**: Sequential IDs
\`\`\`
user_id: 1, 2, 3, 4, 5...
→ Writes go to same partition (hot partition)
\`\`\`

**Good**: UUIDs or composite keys
\`\`\`
user_id: UUID
→ Writes distributed across partitions
\`\`\`

### 2. Use GSIs for Alternate Queries

**Base table**: Query by user_id
**GSI**: Query by email

### 3. Design for Uniform Access

Avoid hot partitions:
- Don't use highly skewed keys (e.g., date, popular user_id)
- Add randomness if needed

### 4. Use Batch Operations

Save costs and improve performance:
- BatchGetItem (up to 100 items)
- BatchWriteItem (up to 25 items)

### 5. Leverage DAX for Caching

**DynamoDB Accelerator (DAX)**:
- In-memory cache
- Microsecond latency
- Transparent (no code changes)
- Reduces RCU consumption

### 6. Monitor with CloudWatch

**Key metrics**:
- ConsumedReadCapacityUnits
- ConsumedWriteCapacityUnits
- ThrottledRequests
- SystemErrors

---

## Performance Optimization

### 1. Use Eventually Consistent Reads

- 50% cheaper (1 RCU vs 2 RCU)
- Lower latency
- Acceptable for most use cases

### 2. Use Projection Expression

**Bad** (read entire item):
\`\`\`python
response = table.get_item(Key={'user_id': '123'})
# Returns all attributes (wasteful if item is large)
\`\`\`

**Good** (read specific attributes):
\`\`\`python
response = table.get_item(
    Key={'user_id': '123'},
    ProjectionExpression='user_id, name, email'
)
# Returns only needed attributes
\`\`\`

### 3. Use Query Instead of Scan

**Scan**: Reads entire table (expensive)
**Query**: Reads only items with matching partition key (efficient)

### 4. Compress Large Attributes

- Use compression (gzip) for large text
- Store compressed data as binary
- Saves storage and reduces RCU

### 5. Parallelize Scans

For large tables, run parallel scans:
\`\`\`python
# Segment 1 of 4
response = table.scan(Segment=0, TotalSegments=4)

# Segment 2 of 4
response = table.scan(Segment=1, TotalSegments=4)
...
\`\`\`

---

## Use Cases

**1. Session Storage**:
- Web application sessions
- Shopping carts
- User preferences

**Why DynamoDB?**
- Fast key-value lookups
- Automatic expiration (TTL)
- Scalable

**2. Gaming Leaderboards**:
- Player scores
- Real-time rankings

**Why DynamoDB?**
- Fast writes
- Sort key for ranking
- Consistent performance

**3. IoT Data**:
- Sensor readings
- Device telemetry

**Why DynamoDB?**
- High write throughput
- Time-series data (partition by device, sort by time)
- Auto-scaling

**Not suitable for**:
- ❌ Complex joins
- ❌ Aggregations (SUM, AVG)
- ❌ OLAP workloads
- ❌ Full-text search

---

## Interview Tips

**Explain DynamoDB in 2 minutes**:
"DynamoDB is AWS's fully managed NoSQL database providing single-digit millisecond latency. Data is partitioned by partition key using consistent hashing. Supports simple and composite primary keys. LSIs provide alternative sort keys, GSIs provide alternative partition keys. Eventually consistent by default, strongly consistent available. On-demand or provisioned capacity modes. Supports ACID transactions across items. DynamoDB Streams for CDC. Global Tables for multi-region replication. Best for key-value lookups, time-series data, session storage. Single-table design pattern stores multiple entities in one table for efficiency."

**Key concepts to mention**:
- Partitioning by partition key
- LSI vs GSI
- Eventually vs strongly consistent
- On-demand vs provisioned capacity
- Single-table design
- Global tables

**Common mistakes**:
- ❌ Hot partitions (bad key choice)
- ❌ Using scan instead of query
- ❌ Not using GSIs for alternate access patterns
- ❌ Treating like SQL database

---

## Key Takeaways

🔑 Fully managed, serverless NoSQL database
🔑 Partition key determines data placement
🔑 LSI = same partition key, different sort key
🔑 GSI = different partition key, eventually consistent
🔑 Eventually consistent (default) vs strongly consistent
🔑 On-demand (pay per request) vs provisioned capacity
🔑 ACID transactions across up to 25 items
🔑 Global Tables for multi-region, active-active replication
🔑 Single-table design: multiple entities in one table
🔑 Best for: key-value lookups, time-series, high throughput
`,
};
