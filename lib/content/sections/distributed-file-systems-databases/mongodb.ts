/**
 * MongoDB Section
 */

export const mongodbSection = {
  id: 'mongodb',
  title: 'MongoDB',
  content: `MongoDB is a popular document-oriented NoSQL database that stores data in flexible, JSON-like documents, making it natural for object-oriented programming and agile development.

## Overview

**MongoDB** = Document-oriented NoSQL database

**Founded**: 2007 by 10gen (now MongoDB Inc.)

**Key features**:
- Document model (JSON-like BSON)
- Flexible schema
- Rich query language
- Horizontal scaling (sharding)
- Replication (replica sets)
- Aggregation framework

**Used by**:
- Facebook (user data)
- eBay (product catalog)
- The Guardian (content management)
- Adobe (user profiles)

**Scale**:
- Single server to thousands of nodes
- Millions of documents
- TBs to PBs of data

---

## Document Model

### Documents are JSON-like

\`\`\`json
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "name": "Alice Smith",
  "email": "alice@example.com",
  "age": 30,
  "address": {
    "street": "123 Main St",
    "city": "New York",
    "zip": "10001"
  },
  "interests": ["reading", "travel", "photography"],
  "created_at": ISODate("2024-01-15T10:30:00Z")
}
\`\`\`

**Benefits**:
- ✅ Natural mapping to objects
- ✅ Flexible schema (fields can vary)
- ✅ Embed related data (no joins)
- ✅ Arrays and nested documents

### BSON (Binary JSON)

**MongoDB stores documents as BSON**:
- Binary encoding of JSON
- Supports more data types (Date, ObjectId, Binary, etc.)
- Efficient traversal
- Fixed-size elements for faster parsing

---

## Database Hierarchy

\`\`\`
MongoDB Server
  └── Database
       └── Collection
            └── Document
                 └── Fields
\`\`\`

**Example**:
\`\`\`
Server: mongodb://localhost:27017
Database: ecommerce
  Collection: users
    Document: { _id: 1, name: "Alice", ... }
  Collection: products
    Document: { _id: 1, title: "Laptop", price: 999.99, ... }
  Collection: orders
    Document: { _id: 1, user_id: 1, items: [...], ... }
\`\`\`

---

## CRUD Operations

### Create (Insert)

\`\`\`javascript
// Insert one
db.users.insertOne({
  name: "Alice",
  email: "alice@example.com",
  age: 30
});

// Insert many
db.users.insertMany([
  { name: "Bob", email: "bob@example.com" },
  { name: "Charlie", email: "charlie@example.com" }
]);
\`\`\`

### Read (Find)

\`\`\`javascript
// Find all
db.users.find();

// Find with filter
db.users.find({ age: { $gte: 25 } });

// Find one
db.users.findOne({ name: "Alice" });

// Projection (select specific fields)
db.users.find({}, { name: 1, email: 1, _id: 0 });

// Sort, limit, skip
db.users.find().sort({ age: -1 }).limit(10).skip(0);
\`\`\`

### Update

\`\`\`javascript
// Update one
db.users.updateOne(
  { name: "Alice" },
  { $set: { age: 31 } }
);

// Update many
db.users.updateMany(
  { age: { $lt: 18 } },
  { $set: { status: "minor" } }
);

// Update operators
$set:      Set field value
$unset:    Remove field
$inc:      Increment number
$push:     Add to array
$pull:     Remove from array
$addToSet: Add unique to array
\`\`\`

### Delete

\`\`\`javascript
// Delete one
db.users.deleteOne({ name: "Alice" });

// Delete many
db.users.deleteMany({ status: "inactive" });
\`\`\`

---

## Query Language

### Query Operators

**Comparison**:
\`\`\`javascript
$eq:  Equal
$ne:  Not equal
$gt:  Greater than
$gte: Greater than or equal
$lt:  Less than
$lte: Less than or equal
$in:  In array
$nin: Not in array

// Example
db.products.find({ price: { $gte: 100, $lte: 1000 } });
\`\`\`

**Logical**:
\`\`\`javascript
$and: Logical AND
$or:  Logical OR
$not: Logical NOT
$nor: Logical NOR

// Example
db.users.find({
  $or: [
    { age: { $lt: 18 } },
    { status: "premium" }
  ]
});
\`\`\`

**Element**:
\`\`\`javascript
$exists: Field exists
$type:   Field type

// Example
db.users.find({ phone: { $exists: true } });
\`\`\`

**Array**:
\`\`\`javascript
$all:      Contains all elements
$elemMatch: Match array element
$size:     Array size

// Example
db.users.find({ interests: { $all: ["reading", "travel"] } });
\`\`\`

---

## Indexing

### Create Indexes

\`\`\`javascript
// Single field index
db.users.createIndex({ email: 1 });  // 1 = ascending, -1 = descending

// Compound index
db.products.createIndex({ category: 1, price: -1 });

// Unique index
db.users.createIndex({ email: 1 }, { unique: true });

// Text index (full-text search)
db.articles.createIndex({ title: "text", content: "text" });

// Geospatial index
db.locations.createIndex({ coordinates: "2dsphere" });

// TTL index (auto-delete after expiration)
db.sessions.createIndex({ createdAt: 1 }, { expireAfterSeconds: 3600 });
\`\`\`

### Index Types

**1. Single Field Index**:
\`\`\`javascript
db.users.createIndex({ age: 1 });
// Fast queries: db.users.find({ age: 30 })
\`\`\`

**2. Compound Index**:
\`\`\`javascript
db.orders.createIndex({ user_id: 1, order_date: -1 });
// Fast: db.orders.find({ user_id: 123 }).sort({ order_date: -1 })
\`\`\`

**3. Multikey Index** (arrays):
\`\`\`javascript
db.users.createIndex({ interests: 1 });
// Fast: db.users.find({ interests: "reading" })
\`\`\`

**4. Text Index**:
\`\`\`javascript
db.articles.createIndex({ content: "text" });
// Fast: db.articles.find({ $text: { $search: "mongodb tutorial" } })
\`\`\`

**5. Geospatial Index**:
\`\`\`javascript
db.places.createIndex({ location: "2dsphere" });
// Fast: db.places.find({ location: { $near: { $geometry: {...} } } })
\`\`\`

### Covered Queries

**Query entirely satisfied by index** (no document lookup):

\`\`\`javascript
db.users.createIndex({ name: 1, email: 1 });

// Covered query
db.users.find(
  { name: "Alice" },
  { name: 1, email: 1, _id: 0 }
);
// No need to read document, all data in index!
\`\`\`

---

## Aggregation Framework

### Pipeline Stages

\`\`\`javascript
db.orders.aggregate([
  // Stage 1: Match (filter)
  { $match: { status: "completed" } },
  
  // Stage 2: Group and aggregate
  { $group: {
      _id: "$user_id",
      total_spent: { $sum: "$amount" },
      order_count: { $sum: 1 }
  }},
  
  // Stage 3: Sort
  { $sort: { total_spent: -1 } },
  
  // Stage 4: Limit
  { $limit: 10 }
]);
\`\`\`

### Common Stages

**$match**: Filter documents
\`\`\`javascript
{ $match: { age: { $gte: 21 } } }
\`\`\`

**$group**: Group and aggregate
\`\`\`javascript
{ $group: {
    _id: "$category",
    count: { $sum: 1 },
    avg_price: { $avg: "$price" }
}}
\`\`\`

**$project**: Reshape documents
\`\`\`javascript
{ $project: {
    name: 1,
    total: { $multiply: ["$price", "$quantity"] }
}}
\`\`\`

**$lookup**: Join collections
\`\`\`javascript
{ $lookup: {
    from: "products",
    localField: "product_id",
    foreignField: "_id",
    as: "product_info"
}}
\`\`\`

**$unwind**: Deconstruct array
\`\`\`javascript
{ $unwind: "$items" }
\`\`\`

**$sort**: Sort documents
\`\`\`javascript
{ $sort: { created_at: -1 } }
\`\`\`

**$limit** / **$skip**: Pagination
\`\`\`javascript
{ $skip: 20 }, { $limit: 10 }
\`\`\`

---

## Replication (Replica Sets)

### Architecture

\`\`\`
         Primary
          (R/W)
         /     \\
        /       \\
   Secondary  Secondary
    (R only)  (R only)
\`\`\`

**Components**:
- **Primary**: Receives all writes, replicates to secondaries
- **Secondary**: Replicates from primary, can serve reads
- **Arbiter**: Voting member, no data (breaks ties)

### Automatic Failover

\`\`\`
Primary crashes
    ↓
Secondaries detect failure (heartbeat timeout)
    ↓
Election (majority vote)
    ↓
One secondary becomes new primary
    ↓
Clients automatically connect to new primary
\`\`\`

**Failover time**: Typically 10-30 seconds

### Read Preference

**Primary** (default):
\`\`\`javascript
db.users.find().readPref("primary");
// Always read from primary (strongly consistent)
\`\`\`

**PrimaryPreferred**:
\`\`\`javascript
db.users.find().readPref("primaryPreferred");
// Read from primary if available, else secondary
\`\`\`

**Secondary**:
\`\`\`javascript
db.users.find().readPref("secondary");
// Read from secondary (may be stale)
\`\`\`

**SecondaryPreferred**:
\`\`\`javascript
db.users.find().readPref("secondaryPreferred");
// Read from secondary if available, else primary
\`\`\`

**Nearest**:
\`\`\`javascript
db.users.find().readPref("nearest");
// Read from nearest member (lowest latency)
\`\`\`

---

## Sharding (Horizontal Scaling)

### Architecture

\`\`\`
      Application
          │
    ┌─────┴─────┐
    │   mongos  │ (Router)
    └─────┬─────┘
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
  Shard1 Shard2 Shard3
   (RS)   (RS)   (RS)
    │
Config Servers (metadata)
\`\`\`

**Components**:
- **mongos**: Query router, routes queries to shards
- **Config servers**: Store metadata (shard key ranges)
- **Shards**: Replica sets storing data

### Shard Keys

**Determines data distribution**:

\`\`\`javascript
// Shard by user_id
sh.shardCollection("mydb.users", { user_id: 1 });

// Shard by compound key
sh.shardCollection("mydb.orders", { user_id: 1, order_date: 1 });

// Hashed shard key (even distribution)
sh.shardCollection("mydb.events", { event_id: "hashed" });
\`\`\`

### Shard Key Selection

**Bad shard keys**:
- ❌ Monotonically increasing (sequential IDs, timestamps)
  → All writes go to last shard (hot shard)
- ❌ Low cardinality (few unique values)
  → Poor distribution
- ❌ Non-queried fields
  → Scatter-gather queries (slow)

**Good shard keys**:
- ✅ High cardinality
- ✅ Even distribution
- ✅ Queried frequently (targeted queries)

**Examples**:
\`\`\`
Good: user_id (UUID)
Good: {country: 1, user_id: 1} (compound)
Good: {_id: "hashed"} (hashed)

Bad: created_at (sequential)
Bad: status (low cardinality)
\`\`\`

### Chunk Management

**Chunks** = contiguous ranges of shard key values

\`\`\`
Shard 1: user_id [0 → 1000]
Shard 2: user_id [1001 → 2000]
Shard 3: user_id [2001 → 3000]
\`\`\`

**Balancer**:
- Monitors chunk distribution
- Migrates chunks between shards
- Runs automatically in background

---

## Write Concerns

### Durability Guarantees

**w: 1** (default):
\`\`\`javascript
{ w: 1 }
// Wait for primary acknowledgment
\`\`\`

**w: "majority"**:
\`\`\`javascript
{ w: "majority" }
// Wait for majority of replica set members
\`\`\`

**w: 0**:
\`\`\`javascript
{ w: 0 }
// Fire-and-forget (no acknowledgment)
\`\`\`

**j: true** (journaling):
\`\`\`javascript
{ w: 1, j: true }
// Wait for write to on-disk journal
\`\`\`

**wtimeout**:
\`\`\`javascript
{ w: "majority", wtimeout: 5000 }
// Timeout after 5 seconds
\`\`\`

---

## Transactions

### Multi-Document ACID Transactions

\`\`\`javascript
const session = client.startSession();

try {
  session.startTransaction();
  
  // Transfer money between accounts
  await accounts.updateOne(
    { _id: "account1" },
    { $inc: { balance: -100 } },
    { session }
  );
  
  await accounts.updateOne(
    { _id: "account2" },
    { $inc: { balance: 100 } },
    { session }
  );
  
  await session.commitTransaction();
} catch (error) {
  await session.abortTransaction();
  throw error;
} finally {
  session.endSession();
}
\`\`\`

**Limitations**:
- Requires replica set or sharded cluster
- Performance overhead
- 60 second timeout (default)

---

## Data Modeling

### Embedding vs Referencing

**Embedding** (denormalization):
\`\`\`javascript
{
  "_id": 1,
  "name": "Alice",
  "address": {
    "street": "123 Main St",
    "city": "NYC"
  },
  "orders": [
    { "order_id": 1, "amount": 99.99 },
    { "order_id": 2, "amount": 49.99 }
  ]
}
\`\`\`

**Benefits**:
- ✅ Single read operation
- ✅ Atomicity (update entire document)
- ✅ Better performance for related data

**Use when**:
- One-to-few relationships
- Data accessed together
- Data doesn't change independently

**Referencing** (normalization):
\`\`\`javascript
// User document
{ "_id": 1, "name": "Alice" }

// Order documents
{ "_id": 101, "user_id": 1, "amount": 99.99 }
{ "_id": 102, "user_id": 1, "amount": 49.99 }
\`\`\`

**Benefits**:
- ✅ Avoid duplication
- ✅ Smaller documents
- ✅ Independent updates

**Use when**:
- One-to-many or many-to-many
- Data accessed independently
- Document size would exceed 16 MB limit

---

## Performance Optimization

### 1. Use Indexes

- Index queried fields
- Use compound indexes for multiple fields
- Monitor slow queries

### 2. Use Projections

\`\`\`javascript
// Bad: Return entire document
db.users.find({ age: 30 });

// Good: Return only needed fields
db.users.find({ age: 30 }, { name: 1, email: 1 });
\`\`\`

### 3. Use Covered Queries

Design indexes to cover queries

### 4. Avoid Large Documents

- 16 MB document size limit
- Large documents slow reads/writes
- Use references if needed

### 5. Use Appropriate Read Preference

- Read from secondaries for analytics (eventual consistency OK)
- Read from primary for critical data

### 6. Use Connection Pooling

\`\`\`javascript
const client = new MongoClient (uri, {
  maxPoolSize: 50,
  minPoolSize: 10
});
\`\`\`

---

## Use Cases

**1. Content Management**:
- Blog posts, articles, comments
- Flexible schema for different content types

**2. Product Catalog**:
- E-commerce products with varying attributes
- Embed reviews, images, specifications

**3. User Profiles**:
- User accounts with diverse data
- Embed preferences, settings, activity

**4. Real-Time Analytics**:
- IoT data, logs, events
- Aggregation framework for analysis

**Not suitable for**:
- ❌ Complex multi-table joins
- ❌ Strict relational data
- ❌ Graph traversals (use Neo4j)

---

## Interview Tips

**Explain MongoDB in 2 minutes**:
"MongoDB is a document-oriented NoSQL database storing JSON-like BSON documents. Flexible schema allows different fields per document. Rich query language with operators, indexes, and aggregation framework. Replica sets provide high availability with automatic failover. Sharding enables horizontal scaling by distributing data across shards based on shard key. Supports embedded documents (denormalization) or references (normalization). Multi-document ACID transactions available. Best for flexible schemas, rapid development, hierarchical data. Not suitable for complex joins or strict relational data."

**Key concepts**:
- Document model (BSON)
- Embedding vs referencing
- Replica sets (primary/secondary)
- Sharding and shard key selection
- Aggregation framework
- Indexing strategies

**Common mistakes**:
- ❌ Treating like SQL database (joins, normalization)
- ❌ Poor shard key selection (hot shards)
- ❌ Not using indexes
- ❌ Documents too large (> 16 MB limit)

---

## Key Takeaways

🔑 Document-oriented NoSQL (JSON-like BSON)
🔑 Flexible schema: fields can vary per document
🔑 Embedding vs referencing: trade-off between duplication and queries
🔑 Replica sets: primary/secondary with automatic failover
🔑 Sharding: horizontal scaling with shard key
🔑 Rich query language: operators, indexes, aggregation
🔑 Indexes critical for performance
🔑 Multi-document transactions (replica set required)
🔑 Best for: flexible schemas, rapid development, hierarchical data
🔑 Not for: complex joins, strict relations, graph data
`,
};
