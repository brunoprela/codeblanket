import { ModuleSection } from '@/lib/types';

const elasticsearchArchitectureSection: ModuleSection = {
  id: 'elasticsearch-architecture',
  title: 'Elasticsearch Architecture',
  content: `
# Elasticsearch Architecture

## Introduction

Elasticsearch is the most popular open-source search and analytics engine, built on Apache Lucene. It powers search for companies like GitHub, Stack Overflow, Netflix, and Uber. Understanding Elasticsearch's distributed architecture is crucial for designing scalable search systems and performing well in system design interviews.

While we covered full-text search fundamentals in the previous section, this section focuses on how Elasticsearch distributes data and queries across multiple nodes, handles failures, and scales to petabytes of data with subsecond response times.

## What is Elasticsearch?

Elasticsearch is:
- **Distributed**: Runs on multiple servers (nodes) forming a cluster
- **RESTful**: All operations via HTTP/JSON APIs
- **Document-oriented**: Stores JSON documents, not rows
- **Schema-free**: Dynamic mapping, no rigid schema required
- **Real-time**: Documents searchable within seconds of indexing
- **Horizontally scalable**: Add nodes to scale capacity and performance

**Key use cases:**
- Full-text search (e-commerce, content sites)
- Log and event analytics (ELK Stack)
- Application performance monitoring
- Business intelligence and analytics
- Geospatial data analysis

## Core Concepts

### Document

The basic unit of data—a JSON object:

\`\`\`json
{
  "_index": "products",
  "_id": "123",
  "_source": {
    "title": "Elastic Coffee Mug",
    "price": 15.99,
    "description": "Elasticsearch branded mug",
    "category": "merchandise",
    "in_stock": true
  }
}
\`\`\`

Each document has:
- **_index**: Which index it belongs to
- **_id**: Unique identifier
- **_source**: The actual document data

### Index

A collection of documents with similar characteristics (like a database table, but more flexible):

\`\`\`
products index → Millions of product documents
logs-2024 index → Logs for 2024
users index → User account documents
\`\`\`

Indices are logical namespaces that:
- Group related documents
- Define mapping (schema) settings
- Configure analyzers and tokenizers
- Set replication and sharding parameters

### Type (Deprecated)

In older versions (< 7.0), each index could have multiple "types" (like tables in a database). This concept was removed because it caused confusion and performance issues. Now each index has a single implicit type.

## Distributed Architecture

Elasticsearch is inherently distributed. Let's understand how it splits data and work across multiple servers.

### Cluster

A **cluster** is one or more nodes (servers) working together:

\`\`\`
Production Cluster:
- Cluster name: "production-es-cluster"
- 15 nodes total
- Stores 10TB of data
- Handles 50,000 queries/second
\`\`\`

All nodes in a cluster share the same cluster name. Clusters are completely isolated from each other.

### Node

A **node** is a single Elasticsearch server instance. Nodes have different roles:

**1. Master Node**
- Manages cluster state
- Creates/deletes indices
- Tracks which nodes are in cluster
- Decides which shards go to which nodes
- Handles cluster-level operations

**Note**: Master node does NOT handle indexing or search—it's purely for coordination.

**2. Data Node**
- Stores data (shards)
- Executes search queries
- Performs CRUD operations
- Executes aggregations
- Most CPU and memory intensive

**3. Coordinating Node**
- Routes requests to appropriate data nodes
- Merges results from multiple data nodes
- Load balances client requests
- Doesn't store data or handle cluster state

**4. Ingest Node**
- Pre-processes documents before indexing
- Runs ingest pipelines (parsing, enrichment)
- Extracts fields, converts formats
- Like Logstash, but built into Elasticsearch

**5. Machine Learning Node** (X-Pack)
- Runs ML jobs
- Anomaly detection
- Forecasting

A single node can have multiple roles, but in production, role separation improves performance and stability:

\`\`\`
Small cluster (3 nodes): Each node does everything
Large cluster (50+ nodes):
  - 3 dedicated master nodes
  - 40 data nodes
  - 5 coordinating nodes
  - 2 ingest nodes
\`\`\`

### Shards

**Shards** are how Elasticsearch distributes an index across nodes. Each shard is a fully functional Lucene index.

**Why sharding?**
- **Horizontal scaling**: Split data across multiple nodes
- **Parallelization**: Execute queries in parallel across shards
- **Performance**: Smaller shards = faster operations

**Types of shards:**

**1. Primary Shards**
- The original shards containing your data
- Each document belongs to exactly one primary shard
- Number of primary shards is set at index creation (can't be changed without reindexing)
- Default: 1 primary shard per index (ES 7.x+)

**2. Replica Shards**
- Copies of primary shards
- Provide high availability (failover)
- Increase read throughput (replicas can serve queries)
- Number can be changed dynamically
- Default: 1 replica per primary shard

**Example Configuration:**

\`\`\`json
{
  "settings": {
    "number_of_shards": 5,        // 5 primary shards
    "number_of_replicas": 1        // 1 replica of each primary = 5 replica shards
  }
}
\`\`\`

**Total shards**: 5 primary + 5 replica = **10 shards**

### Shard Distribution

Elasticsearch automatically distributes shards across nodes:

\`\`\`
3-node cluster with 5 primary shards, 1 replica:

Node 1: [P0, P1, R2, R3]    // P = Primary, R = Replica
Node 2: [P2, P3, R0, R4]
Node 3: [P4, R1]

Rules:
- Primary and its replica never on same node
- Roughly even distribution
- Rebalances automatically when nodes join/leave
\`\`\`

**What happens if Node 1 fails?**
- R2 on Node 2 and R3 on Node 2 are promoted to primaries
- Cluster remains fully functional
- Yellow status until replicas are rebuilt
- No data loss!

## How Indexing Works

When you index a document, several steps occur:

### 1. Document Routing

Elasticsearch needs to determine which primary shard should store the document:

\`\`\`
shard_num = hash(_routing_field) % number_of_primary_shards
\`\`\`

By default, \`_routing_field\` is the document \`_id\`. This ensures:
- Same document always goes to same shard
- Even distribution (hash function)
- Can customize routing (e.g., route by user_id to co-locate user data)

**Example:**
\`\`\`
Document ID: "user123"
Hash: hash("user123") = 2847362
Shards: 5 primary shards
Shard: 2847362 % 5 = 2

→ Document stored in Primary Shard 2
\`\`\`

### 2. Primary Shard Indexing

Once the coordinating node determines the target shard:

1. **Request forwarded** to node containing the primary shard
2. **Document validated** (parsing, field types)
3. **Document indexed** into Lucene segment on primary shard
4. **Stored in transaction log** (for durability)
5. **Write replicated** to all replica shards (in parallel)
6. **Success returned** when primary and all replicas acknowledge

### 3. Refresh Interval

Documents aren't immediately searchable. Elasticsearch uses a **refresh interval** (default: 1 second):

- Documents written to in-memory buffer (very fast)
- Every 1 second, buffer flushed to new Lucene segment
- New segment opened for searching
- This provides "near-real-time" search

**Trade-off**: Faster refresh = more searchability, but more overhead and smaller segments

### 4. Flush and Transaction Log

To ensure durability:
- Every write goes to transaction log on disk
- In-memory segments flushed to disk periodically (called a "flush")
- Transaction log cleared after flush
- If node crashes, replay transaction log to recover un-flushed data

## How Search Works

Search is more complex because it involves multiple shards and nodes.

### Query Phase

1. **Client sends query** to any node (coordinating node)

2. **Broadcast to all shards**
   - Query sent to one copy of each shard (primary or replica)
   - All shards execute query independently

3. **Each shard returns top N results**
   - Shard returns document IDs and scores
   - NOT full documents (just metadata)
   - Sorted by score

4. **Coordinating node merges results**
   - Receives N results from each shard
   - Globally sorts all results by score
   - Determines top N across all shards

### Fetch Phase

5. **Fetch full documents**
   - Coordinating node identifies which shards have the final top N docs
   - Fetches full \`_source\` for those docs only
   - Returns to client

**Example**:
\`\`\`
Query: "elasticsearch"
Top 10 results requested
5 shards

Query Phase:
- Shard 0: Returns IDs and scores for its top 10 (50 total)
- Shard 1: Returns IDs and scores for its top 10
- Shard 2: Returns IDs and scores for its top 10
- Shard 3: Returns IDs and scores for its top 10
- Shard 4: Returns IDs and scores for its top 10

Merge: Coordinating node sorts all 50 results, picks global top 10

Fetch Phase:
- Fetch full documents for those 10 from their respective shards
- Return to client
\`\`\`

**Why two phases?**
- Transferring document IDs + scores is cheap
- Transferring full documents is expensive
- Only fetch full data for final results (minimizes network traffic)

### Query Performance Implications

**More shards = More parallelization BUT more overhead**
- Pro: Each shard searches smaller dataset faster
- Pro: Queries run in parallel across shards
- Con: More shards to coordinate and merge results from
- Con: Each shard has overhead (memory, file handles)

**Best practice**: 
- Shard size: 10-50GB per shard
- Not too many small shards (coordination overhead)
- Not too few large shards (can't parallelize)

## Mappings and Data Types

**Mapping** defines how documents and their fields are stored and indexed (similar to a schema).

### Dynamic Mapping

Elasticsearch automatically detects field types:

\`\`\`json
{
  "title": "Elasticsearch Guide",    // Detected as text
  "price": 39.99,                    // Detected as float
  "publish_date": "2024-01-15",      // Detected as date
  "in_stock": true,                  // Detected as boolean
  "tags": ["search", "elk"]          // Detected as text array
}
\`\`\`

Convenient for getting started, but not ideal for production (can guess wrong).

### Explicit Mapping

Define types explicitly for better control:

\`\`\`json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "standard",
        "fields": {
          "keyword": {
            "type": "keyword"      // For exact matching and sorting
          }
        }
      },
      "price": {
        "type": "scaled_float",
        "scaling_factor": 100      // Store as integer internally
      },
      "publish_date": {
        "type": "date",
        "format": "yyyy-MM-dd"
      },
      "tags": {
        "type": "keyword"          // Exact matching for tags
      },
      "description": {
        "type": "text",
        "analyzer": "english"      // English-specific stemming
      }
    }
  }
}
\`\`\`

### Common Field Types

**1. Text**: Full-text analyzed strings
- Analyzed (tokenized, stemmed, lowercased)
- Used for full-text search
- NOT used for sorting or aggregations

**2. Keyword**: Exact-value strings
- Not analyzed
- Used for exact matching, sorting, aggregations
- IDs, tags, categories, email addresses

**3. Numeric**: byte, short, integer, long, float, double, scaled_float
- For numerical operations and range queries

**4. Date**: Date and datetime
- Multiple format support
- Stored as milliseconds since epoch internally

**5. Boolean**: true/false

**6. Object**: Nested JSON objects
\`\`\`json
{
  "user": {
    "name": "John",
    "age": 30
  }
}
\`\`\`

**7. Nested**: Arrays of objects with independent querying
- Preserves object relationships
- More expensive than flattened objects

**8. Geo-point**: Latitude/longitude
- Geospatial queries (distance, bounding box)

**9. IP**: IPv4 and IPv6 addresses
- CIDR range queries

### Multi-Fields

Index same field in different ways:

\`\`\`json
"title": {
  "type": "text",                  // Full-text search
  "fields": {
    "keyword": {
      "type": "keyword"            // Exact matching
    },
    "suggest": {
      "type": "completion"         // Autocomplete
    }
  }
}
\`\`\`

Access as: \`title\` (analyzed), \`title.keyword\` (exact), \`title.suggest\` (autocomplete)

## Query DSL

Elasticsearch provides a powerful JSON-based query language.

### Match Query

Standard full-text query:

\`\`\`json
{
  "query": {
    "match": {
      "title": "elasticsearch guide"
    }
  }
}
\`\`\`

Analyzes query text ("elasticsearch guide" → ["elasticsearch", "guide"]), then finds documents matching either term (OR) with BM25 scoring.

### Match Phrase Query

Exact phrase:

\`\`\`json
{
  "query": {
    "match_phrase": {
      "title": "elasticsearch guide"
    }
  }
}
\`\`\`

Terms must appear in order and adjacent.

### Bool Query

Combine multiple queries:

\`\`\`json
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "elasticsearch" } }
      ],
      "filter": [
        { "term": { "in_stock": true } },
        { "range": { "price": { "lte": 50 } } }
      ],
      "should": [
        { "match": { "category": "books" } }
      ],
      "must_not": [
        { "term": { "format": "pdf" } }
      ]
    }
  }
}
\`\`\`

- **must**: AND (affects score)
- **filter**: AND (no scoring, cached, faster)
- **should**: OR (affects score, boosts matching docs)
- **must_not**: NOT (excludes docs)

### Range Query

\`\`\`json
{
  "query": {
    "range": {
      "price": {
        "gte": 10,
        "lte": 100
      }
    }
  }
}
\`\`\`

### Term Query

Exact match (no analysis):

\`\`\`json
{
  "query": {
    "term": {
      "category.keyword": "Electronics"
    }
  }
}
\`\`\`

**Warning**: Don't use term query on analyzed text fields—it won't work as expected!

## Aggregations

Aggregations are like SQL GROUP BY on steroids.

### Metric Aggregations

Calculate metrics:

\`\`\`json
{
  "aggs": {
    "avg_price": {
      "avg": { "field": "price" }
    },
    "max_price": {
      "max": { "field": "price" }
    },
    "total_revenue": {
      "sum": { "field": "revenue" }
    }
  }
}
\`\`\`

### Bucket Aggregations

Group documents:

\`\`\`json
{
  "aggs": {
    "products_by_category": {
      "terms": {
        "field": "category.keyword",
        "size": 10
      },
      "aggs": {
        "avg_price_per_category": {
          "avg": { "field": "price" }
        }
      }
    }
  }
}
\`\`\`

Returns top 10 categories with average price for each.

### Histogram Aggregations

Group by numeric/date ranges:

\`\`\`json
{
  "aggs": {
    "prices": {
      "histogram": {
        "field": "price",
        "interval": 10
      }
    }
  }
}
\`\`\`

Returns counts for price ranges: [0-10), [10-20), [20-30), etc.

## Scaling Elasticsearch

### Vertical Scaling

Increase node resources:
- More RAM = larger caches, more segments in memory
- More CPU = faster queries and indexing
- Faster disks (SSD) = faster segment reads

**Limits**: Can't scale single node infinitely

### Horizontal Scaling

Add more nodes:
- Split data across more shards
- Add replicas for read scaling
- Dedicated roles (master, data, coordinating)

**Example Scaling Strategy:**

**100GB, 1000 req/sec**: 3 nodes, 3 shards, 1 replica
**1TB, 10,000 req/sec**: 10 nodes, 10 shards, 2 replicas
**10TB, 100,000 req/sec**: 50 nodes, 30 shards, 3 replicas, dedicated coordinating nodes

### Shard Sizing Best Practices

**Shard size**: 10-50GB is ideal
- Too small = coordination overhead
- Too large = slow recovery, can't redistribute well

**Number of shards**: Consider future growth
- Can't change primary shard count without reindexing
- Start with \`expected_data_size / 30GB\`
- For 300GB expected: 10 primary shards

### Read Scaling

Increase read throughput:
1. Add more replica shards (reads distributed across all replicas)
2. Add more data nodes to host replicas
3. Add dedicated coordinating nodes for request handling
4. Enable query caching

### Write Scaling

Increase write throughput:
1. Add more primary shards (writes distributed)
2. Increase \`refresh_interval\` (less often = faster writes)
3. Increase \`bulk\` API batch sizes
4. Add more data nodes
5. Reduce replicas temporarily during bulk loading

## High Availability

Elasticsearch provides strong HA guarantees:

### Node Failure

- Primary shard fails → Replica promoted to primary
- Replica shard fails → Queries continue using primary
- Master node fails → New master elected within seconds

### Split-Brain Prevention

With 3 master-eligible nodes, require \`minimum_master_nodes = 2\`:
- Prevents split-brain (two separate clusters)
- Ensures cluster always has majority

Modern ES uses quorum-based voting automatically.

### Cluster States

- **Green**: All primary and replica shards active
- **Yellow**: All primaries active, some replicas missing
- **Red**: Some primaries missing (data loss possibility)

## Common Mistakes

### 1. Too Many Shards

Creating 100 shards for 10GB of data:
- Each shard has overhead (memory, file handles)
- Coordination overhead slows queries
- Rule: 10-50GB per shard

### 2. Not Using Replicas

Running with 0 replicas in production:
- No failover
- No read scaling
- Data loss if node fails
- Always use at least 1 replica

### 3. Using Text Fields for Aggregations

Trying to aggregate on analyzed text:
- Doesn't work as expected
- Use keyword fields for exact values

### 4. Not Planning Capacity

Starting with 1 shard, then realizing you need 10:
- Can't change primary shard count
- Must reindex entire dataset
- Plan for growth upfront

### 5. Ignoring Mapping

Relying on dynamic mapping:
- Can guess wrong types
- Can't change mapping for existing fields
- Define explicit mappings in production

## Best Practices

### 1. Plan Your Indices

- Use index-per-time-period for time-series data: \`logs-2024-01\`, \`logs-2024-02\`
- Allows easy deletion of old data
- Optimizes for recent data access patterns

### 2. Monitor Cluster Health

- Track cluster status (green/yellow/red)
- Monitor shard allocation
- Watch query latency (p50, p95, p99)
- Track index and search rates

### 3. Use Bulk API

For indexing multiple documents:
- Single document APIs are inefficient
- Bulk API batches documents (1000-5000 per batch)
- 10-100x faster throughput

### 4. Optimize Mappings

- Disable \`_source\` if not needed (saves 30-50% space)
- Use \`doc_values: false\` for fields not sorted/aggregated
- Use appropriate analyzers per field

### 5. Separate Workloads

In large clusters:
- Dedicated master nodes (stability)
- Dedicated data nodes by workload (hot/warm/cold)
- Dedicated coordinating nodes (protect data nodes)

## Interview Tips

When discussing Elasticsearch:

1. **Start with distributed architecture**: Cluster → Nodes → Indices → Shards
2. **Explain sharding**: Why it's needed, primary vs replica
3. **Cover the query process**: Two-phase (query + fetch)
4. **Discuss scaling**: Horizontal (shards, nodes), vertical (resources)
5. **Mention HA**: Replicas, failover, split-brain prevention
6. **Know the numbers**: 10-50GB per shard, 1 sec refresh interval
7. **Consider the use case**: Logs vs search vs analytics have different requirements

**Example question**: "Design a search system for an e-commerce site with 10 million products."

**Strong answer**: "I'd use Elasticsearch with 10 primary shards (assuming ~300GB of product data at 30GB per shard) and 2 replicas for HA and read scaling. Products would be indexed with explicit mappings: title and description as analyzed text for full-text search, category and brand as keywords for filtering and faceting, and price as numeric for range queries. I'd implement a multi-field strategy where title is indexed both as analyzed text (for search) and keyword (for exact matching and sorting). For scaling, I'd start with 6 data nodes to distribute the 30 total shards (10 primary + 20 replicas), 3 dedicated master nodes for cluster stability, and 2 coordinating nodes for load balancing queries. The search API would use bool queries combining full-text match on title/description with filters on category, price range, and in_stock status. Filters would be cached for performance. I'd implement autocomplete using edge n-grams on the product title. For relevance, I'd use BM25 scoring with field boosting (title 3x description) and function scores to incorporate business metrics like popularity and inventory status. Query performance would be monitored with slow query logs, and I'd aim for p95 latency under 100ms."

## Summary

Elasticsearch architecture is built on:
- **Distributed design**: Cluster of nodes, data split into shards
- **Replication**: Primary and replica shards for HA
- **RESTful API**: JSON over HTTP for all operations
- **Flexible mapping**: Dynamic or explicit schemas
- **Powerful queries**: Full-text, boolean, aggregations
- **Horizontal scalability**: Add nodes and shards to scale
- **Near-real-time**: Documents searchable in seconds

Understanding these fundamentals allows you to design scalable search systems and make informed trade-offs in system design interviews.
`,
};

export default elasticsearchArchitectureSection;
