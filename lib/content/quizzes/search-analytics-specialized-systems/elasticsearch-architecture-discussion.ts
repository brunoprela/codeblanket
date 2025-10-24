import { Quiz } from '@/lib/types';

const elasticsearchArchitectureDiscussionQuiz: Quiz = {
  id: 'elasticsearch-architecture-discussion',
  title: 'Elasticsearch Architecture - Discussion Questions',
  questions: [
    {
      id: 'es-arch-discussion-1',
      type: 'discussion',
      question:
        "You're designing an Elasticsearch cluster for a logging system that will ingest 50GB of logs per day and retain them for 30 days. The system needs to support both recent log searches (last 24 hours) and historical analysis. Users report that searches across multiple days are very slow. How would you architect your index strategy, shard configuration, and node topology to optimize for both recent and historical queries? Discuss the trade-offs and specific Elasticsearch features you would leverage.",
      sampleAnswer: `This scenario requires a **time-based index strategy** with **hot-warm-cold architecture** to balance performance and cost:

**Index Strategy: Time-Based Indices**

Instead of a single "logs" index, create daily indices:
\`\`\`
logs-2024-01-15  (50GB, today, most queried)
logs-2024-01-14  (50GB, yesterday)
logs-2024-01-13  (50GB)
...
logs-2023-12-16  (50GB, 30 days ago, rarely queried)
\`\`\`

**Benefits:**
- Easy deletion: Delete \`logs-2023-12-16\` after 30 days (no per-document deletion)
- Optimized querying: Recent searches hit fewer indices
- Different settings per time period
- Force merge old indices for better read performance

**Shard Configuration:**

For 50GB/day with 30-day retention = **1.5TB total**

Daily index (50GB):
- **Primary shards**: 2 shards (25GB each, within 10-50GB sweet spot)
- **Replicas**: 1 replica (for HA and read scaling)
- **Total per day**: 4 shards (2 primary + 2 replica)

**Total cluster**: 30 days × 4 shards = **120 shards** (manageable)

**Why not 1 shard/day?**
- 50GB is on the edge of recommended size
- 2 shards provides parallelization for queries
- Better distribution across nodes

**Why not 5 shards/day?**
- Would create 150 total shards
- Unnecessary overhead for 50GB
- Coordination cost increases

**Node Topology: Hot-Warm-Cold Architecture**

**Hot Nodes** (last 1-2 days, 100GB):
- 3 nodes with fast SSDs (NVMe)
- High CPU, high RAM (64GB+)
- Handles all incoming writes
- Most recent search queries (80% of traffic)
- Indices with 1 replica

**Warm Nodes** (days 3-7, 250GB):
- 5 nodes with regular SSDs
- Medium resources
- Read-only indices
- Force merged (optimized for reads)
- Indices with 1 replica

**Cold Nodes** (days 8-30, 1.15TB):
- 8 nodes with HDD storage (cheaper)
- Lower resources
- Rarely queried historical data
- Force merged and heavily compressed
- Replicas optional (could use 0 replicas, rely on snapshots)

**Index Lifecycle Management (ILM):**

\`\`\`
Phase: Hot (0-1 days)
- Allocated to hot nodes
- Active indexing
- 1 replica
- Priority: 100 (search priority)

Phase: Warm (2-7 days)
- Rollover to warm nodes
- Mark read-only
- Force merge to 1 segment
- Shrink to 1 shard per index (optional, for small queries)
- Priority: 50

Phase: Cold (8-30 days)
- Move to cold nodes
- Reduce to 0 replicas (rely on snapshots)
- Searchable snapshot (even cheaper storage)
- Priority: 0

Phase: Delete (30+ days)
- Delete index
\`\`\`

**Index Templates:**

\`\`\`json
{
  "index_patterns": ["logs-*"],
  "template": {
    "settings": {
      "number_of_shards": 2,
      "number_of_replicas": 1,
      "refresh_interval": "5s",  // Hot tier
      "routing.allocation.require.data": "hot"
    },
    "mappings": {
      "properties": {
        "timestamp": { "type": "date" },
        "level": { "type": "keyword" },
        "message": { "type": "text" },
        "service": { "type": "keyword" },
        "trace_id": { "type": "keyword" }
      }
    }
  }
}
\`\`\`

**Query Optimization:**

For "recent" queries (last 24 hours):
\`\`\`json
GET /logs-2024-01-15,logs-2024-01-14/_search
{
  "query": { ... }
}
\`\`\`
- Hits only 2 indices (4 shards)
- All on fast hot nodes
- Sub-second response

For "historical" queries (last 7 days):
\`\`\`json
GET /logs-2024-01-*/_search
{
  "query": { ... },
  "size": 100
}
\`\`\`
- Wildcard expands to 7 indices (14 shards)
- Distributed across hot/warm nodes
- Acceptable 1-3 second response

For "deep historical" (30 days):
- Query only when needed
- Use aggregations to reduce data
- Accept slower response (5-10 seconds)
- Consider async search for very large queries

**Why Searches Were Slow (Original Problem):**

1. **All data in one index**: Every query scanned entire 1.5TB
2. **Large shards**: If using few shards, each shard too large
3. **Wrong node types**: Historical data on same nodes as hot data
4. **No optimization**: Old data not force-merged

**Trade-offs:**

1. **Complexity vs Performance**: Time-based indices + ILM adds operational complexity but dramatically improves performance
   - Decision: Worth it for 30-day retention at scale

2. **Storage Redundancy vs Cost**: Hot data needs replicas, cold data could use snapshots only
   - Decision: 1 replica hot/warm, 0 replicas cold (backups instead)

3. **Shard Count vs Overhead**: 2 shards/day = 60 primary shards total
   - Decision: Acceptable, each shard ~25GB, good balance

4. **Hardware Mix vs Simplicity**: Hot/warm/cold nodes require different hardware
   - Decision: Cost savings (10x between NVMe and HDD) justify complexity

**Cost Estimation:**

- Hot: 3 nodes × \`i3.2xlarge\` (NVMe SSD) = $1,800/month
- Warm: 5 nodes × \`r5.xlarge\` (SSD) = $1,500/month
- Cold: 8 nodes × \`d2.xlarge\` (HDD) = $1,600/month
- **Total**: ~$5,000/month vs $15,000/month for all hot storage

**Result**: 80% of queries hit hot tier (sub-second), 15% hit warm tier (1-3 sec), 5% hit cold tier (5-10 sec), with 70% cost savings versus uniform infrastructure.`,
      keyPoints: [
        'Time-based indices enable easy deletion and optimized querying by time range',
        'Hot-warm-cold architecture balances performance and cost for time-series data',
        'Shard sizing: 10-50GB per shard, consider total shard count',
        'Index Lifecycle Management (ILM) automates transitions between tiers',
        'Force merge and compression optimize read performance for old data',
        'Recent data (hot) gets fast storage and replicas, old data (cold) gets cheap storage',
      ],
    },
    {
      id: 'es-arch-discussion-2',
      type: 'discussion',
      question:
        'During a routine deployment, your Elasticsearch cluster with 5 data nodes goes into "yellow" status. Node 3 becomes unresponsive, and you notice that the cluster is trying to reallocate 20 primary shards. Some queries are failing with "no available shards" errors. Walk through your diagnosis and remediation process. What cluster settings, shard allocation decisions, and recovery strategies would you employ to minimize downtime? Also discuss how you would prevent this scenario in the future.',
      sampleAnswer: `This is a critical production incident requiring systematic diagnosis and remediation:

**Immediate Diagnosis (First 5 minutes):**

**1. Check Cluster Health:**
\`\`\`bash
GET /_cluster/health

{
  "status": "yellow",
  "unassigned_shards": 20,
  "active_primary_shards": 80,
  "active_shards": 160,
  "relocating_shards": 0,
  "initializing_shards": 0,
  "number_of_nodes": 4,           // Was 5, now 4!
  "number_of_data_nodes": 4
}
\`\`\`

**Status: Yellow** = All primaries active, some replicas missing (not red = good, no data loss)

**2. Identify Missing Node:**
\`\`\`bash
GET /_cat/nodes?v

ip        name    node.role
10.0.1.1  node-1  data
10.0.1.2  node-2  data
10.0.1.4  node-4  data
10.0.1.5  node-5  data
# node-3 missing!
\`\`\`

**3. Check Unassigned Shards:**
\`\`\`bash
GET /_cat/shards?v&h=index,shard,prirep,state,node,unassigned.reason

logs-2024-01-15  0  r  UNASSIGNED  node-3-missing  NODE_LEFT
logs-2024-01-14  1  r  UNASSIGNED  node-3-missing  NODE_LEFT
...
\`\`\`

**Analysis**: 20 **replica** shards were on node-3. No primaries lost = no data loss!

**4. Why "No Available Shards" Errors?**

If queries specifically requested routing to node-3, or if:
- Custom routing was used pointing to node-3
- Preference parameter specified node-3
- Load balancer still sending to node-3

**Remediation Process:**

**Phase 1: Immediate Stabilization (0-10 minutes)**

**1. Update Load Balancer:**
\`\`\`
Remove node-3 from load balancer pool immediately
Ensures no new queries routed to failed node
\`\`\`

**2. Verify Cluster Can Recover:**
\`\`\`bash
GET /_cluster/allocation/explain
{
  "index": "logs-2024-01-15",
  "shard": 0,
  "primary": false
}
\`\`\`

Check why shards aren't reallocating:
- Disk space on other nodes?
- Shard allocation settings blocking?
- Concurrent recoveries limit?

**3. Check Disk Space:**
\`\`\`bash
GET /_cat/allocation?v

node-1  85%  // High but OK
node-2  92%  // CRITICAL - might trigger read-only
node-4  78%  // OK
node-5  81%  // OK
\`\`\`

If node-2 is near disk threshold (95%), it will refuse new shards!

**4. Adjust Recovery Settings (Temporary):**
\`\`\`json
PUT /_cluster/settings
{
  "transient": {
    "cluster.routing.allocation.node_concurrent_recoveries": "4",  // Default: 2
    "indices.recovery.max_bytes_per_sec": "100mb",                  // Default: 40mb
    "cluster.routing.allocation.cluster_concurrent_rebalance": "4"  // Default: 2
  }
}
\`\`\`

This speeds up recovery but increases load.

**Phase 2: Recovery (10-60 minutes)**

**1. Monitor Recovery Progress:**
\`\`\`bash
GET /_cat/recovery?v&h=index,shard,type,stage,percent,bytes,bytes_percent

logs-2024-01-15  0  replica  done       100%  25.3gb  100%
logs-2024-01-14  1  replica  index      45%   11.2gb  45%
logs-2024-01-13  2  replica  translog   92%   23.1gb  92%
\`\`\`

**2. If Recovery Is Slow:**

Check bottlenecks:
- Network bandwidth: \`GET /_nodes/stats/indices/recovery\`
- Disk I/O: System metrics
- CPU: High during indexing phase

**3. Handle Disk Space Issues (Node-2):**

If node-2 is blocking allocation due to disk:
\`\`\`json
// Option A: Temporarily increase threshold
PUT /_cluster/settings
{
  "transient": {
    "cluster.routing.allocation.disk.watermark.low": "95%",   // Default: 85%
    "cluster.routing.allocation.disk.watermark.high": "97%"   // Default: 90%
  }
}

// Option B: Exclude node-2 from new allocations
PUT /_cluster/settings
{
  "transient": {
    "cluster.routing.allocation.exclude._name": "node-2"
  }
}
\`\`\`

**Phase 3: Investigate Node-3 (Parallel with Recovery)**

**1. Check Node Logs:**
\`\`\`
/var/log/elasticsearch/cluster-name.log

[2024-01-15T10:32:15] ERROR: OutOfMemoryError: Java heap space
[2024-01-15T10:32:16] WARN: Circuit breakers triggered
[2024-01-15T10:32:17] FATAL: JVM crash
\`\`\`

**2. Common Failure Causes:**
- **OOM**: Heap too small, memory leak, heavy aggregations
- **Disk failure**: Hardware issue, read-only file system
- **Network**: Node couldn't communicate with master
- **GC**: Stop-the-world GC pauses exceeding timeout
- **Deployment**: Bad config, corrupted binary

**Phase 4: Restore Node-3 or Add Replacement (30-120 minutes)**

**Option A: Fix and Restart Node-3**
\`\`\`bash
# Fix issue (increase heap, fix config, etc.)
# Restart node
systemctl restart elasticsearch

# Node will rejoin cluster
# Some shards may move back to node-3
\`\`\`

**Option B: Add New Node-6 (Node-3 Unrecoverable)**
\`\`\`bash
# Provision new node with same config
# Will automatically join cluster
# Cluster rebalances shards
\`\`\`

**Phase 5: Return to Normal Operations**

**1. Reset Temporary Settings:**
\`\`\`json
PUT /_cluster/settings
{
  "transient": {
    "cluster.routing.allocation.node_concurrent_recoveries": null,
    "indices.recovery.max_bytes_per_sec": null,
    "cluster.routing.allocation.cluster_concurrent_rebalance": null,
    "cluster.routing.allocation.exclude._name": null
  }
}
\`\`\`

**2. Verify Cluster Health:**
\`\`\`bash
GET /_cluster/health

{
  "status": "green",  // All shards allocated!
  "number_of_nodes": 5
}
\`\`\`

**3. Monitor for 24 Hours:**
- Query latency
- Shard allocation
- Node CPU/memory/disk
- GC pause times

**Prevention Strategies:**

**1. Improve Monitoring:**
\`\`\`
Metrics to track:
- Node availability (alert if node down > 1 min)
- Cluster status (alert if yellow/red)
- JVM heap usage (alert if > 75%)
- GC pause times (alert if > 1 second)
- Disk space (alert if > 80%)
- Shard allocation (alert if unassigned > 5 min)
\`\`\`

**2. Cluster Configuration:**
\`\`\`json
{
  "cluster.routing.allocation.awareness.attributes": "rack",  // Spread replicas across racks
  "cluster.routing.allocation.node_initial_primaries_recoveries": 8,
  "cluster.routing.allocation.node_concurrent_recoveries": 2,
  "gateway.recover_after_nodes": 4,  // Don't start recovery until 4/5 nodes present
  "gateway.recover_after_time": "5m" // Wait 5 min for all nodes before recovery
}
\`\`\`

**3. Resource Management:**
- **Heap sizing**: 50% of RAM, max 31GB (compressed oops)
- **Disk capacity**: Plan for 70% utilization maximum
- **Replica count**: At least 1 replica for critical indices
- **Circuit breakers**: Properly configured to prevent OOM

**4. Deployment Best Practices:**
- **Rolling restart**: Never restart all nodes at once
- **Deploy to one node**: Validate before rolling out
- **Canary deployments**: Deploy to 1 node, wait, then roll out
- **Backup config**: Test config changes in staging
- **Drain node before maintenance**: Reallocate shards before shutdown

**5. Architecture Improvements:**
- **Dedicated master nodes**: 3 masters separate from data nodes
- **Coordinator nodes**: Separate query handling from data storage
- **Cross-cluster replication**: DR cluster in different region
- **Automated snapshots**: Hourly snapshots to S3

**Why Node Failure Didn't Cause Data Loss:**

- **Replica shards**: Each primary had 1 replica on different node
- **Primary shards intact**: All primaries were on nodes 1, 2, 4, 5
- **Yellow not red**: Yellow = replicas missing, red = primaries missing

If node-3 had contained primaries **without replicas**, status would be red and data loss would occur.

**Key Lesson**: This incident highlights why **replicas are non-negotiable** in production. The cluster recovered gracefully because replica shards protected against node failure.`,
      keyPoints: [
        'Yellow status means primaries are active but replicas are missing (no data loss)',
        'Check unassigned shard reasons: NODE_LEFT, ALLOCATION_FAILED, disk space',
        'Temporarily increase concurrent_recoveries to speed up shard reallocation',
        'Monitor disk space carefully; nodes near watermark limits will refuse allocations',
        'Replicas are critical: They prevent data loss and enable zero-downtime recovery',
        'Prevention: Monitor node health, use dedicated masters, automated deployments',
      ],
    },
    {
      id: 'es-arch-discussion-3',
      type: 'discussion',
      question:
        'Your Elasticsearch cluster currently has a single "products" index with 5 primary shards and 1 replica, storing 100GB of product data. The business is projecting 10x growth over the next year (1TB total). You\'re experiencing slow queries and high query latency during peak hours. The team is debating whether to: (A) reindex into an index with 50 primary shards, (B) split the index by category into multiple indices, or (C) add more replica shards and nodes. Analyze each approach, discuss the trade-offs, and provide your recommendation with justification.',
      sampleAnswer: `This is a critical architecture decision that requires analyzing query patterns, growth projections, and operational trade-offs:

**Current State Analysis:**

Current Setup:
- 1 index ("products")
- 5 primary shards (20GB each)
- 1 replica (5 replica shards)
- Total: 10 shards
- 100GB data
- Slow queries during peak

**Problems:**
1. Individual shards will grow to 200GB each (way above 50GB recommended max)
2. Query latency suggests resource contention or undersized cluster
3. Need to scale for 10x growth

**Option A: Reindex to 50 Primary Shards**

**Implementation:**
\`\`\`json
PUT /products_v2
{
  "settings": {
    "number_of_shards": 50,
    "number_of_replicas": 1
  }
}

POST /_reindex
{
  "source": { "index": "products" },
  "dest": { "index": "products_v2" }
}
\`\`\`

**Pros:**
- Handles 10x growth: 1TB / 50 shards = 20GB per shard (ideal)
- Better query parallelization across 50 shards
- Better distribution across nodes
- Single index maintains simple queries

**Cons:**
- **Cannot change shard count after creation** (locked in)
- 50 shards might be overkill now (coordination overhead)
- Every query hits all 50 shards (even for small result sets)
- Higher memory overhead (50 Lucene indices)
- Coordination cost: Merge results from 50 shards
- Downtime required for reindexing (unless using index alias)

**Cost Analysis:**
- Current: 10 shards
- New: 100 shards (50 primary + 50 replica)
- Memory per shard: ~30MB
- Total memory overhead: 3GB
- May need more nodes just for shard overhead

**Verdict**: ❌ **Overkill and inflexible**
- 50 shards is too many for current scale
- Over-sharding now hurts performance
- Better to over-shard slightly (10-15 shards), not 10x

**Option B: Split by Category into Multiple Indices**

**Implementation:**
\`\`\`json
PUT /products_electronics { "settings": { "number_of_shards": 3 } }
PUT /products_clothing { "settings": { "number_of_shards": 2 } }
PUT /products_books { "settings": { "number_of_shards": 2 } }
PUT /products_home { "settings": { "number_of_shards": 2 } }
PUT /products_other { "settings": { "number_of_shards": 1 } }
// Total: 10 primary shards, distributed by category
\`\`\`

**Pros:**
- **Query optimization**: Search within category hits fewer shards
  - "Search electronics" → only 3 shards (was 5)
  - "Search books" → only 2 shards (was 5)
- Independent shard sizing per category (electronics is bigger)
- Can optimize mappings per category
- Can apply different ILM policies
- Can scale categories independently
- Better cache utilization (category-specific caching)

**Cons:**
- **Cross-category queries**: "Search all products" hits all indices
  - \`GET /products_*/_search\` still hits all shards
- Application complexity: Route to correct index based on category
- More indices to manage (backups, settings, mappings)
- Inconsistent if product changes category (rare)
- Harder to do global aggregations

**Use Case Analysis:**

If queries are mostly category-specific (likely for e-commerce):
- 80% of queries: "Search within Electronics" → 3 shards (fast!)
- 15% of queries: "Search all products" → all shards (acceptable)
- 5% of queries: Cross-category aggregations (can optimize differently)

**Growth Handling:**
- Electronics grows to 400GB → Reindex to 10 shards (40GB each)
- Other categories stable → Keep current sharding
- Independent scaling = flexible

**Verdict**: ✅ **Best option if query patterns match**
- Requires understanding query distribution
- Huge performance win for category-specific queries
- Flexible scaling per category

**Option C: Add Replicas and Nodes**

**Implementation:**
\`\`\`json
PUT /products/_settings
{
  "number_of_replicas": 3  // Was 1, now 3
}
\`\`\`

Add nodes to distribute load:
- Current: 3 nodes
- New: 10 nodes (to hold 5 primary + 15 replica = 20 shards)

**Pros:**
- **No reindexing**: Change replicas dynamically
- Immediate improvement for read-heavy workload
- Higher availability (more copies)
- Distributes query load across more nodes
- Simple to implement (just add nodes and change setting)

**Cons:**
- **Doesn't solve shard size problem**: Still have 200GB shards at 10x growth
- 4x storage cost (3 replicas = 4 copies total)
- Only helps read performance, not write performance
- Writes go to all replicas (4x write load)
- Doesn't scale past node capacity (large shards limit)
- Expensive (400GB x 4 copies = 1.6TB, growing to 4TB)

**When This Works:**
- Read-heavy workload (95% reads, 5% writes)
- Temporary solution while planning long-term architecture
- Query latency from node resource contention, not shard size

**Verdict**: ⚠️ **Short-term fix, not long-term solution**
- Helps with read scaling now
- Doesn't address fundamental shard sizing issue
- Can combine with option B

**My Recommendation: Hybrid Approach (Option B + Moderate Sharding + Replicas)**

**Phase 1: Immediate (Weeks 1-2)**
\`\`\`json
// Add 1 more replica temporarily for read scaling
PUT /products/_settings
{
  "number_of_replicas": 2  // Was 1, now 2
}
// Add 3 more nodes to handle load
// Total: 6 nodes, 15 total shards (5 primary + 10 replica)
\`\`\`

**Cost**: Moderate (2x storage)
**Benefit**: Immediate query latency improvement

**Phase 2: Long-term Architecture (Weeks 3-8)**

Analyze query patterns:
\`\`\`sql
SELECT category, COUNT(*) as query_count
FROM query_logs
GROUP BY category
ORDER BY query_count DESC;

Results:
Electronics: 45%
Clothing: 25%
Home: 15%
Books: 10%
Other: 5%
\`\`\`

Create category-specific indices:
\`\`\`json
PUT /products_electronics {
  "settings": {
    "number_of_shards": 10,        // Largest category, needs parallelization
    "number_of_replicas": 2
  }
}

PUT /products_clothing {
  "settings": {
    "number_of_shards": 5,         // Second largest
    "number_of_replicas": 2
  }
}

PUT /products_home {
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 2
  }
}

PUT /products_books {
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 2
  }
}

PUT /products_other {
  "settings": {
    "number_of_shards": 2,
    "number_of_replicas": 2
  }
}

// Total: 23 primary shards
// At 1TB: 23 primary shards = ~43GB per shard (optimal!)
\`\`\`

**Migration Strategy:**
\`\`\`
1. Create new indices with category split
2. Use index alias for zero-downtime migration:
   - "products" alias → old index (reads)
   - Write to new indices based on category
3. Reindex historical data in background
4. Switch alias to new indices
5. Delete old index
\`\`\`

**Query Pattern Optimization:**

Category-specific search (45% of traffic):
\`\`\`json
GET /products_electronics/_search
{
  "query": { "match": { "title": "laptop" } }
}
// Hits only 10 shards instead of all 23 (2.3x faster coordination)
\`\`\`

Cross-category search (remaining 55%):
\`\`\`json
GET /products_*/_search
{
  "query": { "match": { "title": "gift" } }
}
// Hits all 23 shards (slightly more than current 5, but manageable)
// Each shard is smaller and faster
\`\`\`

**Cost-Benefit Analysis:**

**Option A (50 shards):**
- Cost: High (coordination overhead, memory)
- Performance: Poor now, good later
- Flexibility: Locked in
- **Score: 3/10**

**Option B (Category split):**
- Cost: Medium (more indices, but smaller total shard count)
- Performance: Excellent for category queries (most common)
- Flexibility: Can adjust per category
- **Score: 9/10** ✅

**Option C (More replicas):**
- Cost: High (4x storage)
- Performance: Good for reads, doesn't address growth
- Flexibility: Easy to implement
- **Score: 6/10**

**Final Recommendation:**

**Short term**: Add 1 replica + nodes (Option C) for immediate relief

**Long term**: Split by category (Option B) with appropriate shard counts per category

**Rationale**:
- Most e-commerce queries are category-specific
- Independent scaling per category
- Right-sized shards (40-50GB) for 1TB growth
- Flexible to adjust as data grows unevenly
- Better cache utilization and query performance

**Growth Path**:
- Year 1: 1TB across 23 shards (43GB each)
- Year 2: 3TB → Reindex electronics to 20 shards
- Year 3: 5TB → Split largest categories further

This approach balances immediate needs (more replicas) with long-term scalability (category-based indices) while maintaining query performance and operational flexibility.`,
      keyPoints: [
        'Over-sharding (50 shards for 100GB) causes coordination overhead and wastes resources',
        'Category-based indices optimize performance when most queries are category-specific',
        "Adding replicas helps read scaling but doesn't solve fundamental shard size issues",
        "Ideal shard size: 10-50GB; plan for growth but don't over-optimize for distant future",
        'Hybrid approach: immediate relief (replicas) + long-term architecture (split by category)',
        'Analyze query patterns before architectural decisions; 80% of queries should drive design',
      ],
    },
  ],
};

export default elasticsearchArchitectureDiscussionQuiz;
