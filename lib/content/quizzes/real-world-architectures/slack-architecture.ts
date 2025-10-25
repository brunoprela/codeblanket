/**
 * Quiz questions for Slack Architecture section
 */

export const slackarchitectureQuiz = [
  {
    id: 'q1',
    question:
      "Explain Slack\'s real-time messaging architecture using WebSockets. How does it handle message delivery, ordering, and offline clients?",
    sampleAnswer:
      "Slack real-time messaging: (1) Client connects to Gateway service via WebSocket. (2) Gateway maintains persistent connection (heartbeat every 30s). (3) User sends message → Client sends to Gateway → Gateway validates + publishes to Kafka topic (channel_id). (4) Kafka consumers (per channel) process messages, write to database (MySQL), update search index (Elasticsearch). (5) Gateway servers subscribed to channels propagate message to connected clients in that channel. (6) Clients receive message, display immediately. Ordering: Kafka partitions by channel_id ensure messages for channel processed in order. Each message gets timestamp + sequence number. Offline delivery: (1) Gateway stores message in user's message queue (Redis). (2) When user reconnects, Gateway sends queued messages. (3) Client ACKs received messages. (4) Client stores messages in local DB (IndexedDB) for offline viewing. Scaling: 1000s of Gateway servers, each handles 10,000s of connections. Kafka provides durability and scalability.",
    keyPoints: [
      'WebSocket connections to Gateway service (persistent, heartbeat 30s)',
      'Kafka for message bus: Partition by channel_id, ensures ordering',
      'MySQL for storage, Elasticsearch for search, Redis for queues',
      'Offline: Queue messages in Redis, deliver on reconnect, client local DB',
    ],
  },
  {
    id: 'q2',
    question:
      'How does Slack search work across messages, files, and channels? What makes searching billions of messages fast?',
    sampleAnswer:
      'Slack uses Elasticsearch for full-text search. Indexing: (1) User sends message → Stored in MySQL. (2) Published to Kafka. (3) Search indexer (consumer) reads from Kafka, indexes in Elasticsearch. (4) Message searchable within 1-2 seconds. Index structure: (1) One index per workspace (isolation). (2) Documents: messages, files, channels. (3) Fields: content (text), user_id, channel_id, timestamp, attachments. (4) Sharded by workspace_id (large workspaces = multiple shards). Search query: (1) User searches "design mockups". (2) Query Elasticsearch index for user\'s workspace. (3) Filter by accessible channels (user permissions). (4) Rank by relevance (BM25) + recency boost + pinned messages boost. (5) Return top 100 results. Optimizations: (1) Auto-complete - prefix search on Elasticsearch. (2) Filters - search in specific channels, from specific users, date range. (3) Caching - popular searches cached in Redis. Scale: Billions of messages indexed, millions of searches daily, <200ms p95 latency.',
    keyPoints: [
      'Elasticsearch with one index per workspace, sharded by workspace size',
      'Real-time indexing: Message → MySQL → Kafka → Elasticsearch (1-2s)',
      'Ranking: BM25 relevance + recency + pinned boost, permission filtering',
      'Optimizations: Auto-complete, filters (channel/user/date), Redis caching',
    ],
  },
  {
    id: 'q3',
    question:
      "Describe Slack\'s database sharding strategy. How did they evolve from a monolithic database to a sharded architecture?",
    sampleAnswer:
      "Slack started with one MySQL database (2013-2015). As grew to millions of users, single DB became bottleneck: write throughput limited, read load high, schema migrations slow. Sharding strategy (2015-present): Shard by workspace_id (tenant-based sharding). Each workspace = independent unit. Benefits: (1) Isolation - one workspace's load doesn't affect others. (2) Scaling - add shards as workspace count grows. (3) Queries - most queries are within workspace (no cross-shard joins). Implementation: (1) Shard routing service - maps workspace_id → shard. (2) Application queries routing service before DB query. (3) Vitess (MySQL sharding framework) manages shard topology. (4) Large workspaces (>10,000 users) get dedicated shards. Migration: (1) Dual writes to old DB + new shards. (2) Backfill historical data to shards. (3) Shift reads to shards. (4) Deprecate old DB. Result: 1000s of MySQL shards, scaling to millions of workspaces.",
    keyPoints: [
      'Shard by workspace_id (tenant-based), each workspace isolated',
      'Vitess for MySQL sharding, routing service maps workspace → shard',
      'Large workspaces get dedicated shards, small workspaces share shards',
      'Migration: Dual writes, backfill, gradual read shift, deprecate monolith',
    ],
  },
];
