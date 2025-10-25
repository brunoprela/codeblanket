/**
 * Quiz questions for Design Facebook Messenger section
 */

export const facebookMessengerQuiz = [
  {
    id: 'q1',
    question:
      'Messenger needs to maintain 100 million concurrent WebSocket connections. Explain your architecture for connection management: how you partition connections across gateway servers, route messages to the correct gateway, and handle gateway failures.',
    sampleAnswer:
      "WEBSOCKET GATEWAY ARCHITECTURE: (1) PARTITIONING: Deploy 1,000 gateway servers, each handling 100K connections (WebSocket limit per server). Partition connections by user_id: gateway = hash (user_id) % 1000. User always connects to same gateway (sticky routing via load balancer). (2) CONNECTION STATE: Each gateway maintains in-memory map: Map<user_id, WebSocket[]> (users can have multiple devices). On connect: Store WebSocket connection, register in Redis: SADD user:123:gateways {gateway_id}. On disconnect: Remove from map, update Redis. (3) MESSAGE ROUTING: Message Service needs to deliver to Bob (user_id=789): Query Redis: SMEMBERS user:789:gateways → [gateway-5, gateway-12]. Send RPC/gRPC to gateway-5 and gateway-12 with message. Gateways push to Bob\'s WebSocket connections. (4) GATEWAY FAILURE: If gateway-5 crashes (100K connections lost): Clients detect disconnect (heartbeat timeout), reconnect via load balancer to different gateway, fetch missed messages from offline inbox (Redis), Redis automatically cleans up stale gateway entries (TTL). (5) LOAD BALANCER: Use consistent hashing with virtual nodes for even distribution. Health checks detect failed gateways, remove from rotation. New connections redistributed. (6) SCALABILITY: Horizontal: Add more gateway servers, update NUM_GATEWAYS. Vertical: Use high-memory instances (connections are memory-bound: 5KB per connection × 100K = 500 MB). KEY INSIGHT: Gateway is stateful (connections), but Message Service is stateless (easier to scale). Separate concerns - gateways handle TCP connections, message service handles business logic.",
    keyPoints: [
      'Partition by user_id hash: 1,000 gateways × 100K connections each',
      'Redis tracks user_id → gateway mapping for message routing',
      'Gateway crash: Clients reconnect, fetch missed messages from inbox',
      'Stateful gateway (connections) vs stateless message service',
      'Consistent hashing for balanced distribution',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare synchronous direct delivery vs Kafka queue for message delivery in Messenger. Why does Facebook use Kafka despite the added latency?',
    sampleAnswer:
      'SYNCHRONOUS DIRECT DELIVERY: Alice sends message → Message Service writes to DB → Immediately pushes to Bob\'s gateway → Bob receives. PROS: Lower latency (50-100ms total), simpler architecture. CONS: (1) If Bob\'s gateway is down/unreachable, delivery fails and message is lost (unless retry logic added). (2) If Bob offline, sender must wait for timeout. (3) Tight coupling: Sender blocked until delivery completes. (4) Fanout for groups: Sender waits for ALL recipients (100 users = 100 sequential/parallel deliveries). KAFKA QUEUE APPROACH: Alice sends message → Message Service writes to DB → Publishes to Kafka → Returns ACK to Alice (100ms). Kafka → Delivery Workers → Deliver to Bob (async). PROS: (1) Reliability: Message persisted in Kafka (durable), guaranteed delivery even if recipient gateway crashes. (2) Decoupling: Sender not blocked on delivery. Alice sees "sent" immediately. (3) Fanout efficiency: Publish once to Kafka, 100 workers consume in parallel for group messages. (4) Retry logic: Kafka retries automatically if delivery fails. (5) Observability: Kafka lag metrics show delivery backlog. CONS: Added latency: 50-100ms for Kafka roundtrip. Total: 150-200ms vs 50-100ms direct. PRODUCTION CHOICE: Kafka because reliability > latency for messaging. Users prefer message never lost over saving 50ms. For critical messages (payments), durability is non-negotiable. HYBRID: Use direct delivery for online users (fast), Kafka for offline/uncertain cases (reliable). Best of both worlds. KEY INSIGHT: Messaging is write-heavy and must be reliable. Async queue (Kafka) is industry standard for decoupling + durability, despite latency cost.',
    keyPoints: [
      'Direct delivery: Faster (50-100ms) but risky (message loss if recipient unavailable)',
      'Kafka: Reliable (durable queue), decouples sender from delivery',
      'Kafka adds 50-100ms latency but guarantees delivery with retries',
      'Group fanout: Kafka scales to 100s of recipients in parallel',
      'Production choice: Reliability > latency for messaging',
    ],
  },
  {
    id: 'q3',
    question:
      "Why does Messenger use Cassandra instead of MySQL for storing billions of messages? Walk through the query patterns and explain Cassandra\'s advantages.",
    sampleAnswer:
      'MESSAGE STORAGE REQUIREMENTS: (1) Write-heavy: 100B messages/day = 1.16M writes/sec. MySQL primary cannot handle this (maxes at ~10-50K writes/sec). (2) Time-series: Messages ordered by timestamp. Query pattern: "Get last 50 messages in conversation". (3) Scale: 36.5 PB over 5 years. Single MySQL server maxes at ~10 TB. Requires complex sharding. (4) High availability: Multi-region replication for global access. CASSANDRA ADVANTAGES: (1) WRITE OPTIMIZED: LSM-tree architecture, writes to memory (memtable) then flushes to disk sequentially. Handles millions of writes/sec per cluster. No write locks (unlike MySQL row-level locks). (2) SCHEMA DESIGN: PRIMARY KEY (conversation_id, message_id). Partition key: conversation_id → All messages for conversation co-located on same node (query efficiency). Clustering key: message_id (TIMEUUID) → Sorted by time within partition. (3) QUERY EFFICIENCY: SELECT * FROM messages WHERE conversation_id = X AND message_id < Y LIMIT 50. Single partition query (no joins, no scatter-gather). O(log N) lookup within partition. (4) HORIZONTAL SCALING: Add nodes to cluster, data automatically rebalanced. Linear scalability: 10 nodes → 100 nodes = 10x capacity. (5) MULTI-DC REPLICATION: Cassandra natively supports multi-datacenter. Messages replicated to US, EU, Asia datacenters with tunable consistency. (6) NO SINGLE POINT OF FAILURE: Peer-to-peer architecture (no master), any node can accept writes. MYSQL PROBLEMS AT THIS SCALE: (1) Write bottleneck: Single primary handles all writes. (2) Sharding complexity: Manual sharding by conversation_id, cross-shard queries impossible. (3) Replication lag: Replicas can lag minutes behind at high write load. (4) Storage limits: Requires expensive scale-up hardware. KEY INSIGHT: For write-heavy time-series workloads at petabyte scale, NoSQL (Cassandra/DynamoDB) is standard. SQL optimized for ACID transactions and complex joins (not needed for simple message storage).',
    keyPoints: [
      'Cassandra handles 1M+ writes/sec (MySQL maxes at 10-50K)',
      'LSM-tree: No write locks, sequential disk writes',
      'Partition by conversation_id: All messages co-located for fast queries',
      'Horizontal scaling: Add nodes for linear capacity increase',
      'Multi-DC replication built-in for global availability',
    ],
  },
];
