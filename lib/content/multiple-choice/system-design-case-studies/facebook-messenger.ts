/**
 * Multiple choice questions for Design Facebook Messenger section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const facebookmessengerMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Messenger maintains 100 million concurrent WebSocket connections. Each connection uses approximately 5 KB of memory. How much total memory is required just for connection state?',
    options: ['500 MB', '5 GB', '50 GB', '500 GB'],
    correctAnswer: 3,
    explanation:
      '100 million connections × 5 KB = 500,000 MB = 500 GB. This is distributed across ~1,000 gateway servers (500 MB per server). This demonstrates why WebSocket gateways require high-memory instances and why connection management is a significant operational challenge at scale. Each connection stores: user_id, WebSocket object, authentication state, heartbeat timers, etc.',
  },
  {
    id: 'mc2',
    question:
      'Alice sends a message to a group with 100 members. Using Kafka for delivery, what is the fanout strategy?',
    options: [
      'Publish 1 message to Kafka, 1 worker delivers to all 100 members sequentially',
      'Publish 100 messages to Kafka (one per recipient), workers consume in parallel',
      'Store message once, members pull when they connect',
      'Broadcast to all gateways, gateways filter by group membership',
    ],
    correctAnswer: 1,
    explanation:
      'Publish 100 events to Kafka (one per recipient_id). Kafka partitions distribute these across multiple partitions. 100 delivery workers (or fewer with batching) consume in parallel. Each worker: checks recipient online status, delivers via WebSocket or queues for offline user. BENEFITS: Parallelism (100 recipients handled simultaneously), fault tolerance (Kafka retries failures), observability (track per-recipient delivery). This is fanout-on-write pattern for groups. Alternative (pull) would require members to constantly poll (inefficient).',
  },
  {
    id: 'mc3',
    question:
      'Why does Messenger use TIMEUUID (timestamp + UUID) instead of auto-increment ID for message_id?',
    options: [
      'UUIDs are more secure',
      'TIMEUUID provides total ordering across distributed nodes without coordination',
      'Auto-increment is too slow',
      'TIMEUUID uses less storage',
    ],
    correctAnswer: 1,
    explanation:
      'In distributed system (Cassandra cluster), auto-increment requires coordination (single source of truth), creating bottleneck. TIMEUUID = timestamp (milliseconds) + UUID (random). Each node generates TIMEUUIDs independently without coordination. Messages naturally ordered by timestamp, ties broken by UUID. Can sort messages across all nodes/datacenters. Example: Node 1 generates: 2024-10-24-10:00:00-abc123. Node 2 generates: 2024-10-24-10:00:01-def456. Sorted correctly: abc123 before def456. This enables horizontal writes without single point of failure.',
  },
  {
    id: 'mc4',
    question:
      'A gateway server with 100K connections crashes. What happens to those users?',
    options: [
      'They permanently lose connection and must reinstall app',
      'Messages sent to them are lost forever',
      'Clients detect disconnect, reconnect to different gateway, fetch missed messages from inbox',
      'The gateway automatically restarts and reconnects all users',
    ],
    correctAnswer: 2,
    explanation:
      'CLIENT RESILIENCE: (1) Client detects disconnect (heartbeat timeout, typically 30-60 seconds). (2) Client initiates reconnection to load balancer. (3) Load balancer routes to healthy gateway (consistent hashing may route to different server). (4) During disconnect, messages queued in Redis offline inbox: LPUSH inbox:user_123 {message}. (5) On reconnect, client fetches: GET /inbox → Drains all messages. (6) Gateway failure is transparent to user (brief reconnection delay). This is why reliability layer (Kafka, offline inbox) is critical - gateway crashes are common in distributed systems.',
  },
  {
    id: 'mc5',
    question:
      "Messenger stores 100 billion messages per day in Cassandra. Why is Cassandra's LSM-tree architecture better than MySQL's B-tree for this workload?",
    options: [
      'LSM-tree uses less storage than B-tree',
      'LSM-tree has faster reads than B-tree',
      'LSM-tree buffers writes in memory and flushes sequentially to disk (no random writes), ideal for write-heavy workloads',
      'LSM-tree supports SQL queries',
    ],
    correctAnswer: 2,
    explanation:
      'LSM-TREE (Log-Structured Merge-Tree): Writes go to in-memory memtable (extremely fast). When memtable full, flush to disk as immutable SSTable (sequential write, fast). Background compaction merges SSTables. Result: Write throughput of millions/sec, no locks, append-only. B-TREE (MySQL InnoDB): Writes require finding correct page on disk (random I/O), modifying page (read-modify-write), row-level locks (concurrency bottleneck). Result: Write throughput limited to 10-50K/sec per server. MESSAGING WORKLOAD: 1.16M writes/sec, mostly append-only (no updates), time-ordered. LSM-tree perfect fit. Reads are sequential (recent messages), cached in memtable/block cache. KEY INSIGHT: Write amplification (1 write → multiple disk writes in LSM) acceptable when write throughput is bottleneck.',
  },
];
