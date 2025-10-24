/**
 * Quiz questions for Apache Cassandra section
 */

export const cassandraQuiz = [
  {
    id: 'q1',
    question:
      "Explain Cassandra's tunable consistency. How do you achieve strong consistency, and what are the trade-offs?",
    sampleAnswer:
      "Cassandra provides tunable consistency via read and write consistency levels. Strong consistency achieved when: Write CL + Read CL > RF. Example (RF=3): QUORUM write + QUORUM read = strong consistency (2 write + 2 read > 3 guarantees overlap). Or ONE write + ALL read. Or ALL write + ONE read. Process: Write QUORUM writes to 2 replicas → later read QUORUM reads from 2 replicas → at least 1 replica has latest data. Trade-offs: (1) Latency - QUORUM slower than ONE (must wait for multiple nodes). (2) Availability - QUORUM requires majority available (RF=3, 2 must be up). ALL requires all nodes (brittle!). (3) Performance - higher CL = more network round trips. Best practices: Use LOCAL_QUORUM (quorum in local DC) for most operations. Avoid ALL (one node down = failure). Use ONE for non-critical reads. Cassandra's power is flexibility - choose consistency vs availability vs latency per operation!",
    keyPoints: [
      'Strong consistency: Write CL + Read CL > RF',
      'Example: QUORUM write + QUORUM read (RF=3)',
      'Trade-off: Consistency vs latency vs availability',
      'LOCAL_QUORUM recommended for most use cases',
      'Flexibility: Choose per-operation consistency',
    ],
  },
  {
    id: 'q2',
    question:
      'Why does Cassandra use a masterless architecture instead of master-slave? What are the benefits and challenges?',
    sampleAnswer:
      "Cassandra is peer-to-peer with no master. All nodes are equal, any node can handle any request. Benefits: (1) No single point of failure - losing any node doesn't break cluster. Master-slave: master dies = downtime until failover. (2) Linear scalability - add nodes without master bottleneck. (3) High availability - no master election, no failover delay. (4) Multi-datacenter friendly - no master to replicate across DCs. (5) Simple operations - no special node to manage. Challenges: (1) Eventual consistency - no master to serialize operations. Must use consensus (Paxos) for strong consistency (expensive). (2) Complex coordination - gossip protocol for cluster state, read repair for consistency. (3) No global ordering - timestamps used, subject to clock skew. (4) More moving parts - understanding failures harder. (5) Learning curve - no central control. Trade-off: Availability + scalability vs operational simplicity + strong consistency. Cassandra chooses availability. For strong consistency, use systems with master (Spanner, CockroachDB).",
    keyPoints: [
      'All nodes equal, no master, peer-to-peer',
      'Benefits: No SPOF, linear scalability, high availability',
      'Challenges: Eventual consistency, complex coordination',
      'Trade-off: Availability vs strong consistency',
      'Use Cassandra for: High availability, multi-DC',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain partition key design in Cassandra. What makes a good vs bad partition key?',
    sampleAnswer:
      "Partition key determines data distribution via consistent hashing. hash(partition_key) → token → node(s). Critical for performance! Bad keys: (1) Sequential (timestamp, auto-increment) - all writes to latest partition (hot spot). (2) Low cardinality (status='active') - few partitions, no distribution. (3) Highly skewed (popular user_id) - hot partitions. Good keys: (1) High cardinality - UUIDs, user_ids (millions of unique values). (2) Even distribution - random or hash-based. (3) Aligns with query patterns - query by user_id? Use user_id as partition key. Example: Time-series data. Bad: partition_key=date (all today's data on same node). Good: partition_key=(sensor_id, date) (distribute by sensor). Or use bucketing: partition_key=(user_id, bucket) where bucket = timestamp % 100. Guidelines: (1) Partition size < 100 MB. (2) Partition key in WHERE clause (avoid full table scan). (3) Consider access patterns. (4) Use composite partition key for bucketing. Cassandra's performance depends on good partition key design!",
    keyPoints: [
      'Partition key determines data distribution (consistent hashing)',
      'Bad: Sequential, low cardinality, skewed',
      'Good: High cardinality, even distribution, query-aligned',
      'Keep partitions < 100 MB',
      'Use bucketing for unbounded partitions',
    ],
  },
];
