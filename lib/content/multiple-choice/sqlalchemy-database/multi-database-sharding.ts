import { MultipleChoiceQuestion } from '@/lib/types';

export const multiDatabaseShardingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-multi-db-mc-1',
    question: 'What is the main benefit of read replicas?',
    options: [
      'Faster writes',
      'Scale read queries by distributing load across multiple databases',
      'Reduce storage costs',
      'Improve security',
    ],
    correctAnswer: 1,
    explanation:
      'Read replicas scale read queries by distributing load across multiple databases. Master handles writes (limited by single DB capacity). Replicas handle reads (add more replicas = more read capacity). Typical setup: 1 master + 2-3 replicas = 5x read throughput. Async replication: master→replicas (lag ~100ms). Use case: 90% of queries are reads - massive scaling benefit. Reads go to replicas, writes to master. Does not improve write performance or reduce storage (replicas duplicate data).',
  },
  {
    id: 'sql-multi-db-mc-2',
    question: 'What is hash-based sharding?',
    options: [
      'Encrypting data across shards',
      'Distributing data across shards using hash of shard key (e.g., user_id % num_shards)',
      'Storing passwords in shards',
      'Geographic distribution',
    ],
    correctAnswer: 1,
    explanation:
      'Hash-based sharding: Distribute data using hash of shard key. Example: shard_index = user_id % num_shards. User 5 → shard 1 (5 % 4), User 8 → shard 0 (8 % 4). Benefits: Even distribution (no hotspots), deterministic (same user always same shard). Trade-offs: Hard to rebalance (changing num_shards requires migration), range queries across shards. Alternative: Range-based sharding (0-10M → shard 0) - easier rebalancing but potential hotspots. Use hash for even distribution, range for ease of rebalancing.',
  },
  {
    id: 'sql-multi-db-mc-3',
    question: 'What is replication lag?',
    options: [
      'Slow queries',
      'Delay between write to master and replication to replica (~100ms)',
      'Network latency',
      'Shard imbalance',
    ],
    correctAnswer: 1,
    explanation:
      "Replication lag: Delay between write to master and replication to replica. Typical: 50-200ms (async replication). Problem: User writes to master, immediately reads from replica - data not yet replicated, user sees stale data. Solutions: (1) Read-your-writes - after write, read from master (not replica). (2) Sticky session - route user's reads to master for 1 second after write. (3) Check replication lag, if < 100ms use replica. Monitoring: pg_stat_replication shows lag per replica. Alert if lag > 1 second.",
  },
  {
    id: 'sql-multi-db-mc-4',
    question: 'What is the purpose of two-phase commit (2PC)?',
    options: [
      'Faster commits',
      'Atomic transactions across multiple databases - all commit or all rollback',
      'Reduce network calls',
      'Improve security',
    ],
    correctAnswer: 1,
    explanation:
      'Two-phase commit (2PC): Atomic transactions across multiple databases. Phase 1 - prepare all transactions (reserve changes). Phase 2 - commit all or rollback all. Example: Transfer credits from User A (shard 0) to User B (shard 1). Prepare debit on shard 0, prepare credit on shard 1. If both succeed, commit both. If either fails, rollback both. Guarantees atomicity across shards. Trade-offs: Slow (2 round trips), blocking (locks held during prepare), coordinator failure risk. Alternative: Eventual consistency for non-critical operations.',
  },
  {
    id: 'sql-multi-db-mc-5',
    question: 'When should you use sharding?',
    options: [
      'Always',
      'Never',
      'When single database cannot handle load (10M+ users or high write volume)',
      'Only for small applications',
    ],
    correctAnswer: 2,
    explanation:
      'Sharding decision: < 1M users - single database (simplest). 1M-10M users - read replicas + vertical scaling + caching. 10M+ users - horizontal sharding (single DB limits reached). Also: High write volume (> 10K writes/sec) - shard to distribute write load. Geographic: Region-based sharding for latency. Multi-tenant: Tenant-based sharding for isolation. Trade-offs: Sharding adds complexity (cross-shard queries, distributed transactions, more infra). Only shard when necessary. Start simple, scale when needed.',
  },
];
