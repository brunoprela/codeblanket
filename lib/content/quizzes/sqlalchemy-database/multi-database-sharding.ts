import { MultipleChoiceQuestion } from '@/lib/types';

export const multiDatabaseShardingQuiz = [
  {
    id: 'sql-multi-db-q-1',
    question:
      'Design a complete sharding strategy for a social media application with 50M users. Address: (1) sharding key selection, (2) shard routing implementation, (3) cross-shard queries (e.g., timeline), (4) handling replication lag, (5) migration strategy. Include complete code examples and trade-offs.',
    sampleAnswer:
      `Sharding strategy for 50M users: (1) Sharding key: user_id (immutable, evenly distributed). ` +
      `Hash-based: shard_index = user_id % num_shards. ` +
      `Alternative: Range-based (0-10M shard_0, 10M-20M shard_1) - easier rebalancing but hotspots possible. ` +
      `Choose hash for even distribution. ` +
      `(2) Shard routing: class ShardedDatabase: get_shard(user_id) returns engine for shard. ` +
      `ShardedSession(shard_key=user_id) routes all queries to correct shard. ` +
      `Example: user, posts, likes all use user_id as shard key - co-located on same shard. ` +
      `(3) Cross-shard queries: Timeline requires posts from multiple users (multiple shards). ` +
      `Solution: Fan-out query - query each shard in parallel, merge and sort results client-side. ` +
      `Use thread pool for parallel queries. ` +
      `Trade-off: slower than single-shard query, but required for global views. ` +
      `Alternative: Denormalize timeline data into user's shard. ` +
      `(4) Replication lag: Master-replica per shard. ` +
      `Handle lag: Read-your-writes pattern - after write, read from master not replica. ` +
      `For timeline: eventual consistency acceptable, use replicas. For user profile updates: use master. ` +
      `(5) Migration: Run Alembic on all shards in parallel. Script: migrate_all_shards(shard_urls). ` +
      `Test on one shard first, then rollout. Monitoring: track migration status per shard. ` +
      `Rollback plan: downgrade per shard. (6) Adding shards: Start with 10 shards (5M users each). ` +
      `When shard hits 8M users, add more shards and rebalance. ` +
      `Gradual migration: shadow writes to new shards, verify, switch reads. ` +
      `(7) Monitoring: Track shard usage, connection pool per shard, query latency per shard. ` +
      `Alert on imbalanced shards. ` +
      `Result: 50M users distributed across 10 shards, 5M per shard, linear scaling, <100ms query latency.`,
    keyPoints: [
      'Shard key: user_id (immutable, hash-based), co-locate related data',
      'Routing: ShardedSession(shard_key), all queries to one shard',
      'Cross-shard: Fan-out parallel queries, merge client-side, or denormalize',
      'Replication: Read-your-writes pattern, master after write',
      'Migration: Parallel across shards, test one first, monitoring per shard',
    ],
  },
  {
    id: 'sql-multi-db-q-2',
    question:
      'Explain read/write splitting with replicas. Include: (1) master-replica setup, (2) automatic routing implementation, (3) handling replication lag, (4) failover strategy, (5) monitoring. Provide complete code examples.',
    sampleAnswer:
      `Read/write splitting: (1) Setup: Master (writes), 2+ replicas (reads). ` +
      `Async replication: master→replicas (lag ~100ms). ` +
      `Configuration: master_engine = create_engine("postgresql://master"); replica_engines = [create_engine("postgresql://replica1"), create_engine("postgresql://replica2")]. ` +
      `Connection pools: master 20 connections (fewer writes), replicas 50 connections (many reads). ` +
      `(2) Automatic routing: class RoutingSession(Session): get_bind() routes SELECT to replicas (round-robin), INSERT/UPDATE/DELETE to master. ` +
      `Check clause.is_select for read vs write. ` +
      `(3) Replication lag: After write, user expects immediate read. ` +
      `Solution: read_your_writes pattern - after write, subsequent reads go to master (not replica). ` +
      `Implementation: sticky session - after write, set flag, route reads to master for 1 second. ` +
      `Alternative: check replication lag, if < 100ms use replica. ` +
      `(4) Failover: Master fails → promote replica to master. ` +
      `Automatic: Use PostgreSQL streaming replication + pgpool. ` +
      `Manual: Update connection string to new master, restart app. ` +
      `Read replica fails → remove from routing pool, continue with remaining replicas. ` +
      `(5) Monitoring: Track replication lag (pg_stat_replication), alert if > 1s. ` +
      `Track connection pool usage per database. Query latency per database (master vs replica). ` +
      `Automatic health checks: ping databases, remove unhealthy from pool. ` +
      `(6) Trade-offs: Eventual consistency - replica may be stale. Critical reads: use master. ` +
      `95% of reads: use replicas. ` +
      `Cost: 2 replicas doubles infrastructure cost, but enables 5x read throughput. ` +
      `Result: 1000 writes/sec (master), 5000 reads/sec (replicas), <100ms latency, 99.9% uptime.`,
    keyPoints: [
      'Master writes, replicas reads, async replication ~100ms lag',
      'RoutingSession: get_bind() routes SELECT→replica, INSERT/UPDATE/DELETE→master',
      'Read-your-writes: After write, read from master for 1s (sticky session)',
      'Failover: Promote replica to master, remove failed replica from pool',
      'Monitor: Replication lag, connection pools, query latency, health checks',
    ],
  },
  {
    id: 'sql-multi-db-q-3',
    question:
      'You need to implement a distributed transaction across multiple shards (e.g., transferring credits between users on different shards). Explain: (1) why this is challenging, (2) two-phase commit implementation, (3) failure scenarios, (4) alternatives to distributed transactions. Include complete code.',
    sampleAnswer:
      `Distributed transactions challenge: (1) Problem: Transfer credits from User A (shard 0) to User B (shard 1). ` +
      `Need atomic operation: both deduct and add succeed, or both fail. ` +
      `Single transaction can't span multiple databases. ` +
      `Network failures: partial commits possible (A debited, B not credited). ` +
      `(2) Two-phase commit (2PC): Phase 1 - prepare all transactions. Phase 2 - commit all or rollback all. ` +
      `Implementation: def distributed_transaction(engines, operations): connections = []; transactions = []; for engine, op in zip(engines, operations): conn = engine.connect(); txn = conn.begin_twophase(); op(conn); txn.prepare(); # Phase 1. ` +
      `Then: for txn in transactions: txn.commit(); # Phase 2. If any prepare fails: rollback all. ` +
      `(3) Failure scenarios: Network failure during commit phase → some shards committed, others not. ` +
      `Recovery: transaction log, retry or compensating transaction. ` +
      `Coordinator failure → prepared transactions left hanging. Recovery: timeout and rollback. ` +
      `Trade-off: 2PC is slow (2 round trips), can deadlock. ` +
      `(4) Alternatives: Avoid distributed transactions! Design 1: Store credits on single shard (user home shard). ` +
      `Design 2: Eventual consistency - async message queue. Debit A, send message, credit B. ` +
      `If B fails, retry or compensate. ` +
      `Design 3: Saga pattern - series of local transactions with compensating actions. ` +
      `Example: DebitA, CreditB, if CreditB fails, run CompensateDebitA. ` +
      `Design 4: Denormalize - duplicate credit balance on both shards. ` +
      `Trade-off: consistency vs availability. (5) Recommendation: For financial transactions: 2PC or saga. ` +
      `For non-critical: eventual consistency. ` +
      `For most cases: avoid cross-shard writes - design data model to keep related data together. ` +
      `Result: Reliable distributed transactions with fallback strategies.`,
    keyPoints: [
      'Challenge: Atomic operation across DBs, partial failures possible',
      '2PC: Phase 1 prepare all, Phase 2 commit all or rollback all',
      'Failures: Network failure, coordinator failure, requires recovery/retry',
      'Alternatives: Avoid cross-shard writes, eventual consistency, saga pattern',
      'Recommendation: Co-locate related data, use async for non-critical operations',
    ],
  },
];

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
      `Read replicas scale read queries by distributing load across multiple databases. ` +
      `Master handles writes (limited by single DB capacity). ` +
      `Replicas handle reads (add more replicas = more read capacity). ` +
      `Typical setup: 1 master + 2-3 replicas = 5x read throughput. ` +
      `Async replication: master→replicas (lag ~100ms). ` +
      `Use case: 90% of queries are reads - massive scaling benefit. Reads go to replicas, writes to master. ` +
      `Does not improve write performance or reduce storage (replicas duplicate data).`,
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
      `Hash-based sharding: Distribute data using hash of shard key. ` +
      `Example: shard_index = user_id % num_shards. User 5 → shard 1 (5 % 4), User 8 → shard 0 (8 % 4). ` +
      `Benefits: Even distribution (no hotspots), deterministic (same user always same shard). ` +
      `Trade-offs: Hard to rebalance (changing num_shards requires migration), range queries across shards. ` +
      `Alternative: Range-based sharding (0-10M → shard 0) - easier rebalancing but potential hotspots. ` +
      `Use hash for even distribution, range for ease of rebalancing.`,
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
      `Replication lag: Delay between write to master and replication to replica. ` +
      `Typical: 50-200ms (async replication). ` +
      `Problem: User writes to master, immediately reads from replica - data not yet replicated, user sees stale data. ` +
      `Solutions: (1) Read-your-writes - after write, read from master (not replica). ` +
      `(2) Sticky session - route user\`s reads to master for 1 second after write. (3) Check replication lag, if < 100ms use replica. Monitoring: pg_stat_replication shows lag per replica. Alert if lag > 1 second.`,
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
      'Two-phase commit (2PC): Enables atomic transactions across multiple databases so that either all changes are committed or all are rolled back.\n' +
      'Phase 1: Prepare phase—ask all databases to prepare and reserve changes. Phase 2: Commit or rollback phase—if all databases prepared successfully, commit everywhere; if any failed, roll back everywhere.\n' +
      'Example: To transfer credits from User A (shard 0) to User B (shard 1): prepare debit on shard 0 and prepare credit on shard 1. If both prepare steps succeed, commit both; if either fails, rollback both. Ensures atomicity across shards.\n' +
      'Trade-offs: Slow (requires two network round trips), can block (locks held in prepare phase), and has risk if the coordinator fails (needs good recovery/retry logic).\n' +
      'Alternative: For non-critical operations, consider eventual consistency instead of atomic distributed transactions.',
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
      'Sharding decision: For fewer than 1 million users, use a single database (this is the simplest and most maintainable setup). ' +
      'For 1–10 million users, use read replicas plus vertical scaling and caching to handle load. ' +
      'For more than 10 million users, implement horizontal sharding, as a single database will reach its limits. ' +
      'High write volume (e.g., more than 10,000 writes per second) is also a reason to shard—to distribute write load across servers. ' +
      'Geographic sharding (region-based) helps reduce latency, while multi-tenant sharding (tenant-based) improves isolation. ' +
      'Trade-offs: Sharding increases complexity—cross-shard queries, distributed transactions, and more infrastructure to manage. ' +
      'Only shard when necessary: start simple, and scale up your architecture only when justified by growth and need.',
  },
];
