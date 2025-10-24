/**
 * Quiz questions for Redis Deep Dive section
 */

export const redisQuiz = [
  {
    id: 'q1',
    question:
      'Compare RDB vs AOF persistence in Redis. When would you use each or both?',
    sampleAnswer:
      "RDB (Redis Database): Point-in-time snapshots. BGSAVE creates dump.rdb file (forked process, non-blocking). Configure auto-save: save 900 1 (save if 1 key changed in 900 sec). Benefits: (1) Compact single file. (2) Fast restarts. (3) Good for backups. Drawbacks: (1) Can lose data between snapshots. (2) Fork can cause latency spike on large datasets. (3) All-or-nothing (no incremental). AOF (Append Only File): Logs every write operation. appendfsync everysec (fsync every second). Benefits: (1) More durable (lose at most 1 sec). (2) Append-only (no corruption). (3) Human-readable. Drawbacks: (1) Larger files. (2) Slower restarts. (3) Can be slower than RDB. Use RDB when: Acceptable to lose minutes of data, fast restarts critical, backups. Use AOF when: Durability critical, can't lose data. Use both (hybrid): RDB for fast restarts + AOF for durability. Redis 7+ default: hybrid. Best practice: Enable both, use AOF everysec (balance durability and performance).",
    keyPoints: [
      'RDB: Snapshots, fast restarts, can lose data between saves',
      'AOF: Log writes, more durable (lose max 1 sec), slower restarts',
      'Hybrid (both): Fast restarts + durability',
      'Best practice: Enable both, AOF everysec',
      'Trade-off: Durability vs performance vs file size',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain how to implement a distributed lock with Redis. Why is it tricky and what are the pitfalls?',
    sampleAnswer:
      'Simple distributed lock: SET lock:resource unique_id NX EX 30 (set if not exists, expire in 30 sec). Acquire succeeds if returns OK. Release: DEL lock:resource (must verify unique_id matches via Lua script for atomicity). Pitfalls: (1) Clock skew - if server clock jumps, lock may expire early. (2) Process pause - if process pauses (GC), lock expires, another process acquires lock → two processes in critical section! (3) Non-atomic release - if DEL without checking unique_id, wrong process can release. (4) Redis failure - master fails, replica promoted without lock → split-brain. Solution: (1) Use unique_id (UUID) to prevent wrong process releasing. (2) Use Lua script for atomic check-and-release. (3) Set reasonable timeout. (4) Use Redlock algorithm (Martin Kleppmann criticized it). (5) Better: Use ZooKeeper/etcd for critical locks (stronger consistency). Example: SET lock NX EX 30. If acquired: try { critical section } finally { Lua: if GET(lock)==my_id then DEL(lock) }. Distributed locks are hard - many edge cases. Redis works for non-critical locks. For critical (financial), use Chubby/ZooKeeper/etcd.',
    keyPoints: [
      'SET lock NX EX timeout (acquire), Lua script to release',
      'Pitfalls: Clock skew, process pause, Redis failover',
      'Use unique_id to prevent wrong process releasing',
      'Atomic check-and-release via Lua',
      'For critical locks: Use ZooKeeper/etcd (stronger guarantees)',
    ],
  },
  {
    id: 'q3',
    question:
      'How does Redis Cluster achieve horizontal scaling? What are the limitations?',
    sampleAnswer:
      "Redis Cluster: Automatic sharding across nodes. 16,384 hash slots. key → CRC16(key) % 16384 → slot → node. Example: Node1 (slots 0-5460), Node2 (5461-10922), Node3 (10923-16383). Each master has replicas. Automatic failover via gossip protocol. Benefits: (1) Horizontal scaling (add nodes). (2) No proxy (clients connect directly). (3) Automatic sharding. (4) High availability (replica promotion). Limitations: (1) Multi-key ops limited - keys must be on same node (use hash tags: {user123}:key1, {user123}:key2). (2) No database select (only db 0). (3) Client must be cluster-aware. (4) More complex than master-replica. (5) Resharding requires manual trigger. (6) Transaction limitations (MULTI/EXEC only on same slot). Hash tags: Force keys to same slot. Example: user:{123}:profile, user:{123}:settings → both hash on '123', same node. Operations: SET, GET work across cluster. MGET, MSET require same slot. Lua scripts require same slot. Redis Cluster good for: Scaling beyond single machine. Not good for: Multi-key operations across nodes.",
    keyPoints: [
      'Redis Cluster: Automatic sharding, 16,384 hash slots',
      'Each key → hash slot → node',
      'Benefits: Horizontal scaling, no proxy, auto-failover',
      'Limitations: Multi-key ops require same node (hash tags)',
      'Use hash tags to co-locate related keys',
    ],
  },
];
