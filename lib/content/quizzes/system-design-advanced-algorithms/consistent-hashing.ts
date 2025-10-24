/**
 * Quiz questions for Consistent Hashing section
 */

export const consistenthashingQuiz = [
  {
    id: 'q1',
    question:
      'Explain why modulo hashing fails when scaling distributed systems and how consistent hashing solves this problem. Use a concrete example.',
    sampleAnswer:
      'MODULO HASHING FAILURE: With modulo hashing, server = hash(key) % N. Problem: When N changes (add/remove server), MOST keys remap to different servers. Example: 3 servers → 4 servers. Key "user:1" with hash=8473: Before: 8473 % 3 = 2 (Server 2). After: 8473 % 4 = 1 (Server 1). MOVED! Formula: (n/(n+1)) keys remap. 3→4 servers: 75% of keys move. Impact: 75% cache miss rate, need to migrate 75% of data, database gets hammered, potential outage. CONSISTENT HASHING SOLUTION: Map servers AND keys to points on a ring (0 to 2³²-1). Each key stored on first server clockwise from its position. When adding Server 4: Only keys between new server and previous server remap (≈25% for 4 servers, exactly 1/N on average). When removing server: Only its keys remap to next server. This is the fundamental difference: Modulo hashing remaps nearly ALL keys, consistent hashing remaps only 1/N keys. For large scale systems (1000+ servers, billions of keys), this is the difference between downtime and seamless scaling.',
    keyPoints: [
      'Modulo hashing remaps ~100% of keys when server count changes',
      'Formula: (n/(n+1)) keys affected, e.g., 99% for 100→101 servers',
      'Consistent hashing remaps only ~1/N keys (average)',
      'Hash ring: keys map to first server clockwise',
      'Critical for zero-downtime scaling in production systems',
    ],
  },
  {
    id: 'q2',
    question:
      'Virtual nodes solve the load balancing problem in consistent hashing. Explain why simple consistent hashing has uneven load distribution and how virtual nodes fix it. What is the optimal number of virtual nodes?',
    sampleAnswer:
      'PROBLEM WITH SIMPLE CONSISTENT HASHING: With 3 physical servers placed randomly on ring, one server might get 40% of keys while others get 30% each (33% variance). Why? Random hash placement can cluster servers together, creating uneven arc lengths on ring. Worse case: 2 servers cluster near position 0, third server at position 2³¹ → third server gets 50% of load! VIRTUAL NODES SOLUTION: Instead of placing each physical server once, place it 100-200 times with different hash values (Server A-1, Server A-2, ..., Server A-100). This distributes each physical server evenly around the ring. Benefits: (1) Statistical averaging: 100 random placements converge to even distribution (law of large numbers). (2) With 150 vnodes, load variance reduces to <5%. (3) Heterogeneous hardware: Assign 200 vnodes to powerful server, 100 to weak server (proportional load). (4) Smoother rebalancing: When adding server, data migrates from ALL servers (not just one neighbor). OPTIMAL NUMBER: Typically 100-200 vnodes per server. Trade-off: More vnodes = better distribution but more memory/CPU for ring lookups. Cassandra default: 256 vnodes. Amazon DynamoDB: 100-200 vnodes. Diminishing returns beyond 150-200. With 100 vnodes, probability of >10% load imbalance is <1%.',
    keyPoints: [
      'Simple hashing: random placement causes uneven arc lengths (30-40% variance)',
      'Virtual nodes: Place each server 100-200 times on ring',
      'Statistical averaging ensures ~equal load distribution (<5% variance)',
      'Enables weighted distribution for heterogeneous hardware',
      'Optimal: 100-200 vnodes per server (Cassandra: 256, DynamoDB: 100-200)',
    ],
  },
  {
    id: 'q3',
    question:
      'Design the data replication strategy for a distributed key-value store using consistent hashing. How do you ensure data survives server failures? Walk through a concrete scenario.',
    sampleAnswer:
      'REPLICATION STRATEGY: Store each key on R consecutive servers clockwise on the ring (typical: R=3 for fault tolerance). Implementation: hash(key) gives position on ring, store on (1) first server clockwise (primary), (2) second server clockwise (replica 1), (3) third server clockwise (replica 2). CONCRETE EXAMPLE: Ring positions: Server A (20M), Server B (50M), Server C (80M). Key "user:123" hashes to 35M. Store on: (1) Server B (50M) - primary, (2) Server C (80M) - replica 1, (3) Server A (20M) - replica 2 (wraps around). READ STRATEGY: Quorum reads: Read from 2 of 3 replicas, return most recent (based on timestamp/version). If Server B fails, read from C and A. WRITE STRATEGY: Synchronous: Write to 2 of 3 replicas before acknowledging (W=2, N=3). Asynchronous: Write to primary, replicate in background (faster but eventual consistency). SERVER FAILURE SCENARIO: Server B fails at 10:00 AM. Immediately: All reads for keys on Server B route to replicas (C and A). No data loss! No downtime! Background: Cluster detects failure via heartbeat/gossip. Triggers repair process: identify keys that should have 3 replicas but now have 2. Re-replicate from C/A to other servers. Within minutes, replication factor restored to 3. This is why DynamoDB and Cassandra use R=3: Can tolerate 1 server failure with zero downtime and zero data loss. This is consistent hashing + replication working together.',
    keyPoints: [
      'Store each key on R consecutive servers clockwise (typical R=3)',
      'Primary + 2 replicas for fault tolerance',
      'Read/write quorums ensure consistency despite failures',
      'Server failure: immediately read from replicas, no downtime',
      'Background repair restores replication factor',
    ],
  },
];
