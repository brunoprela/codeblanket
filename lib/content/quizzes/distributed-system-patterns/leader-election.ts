/**
 * Quiz questions for Leader Election section
 */

export const leaderelectionQuiz = [
  {
    id: 'q1',
    question:
      'Explain why leader election is necessary in distributed systems and provide three real-world scenarios where it is critical.',
    sampleAnswer:
      'Leader election is necessary to provide a single coordinator in distributed systems, avoiding conflicts and simplifying decision-making. Without a leader, multiple nodes might make conflicting decisions, leading to data inconsistency and split-brain scenarios. Three critical scenarios: (1) Database replication: A primary node must coordinate writes to ensure consistency. MongoDB uses primary election—only the primary accepts writes, secondaries replicate. If all nodes could accept writes, you might have conflicting transactions. (2) Distributed locking: A leader manages distributed locks. If multiple nodes think they can grant locks, two processes might hold the same lock simultaneously, violating mutual exclusion. Google Chubby uses leader election for lock service. (3) Job scheduling: A master node assigns tasks to workers. Hadoop ResourceManager acts as leader—it tracks cluster resources and schedules jobs. Without a leader, jobs might be assigned multiple times or not at all. The leader provides a single source of truth, making coordination possible.',
    keyPoints: [
      'Prevents conflicts: Single coordinator avoids multiple nodes making conflicting decisions',
      'Database replication: Primary node coordinates writes (MongoDB primary election)',
      'Distributed locking: Leader manages locks to prevent multiple processes holding same lock',
      'Job scheduling: Master assigns tasks, tracks resources (Hadoop ResourceManager)',
      'Single source of truth: Simplifies coordination and decision-making',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare the Raft and Bully leader election algorithms. In what scenarios would you choose one over the other?',
    sampleAnswer:
      'Raft and Bully represent different approaches to leader election with distinct trade-offs. Raft: Uses randomized timeouts and majority voting. Nodes start as followers, become candidates after timeout, request votes. Candidate with majority becomes leader. Key properties: (1) Safety—proven correct mathematically. (2) Only nodes with up-to-date logs can win. (3) Prevents split-brain via quorum. Complexity: Moderate, but understandable. Use when: Production systems requiring strong consistency (etcd, Consul, CockroachDB). Bully: Node with highest ID always wins. On leader failure, nodes send election messages to higher IDs. If no response from higher IDs, declare victory. Key properties: (1) Deterministic—always same outcome. (2) Simple to implement. (3) High message overhead (O(n²) in worst case). Use when: Small clusters (5-20 nodes) where deterministic selection is valuable and message overhead is acceptable. Scenario comparison: Raft for large-scale distributed database (need proven correctness, handle network partitions gracefully). Bully for small internal cluster with stable network where simplicity and determinism matter more than efficiency.',
    keyPoints: [
      'Raft: Randomized timeouts, majority voting, proven correct, prevents split-brain via quorum',
      'Bully: Highest ID wins, deterministic, simple but O(n²) message complexity',
      'Raft use cases: Production systems, large clusters, strong consistency (etcd, Consul)',
      'Bully use cases: Small clusters (5-20 nodes), stable networks, need deterministic leader',
      'Trade-offs: Correctness/scalability (Raft) vs simplicity/determinism (Bully)',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the split-brain problem in leader election and explain how quorum-based elections and fencing tokens prevent it.',
    sampleAnswer:
      'Split-brain occurs when a cluster divides into multiple partitions, each electing its own leader, resulting in conflicting operations and data inconsistency. Example: 5-node cluster {A,B,C,D,E}, A is leader. Network partition creates {A,B} and {C,D,E}. Without protection, both partitions might elect leaders (A stays leader, C elected in other partition). Both accept writes, causing divergence. Quorum-based prevention: Require majority (n/2+1) to elect leader. In our example, quorum is 3. Partition {A,B} has 2 nodes (no quorum), cannot elect leader. Partition {C,D,E} has 3 nodes (quorum!), elects C as leader. Only one partition can have majority, ensuring single leader. Trade-off: minority partition becomes unavailable (consistency over availability). Fencing tokens: Each leader gets monotonically increasing token (generation number). All operations include token. Resources reject operations from old tokens. Example: A has token=1, C elected with token=2. Both try to write to database. Database sees token=1 from A (reject—old), token=2 from C (accept—newer). Even if both partitions elected leaders, only the newest can operate. Combination: Quorum prevents election in minority, fencing prevents stale leaders from causing damage if quorum mechanism fails.',
    keyPoints: [
      'Split-brain: Multiple partitions each elect a leader, conflicting operations, data inconsistency',
      'Quorum-based election: Require majority (n/2+1) to elect, only one partition can have majority',
      'Minority partition unavailable: Trade-off consistency (no split-brain) for availability',
      'Fencing tokens: Monotonically increasing tokens, resources reject old tokens',
      'Defense in depth: Quorum prevents split-brain election, fencing prevents stale leader damage',
    ],
  },
];
